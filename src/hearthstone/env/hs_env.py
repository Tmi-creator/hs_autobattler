from __future__ import annotations

import math
import random
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Dict, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from hearthstone.engine.configs import CARD_DB, SPELL_DB
from hearthstone.engine.card_def import TRIGGER_REGISTRY
from hearthstone.engine.cpp_bridge import CARD_ID_MAP, TAG_TO_BIT, TYPE_TO_BIT, get_cpp_engine
from hearthstone.engine.entities import HandCard, Player, Spell, StoreItem, Unit
from hearthstone.engine.enums import UnitType
from hearthstone.engine.event_system import EventType
from hearthstone.engine.game import Game
from hearthstone.engine.spells import SPELLS_REQUIRE_TARGET
from hearthstone.env.es_bot import es_bot_turn
from hearthstone.env.ghost_pool import BoardSnapshot, GhostPool
from hearthstone.env.smart_bot import smart_bot_turn

if TYPE_CHECKING:
    from sb3_contrib import MaskablePPO

# Normalization constants
MAX_ATK = 100.0
MAX_HP = 100.0
MAX_GOLD = 30.0
MAX_TIER = 6.0
MAX_COST = 10.0
MAX_SPELL_DISCOUNT = 10.0
MAX_CARDS_IN_GAME = 500


def count_goldens(player: Player) -> int:
    cnt = 0
    for u in player.board:
        if u.is_golden:
            cnt += 1
    for hc in player.hand:
        if hc.unit and hc.unit.is_golden:
            cnt += 1
    return cnt


def get_minion_types_composition(player: Player, all_types: list[UnitType]) -> list[float]:
    counts = [0.0] * len(all_types)
    for u in player.board:
        for i, t in enumerate(all_types):
            if t in u.types:
                counts[i] += 1.0
    return counts


class HearthstoneEnv(gym.Env[np.ndarray, int]):
    """
    RL Environment for Hearthstone Battlegrounds.

    Action Space (Discrete 32):
    0: End Turn
    1: Roll
    2-8: Buy (Slot 0-6) / DISCOVER_CHOICE (Option 0-2) / TARGET BOARD (Slot 0-6)
    9-15: Sell (Slot 0-6) / TARGET STORE (Slot 0-6)
    [Not implemented in engine yet, usually spells target board]
    16-25: Play Hand (Card 0-9)
    26-31: Swap Right (Slot i <-> Slot i+1)
    26: 0<->1, 27: 1<->2 ... 31: 5<->6
    """

    def __init__(
        self,
        max_tier: int = 6,
        auto_position: bool = True,
        use_oracle_reward: bool = False,
        oracle_reward_scale: float = 10.0,
        use_enemy_board_obs: bool = False,
        use_player_status_obs: bool = False,
    ) -> None:
        super(HearthstoneEnv, self).__init__()

        self._max_tier = max_tier
        self.auto_position = auto_position
        self.use_oracle_reward = use_oracle_reward
        self.oracle_reward_scale = oracle_reward_scale
        self.use_enemy_board_obs = use_enemy_board_obs
        self.use_player_status_obs = use_player_status_obs

        all_ids = sorted(list(CARD_DB.keys()) + list(SPELL_DB.keys()))

        self.static_id_map = {cid: i + 1 for i, cid in enumerate(all_ids)}
        self.num_card_ids = len(all_ids) + 1  # +1 for padding id 0

        self.game = Game(max_tier=max_tier)
        self.my_player_id = 0
        self.enemy_id = 1
        self.max_steps_per_episode = 500
        self.steps_taken = 0

        self.actions_in_turn = 0
        self.max_actions_in_turn = 30
        # Action Space: 0=END, 1=ROLL, 2-8=BUY, 9-15=SELL, 16-25=PLAY, 26-31=SWAP, 32=UPGRADE, 33=FREEZE
        self.action_space = spaces.Discrete(34)

        self.opponent_model: Optional["MaskablePPO"] = None

        # ES-bot opponent (parametric heuristic)
        self._es_bot_weights: Optional[np.ndarray] = None

        # Ghost self-play
        self.ghost_pool: Optional[GhostPool] = None
        self._ghost_trajectory: Optional[Dict[int, BoardSnapshot]] = None
        self._use_ghost: bool = False
        self._ghost_ratio: float = 0.8  # probability of using ghost vs bot
        self._env_id: int = id(self)  # unique per env instance

        # MC Oracle: C++ engine as dense reward oracle
        self._oracle_n_combats: int = 20
        self._oracle_cached_wr: float = 0.5
        self._oracle_seed: int = random.getrandbits(32)
        self._oracle_ghost_flat: list | None = None  # cached C++ flat list for ghost board
        self._oracle_ghost_tier: int = 1

        self.all_types = list(UnitType)
        self.num_types = len(self.all_types)  # 12

        # --- Вектор сущности (35 float) ---
        # [0] Is Present
        # [1] Is Spell
        # [2] Card ID (Norm)
        # [3] Cost
        # [4] Tier
        # [5] Is Frozen
        # [6] ATK
        # [7] HP
        # [8..16] Keywords (Taunt, DS, WF, Poison, Venom, Reborn, Cleave,
        #         Immediate attack, Magnetic)
        # [17] Is Golden
        # [18] Is Token
        # [19] Has Deathrattle (Native)
        # [20] Has Battlecry (Play Effect)
        # [21] Has End of Turn
        # [22] Has Start of Combat
        # [23] Has Sell Effect
        # [24] Has Synergy (Engine)
        # [25] Is Selected (Targeting Source)
        # [26..37] Types (11)

        self.entity_features = 26 + self.num_types

        # Размер вектора наблюдения:
        # Global(7) + Board(7*38) + Hand(10*38) + Store(7*38) + Discover(3*38) + Enemy(3) = 1036
        total_obs_size = (
            7
            + (7 * self.entity_features)
            + (10 * self.entity_features)
            + (7 * self.entity_features)
            + (3 * self.entity_features)
            + 3
        )

        if self.use_enemy_board_obs:
            total_obs_size += 7 * self.entity_features

        if self.use_player_status_obs:
            total_obs_size += 2 * 32  # 2 players * 32 status features

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(total_obs_size,), dtype=np.float32
        )

        # Pre-allocated buffers (avoid per-step allocations)
        self._obs_buffer = np.zeros(total_obs_size, dtype=np.float32)
        self._mask_buffer = np.zeros(34, dtype=np.bool_)
        self._obs_size = total_obs_size

        # Pre-compute zone offsets for fast obs writing
        self._off_global = 0
        self._off_board = 7
        self._off_hand = 7 + 7 * self.entity_features
        self._off_store = 7 + (7 + 10) * self.entity_features
        self._off_discover = 7 + (7 + 10 + 7) * self.entity_features
        self._off_enemy = 7 + (7 + 10 + 7 + 3) * self.entity_features

        next_offset = 7 + (7 + 10 + 7 + 3) * self.entity_features + 3

        if self.use_enemy_board_obs:
            self._off_enemy_board = next_offset
            next_offset += 7 * self.entity_features
        else:
            self._off_enemy_board = -1

        if self.use_player_status_obs:
            self._off_player_status = next_offset
            next_offset += 64
        else:
            self._off_player_status = -1

        # Pre-compute trigger info per card_id (avoids per-entity lookup)
        self._trigger_cache: dict[str, tuple[bool, bool, bool, bool, bool]] = {}
        for cid, triggers in TRIGGER_REGISTRY.items():
            bc = eot = soc = sell = syn = False
            for trig_def in triggers:
                evt = trig_def.event_type
                cond_name = trig_def.condition.__name__
                if evt == EventType.MINION_PLAYED:
                    if cond_name == "_is_self_play":
                        bc = True
                    else:
                        syn = True
                elif evt == EventType.MINION_SUMMONED:
                    syn = True
                elif evt == EventType.END_OF_TURN:
                    eot = True
                elif evt == EventType.START_OF_COMBAT:
                    soc = True
                elif evt == EventType.MINION_SOLD:
                    sell = True
            self._trigger_cache[cid] = (bc, eot, soc, sell, syn)
        self._default_triggers = (False, False, False, False, False)

        self.is_targeting: bool = False
        self.pending_spell_hand_index: Optional[int] = None
        self.pending_target_kind: Optional[str] = None  # "SPELL" | "MAGNETIZE"

    def set_ghost_pool(self, pool: GhostPool) -> None:
        """Set shared ghost pool (called once at env creation)."""
        self.ghost_pool = pool

    def get_ghost_trajectories(self) -> list:
        """Get copy of trajectories from the local ghost pool."""
        if self.ghost_pool is not None:
            return list(self.ghost_pool.trajectories)
        return []

    def set_ghost_trajectories(self, trajectories: list) -> None:
        """Update local ghost pool with trajectories from main process."""
        if self.ghost_pool is not None:
            self.ghost_pool.trajectories.clear()
            for traj in trajectories:
                self.ghost_pool.trajectories.append(traj)

    def enable_ghost_mode(self) -> None:
        """Enable ghost self-play (called by CurriculumCallback)."""
        self._use_ghost = True

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, object] | None = None,
    ) -> tuple[np.ndarray, dict[str, object]]:
        super().reset(seed=seed)

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Finish previous game's ghost recording
        if self.ghost_pool is not None:
            self.ghost_pool.finish_game(self._env_id)
            self.ghost_pool.finish_game(self._env_id + 1_000_000)  # bot

        self.game = Game(max_tier=self._max_tier)

        self.steps_taken = 0
        self.actions_in_turn = 0
        self.is_targeting = False
        self.pending_spell_hand_index = None

        # History & Player Status Tracking
        self.last_combat_board = [[], []]
        self._triples = [0] * len(self.game.players)
        self._streak = [0] * len(self.game.players)
        self._upgrade_turns = [[0.0] * 5 for _ in range(len(self.game.players))]
        self._cumulative_types = [[0.0] * len(self.all_types) for _ in range(len(self.game.players))]
        self._last_goldens = [count_goldens(p) for p in self.game.players]
        self._last_tier = [p.tavern_tier for p in self.game.players]

        # Statistics trackers
        self.stats_reward = 0.0
        self.stats_spells_bought = 0
        self.stats_spells_played = 0
        self.stats_minions_bought = 0
        self.stats_minions_sold = 0
        self.stats_rolls = 0
        self.stats_upgrades = 0

        # Sample ghost trajectory for this episode
        self._ghost_trajectory = None
        if (
            self._use_ghost
            and self.ghost_pool is not None
            and self.ghost_pool.size > 0
            and random.random() < self._ghost_ratio
        ):
            self._ghost_trajectory = self.ghost_pool.sample_trajectory()

        # Reset MC Oracle state
        self._oracle_cached_wr = 0.5
        self._oracle_seed = random.getrandbits(32)
        self._oracle_ghost_flat = None

        return self._get_obs(), {}

    def set_opponent(self, model: MaskablePPO) -> None:
        self.opponent_model = model

    def set_es_bot(self, weights: np.ndarray) -> None:
        """Use a parametric ES bot as the enemy (priority below neural opponent)."""
        self._es_bot_weights = weights.astype(np.float32)

    def get_board_power(self) -> float:
        """Returns current board power for the agent's player."""
        player = self.game.players[self.my_player_id]
        return self._calculate_board_power(player)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]:
        self.steps_taken += 1
        self.actions_in_turn += 1
        truncated = (self.game.turn_count > 50) or (self.steps_taken >= self.max_steps_per_episode)

        player = self.game.players[self.my_player_id]

        def make_transition_result(obs, rew, is_done, is_trunc):
            self.stats_reward += rew
            inf = {}
            if is_done or is_trunc:
                p0_hp = player.health
                p1_hp = self.game.players[self.enemy_id].health
                if p0_hp > 0 and p1_hp <= 0:
                    win = 1
                elif p0_hp <= 0 and p1_hp > 0:
                    win = -1
                else:
                    win = 0

                inf["episode_stats"] = {
                    "reward": self.stats_reward,
                    "length": self.steps_taken,
                    "turns": self.game.turn_count,
                    "win": win,
                    "final_hp": p0_hp,
                    "enemy_final_hp": p1_hp,
                    "final_tier": player.tavern_tier,
                    "final_board_power": self._calculate_board_power(player),
                    "spells_bought": self.stats_spells_bought,
                    "spells_played": self.stats_spells_played,
                    "minions_bought": self.stats_minions_bought,
                    "minions_sold": self.stats_minions_sold,
                    "rolls": self.stats_rolls,
                    "upgrades": self.stats_upgrades,
                    "use_ghost": int(self._ghost_trajectory is not None)
                }
            return obs, rew, is_done, is_trunc, inf

        is_discovering = player.is_discovering
        # === ACTION MAPPING ===
        action_type: str = "UNKNOWN"
        kwargs: dict[str, int] = {}
        done = False

        if is_discovering:
            if 2 <= action <= 4:
                action_type = "DISCOVER_CHOICE"
                kwargs["index"] = action - 2
            else:
                action_type = "INVALID_DURING_DISCOVERY"

        elif self.is_targeting:
            if 2 <= action <= 8:
                action_type = "PLAY"
                kwargs["hand_index"] = (
                    self.pending_spell_hand_index
                    if self.pending_spell_hand_index is not None
                    else 0
                )
                kwargs["target_index"] = action - 2
                self.pending_spell_hand_index = None
                self.is_targeting = False
                self.pending_target_kind = None
            elif action == 0:
                action_type = "CANCEL_CAST"
                self.pending_spell_hand_index = None
                self.is_targeting = False
                self.pending_target_kind = None
            else:
                action_type = "INVALID_NEED_TARGET"
        else:
            if action == 0:
                action_type = "END_TURN"
            elif action == 1:
                action_type = "ROLL"
            elif 2 <= action <= 8:
                action_type = "BUY"
                kwargs["index"] = action - 2
            elif 9 <= action <= 15:
                action_type = "SELL"
                kwargs["index"] = action - 9
            elif 16 <= action <= 25:
                h_idx = action - 16
                if h_idx < len(player.hand):
                    card = player.hand[h_idx]
                    card_id = (
                        card.spell.card_id
                        if card.spell
                        else (card.unit.card_id if card.unit else None)
                    )
                    if card_id in SPELLS_REQUIRE_TARGET:
                        self.pending_spell_hand_index = h_idx
                        self.is_targeting = True
                        action_type = "WAIT_FOR_TARGET"
                    elif card.unit and card.unit.has_magnetic:
                        has_mech = any(UnitType.MECH in u.types for u in player.board)
                        if has_mech:
                            self.is_targeting = True
                            self.pending_target_kind = "MAGNETIZE"
                            self.pending_spell_hand_index = h_idx
                            action_type = "WAIT_FOR_TARGET"
                        else:
                            action_type = "PLAY"
                            kwargs["hand_index"] = h_idx
                            kwargs["insert_index"] = -1
                    else:
                        action_type = "PLAY"
                        kwargs["hand_index"] = h_idx
                        kwargs["insert_index"] = -1
                else:
                    action_type = "INVALID_HAND_INDEX"
            elif 26 <= action <= 31:
                action_type = "SWAP"
                idx_a = action - 26
                idx_b = idx_a + 1
                kwargs["index_a"] = idx_a
                kwargs["index_b"] = idx_b
            elif action == 32:
                action_type = "UPGRADE"
            elif action == 33:
                self.game.step(self.my_player_id, "FREEZE")
                action_type = "END_TURN"

        p0_hp_before = player.health
        p1_hp_before = self.game.players[self.enemy_id].health

        # Engine run

        # Pre-track properties for statistics
        bought_spell = False
        bought_minion = False
        played_spell = False

        if action_type == "BUY":
            idx = kwargs.get("index", -1)
            if 0 <= idx < len(player.store):
                item = player.store[idx]
                if item.spell:
                    bought_spell = True
                elif item.unit:
                    bought_minion = True
        elif action_type == "PLAY":
            idx = kwargs.get("hand_index", -1)
            if 0 <= idx < len(player.hand):
                card = player.hand[idx]
                if card.spell:
                    played_spell = True

        if action_type == "WAIT_FOR_TARGET":
            return make_transition_result(self._get_obs(), 0.0, False, truncated)

        elif action_type == "CANCEL_CAST":
            return make_transition_result(self._get_obs(), 0.0, False, truncated)

        success, done, _ = self.game.step(self.my_player_id, action_type, **kwargs)

        if not success:
            return make_transition_result(self._get_obs(), 0.0, self.game.game_over, truncated)

        # Update triples & upgrades on success
        for p_idx, p in enumerate(self.game.players):
            current_goldens = count_goldens(p)
            diff = current_goldens - self._last_goldens[p_idx]
            if diff > 0:
                self._triples[p_idx] += diff
            self._last_goldens[p_idx] = current_goldens

            current_tier = p.tavern_tier
            last_t = self._last_tier[p_idx]
            if current_tier > last_t:
                for t in range(last_t + 1, current_tier + 1):
                    if 2 <= t <= 6:
                        self._upgrade_turns[p_idx][t - 2] = float(self.game.turn_count)
                self._last_tier[p_idx] = current_tier

        # Update stats on success
        if bought_spell:
            self.stats_spells_bought += 1
        if bought_minion:
            self.stats_minions_bought += 1
        if played_spell:
            self.stats_spells_played += 1
        if action_type == "SELL":
            self.stats_minions_sold += 1
        if action_type == "ROLL":
            self.stats_rolls += 1
        if action_type == "UPGRADE":
            self.stats_upgrades += 1

        # === REWARD: Round Outcome + Action Penalty + Terminal ===
        reward: float = -0.005  # action penalty

        if self.use_oracle_reward and action_type != "END_TURN":
            reward += self._oracle_reward(player)

        if action_type == "END_TURN":
            reward = 0.0  # END_TURN itself is free
            self.actions_in_turn = 0

            if self.auto_position:
                self._auto_position_board(player)

            if self.ghost_pool is not None:
                self.ghost_pool.record_turn(self._env_id, self.game.turn_count, player)

            self._play_enemy_turn()
            done = self.game.game_over

            p0_hp_after = self.game.players[self.my_player_id].health
            p1_hp_after = self.game.players[self.enemy_id].health

            damage_dealt = p1_hp_before - p1_hp_after
            damage_taken = p0_hp_before - p0_hp_after

            # Save the board snapshots right after play_enemy_turn completes (combat resolved, board is pre-combat since combat copies were used)
            self.last_combat_board = [
                [u.combat_copy() for u in p.board]
                for p in self.game.players
            ]

            # Update cumulative minion types
            for p_idx, p in enumerate(self.game.players):
                comp = get_minion_types_composition(p, self.all_types)
                for i in range(len(self.all_types)):
                    self._cumulative_types[p_idx][i] += comp[i]

            # Update streaks for both players
            if damage_dealt > damage_taken:  # Player 0 won
                if self._streak[0] > 0:
                    self._streak[0] += 1
                else:
                    self._streak[0] = 1

                if self._streak[1] < 0:
                    self._streak[1] -= 1
                else:
                    self._streak[1] = -1
            elif damage_taken > damage_dealt:  # Player 1 won
                if self._streak[1] > 0:
                    self._streak[1] += 1
                else:
                    self._streak[1] = 1

                if self._streak[0] < 0:
                    self._streak[0] -= 1
                else:
                    self._streak[0] = -1
            else:  # Draw
                self._streak[0] = 0
                self._streak[1] = 0

            # Round outcome: +1/-1
            if damage_dealt > damage_taken:
                reward += 1.0
            elif damage_taken > damage_dealt:
                reward -= 1.0

            # Terminal: game win/loss
            if done:
                if p0_hp_after > 0:
                    reward += 100.0
                else:
                    reward -= 100.0

            # Prepare oracle for next turn (cache ghost board)
            if not done:
                self._oracle_prepare_ghost()
                self._oracle_cached_wr = self._oracle_eval_winrate(player)

        return make_transition_result(self._get_obs(), reward, done, truncated)

    def _auto_position_board(self, player: Player) -> None:
        """
        Эвристика для авто-расстановки (так как SWAP отключен):
        1. Клив (Cleave) - бьет первым (максимальный вэлью).
        2. Яд/Токсичность - чтобы убить жирного таунта врага.
        3. Божественный щит + Высокая атака.
        4. Просто высокая атака.
        5. В самом конце (справа) - слабые юниты и таунты (если это "стенки").
        """
        if not player.board:
            return

        def sort_key(unit: Unit) -> int:
            score: int = 0
            # Приоритеты (чем выше score, тем левее стоит юнит)
            if unit.has_cleave:
                score += 10000
            if unit.has_poisonous or unit.has_venomous:
                score += 5000
            if unit.has_divine_shield:
                score += 1000

            # Сортируем по атаке (сильные бьют раньше)
            score += unit.cur_atk

            if unit.has_taunt and unit.cur_atk < 5:
                score -= 2000

            return score

        player.board.sort(key=sort_key, reverse=True)

    def _calculate_board_power(self, player: Player) -> float:
        power: float = 0.0
        for unit in player.board:
            u_score = unit.cur_atk * 1.0 + unit.cur_hp * 0.8

            # === MODIFIERS ===

            # 2. Божественный щит
            # Щит дает возможность нанести урон еще раз безнаказанно.
            # Ценность = Атака юнита (он ударит лишний раз) + Немного выживаемости
            if unit.has_divine_shield:
                u_score += (unit.cur_atk * 1.0) + 5.0

            # 3. Яд / Токсичность
            if unit.has_poisonous or unit.has_venomous:
                poison_value = 30.0
                if unit.cur_atk < poison_value:
                    u_score += poison_value - unit.cur_atk

            # 4. Неистовство ветра (Windfury)
            # Потенциально удваивает атаку. Но юнит может умереть после первого удара.
            # Оцениваем как +70% к атаке.
            if unit.has_windfury:
                u_score += unit.cur_atk * 0.7

            # 5. Перерождение (Reborn)
            # Это еще одна тушка с 1 ХП и той же атакой (если нежить).
            if unit.has_reborn:
                u_score += (unit.base_atk * 0.8) + 1.0

            # 6. Клив (Cleave)
            # Бьет троих. Ценность атаки утраивается? Нет, но x2 точно.
            if unit.has_cleave:
                u_score += unit.cur_atk * 1.0

            power += math.sqrt(u_score)

        return power

    # ================================================================
    # MC Oracle: use C++ combat engine as dense reward signal (PBRS)
    # ================================================================

    def _board_to_flat(self, board: List[Unit]) -> list[int]:
        """Convert a list of Units to a flat list of integers for C++ bindings."""
        flat = []
        for u in board:
            cpp_types = 0
            for t in u.types:
                cpp_types |= TYPE_TO_BIT.get(t, 0)
            cpp_tags = 0
            for tag in u.tags:
                cpp_tags |= TAG_TO_BIT.get(tag, 0)
            flat.extend([
                CARD_ID_MAP.get(u.card_id, 0),
                u.cur_atk,
                u.cur_hp,
                cpp_types,
                cpp_tags,
                u.tier,
                int(u.is_golden)
            ])
        return flat

    def _oracle_prepare_ghost(self) -> None:
        """Cache flat representation of the current ghost board (called once per turn)."""
        enemy = self.game.players[self.enemy_id]
        if enemy.board:
            self._oracle_ghost_flat = self._board_to_flat(enemy.board)
            self._oracle_ghost_tier = enemy.tavern_tier
        else:
            self._oracle_ghost_flat = None

    def _oracle_eval_winrate(self, player: Player) -> float:
        """Run N combats via C++ engine using flat lists, return winrate [0, 1]."""
        cpp = get_cpp_engine()
        if cpp is None or not player.board or self._oracle_ghost_flat is None:
            return 0.5

        side0 = self._board_to_flat(player.board)
        results = cpp.fast_combat_batch_flat(
            side0, self._oracle_ghost_flat,
            self._oracle_seed, self._oracle_n_combats,
            tavern_tier_0=player.tavern_tier,
            tavern_tier_1=self._oracle_ghost_tier,
        )
        self._oracle_seed += self._oracle_n_combats

        wins = sum(1 for outcome, _ in results if outcome == 2)  # 2 = WIN for side0
        return wins / len(results)

    def _oracle_reward(self, player: Player) -> float:
        """Compute PBRS reward: delta winrate after action × scale."""
        wr_after = self._oracle_eval_winrate(player)
        delta = wr_after - self._oracle_cached_wr
        self._oracle_cached_wr = wr_after
        return delta * self.oracle_reward_scale

    def _play_enemy_turn(self) -> None:
        p_idx = self.enemy_id
        enemy = self.game.players[p_idx]

        # === GHOST MODE: replay historical board ===
        if self._ghost_trajectory is not None:
            turn = self.game.turn_count
            snap = self._ghost_trajectory.get(turn)
            if snap is not None:
                uid_fn = self.game.tavern.get_next_uid
                GhostPool.materialize_board(snap, enemy, uid_fn)
            # else: keep whatever board enemy has (stale from last turn)
            # Mark enemy as ready so combat triggers
            self.game.step(p_idx, "END_TURN")
            return

        # === NEURAL SELF-PLAY (legacy, kept as fallback) ===
        if self.opponent_model is not None:
            self._neural_enemy_turn(p_idx)
            return

        # === ES BOT (parametric heuristic) ===
        if self._es_bot_weights is not None:
            es_bot_turn(self.game, p_idx, self._es_bot_weights)
            if self.ghost_pool is not None:
                self.ghost_pool.record_turn(
                    self._env_id + 1_000_000,
                    self.game.turn_count,
                    enemy,
                )
            return

        # === SMART BOT (default) ===
        smart_bot_turn(self.game, p_idx)

        # Record bot's board too — bots build decent boards from step 0,
        # so ghost pool gets quality data immediately for faster ramp-up.
        if self.ghost_pool is not None:
            self.ghost_pool.record_turn(
                self._env_id + 1_000_000,  # separate namespace
                self.game.turn_count,
                enemy,
            )

    def _neural_enemy_turn(self, p_idx: int) -> None:
        """Legacy neural self-play. Kept as fallback."""
        player = self.game.players[p_idx]
        self.is_targeting = False
        self.pending_spell_hand_index = None

        max_actions = 30

        for _ in range(max_actions):
            obs = self._get_obs(player_idx=p_idx)
            masks = self.action_masks(player_idx=p_idx)

            import torch

            with torch.no_grad():
                raw_action, _ = self.opponent_model.predict(
                    obs, action_masks=masks, deterministic=False
                )
            action: int = int(raw_action)

            if action == 0:
                if self.is_targeting:
                    self.is_targeting = False
                    self.pending_spell_hand_index = None
                    continue
                else:
                    break

            if self.is_targeting:
                if 2 <= action <= 8:
                    target_idx = action - 2
                    hand_idx: int | None = self.pending_spell_hand_index
                    self.game.step(
                        p_idx,
                        "PLAY",
                        hand_index=hand_idx,
                        target_index=target_idx,
                    )
                    self.is_targeting = False
                    self.pending_spell_hand_index = None
                continue

            action_type, kwargs = self._decode_action_for_engine(action)

            if action_type == "PLAY":
                h_idx = kwargs.get("hand_index")
                if h_idx is not None and h_idx < len(player.hand):
                    card = player.hand[h_idx]
                    card_id: str | None = (
                        card.spell.card_id
                        if card.spell
                        else (card.unit.card_id if card.unit else None)
                    )
                    if card_id in SPELLS_REQUIRE_TARGET:
                        self.pending_spell_hand_index = h_idx
                        self.is_targeting = True
                        self.pending_target_kind = "SPELL"
                        continue
                    if card.unit and card.unit.has_magnetic:
                        has_mech = any(UnitType.MECH in u.types for u in player.board)
                        if has_mech:
                            self.pending_spell_hand_index = h_idx
                            self.is_targeting = True
                            self.pending_target_kind = "MAGNETIZE"
                            continue
            self.game.step(p_idx, action_type, **kwargs)

        self.game.step(p_idx, "END_TURN")
        self.is_targeting = False
        self.pending_spell_hand_index = None

    def _decode_action_for_engine(self, action: int) -> tuple[str, dict[str, int]]:
        if action == 1:
            return "ROLL", {}
        if 2 <= action <= 8:
            return "BUY", {"index": action - 2}
        if 9 <= action <= 15:
            return "SELL", {"index": action - 9}
        if 16 <= action <= 25:
            return "PLAY", {"hand_index": action - 16, "insert_index": -1}
        if 26 <= action <= 31:
            return "SWAP", {"index_a": action - 26, "index_b": action - 26 + 1}
        return "UNKNOWN", {}

    def _simple_bot_turn(self, p_idx: int) -> None:
        """Simple bot: buy random, place all, pick first in discovery"""
        player = self.game.players[p_idx]

        if player.is_discovering and player.discovery.options:
            self.game.step(p_idx, "DISCOVER_CHOICE", index=0)

        attempts = 0
        max_attempts = 15

        while len(player.hand) > 0 and len(player.board) < 7 and attempts < max_attempts:
            hand_size_before = len(player.hand)

            self.game.step(p_idx, "PLAY", hand_index=0, insert_index=-1)
            if len(player.hand) == hand_size_before:
                break

            if player.is_discovering:
                self.game.step(p_idx, "DISCOVER_CHOICE", index=0)

            attempts += 1

        it = 0
        while player.gold >= 3 and player.store and it < 5:
            it += 1
            idx = random.randint(0, len(player.store) - 1)
            self.game.step(p_idx, "BUY", index=idx)

            if player.hand:
                self.game.step(p_idx, "PLAY", hand_index=len(player.hand) - 1, insert_index=-1)
                if player.is_discovering:
                    self.game.step(p_idx, "DISCOVER_CHOICE", index=0)

        self.game.step(p_idx, "END_TURN")

    def _get_obs(self, player_idx: int | None = None) -> np.ndarray:
        p_id = self.my_player_id if player_idx is None else player_idx
        e_id = 1 - p_id

        p = self.game.players[p_id]
        e = self.game.players[e_id]
        buf = self._obs_buffer
        buf[:] = 0.0  # fast zero-fill

        # 1. Global (7) — direct write
        buf[0] = p.gold / MAX_GOLD
        buf[1] = p.tavern_tier / MAX_TIER
        buf[2] = p.health / MAX_HP
        buf[3] = p.up_cost / 10.0
        buf[4] = p.spell_discount / MAX_SPELL_DISCOUNT
        buf[5] = 1.0 if p.is_discovering else 0.0
        buf[6] = 1.0 if self.is_targeting else 0.0

        # 2. Zones — write directly into buffer
        self._encode_zone_fast(p.board, buf, self._off_board, 7, "BOARD")
        self._encode_zone_fast(p.hand, buf, self._off_hand, 10, "HAND")
        self._encode_zone_fast(p.store, buf, self._off_store, 7, "STORE")

        discover_items = p.discovery.options if p.is_discovering else []
        self._encode_zone_fast(discover_items, buf, self._off_discover, 3, "DISCOVER")

        # 3. Enemy (3)
        off = self._off_enemy
        buf[off] = e.health / MAX_HP
        buf[off + 1] = e.tavern_tier / MAX_TIER
        buf[off + 2] = len(e.board) / 7.0

        # 4. Optional: Last Combat Enemy Board (7 * entity_features)
        if self.use_enemy_board_obs:
            enemy_last_board = self.last_combat_board[e_id]
            self._encode_zone_fast(enemy_last_board, buf, self._off_enemy_board, 7, "ENEMY_BOARD")

        # 5. Optional: Player Status History (2 players * 32 features = 64 floats)
        if self.use_player_status_obs:
            off_status = self._off_player_status
            self._encode_player_status(p_id, buf, off_status)
            self._encode_player_status(e_id, buf, off_status + 32)

        return buf

    def _encode_player_status(self, p_idx: int, buf: np.ndarray, off: int) -> None:
        p = self.game.players[p_idx]
        buf[off + 0] = p.health / 30.0
        buf[off + 1] = p.tavern_tier / 6.0
        buf[off + 2] = len(p.board) / 7.0
        buf[off + 3] = self._triples[p_idx] / 10.0
        buf[off + 4] = self._streak[p_idx] / 10.0

        # upgrade turns (5 values)
        for i in range(5):
            buf[off + 5 + i] = self._upgrade_turns[p_idx][i] / 20.0

        # current minion types composition (11 values)
        comp = get_minion_types_composition(p, self.all_types)
        for i in range(11):
            buf[off + 10 + i] = comp[i] / 7.0

        # cumulative minion types composition (11 values)
        for i in range(11):
            buf[off + 21 + i] = self._cumulative_types[p_idx][i] / 20.0

    def _encode_zone_fast(
        self,
        items: Sequence[Unit | HandCard | StoreItem],
        buf: np.ndarray,
        offset: int,
        n_slots: int,
        zone_type: str,
    ) -> None:
        ef = self.entity_features
        n = min(len(items), n_slots)
        for i in range(n):
            self._encode_entity_fast(items[i], buf, offset + i * ef, i, zone_type)

    def _encode_entity_fast(
        self,
        item: Unit | HandCard | StoreItem,
        buf: np.ndarray,
        off: int,
        index_in_zone: int,
        zone_type: str,
    ) -> None:
        """Write entity features directly into buf[off:off+entity_features]."""
        unit: Unit | None
        spell: Spell | None
        is_frozen: bool
        if isinstance(item, Unit):
            unit = item
            spell = None
            is_frozen = False
        else:
            unit = item.unit
            spell = item.spell
            is_frozen = item.is_frozen if isinstance(item, StoreItem) else False

        if unit is not None:
            card_id = unit.card_id
            bc, eot, soc, sell, syn = self._trigger_cache.get(card_id, self._default_triggers)
            db_data: dict[str, Any] = CARD_DB.get(card_id, {})

            buf[off + 0] = 1.0  # Is Present
            # buf[off + 1] = 0.0  # Is Spell (already 0)
            buf[off + 2] = float(self.static_id_map.get(card_id, 0))  # raw int for nn.Embedding
            buf[off + 3] = 3.0 / MAX_COST
            buf[off + 4] = unit.tier / MAX_TIER
            buf[off + 5] = 1.0 if is_frozen else 0.0
            buf[off + 6] = unit.cur_atk / MAX_ATK
            buf[off + 7] = unit.cur_hp / MAX_HP
            # Keywords [8..16]
            buf[off + 8] = 1.0 if unit.has_taunt else 0.0
            buf[off + 9] = 1.0 if unit.has_divine_shield else 0.0
            buf[off + 10] = 1.0 if unit.has_windfury else 0.0
            buf[off + 11] = 1.0 if unit.has_poisonous else 0.0
            buf[off + 12] = 1.0 if unit.has_venomous else 0.0
            buf[off + 13] = 1.0 if unit.has_reborn else 0.0
            buf[off + 14] = 1.0 if unit.has_cleave else 0.0
            buf[off + 15] = 1.0 if unit.has_magnetic else 0.0
            buf[off + 16] = 1.0 if unit.has_immediate_attack else 0.0
            buf[off + 17] = 1.0 if unit.is_golden else 0.0
            buf[off + 18] = 1.0 if db_data.get("is_token", False) else 0.0
            buf[off + 19] = 1.0 if db_data.get("deathrattle", False) else 0.0
            buf[off + 20] = 1.0 if bc else 0.0
            buf[off + 21] = 1.0 if eot else 0.0
            buf[off + 22] = 1.0 if soc else 0.0
            buf[off + 23] = 1.0 if sell else 0.0
            buf[off + 24] = 1.0 if syn else 0.0
            # Is Selected
            if (
                self.is_targeting
                and zone_type == "HAND"
                and self.pending_spell_hand_index is not None
                and index_in_zone == self.pending_spell_hand_index
            ):
                buf[off + 25] = 1.0
            # Types [26..36]
            types = unit.types
            if types:
                base = off + 26
                for i, t_enum in enumerate(self.all_types):
                    if t_enum in types:
                        buf[base + i] = 1.0

        elif spell is not None:
            buf[off + 0] = 1.0  # Is Present
            buf[off + 1] = 1.0  # Is Spell
            buf[off + 2] = float(self.static_id_map.get(spell.card_id, 0))  # raw int for nn.Embedding
            buf[off + 3] = spell.cost / MAX_COST
            buf[off + 4] = spell.tier / MAX_TIER
            buf[off + 5] = 1.0 if is_frozen else 0.0
            buf[off + 20] = 1.0  # spell = has battlecry
            # Is Selected (spells can be targeting source too)
            if (
                self.is_targeting
                and zone_type == "HAND"
                and self.pending_spell_hand_index is not None
                and index_in_zone == self.pending_spell_hand_index
            ):
                buf[off + 25] = 1.0
        # else: all zeros (buf already zeroed)

    def _encode_single_entity(
        self,
        item: Unit | HandCard | StoreItem,
        index_in_zone: int = -1,
        zone_type: str = "UNKNOWN",
    ) -> list[float]:
        """Backwards-compatible wrapper for tests."""
        buf = np.zeros(self.entity_features, dtype=np.float32)
        self._encode_entity_fast(item, buf, 0, index_in_zone, zone_type)
        return buf.tolist()

    def action_masks(self, player_idx: int | None = None) -> np.ndarray:
        """
        Return boolean masks valid actions
        True - action is available, False - banned
        Index order = action_space(Discrete 34)
        """
        p_id = self.my_player_id if player_idx is None else player_idx
        player = self.game.players[p_id]
        masks = self._mask_buffer
        masks[:] = False

        # ban stupid swaps
        if self.actions_in_turn >= self.max_actions_in_turn:
            masks[0] = True  # Only End Turn
            return masks

        # === 1. DISCOVERY PHASE ===
        if player.is_discovering:
            num_options = len(player.discovery.options)
            for i in range(num_options):
                masks[2 + i] = True
            return masks

        # === 2. TARGETING PHASE ===
        if self.is_targeting:
            masks[0] = True  # Cancel cast
            # idx: 2 + i
            board_len = len(player.board)
            valid_targets = 0
            if self.pending_target_kind == "MAGNETIZE":
                # only MECHS
                for i in range(min(board_len, 7)):
                    if UnitType.MECH in player.board[i].types:
                        masks[2 + i] = True
                        valid_targets += 1
            else:
                # СПЕЛЛЫ / БАТТЛКРАИ (любая цель)
                for i in range(min(board_len, 7)):
                    masks[2 + i] = True
                    valid_targets += 1
            if valid_targets > 0:  # stop cast/cancel cycle
                masks[0] = False
            return masks

        # === 3. DEFAULT PHASE ===

        # [0] End Turn - always available
        masks[0] = True

        # [1] Roll - gold >= 1
        if player.gold >= 1:
            masks[1] = True

        # BUY (2-8)
        for i in range(7):
            if i < len(player.store):
                item = player.store[i]
                cost = item.spell.cost - player.spell_discount if item.spell else 3
                # hand overload
                masks[2 + i] = (player.gold >= cost) and (len(player.hand) < 10)

        # SELL (9-15)
        for i in range(7):
            masks[9 + i] = i < len(player.board)

        # PLAY (16-25)
        for i in range(10):
            masks[16 + i] = self._can_play_card(player, i)

        # SWAP (26-31)
        for i in range(6):
            if self.auto_position:
                masks[26 + i] = False  # positioning handled by auto_position / positioning module
            else:
                masks[26 + i] = (i + 1 < len(player.board))

        # UPGRADE (32) — disabled if tavern already at max_tier
        masks[32] = (
            player.gold >= player.up_cost
            and player.tavern_tier < 6
            and player.tavern_tier < self._max_tier
        )

        # FREEZE_AND_END_TURN (33) — same availability as END_TURN
        masks[33] = True

        return masks

    def _can_play_card(self, player: Player, card_index: int) -> bool:
        """
        Centralized check: can you play card from hand with index "card_index"
        """
        if card_index >= len(player.hand):
            return False

        card = player.hand[card_index]

        if card.spell:
            spell_id = card.spell.card_id

            # Condition A: need Target
            # Check from list
            if spell_id in SPELLS_REQUIRE_TARGET:
                if not player.board:
                    return False

            # Condition B: unique spells (future example)
            # if spell_id == SpellIDs.SOME_CONDITIONAL_SPELL:
            #     if not condition: return False

            return True

        if card.unit:
            if len(player.board) >= 7:
                return False

            # Условие C: Battlecry with target
            # card_id = card.unit.card_id
            # if card_id in UNITS_REQUIRE_TARGET and not player.board:
            #    return False

            return True

        return False
