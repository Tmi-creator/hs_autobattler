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

    def __init__(self) -> None:
        super(HearthstoneEnv, self).__init__()

        all_ids = sorted(list(CARD_DB.keys()) + list(SPELL_DB.keys()))

        self.static_id_map = {cid: i + 1 for i, cid in enumerate(all_ids)}
        self.num_card_ids = len(all_ids) + 1  # +1 for padding id 0

        self.game = Game()
        self.my_player_id = 0
        self.enemy_id = 1
        self.max_steps_per_episode = 500
        self.steps_taken = 0

        self.actions_in_turn = 0
        self.max_actions_in_turn = 30
        # Action Space: 0=END, 1=ROLL, 2-8=BUY, 9-15=SELL, 16-25=PLAY, 26-31=SWAP, 32=UPGRADE, 33=FREEZE
        self.action_space = spaces.Discrete(34)

        self.opponent_model: Optional["MaskablePPO"] = None

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
        self._oracle_ghost_cpp: list | None = None  # cached C++ tuples for ghost board
        self._oracle_ghost_tier: int = 1

        self.all_types = list(UnitType)
        self.num_types = len(self.all_types)  # 11

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
        # Global(7) + Board(7*37) + Hand(10*37) + Store(7*37) + Discover(3*37) + Enemy(3)
        # 7 + 259 + 370 + 259 + 111 + 3 = 1009
        total_obs_size = (
            7
            + (7 * self.entity_features)
            + (10 * self.entity_features)
            + (7 * self.entity_features)
            + (3 * self.entity_features)
            + 3
        )

        self.observation_space = spaces.Box(
            low=0, high=MAX_CARDS_IN_GAME, shape=(total_obs_size,), dtype=np.float32
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

        self.game = Game()

        self.steps_taken = 0
        self.actions_in_turn = 0
        self.is_targeting = False
        self.pending_spell_hand_index = None

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
        self._oracle_ghost_cpp = None

        return self._get_obs(), {}

    def set_opponent(self, model: MaskablePPO) -> None:
        self.opponent_model = model

    def get_board_power(self) -> float:
        """Returns current board power for the agent's player."""
        player = self.game.players[self.my_player_id]
        return self._calculate_board_power(player)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]:
        self.steps_taken += 1
        self.actions_in_turn += 1
        truncated = (self.game.turn_count > 50) or (self.steps_taken >= self.max_steps_per_episode)

        player = self.game.players[self.my_player_id]
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

        if action_type == "WAIT_FOR_TARGET":
            return self._get_obs(), 0.0, False, truncated, {}

        elif action_type == "CANCEL_CAST":
            return self._get_obs(), 0.0, False, truncated, {}

        success, done, _ = self.game.step(self.my_player_id, action_type, **kwargs)

        if not success:
            return self._get_obs(), 0.0, self.game.game_over, truncated, {}

        # === REWARD: MC Oracle PBRS + Round Outcome + Terminal ===
        # Dense reward from C++ combat oracle on board-changing actions.
        # Round outcome (+1/-1) at END_TURN. Terminal ±100.

        reward: float = -0.005  # action penalty: discourages infinite loops

        if action_type in ("BUY", "SELL", "PLAY"):
            # MC Oracle: run N combats to measure board strength change
            reward += self._oracle_reward(player)

        elif action_type == "END_TURN":
            reward = 0.0  # END_TURN itself is free
            self.actions_in_turn = 0

            self._auto_position_board(player)

            if self.ghost_pool is not None:
                self.ghost_pool.record_turn(self._env_id, self.game.turn_count, player)

            self._play_enemy_turn()
            done = self.game.game_over

            p0_hp_after = self.game.players[self.my_player_id].health
            p1_hp_after = self.game.players[self.enemy_id].health

            damage_dealt = p1_hp_before - p1_hp_after
            damage_taken = p0_hp_before - p0_hp_after

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

        return self._get_obs(), reward, done, truncated, {}

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

    @staticmethod
    def _unit_to_cpp(unit: Unit) -> tuple:
        """Convert Unit → C++ tuple. Mirrors CombatManager._unit_to_cpp."""
        cpp_types = 0
        for t in unit.types:
            cpp_types |= TYPE_TO_BIT.get(t, 0)
        cpp_tags = 0
        for tag in unit.tags:
            cpp_tags |= TAG_TO_BIT.get(tag, 0)
        return (
            CARD_ID_MAP.get(unit.card_id, 0),
            unit.cur_atk, unit.cur_hp,
            cpp_types, cpp_tags,
            unit.tier, unit.is_golden,
        )

    def _oracle_prepare_ghost(self) -> None:
        """Cache C++ tuples for the current ghost board (called once per turn)."""
        enemy = self.game.players[self.enemy_id]
        if enemy.board:
            self._oracle_ghost_cpp = [self._unit_to_cpp(u) for u in enemy.board]
            self._oracle_ghost_tier = enemy.tavern_tier
        else:
            self._oracle_ghost_cpp = None

    def _oracle_eval_winrate(self, player: Player) -> float:
        """Run N combats via C++ engine, return winrate [0, 1]."""
        cpp = get_cpp_engine()
        if cpp is None or not player.board or self._oracle_ghost_cpp is None:
            return 0.5

        side0 = [self._unit_to_cpp(u) for u in player.board]
        results = cpp.fast_combat_batch(
            side0, self._oracle_ghost_cpp,
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
        return delta * 10.0

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

        return buf

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
            masks[26 + i] = False  # positioning handled by auto_position / positioning module
            # masks[26 + i] = (i + 1 < len(player.board))

        # UPGRADE (32)
        masks[32] = (player.gold >= player.up_cost and player.tavern_tier < 6)

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
