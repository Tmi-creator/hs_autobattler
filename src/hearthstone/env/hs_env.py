import gymnasium as gym
import numpy as np
import random
from gymnasium import spaces

from hearthstone.engine.game import Game
from hearthstone.engine.enums import UnitType
from hearthstone.engine.effects import TRIGGER_REGISTRY
from hearthstone.engine.event_system import EventType
from hearthstone.engine.spells import SPELLS_REQUIRE_TARGET
from hearthstone.engine.configs import CARD_DB, SPELL_DB

# Normalization constants
MAX_ATK = 100.0
MAX_HP = 100.0
MAX_GOLD = 30.0
MAX_TIER = 6.0
MAX_COST = 10.0
MAX_SPELL_DISCOUNT = 10.0
MAX_CARDS_IN_GAME = 500


class HearthstoneEnv(gym.Env):
    """
    RL Environment for Hearthstone Battlegrounds.

    Action Space (Discrete 32):
    0: End Turn
    1: Roll
    2-8: Buy (Slot 0-6) / DISCOVER_CHOICE (Option 0-2) / TARGET BOARD (Slot 0-6)
    9-15: Sell (Slot 0-6) / TARGET STORE (Slot 0-6) [Not implemented in engine yet, usually spells target board]
    16-25: Play Hand (Card 0-9)
    26-31: Swap Right (Slot i <-> Slot i+1)
           26: 0<->1, 27: 1<->2 ... 31: 5<->6
    """

    def __init__(self):
        super(HearthstoneEnv, self).__init__()

        all_ids = sorted(list(CARD_DB.keys()) + list(SPELL_DB.keys()))

        self.static_id_map = {cid: i + 1 for i, cid in enumerate(all_ids)}

        self.game = Game()
        self.my_player_id = 0
        self.enemy_id = 1
        self.max_steps_per_episode = 500
        self.steps_taken = 0

        self.actions_in_turn = 0
        self.max_actions_in_turn = 30
        # Action Space 26 -> 32
        self.action_space = spaces.Discrete(32)

        self.opponent_model = None

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
        # [8..16] Keywords (Taunt, DS, WF, Poison, Venom, Reborn, Cleave, Immediate attack, Magnetic)
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
        total_obs_size = 7 + \
                         (7 * self.entity_features) + \
                         (10 * self.entity_features) + \
                         (7 * self.entity_features) + \
                         (3 * self.entity_features) + \
                         3

        self.observation_space = spaces.Box(
            low=0, high=1, shape=(total_obs_size,), dtype=np.float32
        )

        self.is_targeting = False
        self.pending_spell_hand_index = None
        self.pending_target_kind = None  # "SPELL" | "MAGNETIZE"

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.game = Game()

        self.steps_taken = 0
        self.actions_in_turn = 0
        self.is_targeting = False
        self.pending_spell_hand_index = None

        return self._get_obs(), {}

    def set_opponent(self, model):
        self.opponent_model = model

    def step(self, action):
        self.steps_taken += 1
        self.actions_in_turn += 1
        truncated = (self.game.turn_count > 50) or (self.steps_taken >= self.max_steps_per_episode)

        player = self.game.players[self.my_player_id]
        is_discovering = player.is_discovering
        prev_board_power = self._calculate_board_power(self.game.players[self.my_player_id])
        # === ACTION MAPPING ===
        action_type = "UNKNOWN"
        kwargs = {}

        reward = 0
        done = False
        info = "Unknown"

        if is_discovering:
            # (2-4 -> 0-2)
            if 2 <= action <= 4:
                action_type = "DISCOVER_CHOICE"
                kwargs['index'] = action - 2
            else:
                action_type = "INVALID_DURING_DISCOVERY"

        elif self.is_targeting:
            # Use button BUY (2-8) for target on board (0-6)
            if 2 <= action <= 8:
                action_type = "PLAY"  # Continue playing
                kwargs['hand_index'] = self.pending_spell_hand_index
                kwargs['target_index'] = action - 2

                # Reset targeting after play
                self.pending_spell_hand_index = None
                self.is_targeting = False
                self.pending_target_kind = None

            # button END TURN (0) as CANCEL
            elif action == 0:
                action_type = "CANCEL_CAST"
                self.pending_spell_hand_index = None
                self.is_targeting = False
                self.pending_target_kind = None
                # Just reset state

            else:
                action_type = "INVALID_NEED_TARGET"
        else:
            # basic mapping
            if action == 0:
                action_type = "END_TURN"
            elif action == 1:
                action_type = "ROLL"
            elif 2 <= action <= 8:
                action_type = "BUY"
                kwargs['index'] = action - 2

                # === REWARD BEFORE BUY TRIPLE ===
                buy_index = action - 2
                if buy_index < len(player.store):
                    store_item = player.store[buy_index]
                    if store_item.unit:
                        card_id = store_item.unit.card_id

                        # count cards
                        count_on_board = sum(1 for u in player.board if u.card_id == card_id)
                        count_in_hand = sum(1 for c in player.hand if c.unit and c.unit.card_id == card_id)

                        total_copies = count_on_board + count_in_hand
                        # big reward for triple
                        if total_copies == 2:
                            reward += 2.5
                        # not really big for pair
                        elif total_copies == 1:
                            reward += 0.5
            elif 9 <= action <= 15:
                action_type = "SELL"
                kwargs['index'] = action - 9
            elif 16 <= action <= 25:
                h_idx = action - 16
                # Check, is there a card and require it target or not
                if h_idx < len(player.hand):
                    card = player.hand[h_idx]
                    card_id = getattr(card.spell, 'card_id', None) if card.spell else getattr(card.unit, 'card_id',
                                                                                              None)
                    if card_id in SPELLS_REQUIRE_TARGET:
                        # GO INTO TARGET MODE
                        self.pending_spell_hand_index = h_idx
                        self.is_targeting = True
                        action_type = "WAIT_FOR_TARGET"
                    elif card.unit and card.unit.has_magnetic:  # try magnet mech
                        has_mech = any(UnitType.MECH in u.types for u in player.board)
                        if has_mech:
                            self.is_targeting = True
                            self.pending_target_kind = "MAGNETIZE"
                            self.pending_spell_hand_index = h_idx
                            action_type = "WAIT_FOR_TARGET"
                        else:
                            action_type = "PLAY"
                            kwargs['hand_index'] = h_idx
                            kwargs['insert_index'] = -1
                    else:
                        # default
                        action_type = "PLAY"
                        kwargs['hand_index'] = h_idx
                        kwargs['insert_index'] = -1
                        if card.spell and card.spell.card_id == "S999":
                            reward += 3.0
                else:
                    action_type = "INVALID_HAND_INDEX"
            elif 26 <= action <= 31:
                action_type = "SWAP"
                idx_a = action - 26
                idx_b = idx_a + 1
                kwargs['index_a'] = idx_a
                kwargs['index_b'] = idx_b

        p0_hp_before = player.health
        p1_hp_before = self.game.players[self.enemy_id].health

        # Engine run

        if action_type == "WAIT_FOR_TARGET":
            # Don't go into game.step, just change state
            # Return reward 0 and new obs (where is_targeting=1 and is_selected=1)
            return self._get_obs(), 0.0, False, truncated, {}

        elif action_type == "CANCEL_CAST":
            return self._get_obs(), -0.01, False, truncated, {}

        success, done, info = self.game.step(self.my_player_id, action_type, **kwargs)

        # Errors
        if not success:
            return self._get_obs(), 0, self.game.game_over, truncated, {}

        current_board_power = self._calculate_board_power(self.game.players[self.my_player_id])
        power_delta = (current_board_power - prev_board_power)
        if power_delta > 0:
            reward += power_delta * 0.05
        if action_type == "UPGRADE":  # give reward for up tavern
            pass  # right now no need to up tavern
            reward += 0.5
        if action_type == "SWAP":
            reward -= 0.01

        # Combat + EndOfTurn
        if action_type == "END_TURN" and not done:
            self.actions_in_turn = 0
            if player.gold > 2:
                reward -= 0.1 * player.gold

            if self.game.turn_count < 9:  # penalty in early game for waiting
                if len(player.board) < 7:
                    has_unit_in_hand = False
                    for card in player.hand:
                        if card.unit:
                            has_unit_in_hand = True
                            break

                    if has_unit_in_hand:
                        penalty = 1.5 if self.game.turn_count < 4 else 0.8
                        reward -= penalty

            # less pain for spells
            has_utility_spell = False
            for card in player.hand:
                if card.spell and card.spell.card_id not in ["S999"]:
                    has_utility_spell = True
                    break
            if has_utility_spell and len(player.board) > 0:
                reward -= 0.5

            self._auto_position_board(player)
            self._play_enemy_turn()
            done = self.game.game_over

            # Reward for fight
            p0_hp_after = self.game.players[self.my_player_id].health
            p1_hp_after = self.game.players[self.enemy_id].health

            damage_dealt = p1_hp_before - p1_hp_after
            damage_taken = p0_hp_before - p0_hp_after

            reward += (damage_dealt * 0.2)
            reward -= (damage_taken * 0.2)

            if done:
                if p0_hp_after > 0:
                    reward += 5.0
                else:
                    reward -= 5.0

        if len(player.hand) >= 10:  # punish for bullshit blocking good cards in hand
            reward -= 0.5
        for card in player.hand:
            if card.spell and card.spell.card_id == "S999":
                reward -= 0.2  # punish for every move while not played

        return self._get_obs(), reward, done, truncated, {}

    def _auto_position_board(self, player):
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

        def sort_key(unit):
            score = 0
            # Приоритеты (чем выше score, тем левее стоит юнит)
            if unit.has_cleave: score += 10000
            if unit.has_poisonous or unit.has_venomous: score += 5000
            if unit.has_divine_shield: score += 1000

            # Сортируем по атаке (сильные бьют раньше)
            score += unit.cur_atk

            if unit.has_taunt and unit.cur_atk < 5: score -= 2000

            return score

        player.board.sort(key=sort_key, reverse=True)

    def _calculate_board_power(self, player):
        power = 0
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
                    u_score += (poison_value - unit.cur_atk)

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

            power += u_score

        return power

    def _play_enemy_turn(self):
        p_idx = self.enemy_id

        if self.opponent_model is None:
            self._simple_bot_turn(p_idx)
            return
        player = self.game.players[p_idx]
        self.is_targeting = False
        self.pending_spell_hand_index = None

        max_actions = 30

        for _ in range(max_actions):
            obs = self._get_obs(player_idx=p_idx)

            masks = self.action_masks(player_idx=p_idx)

            import torch
            with torch.no_grad():  # no gradients
                action, _ = self.opponent_model.predict(obs, action_masks=masks, deterministic=False)
            action = int(action)

            if action == 0:
                if self.is_targeting:  # CANCEL
                    self.is_targeting = False
                    self.pending_spell_hand_index = None
                    continue
                else:
                    break  # END TURN

            # TARGET SPELL
            if self.is_targeting:
                if 2 <= action <= 8:
                    target_idx = action - 2
                    hand_idx = self.pending_spell_hand_index
                    self.game.step(p_idx, "PLAY", hand_index=hand_idx, target_index=target_idx)
                    self.is_targeting = False
                    self.pending_spell_hand_index = None
                continue

            action_type, kwargs = self._decode_action_for_engine(action)

            if action_type == "PLAY":
                h_idx = kwargs.get('hand_index')
                if h_idx is not None and h_idx < len(player.hand):
                    card = player.hand[h_idx]
                    card_id = getattr(card.spell, 'card_id', None) if card.spell else getattr(card.unit, 'card_id',
                                                                                              None)

                    if card_id in SPELLS_REQUIRE_TARGET:
                        self.pending_spell_hand_index = h_idx
                        self.is_targeting = True
                        continue

            self.game.step(p_idx, action_type, **kwargs)

        self.game.step(p_idx, "END_TURN")
        self.is_targeting = False
        self.pending_spell_hand_index = None

    def _decode_action_for_engine(self, action):
        if action == 1: return "ROLL", {}
        if 2 <= action <= 8: return "BUY", {'index': action - 2}
        if 9 <= action <= 15: return "SELL", {'index': action - 9}
        if 16 <= action <= 25:
            return "PLAY", {'hand_index': action - 16, 'insert_index': -1}
        if 26 <= action <= 31:
            return "SWAP", {'index_a': action - 26, 'index_b': action - 26 + 1}
        return "UNKNOWN", {}

    def _simple_bot_turn(self, p_idx):
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

    def _get_obs(self, player_idx=None):
        p_id = self.my_player_id if player_idx is None else player_idx
        e_id = 1 - p_id

        p = self.game.players[p_id]
        e = self.game.players[e_id]

        # 1. Global (7)
        global_features = [
            p.gold / MAX_GOLD,
            p.tavern_tier / MAX_TIER,
            p.health / MAX_HP,
            p.up_cost / 10.0,
            p.spell_discount / MAX_SPELL_DISCOUNT,
            1.0 if p.is_discovering else 0.0,
            1.0 if self.is_targeting else 0.0
        ]

        # 2. Zones
        board_vec = self._encode_zone(p.board, 7, zone_type="BOARD")
        hand_vec = self._encode_zone(p.hand, 10, zone_type="HAND")
        store_vec = self._encode_zone(p.store, 7, zone_type="STORE")

        discover_items = p.discovery.options if p.is_discovering else []
        discover_vec = self._encode_zone(discover_items, 3, zone_type="DISCOVER")

        # 3. Enemy (3)
        enemy_vec = [
            e.health / MAX_HP,
            e.tavern_tier / MAX_TIER,
            len(e.board) / 7.0
        ]

        return np.concatenate([
            global_features,
            board_vec,
            hand_vec,
            store_vec,
            discover_vec,
            enemy_vec
        ], dtype=np.float32)

    def _encode_zone(self, items, n_slots, zone_type="UNKNOWN"):
        zone_features = []
        for i in range(n_slots):
            if i < len(items):
                zone_features.extend(self._encode_single_entity(items[i], index_in_zone=i, zone_type=zone_type))
            else:
                zone_features.extend([0.0] * self.entity_features)
        return zone_features

    def _encode_single_entity(self, item, index_in_zone=-1, zone_type="UNKNOWN"):
        """Create entity vector with trigger analysis"""
        unit = getattr(item, 'unit', None)
        spell = getattr(item, 'spell', None)
        is_frozen = getattr(item, 'is_frozen', False)

        if unit is None and spell is None and hasattr(item, 'cur_hp'):
            unit = item

        if unit is None and spell is None:
            return [0.0] * self.entity_features

        # Initialize effect flags
        has_battlecry = False
        has_eot = False
        has_soc = False
        has_sell = False
        has_synergy = False

        if unit:
            card_id = unit.card_id
            cost = 3.0
            tier = unit.tier
            is_spell = 0.0

            cur_atk = unit.cur_atk
            cur_hp = unit.cur_hp
            is_golden = unit.is_golden

            # --- Анализ триггеров ---
            triggers = TRIGGER_REGISTRY.get(card_id, [])
            for trig_def in triggers:
                evt = trig_def.event_type
                cond_name = trig_def.condition.__name__

                if evt == EventType.MINION_PLAYED:
                    if cond_name == "_is_self_play":
                        has_battlecry = True
                    else:
                        has_synergy = True

                elif evt == EventType.MINION_SUMMONED:
                    has_synergy = True

                elif evt == EventType.END_OF_TURN:
                    has_eot = True

                elif evt == EventType.START_OF_COMBAT:
                    has_soc = True

                elif evt == EventType.MINION_SOLD:
                    has_sell = True
            db_data = CARD_DB.get(card_id, {})
            flags = [
                unit.has_taunt,
                unit.has_divine_shield,
                unit.has_windfury,
                unit.has_poisonous,
                unit.has_venomous,
                unit.has_reborn,
                unit.has_cleave,
                unit.has_magnetic,
                unit.has_immediate_attack,
                is_golden,
                db_data.get('is_token', False),
                db_data.get('deathrattle', False),

                has_battlecry,
                has_eot,
                has_soc,
                has_sell,
                has_synergy
            ]

            u_types = unit.types

        else:  # Spell
            card_id = spell.card_id
            cost = spell.cost
            tier = spell.tier
            is_spell = 1.0
            cur_atk = 0.0
            cur_hp = 0.0
            flags = [False] * 17
            flags[12] = True  # spell = battlecry

            u_types = []

        # Card ID
        cid_val = self.static_id_map.get(card_id, 0) / MAX_CARDS_IN_GAME

        # Build vector
        vec = [
            1.0,  # [0] Is Present
            is_spell,  # [1] Is Spell
            cid_val,  # [2] ID
            cost / MAX_COST,  # [3] Cost
            tier / MAX_TIER,  # [4] Tier
            1.0 if is_frozen else 0.0,  # [5] Frozen
            cur_atk / MAX_ATK,  # [6] ATK
            cur_hp / MAX_HP,  # [7] HP
        ]

        # Flags (17)
        vec.extend([1.0 if f else 0.0 for f in flags])

        # === Is Selected (Targeting) ===
        is_selected = 0.0
        if self.is_targeting and \
                zone_type == "HAND" and \
                self.pending_spell_hand_index is not None and \
                index_in_zone == self.pending_spell_hand_index:
            is_selected = 1.0

        vec.append(is_selected)  # [23]

        # Types (11)
        type_vec = [0.0] * self.num_types
        if u_types:
            for i, t_enum in enumerate(self.all_types):
                if t_enum in u_types:
                    type_vec[i] = 1.0
        vec.extend(type_vec)

        return vec

    def action_masks(self, player_idx=None):
        """
        Return boolean masks valid actions
        True - action is available, False - banned
        Index order = action_space(Discrete 32)
        """
        p_id = self.my_player_id if player_idx is None else player_idx
        player = self.game.players[p_id]
        masks = [False] * 32

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
            masks[9 + i] = (i < len(player.board))

        # PLAY (16-25)
        for i in range(10):
            masks[16 + i] = self._can_play_card(player, i)

        # SWAP (26-31)
        for i in range(6):
            masks[26 + i] = False  # now he's very stupid, so he cant handle this power of choice
            # masks[26 + i] = (i + 1 < len(player.board))

        return masks

    def _can_play_card(self, player, card_index):
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
