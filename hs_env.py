import gymnasium as gym
import numpy as np
import random
from gymnasium import spaces

from engine.game import Game
from engine.enums import UnitType
from engine.configs import CARD_DB
from engine.effects import TRIGGER_REGISTRY
from engine.event_system import EventType
from engine.spells import SPELLS_REQUIRE_TARGET

# Константы нормализации
MAX_ATK = 50.0
MAX_HP = 50.0
MAX_GOLD = 10.0
MAX_TIER = 6.0
MAX_COST = 10.0
MAX_SPELL_DISCOUNT = 10.0
MAX_CARDS_IN_GAME = 200


class HearthstoneEnv(gym.Env):
    """
    RL Среда для Hearthstone Battlegrounds.

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

        self.game = Game()
        self.my_player_id = 0
        self.enemy_id = 1

        # Action Space увеличен с 26 до 32
        self.action_space = spaces.Discrete(32)

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
        # [8..14] Keywords (Taunt, DS, WF, Poison, Venom, Reborn, Cleave)
        # [15] Is Golden
        # [16] Is Token
        # [17] Has Deathrattle (Native)
        # [18] Has Battlecry (Play Effect)
        # [19] Has End of Turn
        # [20] Has Start of Combat
        # [21] Has Sell Effect
        # [22] Has Synergy (Engine)
        # [23] Is Selected (Targeting Source) <-- НОВОЕ
        # [24..34] Types (11 шт)

        self.entity_features = 24 + self.num_types  # Было 34, стало 35

        # Размер вектора наблюдения:
        # Global(7) + Board(7*35) + Hand(10*35) + Store(7*35) + Discover(3*35) + Enemy(3)
        # 7 + 245 + 350 + 245 + 105 + 3 = 955
        total_obs_size = 7 + \
                         (7 * self.entity_features) + \
                         (10 * self.entity_features) + \
                         (7 * self.entity_features) + \
                         (3 * self.entity_features) + \
                         3

        self.observation_space = spaces.Box(
            low=0, high=1, shape=(total_obs_size,), dtype=np.float32
        )

        self._card_id_map = {}
        self._next_card_int = 1

        self.is_targeting = False
        self.pending_spell_hand_index = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.game = Game()

        self.is_targeting = False
        self.pending_spell_hand_index = None

        return self._get_obs(), {}

    def step(self, action):

        player = self.game.players[self.my_player_id]
        is_discovering = player.is_discovering
        prev_board_power = self._calculate_board_power(self.game.players[self.my_player_id])
        # --- МЭППИНГ ДЕЙСТВИЙ ---
        action_type = "UNKNOWN"
        kwargs = {}

        if is_discovering:
            # В раскопке работают только слоты покупки как выбор (2-4 -> 0-2)
            if 2 <= action <= 4:
                action_type = "DISCOVER_CHOICE"
                kwargs['index'] = action - 2
            else:
                action_type = "INVALID_DURING_DISCOVERY"

        elif self.is_targeting:
            # Используем кнопки BUY (2-8) как выбор цели на ДОСКЕ (0-6)
            if 2 <= action <= 8:
                action_type = "PLAY"  # Продолжаем розыгрыш
                kwargs['hand_index'] = self.pending_spell_hand_index
                kwargs['target_index'] = action - 2

                # Сбрасываем состояние таргетинга после выбора
                self.pending_spell_hand_index = None
                self.is_targeting = False

            # Кнопка END TURN (0) как ОТМЕНА (Cancel)
            elif action == 0:
                action_type = "CANCEL_CAST"
                self.pending_spell_hand_index = None
                self.is_targeting = False
                # В движок ничего не шлем, просто сброс состояния

            else:
                action_type = "INVALID_NEED_TARGET"
        else:
            # Обычный режим
            if action == 0:
                action_type = "END_TURN"
            elif action == 1:
                action_type = "ROLL"
            elif 2 <= action <= 8:
                action_type = "BUY"
                kwargs['index'] = action - 2
            elif 9 <= action <= 15:
                action_type = "SELL"
                kwargs['index'] = action - 9
            elif 16 <= action <= 25:
                h_idx = action - 16
                # Проверяем, есть ли карта и требует ли она цель
                if h_idx < len(player.hand):
                    card = player.hand[h_idx]
                    card_id = getattr(card.spell, 'card_id', None) if card.spell else getattr(card.unit, 'card_id',
                                                                                              None)
                    if card_id in SPELLS_REQUIRE_TARGET:
                        # ПЕРЕХОД В РЕЖИМ ЦЕЛИ
                        self.pending_spell_hand_index = h_idx
                        self.is_targeting = True
                        action_type = "WAIT_FOR_TARGET"
                    else:
                        # default
                        action_type = "PLAY"
                        kwargs['hand_index'] = h_idx
                        kwargs['insert_index'] = -1
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
        reward = 0
        done = False
        info = "Unknown"

        if action_type == "WAIT_FOR_TARGET":
            # Мы не идем в game.step, мы просто обновили внутреннее состояние
            # Возвращаем награду 0 и обновленный obs (где is_targeting=1 и is_selected=1)
            return self._get_obs(), 0.0, False, False, {}

        elif action_type == "CANCEL_CAST":
            return self._get_obs(), 0.0, False, False, {}

        _, done, info = self.game.step(self.my_player_id, action_type, **kwargs)

        # Errors
        is_valid_action = (
                info != "Not enough gold" and
                info != "Board is full" and
                info != "Hand is full" and
                info != "Invalid index" and
                info != "Invalid hand index" and
                info != "Unknown Action" and
                info != "Must choose discovery" and
                info != "Player already ready" and
                info != "Max tier reached" and
                info != "Empty slot" and
                info != "Invalid indices" and
                info != "Same index" and
                info != "Not discovering" and
                info != "No spell to cast" and
                info != "Invalid target"
        )

        if not is_valid_action:
            return self._get_obs(), 0, False, False, {}
        current_board_power = self._calculate_board_power(self.game.players[self.my_player_id])
        power_delta = (current_board_power - prev_board_power)
        if power_delta > 0:
            reward += power_delta * 0.05
        if action_type == "UPGRADE":  # Поощряем прокачку таверны
            pass  # right now no need to up tavern
            reward += 0.5

        if action_type == "DISCOVER_CHOICE":  # Раскопка это почти всегда хорошо
            reward += 0.5

        # Combat + EndOfTurn
        if action_type == "END_TURN" and not done:
            if player.gold > 2:
                reward -= 0.1 * player.gold
            self._play_enemy_turn()
            done = self.game.game_over

            # Награда за бой
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

        truncated = self.game.turn_count > 50
        return self._get_obs(), reward, done, truncated, {}

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
        """Простейший бот: покупает рандомно, ставит всё, выбирает первое в раскопке"""
        p_idx = self.enemy_id
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

    def _get_obs(self):
        p = self.game.players[self.my_player_id]
        e = self.game.players[self.enemy_id]

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
        """Создание вектора сущности с анализом триггеров"""
        unit = getattr(item, 'unit', None)
        spell = getattr(item, 'spell', None)
        is_frozen = getattr(item, 'is_frozen', False)

        if unit is None and spell is None and hasattr(item, 'cur_hp'):
            unit = item

        if unit is None and spell is None:
            return [0.0] * self.entity_features

        # Инициализация флагов эффектов
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

            flags = [
                unit.has_taunt,
                unit.has_divine_shield,
                unit.has_windfury,
                unit.has_poisonous,
                unit.has_venomous,
                unit.has_reborn,
                unit.has_cleave,
                is_golden,
                False,  # is_token
                False,  # has_deathrattle

                has_battlecry,
                has_eot,
                has_soc,
                has_sell,
                has_synergy
            ]

            db_data = CARD_DB.get(card_id, {})
            flags[8] = db_data.get('is_token', False)
            flags[9] = db_data.get('deathrattle', False)

            u_types = unit.type

        else:  # Spell
            card_id = spell.card_id
            cost = spell.cost
            tier = spell.tier
            is_spell = 1.0
            cur_atk = 0.0
            cur_hp = 0.0
            flags = [False] * 15
            flags[10] = True  # Спелл как Battlecry

            u_types = []

        # Card ID
        if card_id not in self._card_id_map:
            self._card_id_map[card_id] = self._next_card_int
            self._next_card_int += 1
        cid_val = self._card_id_map[card_id] / MAX_CARDS_IN_GAME

        # Сборка вектора
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

        # Flags (15 шт)
        vec.extend([1.0 if f else 0.0 for f in flags])

        # --- Is Selected (Targeting) ---
        is_selected = 0.0
        if self.is_targeting and \
                zone_type == "HAND" and \
                self.pending_spell_hand_index is not None and \
                index_in_zone == self.pending_spell_hand_index:
            is_selected = 1.0

        vec.append(is_selected)  # [23]

        # Types (11 шт)
        type_vec = [0.0] * self.num_types
        if u_types:
            for i, t_enum in enumerate(self.all_types):
                if t_enum in u_types:
                    type_vec[i] = 1.0
        vec.extend(type_vec)

        return vec

    def action_masks(self):
        """
        Возвращает булеву маску валидных действий:
        True - действие доступно, False - запрещено.
        Порядок индексов соответствует action_space (Discrete 32).
        """
        player = self.game.players[self.my_player_id]
        masks = [False] * 32

        # --- 1. ФАЗА РАСКОПКИ (Discovery) ---
        if player.is_discovering:
            num_options = len(player.discovery.options)
            for i in range(num_options):
                masks[2 + i] = True
            return masks

        # --- 2. ФАЗА ВЫБОРА ЦЕЛИ (Targeting) ---
        if self.is_targeting:
            masks[0] = True  # Cancel cast
            # idx: 2 + i
            board_len = len(player.board)
            for i in range(board_len):
                masks[2 + i] = True
            return masks

        # --- 3. ОБЫЧНАЯ ФАЗА ---

        # [0] End Turn - always available
        masks[0] = True

        # [1] Roll - доступно, если есть 1 золотой
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
            masks[26 + i] = (i + 1 < len(player.board))

        return masks

    def _can_play_card(self, player, card_index):
        """
        Централизованная проверка: можно ли сыграть карту из руки с индексом card_index.
        """
        if card_index >= len(player.hand):
            return False

        card = player.hand[card_index]

        if card.spell:
            spell_id = card.spell.card_id

            # Условие А: Нужна цель (Target)
            # Берем список из движка, но проверяем универсально
            if spell_id in SPELLS_REQUIRE_TARGET:
                if not player.board:
                    return False

            # Условие Б: Особые спеллы (пример на будущее)
            # if spell_id == SpellIDs.SOME_CONDITIONAL_SPELL:
            #     if not condition: return False

            return True

        if card.unit:
            if len(player.board) >= 7:
                return False

            # Условие Б: Боевые кличи с целью (Battlecry Target)
            # В ТВОЕМ текущем движке нет юнитов, требующих цель (как Coin Naga или Weaver).
            # Но если появятся, мы добавим их в список UNITS_REQUIRE_TARGET и проверим тут.
            # card_id = card.unit.card_id
            # if card_id in UNITS_REQUIRE_TARGET and not player.board:
            #    return False

            return True

        return False
