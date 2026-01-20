import gymnasium as gym
import numpy as np
import random
from gymnasium import spaces

from engine.game import Game
from engine.enums import UnitType
from engine.configs import CARD_DB
from engine.effects import TRIGGER_REGISTRY
from engine.event_system import EventType

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
    2-8: Buy (Slot 0-6) / DISCOVER_CHOICE (Option 0-2)
    9-15: Sell (Slot 0-6)
    16-25: Play Hand (Card 0-9)
    26-31: Swap Right (Slot i <-> Slot i+1) [New]
           26: 0<->1, 27: 1<->2 ... 31: 5<->6
    """

    def __init__(self):
        super(HearthstoneEnv, self).__init__()

        self.game = Game()
        self.my_player_id = 0
        self.enemy_id = 1
        self.consecutive_errors = 0

        # Action Space увеличен с 26 до 32
        self.action_space = spaces.Discrete(32)

        self.all_types = list(UnitType)
        self.num_types = len(self.all_types)  # 11

        # --- Вектор сущности (34 float) ---
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
        # [23..33] Types (11 шт)

        self.entity_features = 23 + self.num_types  # 34

        # Размер вектора наблюдения:
        # Global(6) + Board(7*34) + Hand(10*34) + Store(7*34) + Discover(3*34) + Enemy(3)
        # 6 + 238 + 340 + 238 + 102 + 3 = 927
        total_obs_size = 6 + \
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

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.game = Game()
        self.consecutive_errors = 0
        return self._get_obs(), {}

    def step(self, action):
        if not hasattr(self, "consecutive_errors"):
            self.consecutive_errors = 0

        player = self.game.players[self.my_player_id]
        is_discovering = player.is_discovering

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
                action_type = "PLAY"
                kwargs['hand_index'] = action - 16
                kwargs['insert_index'] = -1
            elif 26 <= action <= 31:  # --- SWAP LOGIC ---
                action_type = "SWAP"
                idx_a = action - 26
                idx_b = idx_a + 1
                kwargs['index_a'] = idx_a
                kwargs['index_b'] = idx_b

        p0_hp_before = player.health
        p1_hp_before = self.game.players[self.enemy_id].health

        # --- ВЫПОЛНЕНИЕ В ДВИЖКЕ ---
        reward = 0
        done = False
        info = "Unknown"

        if action_type == "INVALID_DURING_DISCOVERY":
            reward = -2.0
            info = "Must choose discovery"
            self.consecutive_errors += 1
        else:
            _, done, info = self.game.step(self.my_player_id, action_type, **kwargs)

            # Список ошибок движка, которые считаются "плохим действием"
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
                    info != "Invalid indices" and  # Ошибка при неверном свапе
                    info != "Same index" and
                    info != "Not discovering" and
                    info != "No spell to cast" and
                    info != "Invalid target"
            )

            if not is_valid_action:
                reward = -2.0
                self.consecutive_errors += 1
            else:
                self.consecutive_errors = 0
                if action_type != "END_TURN":
                    reward = 0.1
                if action_type == "DISCOVER_CHOICE":
                    reward += 1.0
                # Можно добавить небольшой reward за swap, но лучше 0.1,
                # чтобы не поощрять бессмысленное перекладывание

        # --- ЗАЩИТА ОТ ЗАВИСАНИЯ (Training Wheels) ---
        force_end_turn = False
        if self.consecutive_errors >= 10:
            if not is_discovering:
                action_type = "END_TURN"
                force_end_turn = True
                self.consecutive_errors = 0
            else:
                self.game.step(self.my_player_id, "DISCOVER_CHOICE", index=0)
                reward -= 1.0
                self.consecutive_errors = 0

        # --- КОНЕЦ ХОДА И БОЙ ---
        if action_type == "END_TURN" and not done:
            if not force_end_turn:
                reward += 2.0

            if force_end_turn:
                self.game.step(self.my_player_id, "END_TURN")

            self._play_enemy_turn()
            done = self.game.game_over

            # Награда за бой
            p0_hp_after = self.game.players[self.my_player_id].health
            p1_hp_after = self.game.players[self.enemy_id].health

            damage_dealt = p1_hp_before - p1_hp_after
            damage_taken = p0_hp_before - p0_hp_after

            reward += (damage_dealt * 2.0)
            reward -= (damage_taken * 1.0)

            if done:
                if p0_hp_after > 0:
                    reward += 20.0
                else:
                    reward -= 20.0

        truncated = self.game.turn_count > 50
        return self._get_obs(), reward, done, truncated, {}

    def _play_enemy_turn(self):
        """Простейший бот: покупает рандомно, ставит всё, выбирает первое в раскопке"""
        p_idx = self.enemy_id
        player = self.game.players[p_idx]

        # 1. Раскопка
        if player.is_discovering and player.discovery.options:
            self.game.step(p_idx, "DISCOVER_CHOICE", index=0)

        # 2. Рука -> Стол
        while len(player.hand) > 0 and len(player.board) < 7:
            self.game.step(p_idx, "PLAY", hand_index=0, insert_index=-1)
            if player.is_discovering:
                self.game.step(p_idx, "DISCOVER_CHOICE", index=0)

        # 3. Магазин -> Покупка
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

        # 1. Global (6)
        global_features = [
            p.gold / MAX_GOLD,
            p.tavern_tier / MAX_TIER,
            p.health / MAX_HP,
            p.up_cost / 10.0,
            p.spell_discount / MAX_SPELL_DISCOUNT,
            1.0 if p.is_discovering else 0.0
        ]

        # 2. Zones
        board_vec = self._encode_zone(p.board, 7)
        hand_vec = self._encode_zone(p.hand, 10)
        store_vec = self._encode_zone(p.store, 7)

        discover_items = p.discovery.options if p.is_discovering else []
        discover_vec = self._encode_zone(discover_items, 3)

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

    def _encode_zone(self, items, n_slots):
        zone_features = []
        for i in range(n_slots):
            if i < len(items):
                zone_features.extend(self._encode_single_entity(items[i]))
            else:
                zone_features.extend([0.0] * self.entity_features)
        return zone_features

    def _encode_single_entity(self, item):
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
                        has_synergy = True  # Wrath Weaver etc.

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

            # Данные из базы (is_token, deathrattle)
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
            # Спелл считаем как Battlecry (мгновенный эффект)
            flags = [False] * 15
            flags[10] = True

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

        # Types (11 шт)
        type_vec = [0.0] * self.num_types
        if u_types:
            for i, t_enum in enumerate(self.all_types):
                if t_enum in u_types:
                    type_vec[i] = 1.0
        vec.extend(type_vec)

        return vec
