import gymnasium as gym
import numpy as np
import random
from gymnasium import spaces

from engine.game import Game
from engine.enums import UnitType
from engine.configs import CARD_DB

# Константы нормализации
MAX_ATK = 50.0
MAX_HP = 50.0
MAX_GOLD = 10.0
MAX_TIER = 6.0
MAX_CARDS_IN_GAME = 200


class HearthstoneEnv(gym.Env):
    """
    RL Среда для Hearthstone Battlegrounds.

    Action Space (Discrete 26):
    0: End Turn
    1: Roll
    2-8: Buy (Slot 0-6)
    9-15: Sell (Slot 0-6)
    16-25: Play Hand (Card 0-9) -> ставит в конец стола
    """

    def __init__(self):
        super(HearthstoneEnv, self).__init__()

        self.game = Game()
        self.my_player_id = 0
        self.enemy_id = 1
        self.consecutive_errors = 0
        self.action_space = spaces.Discrete(26)
        self.all_types = list(UnitType)
        num_types = len(self.all_types)  # 11

        # --- Вектор признаков юнита ---
        #  0. ATK
        #  1. HP
        #  2. Card ID (Hash)
        #  3. Tier
        #  4. Taunt
        #  5. Divine Shield
        #  6. Windfury
        #  7. Poisonous (Вечный яд)
        #  8. Venomous (Одноразовый яд)
        #  9. Reborn
        # 10. Cleave
        # 11. Is Golden
        # 12. Is Token
        # 13. Has Deathrattle
        # 14...24. Types (11 слотов под каждый тип)

        self.base_features = 14
        self.unit_features = self.base_features + num_types  # 25

        # Размер вектора наблюдения:
        # Ресурсы(3) + Стол(7*25) + Рука(10*25) + Магазин(7*25) + Враг(3)
        # 3 + 175 + 250 + 175 + 3 = 606 входов
        total_obs_size = 3 + (7 * self.unit_features) + (10 * self.unit_features) + (7 * self.unit_features) + 3

        self.observation_space = spaces.Box(
            low=0, high=1, shape=(total_obs_size,), dtype=np.float32
        )

        self._card_id_map = {}
        self._next_card_int = 1

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.game = Game()
        return self._get_obs(), {}

    def step(self, action):
        if not hasattr(self, "consecutive_errors"):
            self.consecutive_errors = 0

        action_type = "UNKNOWN"
        kwargs = {}

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

        p0_hp_before = self.game.players[self.my_player_id].health
        p1_hp_before = self.game.players[self.enemy_id].health

        # --- ВЫПОЛНЕНИЕ ДЕЙСТВИЯ ---
        # Теперь game.step возвращает награду. Мы ее переопределим для обучения.
        # game.step возвращает -0.1 за ошибку. Мы хотим жестче.
        _, done, info = self.game.step(self.my_player_id, action_type, **kwargs)

        reward = 0
        is_valid_action = (info != "Not enough gold" and
                           info != "Board is full" and
                           info != "Hand is full" and
                           info != "Invalid index" and
                           info != "Invalid hand index" and
                           info != "Unknown Action")

        # 1. СИСТЕМА НАГРАД (Reward Shaping)
        if not is_valid_action:
            reward = -2.0  # ЖЕСТКИЙ ШТРАФ за клик в никуда
            self.consecutive_errors += 1
        else:
            self.consecutive_errors = 0  # Сброс счетчика ошибок
            # Небольшое поощрение за валидные действия (покупка/продажа), чтобы он не боялся играть
            if action_type != "END_TURN":
                reward = 0.1

                # 2. АВТО-ЗАВЕРШЕНИЕ ХОДА (Training Wheels)
        # Если агент сделал 10 ошибок подряд, мы считаем, что он затупил, и пинаем его в следующий ход.
        force_end_turn = False
        if self.consecutive_errors >= 10:
            action_type = "END_TURN"
            force_end_turn = True
            self.consecutive_errors = 0
            # Мы НЕ даем награду за принудительный конец, но и не штрафуем дополнительно.
            # Главное - он попадет в бой.

        # 3. ОБРАБОТКА КОНЦА ХОДА (Добровольного или Принудительного)
        if action_type == "END_TURN" and not done:
            # Если он САМ нажал End Turn - даем печеньку
            if not force_end_turn:
                reward += 2.0

                # Если это было принудительно, нам нужно "прожать" это в движке,
            # так как предыдущий вызов game.step был ошибочным действием.
            if force_end_turn:
                self.game.step(self.my_player_id, "END_TURN")

            # Ход врага -> Бой
            self._play_enemy_turn()
            done = self.game.game_over

            # Расчет боевой награды
            p0_hp_after = self.game.players[self.my_player_id].health
            p1_hp_after = self.game.players[self.enemy_id].health
            damage_dealt = p1_hp_before - p1_hp_after
            damage_taken = p0_hp_before - p0_hp_after

            # Боевая награда
            reward += (damage_dealt * 2)  # Удваиваем радость от урона
            reward -= damage_taken

            if done:
                if p0_hp_after > 0:
                    reward += 20  # Большая награда за победу
                else:
                    reward -= 20

        # Условие прерывания (чтобы эпизоды не длились вечно)
        # 50 ходов максимум
        truncated = self.game.turn_count > 50

        obs = self._get_obs()

        return obs, reward, done, truncated, {}

    def _play_enemy_turn(self):
        """Простейший бот за противника"""
        p_idx = self.enemy_id
        player = self.game.players[p_idx]
        while len(player.hand) > 0 and len(player.board) < 7:
            self.game.step(p_idx, "PLAY", hand_index=0, insert_index=-1)
        it = 0
        while player.gold >= 3 and player.store and it < 5:
            it += 1
            idx = random.randint(0, len(player.store) - 1)
            self.game.step(p_idx, "BUY", index=idx)
            if player.hand:
                self.game.step(p_idx, "PLAY", hand_index=len(player.hand) - 1, insert_index=-1)
        self.game.step(p_idx, "END_TURN")

    def _get_obs(self):
        p = self.game.players[self.my_player_id]
        e = self.game.players[self.enemy_id]

        obs = []
        # 1. Resources
        obs.extend([p.gold / MAX_GOLD, p.tavern_tier / MAX_TIER, p.health / MAX_HP])

        # 2. Zones
        obs.extend(self._encode_units(p.board, 7))
        obs.extend(self._encode_units([c.unit for c in p.hand if c.unit], 10))
        obs.extend(self._encode_units(p.store, 7))

        # 3. Enemy Public Info
        obs.extend([e.health / MAX_HP, e.tavern_tier / MAX_TIER, len(e.board) / 7.0])

        return np.array(obs, dtype=np.float32)

    def _encode_units(self, units, n_slots):
        res = []
        for i in range(n_slots):
            if i < len(units):
                u = units[i]

                # Маппинг ID
                if u.card_id not in self._card_id_map:
                    self._card_id_map[u.card_id] = self._next_card_int
                    self._next_card_int += 1
                cid_val = self._card_id_map[u.card_id] / MAX_CARDS_IN_GAME

                # Доп данные из БД
                db_data = CARD_DB.get(u.card_id, {})
                is_token = 1.0 if db_data.get('is_token', False) else 0.0
                has_deathrattle = 1.0 if db_data.get('deathrattle', False) else 0.0

                # Базовые признаки (14 чисел)
                features = [
                    u.cur_atk / MAX_ATK,
                    u.cur_hp / MAX_HP,
                    cid_val,
                    u.tier / MAX_TIER,
                    1.0 if u.has_taunt else 0.0,
                    1.0 if u.has_divine_shield else 0.0,
                    1.0 if u.has_windfury else 0.0,
                    1.0 if u.has_poisonous else 0.0,
                    1.0 if u.has_venomous else 0.0,
                    1.0 if u.has_reborn else 0.0,
                    1.0 if u.has_cleave else 0.0,
                    1.0 if u.is_golden else 0.0,
                    is_token,
                    has_deathrattle
                ]

                # Типы существ (Multi-Hot Encoding) (11 чисел)
                # Если у существа тип [BEAST, PIRATE], то будут единицы в соответствующих слотах
                type_features = [0.0] * len(self.all_types)
                for t_idx, t_enum in enumerate(self.all_types):
                    if t_enum in u.type:
                        type_features[t_idx] = 1.0

                res.extend(features + type_features)
            else:
                res.extend([0.0] * self.unit_features)
        return res
