import random
from typing import List, Tuple, Dict
from .entities import Player, Unit
from .pool import CardPool
from .tavern import TavernManager
from .combat import Combat_Manager


class Game:
    def __init__(self):
        self.pool = CardPool()
        self.tavern = TavernManager(self.pool)
        self.combat = Combat_Manager()

        self.players: List[Player] = [
            Player(uid=0, board=[], hand=[], tavern_tier=1, gold=3, health=30),
            Player(uid=1, board=[], hand=[], tavern_tier=1, gold=3, health=30)
        ]

        self.turn_count = 1
        self.game_over = False
        self.winner_id = None

        self.players_ready = {0: False, 1: False}

        for p in self.players:
            self.tavern.start_turn(p, self.turn_count)

    def step(self, player_idx: int, action_type: str, **kwargs) -> Tuple[float, bool, str]:
        """
        Выполняет действие агента.
        Возвращает: (Reward, Done, Info)
        """
        if self.game_over:
            return 0, True, "Game Over"

        player = self.players[player_idx]
        reward = 0

        if action_type == "END_TURN":
            self.players_ready[player_idx] = True
            info = "Ready"

        elif action_type == "BUY":
            success, msg = self.tavern.buy_unit(player, kwargs.get('index', -1))
            if not success: reward = -0.1
            info = msg

        elif action_type == "SELL":
            success, msg = self.tavern.sell_unit(player, kwargs.get('index', -1))
            if not success: reward = -0.1
            info = msg

        elif action_type == "ROLL":
            success, msg = self.tavern.roll_tavern(player)
            if not success: reward = -0.1
            info = msg

        elif action_type == "UPGRADE":
            success, msg = self.tavern.upgrade_tavern(player)
            if not success: reward = -0.1
            info = msg

        elif action_type == "FREEZE":
            success, msg = self.tavern.toggle_freeze(player)
            info = msg

        elif action_type == "PLAY":
            # kwargs: hand_index, insert_index, target_index
            h_idx = kwargs.get('hand_index', -1)
            i_idx = kwargs.get('insert_index', len(player.board))  # По умолчанию в конец
            t_idx = kwargs.get('target_index', -1)

            success, msg = self.tavern.play_unit(player, h_idx, i_idx, t_idx)
            if not success:
                reward = -0.1
            else:
                reward = 0.1
            info = msg
        else:
            reward = -0.1
            info = "Unknown Action"

        combat_reward = 0
        if all(self.players_ready.values()):
            combat_reward = self._resolve_combat_phase(player_idx)
            if not self.game_over:
                self.players_ready = {0: False, 1: False}

        total_reward = reward + combat_reward

        return total_reward, self.game_over, info

    def _resolve_combat_phase(self, current_agent_idx: int) -> float:
        """
        Проводит бой, начисляет урон, обновляет ход.
        Возвращает награду ДЛЯ ТЕКУЩЕГО АГЕНТА (player_idx).
        """
        p0, p1 = self.players[0], self.players[1]

        result, damage = self.combat.resolve_combat(p0, p1)

        damage_val = abs(damage)

        if result == "WIN":
            p1.health -= damage_val
        elif result == "LOSE":
            p0.health -= damage_val

        agent_reward = 0

        if current_agent_idx == 0:
            if result == "WIN":
                agent_reward = damage_val
            elif result == "LOSE":
                agent_reward = -damage_val
        else:
            if result == "LOSE":
                agent_reward = damage_val
            elif result == "WIN":
                agent_reward = -damage_val

        if p0.health <= 0 and p1.health > 0:
            if current_agent_idx == 1:
                agent_reward += 10
            else:
                agent_reward -= 10
        elif p1.health <= 0 and p0.health > 0:
            if current_agent_idx == 0:
                agent_reward += 10
            else:
                agent_reward -= 10

        if p0.health <= 0 or p1.health <= 0:
            self.game_over = True
            return agent_reward

        self.turn_count += 1
        for p in self.players:
            self.tavern.start_turn(p, self.turn_count)

        return agent_reward
