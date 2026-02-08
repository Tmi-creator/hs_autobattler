from typing import List, Tuple

from .combat import CombatManager
from .effects import TRIGGER_REGISTRY, GOLDEN_TRIGGER_REGISTRY
from .entities import Player
from .enums import BattleOutcome
from .event_system import EventManager
from .pool import CardPool, SpellPool
from .tavern import TavernManager


class Game:
    def __init__(self):
        self.pool = CardPool()
        self.spell_pool = SpellPool()
        self.event_manager = EventManager(TRIGGER_REGISTRY, GOLDEN_TRIGGER_REGISTRY)
        self.tavern = TavernManager(self.pool, self.spell_pool, event_manager=self.event_manager)
        self.combat = CombatManager(event_manager=self.event_manager)

        self.players: List[Player] = [
            Player(uid=0, board=[], hand=[], health=30),
            Player(uid=1, board=[], hand=[], health=30)
        ]

        self.turn_count = 1
        self.game_over = False
        self.winner_id = None

        self.players_ready = {0: False, 1: False}

        for p in self.players:
            self.tavern.start_turn(p, self.turn_count)

    def step(self, player_idx: int, action_type: str, **kwargs) -> Tuple[bool, str]:
        """
        Выполняет действие агента.
        Возвращает: (Done, Info)
        """
        if self.game_over:
            return True, "Game Over"
        player = self.players[player_idx]
        info = "Unknown Action"

        if player.is_discovering and action_type != "DISCOVER_CHOICE":
            return False, "Must choose discovery"

        if self.players_ready.get(player_idx, False) and action_type != "END_TURN":
            return False, "Player already ready"

        if action_type == "END_TURN":
            self.tavern.end_turn(player)
            self.players_ready[player_idx] = True
            info = "Ready"
        elif action_type == "BUY":
            success, info = self.tavern.buy_unit(player, kwargs.get('index', -1))
        elif action_type == "SELL":
            success, info = self.tavern.sell_unit(player, kwargs.get('index', -1))
        elif action_type == "ROLL":
            success, info = self.tavern.roll_tavern(player)
        elif action_type == "UPGRADE":
            success, info = self.tavern.upgrade_tavern(player)
        elif action_type == "FREEZE":
            success, info = self.tavern.toggle_freeze(player)
        elif action_type == "PLAY":
            # kwargs: hand_index, insert_index, target_index
            h_idx = kwargs.get('hand_index', -1)
            i_idx = kwargs.get('insert_index', len(player.board))  # По умолчанию в конец
            t_idx = kwargs.get('target_index', -1)
            success, info = self.tavern.play_unit(player, h_idx, i_idx, t_idx)
        elif action_type == "SWAP":
            # kwargs: index_a, index_b
            a = kwargs.get('index_a', -1)
            b = kwargs.get('index_b', -1)
            success, info = self.tavern.swap_units(player, a, b)
        elif action_type == "DISCOVER_CHOICE":
            # kwargs: index
            success, info = self.tavern.make_discovery_choice(player, kwargs.get('index', -1))

        if all(self.players_ready.values()):
            self._resolve_combat_phase(player_idx)
            if not self.game_over:
                self.players_ready = {0: False, 1: False}

        return self.game_over, info

    def _resolve_combat_phase(self, current_agent_idx: int):
        """
        Проводит бой, начисляет урон, обновляет ход.
        """
        p0, p1 = self.players[0], self.players[1]

        result, damage = self.combat.resolve_combat(p0, p1)

        damage_val = abs(damage)

        if result == BattleOutcome.WIN:
            p1.health -= damage_val
        elif result == BattleOutcome.LOSE:
            p0.health -= damage_val

        if p0.health <= 0 or p1.health <= 0:
            self.game_over = True
            self.winner_id = 0 if (p0.health > 0 >= p1.health) else (
                1 if (p1.health > 0 >= p0.health) else None)
            return

        self.turn_count += 1
        for p in self.players:
            self.tavern.start_turn(p, self.turn_count)
