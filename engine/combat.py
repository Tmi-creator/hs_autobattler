from .entities import Unit, Player
import random

from .effects import TRIGGER_REGISTRY
from .event_system import Event, EventManager, EventType, TargetRef, TriggerDef, TriggerInstance, Zone


class Combat_Manager:
    def __init__(self):
        self.uid = 10000
        self.event_manager = EventManager(TRIGGER_REGISTRY)

    def get_uid(self):
        self.uid += 1
        return self.uid

    def resolve_combat(self, player_1, player_2):
        board_1 = [u.combat_copy() for u in player_1.board]
        board_2 = [u.combat_copy() for u in player_2.board]
        combat_players = {
            player_1.uid: player_1.__class__(
                uid=player_1.uid,
                board=board_1,
                hand=[],
                store=[],
                tavern_tier=player_1.tavern_tier,
                gold=0,
                health=player_1.health,
                up_cost=player_1.up_cost,
            ),
            player_2.uid: player_2.__class__(
                uid=player_2.uid,
                board=board_2,
                hand=[],
                store=[],
                tavern_tier=player_2.tavern_tier,
                gold=0,
                health=player_2.health,
                up_cost=player_2.up_cost,
            ),
        }

        boards = [board_1, board_2]

        # 2. Фаза "Start of Combat" (Начало боя)

        if len(board_1) > len(board_2):
            attacker_player_idx = 0
        elif len(board_2) > len(board_1):
            attacker_player_idx = 1
        else:
            attacker_player_idx = random.choice([0, 1])

        attack_indices = [0, 0]

        while True:
            if not board_1 and not board_2:
                return "DRAW", 0
            if not board_1:
                damage = sum(u.tier for u in board_2) + player_2.tavern_tier
                return "LOSE", -damage
            if not board_2:
                damage = sum(u.tier for u in board_1) + player_1.tavern_tier
                return "WIN", damage

            attacker_board = boards[attacker_player_idx]
            defender_board = boards[1 - attacker_player_idx]

            if attack_indices[attacker_player_idx] >= len(attacker_board):
                attack_indices[attacker_player_idx] = 0

            attacker_idx = attack_indices[attacker_player_idx]
            attacker_unit = attacker_board[attacker_idx]
            num_attacks = 1
            if attacker_unit.has_windfury:
                num_attacks += 1
            for i in range(num_attacks):
                taunts = [u for u in defender_board if u.has_taunt]
                if taunts:
                    target = random.choice(taunts)
                else:
                    target = random.choice(defender_board)

                self.perform_attack(attacker_unit, target)

                self.cleanup_dead(boards, attack_indices, combat_players)

                if not attacker_unit.is_alive:
                    break
                if not board_1 and not board_2:
                    return "DRAW", 0
                if not board_1:
                    damage = sum(u.tier for u in board_2) + player_2.tavern_tier
                    return "LOSE", -damage
                if not board_2:
                    damage = sum(u.tier for u in board_1) + player_1.tavern_tier
                    return "WIN", damage

            if attacker_unit.is_alive:
                attack_indices[attacker_player_idx] += 1

            attacker_player_idx = 1 - attacker_player_idx

    def perform_attack(self, attacker, target):
        """
        Реализация атаки существа со всеми доп механиками
        """
        dmg_to_target = attacker.cur_atk
        dmg_to_attacker = target.cur_atk
        if dmg_to_target > 0 and target.has_divine_shield:
            target.has_divine_shield = False
        else:
            target.cur_hp -= dmg_to_target
            if attacker.has_poisonous or attacker.has_venomous:
                target.cur_hp = 0
                attacker.has_venomous = False
        if dmg_to_attacker > 0 and attacker.has_divine_shield:
            attacker.has_divine_shield = False
        else:
            attacker.cur_hp -= dmg_to_attacker
            if target.has_poisonous or target.has_venomous:
                attacker.cur_hp = 0
                target.has_venomous = False

    def _collect_death_triggers(self, unit, slot_index):
        trigger_defs = self.event_manager.trigger_registry.get(unit.card_id, [])
        triggers = []
        for trigger_def in trigger_defs:
            if trigger_def.event_type == EventType.MINION_DIED:
                triggers.append(
                    TriggerInstance(
                        trigger_def=trigger_def,
                        trigger_ref=TargetRef(side=unit.owner_id, zone=Zone.BOARD, slot=slot_index),
                    )
                )
        if unit.has_reborn:
            def _reborn_effect(ctx, event, trigger_ref, card_id=unit.card_id):
                summoned_ref = ctx.summon(trigger_ref.side, card_id, trigger_ref.slot)
                if summoned_ref:
                    reborn_unit = ctx.resolve_unit(summoned_ref)
                    if reborn_unit:
                        reborn_unit.cur_hp = 1
                        reborn_unit.has_reborn = False

            triggers.append(
                TriggerInstance(
                    trigger_def=TriggerDef(
                        event_type=EventType.MINION_DIED,
                        condition=lambda ctx, event, ref: event.source == ref,
                        effect=_reborn_effect,
                        name="Reborn",
                    ),
                    trigger_ref=TargetRef(side=unit.owner_id, zone=Zone.BOARD, slot=slot_index),
                )
            )
        return triggers

    def cleanup_dead(self, boards, attack_indices, combat_players):
        """
        Чистим стол при смерти существ и двигаем индекс атаки куда надо
        """
        for p_idx in range(2):
            board = boards[p_idx]
            i = 0
            while i < len(board):
                unit = board[i]

                if not unit.is_alive:
                    death_event = Event(
                        event_type=EventType.MINION_DIED,
                        source=TargetRef(side=unit.owner_id, zone=Zone.BOARD, slot=i),
                    )
                    extra_triggers = self._collect_death_triggers(unit, i)

                    board.pop(i)

                    if i < attack_indices[p_idx]:
                        attack_indices[p_idx] -= 1

                    before_len = len(board)
                    self.event_manager.process_event(
                        death_event,
                        combat_players,
                        self.get_uid,
                        extra_triggers=extra_triggers,
                    )
                    units_added = len(board) - before_len

                    if i < attack_indices[p_idx]:
                        attack_indices[p_idx] += units_added
                    i += units_added

                else:
                    i += 1
