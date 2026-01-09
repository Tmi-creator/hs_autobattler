from .entities import Unit
import random

from .effects import TRIGGER_REGISTRY
from .enums import Tags
from .event_system import (
    EntityRef,
    Event,
    EventManager,
    EventType,
    MinionSnapshot,
    PosRef,
    TriggerDef,
    TriggerInstance,
    Zone,
)


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
            player_1.uid: player_1.combat_copy(),
            player_2.uid: player_2.combat_copy(),
        }

        boards = [board_1, board_2]

        self.event_manager.process_event(
            Event(event_type=EventType.START_OF_COMBAT),
            combat_players,
            self.get_uid,
        )

        if len(board_1) > len(board_2):
            attacker_player_idx = 0
        elif len(board_2) > len(board_1):
            attacker_player_idx = 1
        else:
            attacker_player_idx = random.choice([0, 1])

        attack_indices = [0, 0]

        while True:
            end_battle = self.check_end_of_battle(board_1, board_2, player_1, player_2, combat_players)
            if end_battle[0] != "NO END":
                return end_battle
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

                self.perform_attack(attacker_unit, target, combat_players)

                self.cleanup_dead(boards, attack_indices, combat_players)

                if not attacker_unit.is_alive:
                    break
                end_battle = self.check_end_of_battle(board_1, board_2, player_1, player_2, combat_players)
                if end_battle[0] != "NO END":
                    return end_battle

            if attacker_unit.is_alive:
                attack_indices[attacker_player_idx] += 1

            attacker_player_idx = 1 - attacker_player_idx

    def check_end_of_battle(self, board_1, board_2, player_1, player_2, combat_players):
        if not board_1 and not board_2:
            self.event_manager.process_event(
                Event(event_type=EventType.END_OF_COMBAT),
                combat_players,
                self.get_uid,
            )
            return "DRAW", 0
        if not board_1:
            damage = sum(u.tier for u in board_2) + player_2.tavern_tier
            self.event_manager.process_event(
                Event(event_type=EventType.END_OF_COMBAT),
                combat_players,
                self.get_uid,
            )
            return "LOSE", -damage
        if not board_2:
            damage = sum(u.tier for u in board_1) + player_1.tavern_tier
            self.event_manager.process_event(
                Event(event_type=EventType.END_OF_COMBAT),
                combat_players,
                self.get_uid,
            )
            return "WIN", damage
        return "NO END", 0

    def perform_attack(self, attacker, target, combat_players):
        """
        Реализация атаки существа со всеми доп механиками
        """
        attacker_ref = EntityRef(attacker.uid)
        target_ref = EntityRef(target.uid)
        attacker_pos = self._find_pos(combat_players, attacker.uid)
        target_pos = self._find_pos(combat_players, target.uid)
        self.event_manager.process_event(
            Event(
                event_type=EventType.ATTACK_DECLARED,
                source=attacker_ref,
                target=target_ref,
                source_pos=attacker_pos,
                target_pos=target_pos,
            ),
            combat_players,
            self.get_uid,
        )
        dmg_to_target = attacker.cur_atk
        dmg_to_attacker = target.cur_atk
        pre_target_hp = target.cur_hp
        pre_attacker_hp = attacker.cur_hp
        if dmg_to_target > 0 and target.has_divine_shield:
            target.tags.discard(Tags.DIVINE_SHIELD)
        else:
            target.cur_hp -= dmg_to_target
            if attacker.has_poisonous or attacker.has_venomous:
                target.cur_hp = 0
                attacker.tags.discard(Tags.VENOMOUS)
        if dmg_to_attacker > 0 and attacker.has_divine_shield:
            attacker.tags.discard(Tags.DIVINE_SHIELD)
        else:
            attacker.cur_hp -= dmg_to_attacker
            if target.has_poisonous or target.has_venomous:
                attacker.cur_hp = 0
                target.tags.discard(Tags.VENOMOUS)

        actual_damage_to_target = max(0, pre_target_hp - target.cur_hp)
        actual_damage_to_attacker = max(0, pre_attacker_hp - attacker.cur_hp)

        if actual_damage_to_target > 0:
            self.event_manager.process_event(
                Event(
                    event_type=EventType.MINION_DAMAGED,
                    source=attacker_ref,
                    target=target_ref,
                    source_pos=attacker_pos,
                    target_pos=target_pos,
                    value=actual_damage_to_target,
                ),
                combat_players,
                self.get_uid,
            )
            self.event_manager.process_event(
                Event(
                    event_type=EventType.DAMAGE_DEALT,
                    source=attacker_ref,
                    target=target_ref,
                    source_pos=attacker_pos,
                    target_pos=target_pos,
                    value=actual_damage_to_target,
                ),
                combat_players,
                self.get_uid,
            )
        if actual_damage_to_attacker > 0:
            self.event_manager.process_event(
                Event(
                    event_type=EventType.MINION_DAMAGED,
                    source=target_ref,
                    target=attacker_ref,
                    source_pos=target_pos,
                    target_pos=attacker_pos,
                    value=actual_damage_to_attacker,
                ),
                combat_players,
                self.get_uid,
            )
            self.event_manager.process_event(
                Event(
                    event_type=EventType.DAMAGE_DEALT,
                    source=target_ref,
                    target=attacker_ref,
                    source_pos=target_pos,
                    target_pos=attacker_pos,
                    value=actual_damage_to_attacker,
                ),
                combat_players,
                self.get_uid,
            )

        self.event_manager.process_event(
            Event(
                event_type=EventType.AFTER_ATTACK,
                source=attacker_ref,
                target=target_ref,
                source_pos=attacker_pos,
                target_pos=target_pos,
            ),
            combat_players,
            self.get_uid,
        )

    def _collect_death_triggers(self, unit, slot_index):
        trigger_defs = self.event_manager.trigger_registry.get(unit.card_id, [])
        triggers = []
        for trigger_def in trigger_defs:
            if trigger_def.event_type == EventType.MINION_DIED:
                triggers.append(
                    TriggerInstance(
                        trigger_def=trigger_def,
                        trigger_uid=unit.uid,
                    )
                )
        for attached in (unit.attached_perm, unit.attached_turn, unit.attached_combat):
            for index, count in attached.items():
                if count <= 0:
                    continue
                trigger_defs = self.event_manager.trigger_registry.get(index, [])
                for trigger_def in trigger_defs:
                    if trigger_def.event_type == EventType.MINION_DIED:
                        triggers.append(
                            TriggerInstance(
                                trigger_def=trigger_def,
                                trigger_uid=unit.uid,
                                stacks=count,
                            )
                        )
        if unit.has_reborn:
            def _reborn_effect(ctx, event, trigger_uid, card_id=unit.card_id):
                if not event.source_pos:
                    return
                summoned_ref = ctx.summon(event.source_pos.side, card_id, event.source_pos.slot)
                if summoned_ref:
                    reborn_unit = ctx.resolve_unit(summoned_ref)
                    if reborn_unit:
                        reborn_unit.cur_hp = 1
                        reborn_unit.tags.discard(Tags.REBORN)

            triggers.append(
                TriggerInstance(
                    trigger_def=TriggerDef(
                        event_type=EventType.MINION_DIED,
                        condition=lambda ctx, event,
                                         trigger_uid: event.source is not None and event.source.uid == trigger_uid,
                        effect=_reborn_effect,
                        name="Reborn",
                    ),
                    trigger_uid=unit.uid,
                )
            )
        return triggers

    def _find_pos(self, combat_players, uid):
        for side, player in combat_players.items():
            for slot, unit in enumerate(player.board):
                if unit.uid == uid:
                    return PosRef(side=side, zone=Zone.BOARD, slot=slot)
        return None

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
                    death_snapshot = MinionSnapshot(
                        uid=unit.uid,
                        card_id=unit.card_id,
                        owner_id=unit.owner_id,
                        pos=PosRef(side=unit.owner_id, zone=Zone.BOARD, slot=i),
                        atk=unit.cur_atk,
                        hp=unit.cur_hp,
                        types=list(unit.type),
                        tags=set(unit.tags),
                    )
                    death_event = Event(
                        event_type=EventType.MINION_DIED,
                        source=EntityRef(unit.uid),
                        source_pos=death_snapshot.pos,
                        snapshot=death_snapshot,
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
