from typing import List

from .entities import Unit, Player
import random
from .auras import recalculate_board_auras
from .effects import TRIGGER_REGISTRY, GOLDEN_TRIGGER_REGISTRY
from .enums import Tags, BattleOutcome
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


class CombatManager:
    def __init__(self, event_manager: EventManager | None = None):
        self.uid = 10000
        self.event_manager = event_manager or EventManager(TRIGGER_REGISTRY, GOLDEN_TRIGGER_REGISTRY)

    def get_uid(self):
        self.uid += 1
        return self.uid

    def resolve_combat(self, player_1: Player, player_2: Player) -> tuple[BattleOutcome, int]:
        combat_players = {
            player_1.uid: player_1.combat_copy(),
            player_2.uid: player_2.combat_copy(),
        }

        boards = [i.board for i in combat_players.values()]
        board_1, board_2 = boards
        recalculate_board_auras(board_1)
        recalculate_board_auras(board_2)
        self.event_manager.process_event(
            Event(event_type=EventType.START_OF_COMBAT),
            combat_players,
            self.get_uid,
        )
        attack_indices = [0, 0]
        self.cleanup_dead(boards, attack_indices, combat_players)

        if len(board_1) > len(board_2):
            attacker_player_idx = 0
        elif len(board_2) > len(board_1):
            attacker_player_idx = 1
        else:
            attacker_player_idx = random.choice([0, 1])

        def _find_target(target_board):
            taunts = [u for u in target_board if u.has_taunt]
            if taunts:
                return random.choice(taunts)
            return random.choice(target_board)

        can_attack = [1, 1]
        while True:
            end_battle = self.check_end_of_battle(board_1, board_2, player_1, player_2, combat_players)
            if end_battle[0] != BattleOutcome.NO_END:
                return end_battle
            if can_attack == [0, 0]:
                return BattleOutcome.DRAW, 0
            if can_attack[attacker_player_idx] == 0:
                attacker_player_idx = 1 - attacker_player_idx
                continue
            # 1. Immediate Attack Batch
            while True:
                attack_queue = []
                # Active Player First
                scan_order = [attacker_player_idx, 1 - attacker_player_idx]
                for side in scan_order:
                    board = boards[side]
                    for unit in board:
                        if unit.is_alive and Tags.IMMEDIATE_ATTACK in unit.tags:
                            attack_queue.append(unit)
                            # discard RIGHT NOW because it goes infinite
                            unit.tags.discard(Tags.IMMEDIATE_ATTACK)
                if not attack_queue:
                    break
                # 1.2 Execute attacks
                for unit in attack_queue:
                    # Unit can die while wait its order
                    if not unit.is_alive:
                        continue

                    attacker_side = -1  # find side by unit side
                    if unit in boards[0]:
                        attacker_side = 0
                    elif unit in boards[1]:
                        attacker_side = 1

                    # how?
                    if attacker_side == -1:
                        print("what the f")
                        continue

                    enemy_side = 1 - attacker_side
                    target = _find_target(boards[enemy_side])

                    if target:
                        self.perform_attack(unit, target, combat_players)
                        # clean after every attack
                        self.cleanup_dead(boards, attack_indices, combat_players)
                # re-scan
                continue
            attacker_board = boards[attacker_player_idx]
            defender_board = boards[1 - attacker_player_idx]

            if attack_indices[attacker_player_idx] >= len(attacker_board):
                attack_indices[attacker_player_idx] = 0

            attacker_idx = attack_indices[attacker_player_idx]
            make_attack = False
            for i in range(len(attacker_board)):
                if attacker_board[attacker_idx].cur_atk == 0:
                    attacker_idx += 1
                    if attacker_idx >= len(attacker_board):
                        attacker_idx = 0
                else:
                    make_attack = True
                    break

            if not make_attack:
                can_attack[attacker_player_idx] = 0
                continue
            attacker_unit = attacker_board[attacker_idx]
            num_attacks = 1
            if attacker_unit.has_windfury:
                num_attacks += 1

            for i in range(num_attacks):
                target = _find_target(defender_board)

                self.perform_attack(attacker_unit, target, combat_players)

                self.cleanup_dead(boards, attack_indices, combat_players)

                if not attacker_unit.is_alive:
                    break
                end_battle = self.check_end_of_battle(board_1, board_2, player_1, player_2, combat_players)
                if end_battle[0] != BattleOutcome.NO_END:
                    return end_battle

            if attacker_unit.is_alive:
                attack_indices[attacker_player_idx] = attacker_idx + 1

            attacker_player_idx = 1 - attacker_player_idx

    def check_end_of_battle(self, board_1: List[Unit], board_2: List[Unit], player_1: Player, player_2: Player,
                            combat_players: dict[int, Player]) -> tuple[BattleOutcome, int]:
        if not board_1 or not board_2:
            self.event_manager.process_event(
                Event(event_type=EventType.END_OF_COMBAT),
                combat_players,
                self.get_uid,
            )
        if not board_1 and not board_2:
            return BattleOutcome.DRAW, 0
        if not board_1:
            damage = sum(u.tier for u in board_2) + player_2.tavern_tier
            return BattleOutcome.LOSE, -damage
        if not board_2:
            damage = sum(u.tier for u in board_1) + player_1.tavern_tier
            return BattleOutcome.WIN, damage
        return BattleOutcome.NO_END, 0

    def perform_attack(self, attacker: Unit, target: Unit, combat_players: dict[int, Player]) -> None:
        """
        Perform attack with all additional mechanics
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
        victims_data = []

        if target_pos:
            victims_data.append((target, target_pos, target_ref))

        if attacker.has_cleave and target_pos:
            defender_player = combat_players[target_pos.side]
            defender_board = defender_player.board

            real_idx = -1
            for i, u in enumerate(defender_board):
                if u.uid == target.uid:
                    real_idx = i
                    break

            if real_idx != -1:
                if real_idx > 0:
                    left_u = defender_board[real_idx - 1]
                    left_pos = self._find_pos(combat_players, left_u.uid)
                    left_ref = EntityRef(left_u.uid)
                    victims_data.insert(0, (left_u, left_pos, left_ref))

                if real_idx < len(defender_board) - 1:
                    right_u = defender_board[real_idx + 1]
                    right_pos = self._find_pos(combat_players, right_u.uid)
                    right_ref = EntityRef(right_u.uid)
                    victims_data.append((right_u, right_pos, right_ref))

        def _apply_damage_batch(source_unit, source_ref, source_pos, targets_list):
            dmg_amount = source_unit.cur_atk
            if dmg_amount <= 0:
                return
            has_poison = source_unit.has_poisonous
            has_venom = source_unit.has_venomous
            venom_used = False
            for (victim_unit, victim_pos, victim_ref) in targets_list:
                if not victim_unit.is_alive:
                    continue
                hp_before = victim_unit.cur_hp
                if victim_unit.has_divine_shield:
                    victim_unit.tags.discard(Tags.DIVINE_SHIELD)
                    actual_damage = 0
                    self.event_manager.process_event(
                        Event(
                            event_type=EventType.DIVINE_SHIELD_LOST,
                            source=victim_ref,
                            target=source_ref,
                            source_pos=victim_pos,
                            target_pos=source_pos,
                        ),
                        combat_players,
                        self.get_uid,
                    )
                else:
                    victim_unit.cur_hp -= dmg_amount
                    actual_damage = dmg_amount
                    if has_poison or has_venom:
                        victim_unit.cur_hp = 0
                        if has_venom:
                            venom_used = True

                if actual_damage > 0 and actual_damage > hp_before:
                    self.event_manager.process_event(
                        Event(
                            event_type=EventType.OVERKILL,
                            source=source_ref,
                            target=victim_ref,
                            source_pos=source_pos,
                            target_pos=victim_pos,
                            value=actual_damage - hp_before  # how much overdmg
                        ),
                        combat_players,
                        self.get_uid,
                    )

                if actual_damage > 0:
                    self.event_manager.process_event(
                        Event(
                            event_type=EventType.MINION_DAMAGED,
                            source=source_ref,
                            target=victim_ref,
                            source_pos=source_pos,
                            target_pos=victim_pos,
                            value=actual_damage,
                        ),
                        combat_players,
                        self.get_uid,
                    )
                    self.event_manager.process_event(
                        Event(
                            event_type=EventType.DAMAGE_DEALT,
                            source=source_ref,
                            target=victim_ref,
                            source_pos=source_pos,
                            target_pos=victim_pos,
                            value=actual_damage,
                        ),
                        combat_players,
                        self.get_uid,
                    )
            if venom_used:
                source_unit.tags.discard(Tags.VENOMOUS)

        _apply_damage_batch(attacker, attacker_ref, attacker_pos, victims_data)
        _apply_damage_batch(target, target_ref, target_pos, [(attacker, attacker_pos, attacker_ref)])
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

    def _collect_death_triggers(self, unit: Unit, slot_index: int) -> List[TriggerInstance]:
        triggers = []

        stacks_multiplier = 1

        if unit.is_golden:
            if unit.card_id in self.event_manager.golden_trigger_registry:
                trigger_defs = self.event_manager.golden_trigger_registry[unit.card_id]
            else:
                trigger_defs = self.event_manager.trigger_registry.get(unit.card_id, [])
                stacks_multiplier = 2
        else:
            trigger_defs = self.event_manager.trigger_registry.get(unit.card_id, [])

        for trigger_def in trigger_defs:
            if trigger_def.event_type == EventType.MINION_DIED:
                triggers.append(
                    TriggerInstance(
                        trigger_def=trigger_def,
                        trigger_uid=unit.uid,
                        stacks=stacks_multiplier
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
                summoned_ref = ctx.summon(event.source_pos.side, card_id, event.source_pos.slot,
                                          is_golden=unit.is_golden)
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

    def _find_pos(self, combat_players: dict[int, Player], uid: int) -> PosRef | None:
        for side, player in combat_players.items():
            for slot, unit in enumerate(player.board):
                if unit.uid == uid:
                    return PosRef(side=side, zone=Zone.BOARD, slot=slot)
        return None

    def cleanup_dead(self, boards: List[List[Unit]], attack_indices: List[int],
                     combat_players: dict[int, Player]) -> None:
        """
        Clean board after death and move attack indexes where they should be
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
                        types=list(unit.types),
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
                    recalculate_board_auras(board)

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
        recalculate_board_auras(boards[0])
        recalculate_board_auras(boards[1])
