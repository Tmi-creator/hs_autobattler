from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum, auto
from typing import Callable, Deque, Dict, Iterable, List, Optional

from .attached_effects import EFFECT_ID_TO_INDEX, EFFECT_INDEX_TO_ID
from .entities import HandCard, Player, Spell, Unit


class Zone(Enum):
    BOARD = auto()
    HAND = auto()
    SHOP = auto()
    HERO = auto()


class EventType(Enum):
    MINION_PLAYED = auto()
    MINION_BOUGHT = auto()
    MINION_SOLD = auto()
    MINION_SUMMONED = auto()
    MINION_DIED = auto()
    MINION_DAMAGED = auto()
    DAMAGE_DEALT = auto()
    ATTACK_DECLARED = auto()
    AFTER_ATTACK = auto()
    START_OF_COMBAT = auto()
    END_OF_COMBAT = auto()
    START_OF_TURN = auto()
    END_OF_TURN = auto()
    SPELL_CAST = auto()


@dataclass(frozen=True)
class EntityRef:
    uid: int


@dataclass(frozen=True)
class PosRef:
    side: int
    zone: Zone
    slot: int


@dataclass(frozen=True)
class MinionSnapshot:
    uid: int
    card_id: str
    owner_id: int
    pos: Optional[PosRef]
    atk: int
    hp: int
    types: List
    flags: Dict[str, bool]


@dataclass(frozen=True)
class Event:
    event_type: EventType
    source: Optional[EntityRef] = None
    target: Optional[EntityRef] = None
    source_pos: Optional[PosRef] = None
    target_pos: Optional[PosRef] = None
    value: Optional[int] = None
    meta: Optional[int] = None
    snapshot: Optional[MinionSnapshot] = None


ConditionFn = Callable[["EffectContext", Event, int], bool]
EffectFn = Callable[["EffectContext", Event, int], None]


@dataclass(frozen=True)
class TriggerDef:
    event_type: EventType
    condition: ConditionFn
    effect: EffectFn
    name: str = ""


@dataclass(frozen=True)
class TriggerInstance:
    trigger_def: TriggerDef
    trigger_uid: int
    stacks: int = 1


class EffectContext:
    def __init__(
            self,
            players_by_uid: Dict[int, Player],
            uid_provider: Callable[[], int],
            event_queue: Deque[Event],
    ):
        self.players_by_uid = players_by_uid
        self._uid_provider = uid_provider
        self._event_queue = event_queue
        self._uid_to_pos: Dict[int, PosRef] = {}
        for player_id, player in players_by_uid.items():
            self._reindex_side(player_id, player.board)

    def resolve_unit(self, ref: Optional[EntityRef]) -> Optional[Unit]:
        if not ref:
            return None
        pos = self._uid_to_pos.get(ref.uid)
        if not pos or pos.zone != Zone.BOARD:
            return None
        player = self.players_by_uid.get(pos.side)
        if not player:
            return None
        if pos.slot < 0 or pos.slot >= len(player.board):
            return None
        return player.board[pos.slot]

    def iter_board_units(self, side: int) -> Iterable[tuple[int, Unit]]:
        player = self.players_by_uid.get(side)
        if not player:
            return []
        return list(enumerate(player.board))

    def resolve_pos(self, ref: Optional[EntityRef]) -> Optional[PosRef]:
        if not ref:
            return None
        return self._uid_to_pos.get(ref.uid)

    def _reindex_side(self, side: int, board: List[Unit]) -> None:
        for idx, unit in enumerate(board):
            self._uid_to_pos[unit.uid] = PosRef(side=side, zone=Zone.BOARD, slot=idx)

    def _reindex_all(self) -> None:
        for player_id, player in self.players_by_uid.items():
            self._reindex_side(player_id, player.board)

    def gain_gold(self, side: int, amount: int) -> None:
        player = self.players_by_uid.get(side)
        if player:
            player.gold += amount

    def add_spell_to_hand(self, side: int, spell_id: str) -> None:
        player = self.players_by_uid.get(side)
        if not player:
            return
        if len(player.hand) >= 10:
            return
        spell = Spell.create_from_db(spell_id)
        player.hand.append(HandCard(uid=self._uid_provider(), spell=spell))

    def damage_hero(self, side: int, amount: int) -> None:
        player = self.players_by_uid.get(side)
        if player:
            player.health -= amount

    def buff_perm(self, target_ref: EntityRef, atk: int, hp: int) -> None:
        unit = self.resolve_unit(target_ref)
        if not unit:
            return
        unit.perm_atk_add += atk
        unit.perm_hp_add += hp
        unit.recalc_stats()

    def buff(self, target_ref: EntityRef, atk: int, hp: int) -> None:
        self.buff_perm(target_ref, atk, hp)

    def buff_turn(self, target_ref: EntityRef, atk: int, hp: int) -> None:
        unit = self.resolve_unit(target_ref)
        if not unit:
            return
        unit.turn_atk_add += atk
        unit.turn_hp_add += hp
        unit.recalc_stats()

    def buff_combat(self, target_ref: EntityRef, atk: int, hp: int) -> None:
        unit = self.resolve_unit(target_ref)
        if not unit:
            return
        unit.combat_atk_add += atk
        unit.combat_hp_add += hp
        unit.recalc_stats()

    def attach_effect_perm(self, target_ref: EntityRef, effect_id: str, count: int = 1) -> None:
        unit = self.resolve_unit(target_ref)
        if not unit:
            return
        index = EFFECT_ID_TO_INDEX.get(effect_id)
        if index is None:
            return
        unit.attached_perm[index] += count

    def attach_effect_turn(self, target_ref: EntityRef, effect_id: str, count: int = 1) -> None:
        unit = self.resolve_unit(target_ref)
        if not unit:
            return
        index = EFFECT_ID_TO_INDEX.get(effect_id)
        if index is None:
            return
        unit.attached_turn[index] += count

    def attach_effect_combat(self, target_ref: EntityRef, effect_id: str, count: int = 1) -> None:
        unit = self.resolve_unit(target_ref)
        if not unit:
            return
        index = EFFECT_ID_TO_INDEX.get(effect_id)
        if index is None:
            return
        unit.attached_combat[index] += count

    def summon(self, side: int, card_id: str, insert_index: int) -> Optional[EntityRef]:
        player = self.players_by_uid.get(side)
        if not player:
            return None
        if len(player.board) >= 7:
            return None
        index = max(0, min(insert_index, len(player.board)))
        unit = Unit.create_from_db(card_id, self._uid_provider(), side)
        player.board.insert(index, unit)
        self._reindex_side(side, player.board)
        summoned = EntityRef(uid=unit.uid)
        pos = self._uid_to_pos.get(unit.uid)
        self.emit_event(
            Event(
                event_type=EventType.MINION_SUMMONED,
                source=summoned,
                source_pos=pos,
            )
        )
        return summoned

    def emit_event(self, event: Event) -> None:
        self._event_queue.append(event)


class EffectExecutor:
    def run(self, effect: EffectFn, ctx: EffectContext, event: Event, trigger_uid: int) -> None:
        effect(ctx, event, trigger_uid)


class EventManager:
    def __init__(self, trigger_registry: Dict[str, List[TriggerDef]], executor: Optional[EffectExecutor] = None):
        self.trigger_registry = trigger_registry
        self.executor = executor or EffectExecutor()

    def process_event(
            self,
            event: Event,
            players_by_uid: Dict[int, Player],
            uid_provider: Callable[[], int],
            extra_triggers: Optional[List[TriggerInstance]] = None,
    ) -> None:
        queue: Deque[Event] = deque([event])
        ctx = EffectContext(players_by_uid, uid_provider, queue)
        initial_event = event
        while queue:
            current_event = queue.popleft()
            triggers = self.collect_triggers(current_event, ctx)
            if extra_triggers and current_event == initial_event:
                triggers.extend(extra_triggers)
            for trigger in self.order_triggers(triggers, current_event, ctx):
                if trigger.trigger_def.condition(ctx, current_event, trigger.trigger_uid):
                    for _ in range(trigger.stacks):
                        self.executor.run(trigger.trigger_def.effect, ctx, current_event, trigger.trigger_uid)

    def collect_triggers(self, event: Event, ctx: EffectContext) -> List[TriggerInstance]:
        triggers: List[TriggerInstance] = []
        for player_id, player in ctx.players_by_uid.items():
            for slot, unit in enumerate(player.board):
                trigger_defs = self.trigger_registry.get(unit.card_id, [])
                for trigger_def in trigger_defs:
                    if trigger_def.event_type == event.event_type:
                        triggers.append(
                            TriggerInstance(
                                trigger_def=trigger_def,
                                trigger_uid=unit.uid,
                            )
                        )
                for attached in (unit.attached_perm, unit.attached_turn, unit.attached_combat):
                    for index, count in enumerate(attached):
                        if count <= 0:
                            continue
                        effect_id = EFFECT_INDEX_TO_ID[index]
                        if not effect_id:
                            continue
                        trigger_defs = self.trigger_registry.get(effect_id, [])
                        for trigger_def in trigger_defs:
                            if trigger_def.event_type == event.event_type:
                                triggers.append(
                                    TriggerInstance(
                                        trigger_def=trigger_def,
                                        trigger_uid=unit.uid,
                                        stacks=count,
                                    )
                                )
        return triggers

    def order_triggers(
            self,
            triggers: List[TriggerInstance],
            event: Event,
            ctx: EffectContext,
    ) -> List[TriggerInstance]:
        active_side = None
        if event.source_pos:
            active_side = event.source_pos.side
        elif event.source:
            pos = ctx.resolve_pos(event.source)
            active_side = pos.side if pos else None

        def sort_key(trigger: TriggerInstance) -> tuple:
            pos = ctx.resolve_pos(EntityRef(trigger.trigger_uid))
            unit = ctx.resolve_unit(EntityRef(trigger.trigger_uid))
            unit_uid = unit.uid if unit else 0
            slot = pos.slot if pos else 999
            side = pos.side if pos else -1
            side_priority = 0 if active_side is None or side == active_side else 1
            return (side_priority, slot, unit_uid)

        return sorted(triggers, key=sort_key)
