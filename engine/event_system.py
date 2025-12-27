from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum, auto
from typing import Callable, Deque, Dict, Iterable, List, Optional

from .entities import Player, Unit


class Zone(Enum):
    BOARD = auto()
    HAND = auto()
    SHOP = auto()
    HERO = auto()


class EventType(Enum):
    MINION_PLAYED = auto()
    MINION_DIED = auto()


@dataclass(frozen=True)
class TargetRef:
    side: int
    zone: Zone
    slot: int


@dataclass(frozen=True)
class Event:
    event_type: EventType
    source: TargetRef
    target: Optional[TargetRef] = None
    value: Optional[int] = None
    meta: Optional[int] = None


ConditionFn = Callable[["EffectContext", Event, TargetRef], bool]
EffectFn = Callable[["EffectContext", Event, TargetRef], None]


@dataclass(frozen=True)
class TriggerDef:
    event_type: EventType
    condition: ConditionFn
    effect: EffectFn
    name: str = ""


@dataclass(frozen=True)
class TriggerInstance:
    trigger_def: TriggerDef
    trigger_ref: TargetRef


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

    def resolve_unit(self, ref: TargetRef) -> Optional[Unit]:
        if ref.zone != Zone.BOARD:
            return None
        player = self.players_by_uid.get(ref.side)
        if not player:
            return None
        if ref.slot < 0 or ref.slot >= len(player.board):
            return None
        return player.board[ref.slot]

    def iter_board_units(self, side: int) -> Iterable[tuple[int, Unit]]:
        player = self.players_by_uid.get(side)
        if not player:
            return []
        return list(enumerate(player.board))

    def gain_gold(self, side: int, amount: int) -> None:
        player = self.players_by_uid.get(side)
        if player:
            player.gold += amount

    def damage_hero(self, side: int, amount: int) -> None:
        player = self.players_by_uid.get(side)
        if player:
            player.health -= amount

    def buff(self, target_ref: TargetRef, atk: int, hp: int) -> None:
        unit = self.resolve_unit(target_ref)
        if not unit:
            return
        unit.max_atk += atk
        unit.cur_atk += atk
        unit.max_hp += hp
        unit.cur_hp += hp

    def summon(self, side: int, card_id: str, insert_index: int) -> Optional[TargetRef]:
        player = self.players_by_uid.get(side)
        if not player:
            return None
        if len(player.board) >= 7:
            return None
        index = max(0, min(insert_index, len(player.board)))
        unit = Unit.create_from_db(card_id, self._uid_provider(), side)
        player.board.insert(index, unit)
        return TargetRef(side=side, zone=Zone.BOARD, slot=index)

    def emit_event(self, event: Event) -> None:
        self._event_queue.append(event)


class EffectExecutor:
    def run(self, effect: EffectFn, ctx: EffectContext, event: Event, trigger_ref: TargetRef) -> None:
        effect(ctx, event, trigger_ref)


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
                if trigger.trigger_def.condition(ctx, current_event, trigger.trigger_ref):
                    self.executor.run(trigger.trigger_def.effect, ctx, current_event, trigger.trigger_ref)

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
                                trigger_ref=TargetRef(side=player_id, zone=Zone.BOARD, slot=slot),
                            )
                        )
        return triggers

    def order_triggers(
        self,
        triggers: List[TriggerInstance],
        event: Event,
        ctx: EffectContext,
    ) -> List[TriggerInstance]:
        active_side = event.source.side if event.source else None

        def sort_key(trigger: TriggerInstance) -> tuple:
            unit = ctx.resolve_unit(trigger.trigger_ref)
            unit_uid = unit.uid if unit else 0
            side_priority = 0 if active_side is None or trigger.trigger_ref.side == active_side else 1
            return (side_priority, trigger.trigger_ref.slot, unit_uid)

        return sorted(triggers, key=sort_key)
