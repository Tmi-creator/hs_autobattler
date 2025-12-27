from __future__ import annotations

from typing import Dict, List

from .enums import UnitType
from .event_system import Event, EventType, TriggerDef, Zone, TargetRef


def _is_self_play(ctx, event: Event, trigger_ref: TargetRef) -> bool:
    return event.source == trigger_ref


def _played_unit(ctx, event: Event):
    return ctx.resolve_unit(event.source)


def _gain_coin(ctx, event: Event, trigger_ref: TargetRef) -> None:
    ctx.gain_gold(trigger_ref.side, 1)


def _summon_tabbycat(ctx, event: Event, trigger_ref: TargetRef) -> None:
    ctx.summon(trigger_ref.side, "102t", trigger_ref.slot + 1)


def _wrath_weaver_buff(ctx, event: Event, trigger_ref: TargetRef) -> None:
    unit = _played_unit(ctx, event)
    if not unit:
        return
    if UnitType.DEMON not in unit.type:
        return
    if unit.card_id == "101":
        return
    for slot, board_unit in ctx.iter_board_units(trigger_ref.side):
        if board_unit.card_id == "101":
            ctx.damage_hero(trigger_ref.side, 1)
            ctx.buff(TargetRef(side=trigger_ref.side, zone=Zone.BOARD, slot=slot), 2, 1)


def _swampstriker_buff(ctx, event: Event, trigger_ref: TargetRef) -> None:
    unit = _played_unit(ctx, event)
    if not unit:
        return
    if UnitType.MURLOC not in unit.type:
        return
    for slot, board_unit in ctx.iter_board_units(trigger_ref.side):
        if board_unit.card_id == "104" and board_unit.uid != unit.uid:
            ctx.buff(TargetRef(side=trigger_ref.side, zone=Zone.BOARD, slot=slot), 1, 0)


TRIGGER_REGISTRY: Dict[str, List[TriggerDef]] = {
    "107": [
        TriggerDef(
            event_type=EventType.MINION_PLAYED,
            condition=_is_self_play,
            effect=_gain_coin,
            name="Shell Collector Battlecry",
        )
    ],
    "102": [
        TriggerDef(
            event_type=EventType.MINION_PLAYED,
            condition=_is_self_play,
            effect=_summon_tabbycat,
            name="Alleycat Battlecry",
        )
    ],
    "101": [
        TriggerDef(
            event_type=EventType.MINION_PLAYED,
            condition=lambda ctx, event, ref: True,
            effect=_wrath_weaver_buff,
            name="Wrath Weaver Trigger",
        )
    ],
    "104": [
        TriggerDef(
            event_type=EventType.MINION_PLAYED,
            condition=lambda ctx, event, ref: True,
            effect=_swampstriker_buff,
            name="Swampstriker Trigger",
        )
    ],
}
