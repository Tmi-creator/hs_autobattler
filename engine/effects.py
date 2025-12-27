from __future__ import annotations

from typing import Dict, List

from .enums import UnitType
from .event_system import EntityRef, Event, EventType, TriggerDef


def _is_self_play(ctx, event: Event, trigger_uid: int) -> bool:
    return event.source is not None and event.source.uid == trigger_uid


def _played_unit(ctx, event: Event):
    return ctx.resolve_unit(event.source)


def _gain_coin(ctx, event: Event, trigger_uid: int) -> None:
    pos = ctx.resolve_pos(EntityRef(trigger_uid))
    if not pos:
        return
    ctx.gain_gold(pos.side, 1)


def _summon_tabbycat(ctx, event: Event, trigger_uid: int) -> None:
    pos = ctx.resolve_pos(EntityRef(trigger_uid))
    if not pos:
        return
    ctx.summon(pos.side, "102t", pos.slot + 1)


def _summon_scallywag_token(ctx, event: Event, trigger_uid: int) -> None:
    pos = event.source_pos or (event.snapshot.pos if event.snapshot else None)
    if pos:
        ctx.summon(pos.side, "103t", pos.slot)


def _summon_imprisoner_token(ctx, event: Event, trigger_uid: int) -> None:
    pos = event.source_pos or (event.snapshot.pos if event.snapshot else None)
    if pos:
        ctx.summon(pos.side, "108t", pos.slot)


def _wrath_weaver_buff(ctx, event: Event, trigger_uid: int) -> None:
    unit = _played_unit(ctx, event)
    if not unit:
        return
    if UnitType.DEMON not in unit.type:
        return
    if unit.card_id == "101":
        return
    pos = ctx.resolve_pos(EntityRef(trigger_uid))
    if not pos:
        return
    for slot, board_unit in ctx.iter_board_units(pos.side):
        if board_unit.card_id == "101":
            ctx.damage_hero(pos.side, 1)
            ctx.buff(EntityRef(board_unit.uid), 2, 1)


def _swampstriker_buff(ctx, event: Event, trigger_uid: int) -> None:
    unit = _played_unit(ctx, event)
    if not unit:
        return
    if UnitType.MURLOC not in unit.type:
        return
    pos = ctx.resolve_pos(EntityRef(trigger_uid))
    if not pos:
        return
    for slot, board_unit in ctx.iter_board_units(pos.side):
        if board_unit.card_id == "104" and board_unit.uid != unit.uid:
            ctx.buff(EntityRef(board_unit.uid), 1, 0)


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
    "103": [
        TriggerDef(
            event_type=EventType.MINION_DIED,
            condition=_is_self_play,
            effect=_summon_scallywag_token,
            name="Scallywag Deathrattle",
        )
    ],
    "108": [
        TriggerDef(
            event_type=EventType.MINION_DIED,
            condition=_is_self_play,
            effect=_summon_imprisoner_token,
            name="Imprisoner Deathrattle",
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
