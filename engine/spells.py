from __future__ import annotations

from typing import Dict, List

from .event_system import EntityRef, Event, EventType, TriggerDef


def _spell_coin(ctx, event: Event, trigger_uid: int) -> None:
    if not event.source_pos:
        return
    ctx.gain_gold(event.source_pos.side, 1)


def _spell_banana(ctx, event: Event, trigger_uid: int) -> None:
    if not event.target:
        return
    ctx.buff(EntityRef(event.target.uid), 2, 2)


def _spell_bloodgem(ctx, event: Event, trigger_uid: int) -> None:
    if not event.target:
        return
    player = ctx.players_by_uid.get(event.source_pos.side)
    gem_atk, gem_hp = player.gem_atk, player.gem_hp
    ctx.buff(EntityRef(event.target.uid), 1 + gem_atk, 1 + gem_hp)


def _spell_arrow(ctx, event: Event, trigger_uid: int) -> None:
    if not event.target:
        return
    ctx.buff(EntityRef(event.target.uid), 4, 0)


def _spell_fortify(ctx, event: Event, trigger_uid: int) -> None:
    if not event.target:
        return
    ctx.buff(EntityRef(event.target.uid), 0, 3)
    unit = ctx.resolve_unit(EntityRef(event.target.uid))
    unit.has_taunt = True


def _spell_apple(ctx, event: Event, trigger_uid: int) -> None:
    if not event.target:
        return
    player = ctx.players_by_uid.get(event.source_pos.side)
    for item in player.store:
        if item.unit:
            ctx.buff(EntityRef(item.unit.uid), 1, 2)


SPELL_TRIGGER_REGISTRY: Dict[str, List[TriggerDef]] = {
    "S001": [
        TriggerDef(
            event_type=EventType.SPELL_CAST,
            condition=lambda ctx, event, ref: True,
            effect=_spell_coin,
            name="Tavern Coin",
        )
    ],
    "S002": [
        TriggerDef(
            event_type=EventType.SPELL_CAST,
            condition=lambda ctx, event, ref: True,
            effect=_spell_banana,
            name="Banana",
        )
    ],
    "S003": [
        TriggerDef(
            event_type=EventType.SPELL_CAST,
            condition=lambda ctx, event, ref: True,
            effect=_spell_bloodgem,
            name="Blood Gem",
        )
    ],
    "S004": [
        TriggerDef(
            event_type=EventType.SPELL_CAST,
            condition=lambda ctx, event, ref: True,
            effect=_spell_arrow,
            name="Pointy Arrow",
        )
    ],
    "S005": [
        TriggerDef(
            event_type=EventType.SPELL_CAST,
            condition=lambda ctx, event, ref: True,
            effect=_spell_fortify,
            name="Fortify",
        )
    ],
    "S006": [
        TriggerDef(
            event_type=EventType.SPELL_CAST,
            condition=lambda ctx, event, ref: True,
            effect=_spell_apple,
            name="Apple",
        )
    ],
}

SPELLS_REQUIRE_TARGET = {"S002", "S003", "S004", "S005"}
