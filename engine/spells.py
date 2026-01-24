from __future__ import annotations

from typing import Dict, List

from .enums import Tags, MechanicType, SpellIDs, EffectIDs
from .event_system import EntityRef, Event, EventType, TriggerDef


def _spell_coin(ctx, event: Event, trigger_uid: int) -> None:
    if not event.source_pos:
        return
    ctx.gain_gold(event.source_pos.side, 1)


def _spell_banana(ctx, event: Event, trigger_uid: int) -> None:
    if not event.target:
        return
    ctx.buff_perm(EntityRef(event.target.uid), 2, 2)


def _spell_bloodgem(ctx, event: Event, trigger_uid: int) -> None:
    if not event.target:
        return
    player = ctx.players_by_uid.get(event.source_pos.side)
    atk, hp = player.mechanics.get_stat(MechanicType.BLOOD_GEM)
    ctx.buff_perm(EntityRef(event.target.uid), atk, hp)


def _spell_arrow(ctx, event: Event, trigger_uid: int) -> None:
    if not event.target:
        return
    ctx.buff_perm(EntityRef(event.target.uid), 4, 0)


def _spell_fortify(ctx, event: Event, trigger_uid: int) -> None:
    if not event.target:
        return
    ctx.buff_perm(EntityRef(event.target.uid), 0, 3)
    unit = ctx.resolve_unit(EntityRef(event.target.uid))
    unit.tags.add(Tags.TAUNT)


def _spell_apple(ctx, event: Event, trigger_uid: int) -> None:
    if not event.source_pos:
        return
    side = event.source_pos.side

    for idx, _ in ctx.iter_store_units(side):
        ctx.buff_tavern_minion_at_index(side, idx, 1, 2)


def _spell_surf_spellcraft(ctx, event: Event, trigger_uid: int) -> None:
    if not event.target:
        return
    ctx.attach_effect_turn(EntityRef(event.target.uid), EffectIDs.CRAB_DEATHRATTLE, 1)


SPELL_TRIGGER_REGISTRY: Dict[SpellIDs, List[TriggerDef]] = {
    SpellIDs.TAVERN_COIN: [
        TriggerDef(
            event_type=EventType.SPELL_CAST,
            condition=lambda ctx, event, ref: True,
            effect=_spell_coin,
            name="Tavern Coin",
        )
    ],
    SpellIDs.BANANA: [
        TriggerDef(
            event_type=EventType.SPELL_CAST,
            condition=lambda ctx, event, ref: True,
            effect=_spell_banana,
            name="Banana",
        )
    ],
    SpellIDs.BLOOD_GEM: [
        TriggerDef(
            event_type=EventType.SPELL_CAST,
            condition=lambda ctx, event, ref: True,
            effect=_spell_bloodgem,
            name="Blood Gem",
        )
    ],
    SpellIDs.POINTY_ARROW: [
        TriggerDef(
            event_type=EventType.SPELL_CAST,
            condition=lambda ctx, event, ref: True,
            effect=_spell_arrow,
            name="Pointy Arrow",
        )
    ],
    SpellIDs.FORTIFY: [
        TriggerDef(
            event_type=EventType.SPELL_CAST,
            condition=lambda ctx, event, ref: True,
            effect=_spell_fortify,
            name="Fortify",
        )
    ],
    SpellIDs.APPLE: [
        TriggerDef(
            event_type=EventType.SPELL_CAST,
            condition=lambda ctx, event, ref: True,
            effect=_spell_apple,
            name="Apple",
        )
    ],
    SpellIDs.SURF_SPELLCRAFT: [
        TriggerDef(
            event_type=EventType.SPELL_CAST,
            condition=lambda ctx, event, ref: True,
            effect=_spell_surf_spellcraft,
            name="Surf Spellcraft",
        )
    ],

}

SPELLS_REQUIRE_TARGET = {
    SpellIDs.BANANA,
    SpellIDs.BLOOD_GEM,
    SpellIDs.POINTY_ARROW,
    SpellIDs.FORTIFY,
    SpellIDs.SURF_SPELLCRAFT
}
