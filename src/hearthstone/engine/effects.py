from __future__ import annotations

import random
from typing import TYPE_CHECKING, Dict, List, Optional

if TYPE_CHECKING:
    from .entities import Unit
    from .event_system import EffectContext

from .configs import CARD_DB
from .enums import CardIDs, EffectIDs, MechanicType, SpellIDs, Tags, UnitType
from .event_system import (
    EntityRef,
    Event,
    EventType,
    TriggerDef,
)

# =====================================================================
# Helpers
# =====================================================================


def _is_self_play(ctx: EffectContext, event: Event, trigger_uid: int) -> bool:
    return event.source is not None and event.source.uid == trigger_uid


def _played_unit(ctx: EffectContext, event: Event) -> Optional[Unit]:
    return ctx.resolve_unit(event.source)


def _is_friendly_death_exclude_self(ctx: EffectContext, event: Event, trigger_uid: int) -> bool:
    if event.event_type != EventType.MINION_DIED:
        return False
    dead_pos = event.source_pos or (event.snapshot.pos if event.snapshot else None)
    if not dead_pos:
        return False
    owner_pos = ctx.resolve_pos(EntityRef(trigger_uid))
    if not owner_pos:
        return False
    return (dead_pos.side == owner_pos.side) and (
        event.source is not None and event.source.uid != trigger_uid
    )


def _is_self_death(ctx: EffectContext, event: Event, trigger_uid: int) -> bool:
    return event.source is not None and event.source.uid == trigger_uid


def _is_friendly_soc(ctx: EffectContext, event: Event, trigger_uid: int) -> bool:
    """Condition: Start of Combat, and I'm on a board."""
    pos = ctx.resolve_pos(EntityRef(trigger_uid))
    return pos is not None


# =====================================================================
# Effect functions — Tier 1
# =====================================================================


# --- Wrath Weaver: After you play a Demon, deal 1 dmg to hero, gain +2/+1 ---
def _wrath_weaver_buff(ctx: EffectContext, event: Event, trigger_uid: int) -> None:
    played = _played_unit(ctx, event)
    if not played or UnitType.DEMON not in played.types:
        return
    if _is_self_play(ctx, event, trigger_uid):
        return
    weaver = ctx.resolve_unit(EntityRef(trigger_uid))
    if not weaver:
        return
    pos = ctx.resolve_pos(EntityRef(trigger_uid))
    if not pos:
        return
    ctx.damage_hero(pos.side, 1)
    ctx.buff_perm(EntityRef(weaver.uid), 2, 1)


# --- Swampstriker: After you summon a Murloc, gain +1 Attack ---
def _swampstriker_buff(ctx: EffectContext, event: Event, trigger_uid: int) -> None:
    unit = _played_unit(ctx, event)
    if not unit:
        return
    if UnitType.MURLOC not in unit.types:
        return
    if _is_self_play(ctx, event, trigger_uid):
        return
    ctx.buff_perm(EntityRef(trigger_uid), 1, 0)


# --- Minted Corsair: On sell, get Tavern Coin ---
def _minted_corsair_coin(ctx: EffectContext, event: Event, trigger_uid: int) -> None:
    pos = ctx.resolve_pos(EntityRef(trigger_uid))
    if not pos:
        return
    ctx.add_spell_to_hand(pos.side, SpellIDs.TAVERN_COIN)


# --- Dune Dweller: BC: Elementals in Tavern get +1/+1 this game ---
def _dune_dweller_bc(ctx: EffectContext, event: Event, trigger_uid: int) -> None:
    pos = ctx.resolve_pos(EntityRef(trigger_uid))
    if not pos:
        return
    player = ctx.players_by_uid.get(pos.side)
    if not player:
        return
    player.mechanics.modify_stat(MechanicType.ELEMENTAL_BUFF, 1, 1)


# --- Ominous Seer: BC: Next tavern spell costs (1) less ---
def _ominous_seer_bc(ctx: EffectContext, event: Event, trigger_uid: int) -> None:
    pos = ctx.resolve_pos(EntityRef(trigger_uid))
    if not pos:
        return
    player = ctx.players_by_uid.get(pos.side)
    if not player:
        return
    player.spell_discount += 1


# --- Razorfen Geomancer: BC: Get 2 Blood Gems ---
def _razorfen_geomancer_bc(ctx: EffectContext, event: Event, trigger_uid: int) -> None:
    pos = ctx.resolve_pos(EntityRef(trigger_uid))
    if not pos:
        return
    ctx.add_spell_to_hand(pos.side, SpellIDs.BLOOD_GEM)
    ctx.add_spell_to_hand(pos.side, SpellIDs.BLOOD_GEM)


# --- Picky Eater: BC: Consume random tavern minion, gain its stats ---
def _picky_eater_bc(ctx: EffectContext, event: Event, trigger_uid: int) -> None:
    pos = ctx.resolve_pos(EntityRef(trigger_uid))
    if not pos:
        return
    result = ctx.consume_random_store_unit(pos.side)
    if result:
        atk, hp = result
        ctx.buff_perm(EntityRef(trigger_uid), atk, hp)


# --- River Skipper: On sell, get a random Tier 1 minion ---
def _river_skipper_sell(ctx: EffectContext, event: Event, trigger_uid: int) -> None:
    pos = ctx.resolve_pos(EntityRef(trigger_uid))
    if not pos:
        return
    # Pick a random T1 pool card
    t1_ids = [
        cid for cid, data in CARD_DB.items() if data["tier"] == 1 and not data.get("is_token")
    ]
    if not t1_ids:
        return
    chosen_id = random.choice(t1_ids)
    ctx.add_unit_to_hand(pos.side, chosen_id)


# --- Cord Puller: DR: Summon 1/1 Microbot ---
def _cord_puller_dr(ctx: EffectContext, event: Event, trigger_uid: int) -> None:
    pos = event.source_pos or (event.snapshot.pos if event.snapshot else None)
    if pos:
        ctx.summon(pos.side, CardIDs.MICROBOT, pos.slot)


# --- Harmless Bonehead: DR: Summon TWO 1/1 Skeletons ---
def _harmless_bonehead_dr(ctx: EffectContext, event: Event, trigger_uid: int) -> None:
    pos = event.source_pos or (event.snapshot.pos if event.snapshot else None)
    if pos:
        ctx.summon(pos.side, CardIDs.SKELETON, pos.slot)
        ctx.summon(pos.side, CardIDs.SKELETON, pos.slot)


# --- Manasaber: DR: Summon TWO 0/1 Cublings with Taunt ---
def _manasaber_dr(ctx: EffectContext, event: Event, trigger_uid: int) -> None:
    pos = event.source_pos or (event.snapshot.pos if event.snapshot else None)
    if pos:
        ctx.summon(pos.side, CardIDs.CUBLING, pos.slot)
        ctx.summon(pos.side, CardIDs.CUBLING, pos.slot)


# --- Misfit Dragonling: SoC: Gain +tier/+tier ---
def _misfit_dragonling_soc(ctx: EffectContext, event: Event, trigger_uid: int) -> None:
    unit = ctx.resolve_unit(EntityRef(trigger_uid))
    if not unit or not unit.is_alive:
        return
    pos = ctx.resolve_pos(EntityRef(trigger_uid))
    if not pos:
        return
    player = ctx.players_by_uid.get(pos.side)
    if not player:
        return
    tier = player.tavern_tier
    ctx.buff_combat(EntityRef(trigger_uid), tier, tier)


# --- Rot Hide Gnoll: +1 Atk per friendly death this combat ---
def _rot_hide_gnoll_buff(ctx: EffectContext, event: Event, trigger_uid: int) -> None:
    gnoll = ctx.resolve_unit(EntityRef(trigger_uid))
    if not gnoll or not gnoll.is_alive:
        return
    ctx.buff_combat(EntityRef(trigger_uid), 1, 0)


# --- Twilight Hatchling: DR: Summon 3/3 Whelp that attacks immediately ---
def _twilight_hatchling_dr(ctx: EffectContext, event: Event, trigger_uid: int) -> None:
    pos = event.source_pos or (event.snapshot.pos if event.snapshot else None)
    if not pos:
        return
    ref = ctx.summon(pos.side, CardIDs.TWILIGHT_WHELP, pos.slot)
    if ref:
        whelp = ctx.resolve_unit(ref)
        if whelp:
            whelp.tags.add(Tags.IMMEDIATE_ATTACK)


# --- Crab Deathrattle (from Surf n' Surf spellcraft) ---
def _summon_crab_token(ctx: EffectContext, event: Event, trigger_uid: int) -> None:
    pos = event.source_pos or (event.snapshot.pos if event.snapshot else None)
    if pos:
        ctx.summon(pos.side, CardIDs.CRAB_TOKEN, pos.slot)


# =====================================================================
# Trigger Registry
# =====================================================================

TRIGGER_REGISTRY: Dict[str, List[TriggerDef]] = {
    CardIDs.WRATH_WEAVER: [
        TriggerDef(
            event_type=EventType.MINION_PLAYED,
            condition=lambda ctx, event, ref: True,
            effect=_wrath_weaver_buff,
            name="Wrath Weaver Trigger",
        )
    ],
    CardIDs.SWAMPSTRIKER: [
        TriggerDef(
            event_type=EventType.MINION_PLAYED,
            condition=lambda ctx, event, ref: True,
            effect=_swampstriker_buff,
            name="Swampstriker Trigger",
        )
    ],
    CardIDs.MINTED_CORSAIR: [
        TriggerDef(
            event_type=EventType.MINION_SOLD,
            condition=_is_self_play,
            effect=_minted_corsair_coin,
            name="Minted Corsair Sell",
        )
    ],
    CardIDs.DUNE_DWELLER: [
        TriggerDef(
            event_type=EventType.MINION_PLAYED,
            condition=_is_self_play,
            effect=_dune_dweller_bc,
            name="Dune Dweller Battlecry",
            priority=10,
        )
    ],
    CardIDs.OMINOUS_SEER: [
        TriggerDef(
            event_type=EventType.MINION_PLAYED,
            condition=_is_self_play,
            effect=_ominous_seer_bc,
            name="Ominous Seer Battlecry",
            priority=10,
        )
    ],
    CardIDs.RAZORFEN_GEOMANCER: [
        TriggerDef(
            event_type=EventType.MINION_PLAYED,
            condition=_is_self_play,
            effect=_razorfen_geomancer_bc,
            name="Razorfen Geomancer Battlecry",
            priority=10,
        )
    ],
    CardIDs.PICKY_EATER: [
        TriggerDef(
            event_type=EventType.MINION_PLAYED,
            condition=_is_self_play,
            effect=_picky_eater_bc,
            name="Picky Eater Battlecry",
            priority=10,
        )
    ],
    CardIDs.RIVER_SKIPPER: [
        TriggerDef(
            event_type=EventType.MINION_SOLD,
            condition=_is_self_play,
            effect=_river_skipper_sell,
            name="River Skipper Sell",
        )
    ],
    CardIDs.CORD_PULLER: [
        TriggerDef(
            event_type=EventType.MINION_DIED,
            condition=_is_self_death,
            effect=_cord_puller_dr,
            name="Cord Puller Deathrattle",
        )
    ],
    CardIDs.HARMLESS_BONEHEAD: [
        TriggerDef(
            event_type=EventType.MINION_DIED,
            condition=_is_self_death,
            effect=_harmless_bonehead_dr,
            name="Harmless Bonehead Deathrattle",
        )
    ],
    CardIDs.MANASABER: [
        TriggerDef(
            event_type=EventType.MINION_DIED,
            condition=_is_self_death,
            effect=_manasaber_dr,
            name="Manasaber Deathrattle",
        )
    ],
    CardIDs.MISFIT_DRAGONLING: [
        TriggerDef(
            event_type=EventType.START_OF_COMBAT,
            condition=_is_friendly_soc,
            effect=_misfit_dragonling_soc,
            name="Misfit Dragonling SoC",
        )
    ],
    CardIDs.ROT_HIDE_GNOLL: [
        TriggerDef(
            event_type=EventType.MINION_DIED,
            condition=_is_friendly_death_exclude_self,
            effect=_rot_hide_gnoll_buff,
            name="Rot Hide Gnoll Buff",
        )
    ],
    CardIDs.TWILIGHT_HATCHLING: [
        TriggerDef(
            event_type=EventType.MINION_DIED,
            condition=_is_self_death,
            effect=_twilight_hatchling_dr,
            name="Twilight Hatchling Deathrattle",
        )
    ],
    EffectIDs.CRAB_DEATHRATTLE: [
        TriggerDef(
            event_type=EventType.MINION_DIED,
            condition=_is_self_death,
            effect=_summon_crab_token,
            name="Attached Crab Deathrattle",
        )
    ],
}

GOLDEN_TRIGGER_REGISTRY: Dict[str, List[TriggerDef]] = {}


# =====================================================================
# System triggers (global)
# =====================================================================


def _apply_elemental_buff(ctx: EffectContext, event: Event, trigger_uid: int) -> None:
    unit = ctx.resolve_unit(event.source)
    if not unit:
        return
    if UnitType.ELEMENTAL not in unit.types:
        return
    pos = ctx.resolve_pos(event.source)
    if not pos:
        return
    player = ctx.players_by_uid.get(pos.side)
    if not player:
        return
    buff_atk, buff_hp = player.mechanics.get_stat(MechanicType.ELEMENTAL_BUFF)
    if (buff_atk > 0 or buff_hp > 0) and event.source:
        ctx.buff_perm(event.source, buff_atk, buff_hp)


SYSTEM_TRIGGER_REGISTRY = {
    EventType.MINION_ADDED_TO_SHOP: [
        TriggerDef(
            event_type=EventType.MINION_ADDED_TO_SHOP,
            condition=lambda ctx, e, ref: True,
            effect=_apply_elemental_buff,
            name="Global Elemental Buff",
        )
    ]
}
