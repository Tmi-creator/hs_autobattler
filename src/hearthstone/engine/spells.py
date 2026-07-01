from __future__ import annotations

from typing import Dict, List, Set

from .enums import EffectIDs, MechanicType, SpellIDs, Tags
from .event_system import EffectContext, EntityRef, Event, EventType, TriggerDef


# ---------------------------------------------------------------------------
# Legacy hand-rolled handlers (kept for backward compat; still used by the
# old hardcoded registry entries that are now superseded by build_spell_registry,
# but they are NOT removed so existing import paths don't break).
# ---------------------------------------------------------------------------

def _spell_coin(ctx: EffectContext, event: Event, trigger_uid: int) -> None:
    if not event.source_pos:
        return
    ctx.gain_gold(event.source_pos.side, 1)


def _spell_banana(ctx: EffectContext, event: Event, trigger_uid: int) -> None:
    if not event.target:
        return
    ctx.buff_perm(EntityRef(event.target.uid), 2, 2)


def _spell_bloodgem(ctx: EffectContext, event: Event, trigger_uid: int) -> None:
    if not event.target or not event.source_pos:
        return
    player = ctx.players_by_uid.get(event.source_pos.side)
    if not player:
        return
    atk, hp = player.mechanics.get_stat(MechanicType.BLOOD_GEM)
    ctx.buff_perm(EntityRef(event.target.uid), atk, hp)


def _spell_arrow(ctx: EffectContext, event: Event, trigger_uid: int) -> None:
    if not event.target:
        return
    ctx.buff_perm(EntityRef(event.target.uid), 4, 0)


def _spell_fortify(ctx: EffectContext, event: Event, trigger_uid: int) -> None:
    if not event.target:
        return
    ctx.buff_perm(EntityRef(event.target.uid), 0, 3)
    unit = ctx.resolve_unit(EntityRef(event.target.uid))
    if unit:
        unit.tags.add(Tags.TAUNT)


def _spell_apple(ctx: EffectContext, event: Event, trigger_uid: int) -> None:
    if not event.source_pos:
        return
    side = event.source_pos.side

    for _, unit in ctx.iter_store_units(side):
        ctx.buff_perm(EntityRef(unit.uid), 1, 2)


def _spell_surf_spellcraft(ctx: EffectContext, event: Event, trigger_uid: int) -> None:
    if not event.target:
        return
    ctx.attach_effect_turn(EntityRef(event.target.uid), EffectIDs.CRAB_DEATHRATTLE, 1)


# ---------------------------------------------------------------------------
# Factory functions — one per effect code.
# Each factory receives spell_id so it can close over SPELL_DB params.
# ---------------------------------------------------------------------------

def _make_gain_gold_handler(spell_id: str):
    """Spell effect: GAIN_GOLD — gives player gold equal to params['gold']."""
    from .configs import SPELL_DB
    params = SPELL_DB[spell_id].get("params", {})
    gold_amount = params.get("gold", 1)

    def _handler(ctx: EffectContext, event: Event, trigger_uid: int) -> None:
        if not event.source_pos:
            return
        ctx.gain_gold(event.source_pos.side, gold_amount)

    return _handler


def _make_buff_minion_handler(spell_id: str):
    """Spell effect: BUFF_MINION — buffs targeted board unit with atk/hp/tags from params."""
    from .configs import SPELL_DB
    params = SPELL_DB[spell_id].get("params", {})

    # Blood Gem is special: its atk/hp come from player mechanic state, not fixed params.
    # We detect it by the absence of fixed atk/hp (both 0 and no 'tags' means dynamic).
    # Actually the simplest check: if spell_id is BLOOD_GEM, delegate to dynamic path.
    is_blood_gem = (spell_id == SpellIDs.BLOOD_GEM)

    fixed_atk: int = params.get("atk", 0)
    fixed_hp: int = params.get("hp", 0)
    extra_tags: set = params.get("tags", set())

    def _handler(ctx: EffectContext, event: Event, trigger_uid: int) -> None:
        if not event.target:
            return
        ref = EntityRef(event.target.uid)
        if is_blood_gem:
            if not event.source_pos:
                return
            player = ctx.players_by_uid.get(event.source_pos.side)
            if not player:
                return
            atk, hp = player.mechanics.get_stat(MechanicType.BLOOD_GEM)
            ctx.buff_perm(ref, atk, hp)
        else:
            ctx.buff_perm(ref, fixed_atk, fixed_hp)
            if extra_tags:
                unit = ctx.resolve_unit(ref)
                if unit:
                    unit.tags |= extra_tags
                    unit.recalc_stats()

    return _handler


def _make_buff_board_handler(spell_id: str):
    """Spell effect: BUFF_BOARD — buffs ALL friendly board units with atk/hp from params."""
    from .configs import SPELL_DB
    params = SPELL_DB[spell_id].get("params", {})
    atk: int = params.get("atk", 0)
    hp: int = params.get("hp", 0)
    extra_tags: set = params.get("tags", set())

    def _handler(ctx: EffectContext, event: Event, trigger_uid: int) -> None:
        if not event.source_pos:
            return
        side = event.source_pos.side
        for _, unit in ctx.iter_board_units(side):
            ctx.buff_perm(EntityRef(unit.uid), atk, hp)
            if extra_tags:
                unit.tags |= extra_tags
                unit.recalc_stats()

    return _handler


def _make_buff_board_type_handler(spell_id: str):
    """Spell effect: BUFF_BOARD_TYPE — buffs friendly board units of a given type."""
    from .configs import SPELL_DB
    from .enums import UnitType
    params = SPELL_DB[spell_id].get("params", {})
    atk: int = params.get("atk", 0)
    hp: int = params.get("hp", 0)
    type_filter = params.get("type", None)
    extra_tags: set = params.get("tags", set())

    def _handler(ctx: EffectContext, event: Event, trigger_uid: int) -> None:
        if not event.source_pos:
            return
        side = event.source_pos.side
        for _, unit in ctx.iter_board_units(side):
            if type_filter is not None and type_filter not in unit.types:
                continue
            ctx.buff_perm(EntityRef(unit.uid), atk, hp)
            if extra_tags:
                unit.tags |= extra_tags
                unit.recalc_stats()

    return _handler


def _make_discover_handler(spell_id: str):
    """Spell effect: DISCOVER — sets player.pending_discovery_request for deferred resolution."""
    from .configs import SPELL_DB
    from .entities import DiscoveryRequest
    params = SPELL_DB[spell_id].get("params", {})
    tier: int = params.get("tier", 1)
    exact_tier: bool = params.get("exact_tier", False)

    def _handler(ctx: EffectContext, event: Event, trigger_uid: int) -> None:
        if not event.source_pos:
            return
        player = ctx.players_by_uid.get(event.source_pos.side)
        if not player:
            return
        player.pending_discovery_request = DiscoveryRequest(
            tier=tier,
            exact_tier=exact_tier,
            source=f"Spell:{spell_id}",
        )

    return _handler


def _make_discover_tier_up_handler(spell_id: str):
    """Spell effect: DISCOVER_TIER_UP — discover using dynamic tier stored in spell.params.

    This is used by TRIPLET_REWARD whose tier is set dynamically at cast time.
    However, TRIPLET_REWARD is handled directly in TavernManager._cast_spell,
    so this factory covers any future DISCOVER_TIER_UP spells with static tier.
    """
    from .configs import SPELL_DB
    from .entities import DiscoveryRequest
    params = SPELL_DB[spell_id].get("params", {})
    tier: int = params.get("tier", 1)
    exact_tier: bool = params.get("exact_tier", True)

    def _handler(ctx: EffectContext, event: Event, trigger_uid: int) -> None:
        if not event.source_pos:
            return
        player = ctx.players_by_uid.get(event.source_pos.side)
        if not player:
            return
        player.pending_discovery_request = DiscoveryRequest(
            tier=tier,
            exact_tier=exact_tier,
            source=f"Spell:{spell_id}",
        )

    return _handler


def _make_get_random_unit_handler(spell_id: str):
    """Spell effect: GET_RANDOM_UNIT — draws a random unit from pool into hand."""
    from .configs import SPELL_DB
    params = SPELL_DB[spell_id].get("params", {})
    tier: int = params.get("tier", 1)
    count: int = params.get("count", 1)

    def _handler(ctx: EffectContext, event: Event, trigger_uid: int) -> None:
        if not event.source_pos:
            return
        side = event.source_pos.side
        ctx.draw_from_pool(side, tier, count)

    return _handler


def _make_free_refresh_handler(spell_id: str):
    """Spell effect: FREE_REFRESH — grants player one free tavern refresh."""
    from .configs import SPELL_DB
    params = SPELL_DB[spell_id].get("params", {})
    count: int = params.get("count", 1)

    def _handler(ctx: EffectContext, event: Event, trigger_uid: int) -> None:
        if not event.source_pos:
            return
        player = ctx.players_by_uid.get(event.source_pos.side)
        if not player:
            return
        player.free_refreshes += count

    return _handler


def _make_buff_tavern_handler(spell_id: str):
    """Spell effect: BUFF_TAVERN — buffs all units currently in the shop."""
    from .configs import SPELL_DB
    params = SPELL_DB[spell_id].get("params", {})
    atk: int = params.get("atk", 0)
    hp: int = params.get("hp", 0)
    extra_tags: set = params.get("tags", set())

    def _handler(ctx: EffectContext, event: Event, trigger_uid: int) -> None:
        if not event.source_pos:
            return
        side = event.source_pos.side
        for _, unit in ctx.iter_store_units(side):
            ctx.buff_perm(EntityRef(unit.uid), atk, hp)
            if extra_tags:
                unit.tags |= extra_tags
                unit.recalc_stats()

    return _handler


def _make_attach_effect_handler(spell_id: str):
    """Spell effect: ATTACH_EFFECT — attaches an effect to the targeted unit (turn-scoped)."""
    from .configs import SPELL_DB
    params = SPELL_DB[spell_id].get("params", {})
    effect_id: str = params.get("effect_id", "")
    count: int = params.get("count", 1)

    def _handler(ctx: EffectContext, event: Event, trigger_uid: int) -> None:
        if not event.target or not effect_id:
            return
        ctx.attach_effect_turn(EntityRef(event.target.uid), effect_id, count)

    return _handler


def _make_set_stats_20_20_handler(spell_id: str):
    """Spell effect: SET_STATS_20_20 — sets targeted minion's stats to 20/20."""
    def _handler(ctx: EffectContext, event: Event, trigger_uid: int) -> None:
        if not event.target:
            return
        unit = ctx.resolve_unit(EntityRef(event.target.uid))
        if not unit:
            return
        unit.perm_atk_add = 20 - unit.base_atk
        unit.perm_hp_add = 20 - unit.base_hp
        unit.recalc_stats()
        unit.restore_stats()

    return _handler


def _make_attach_crab_dr_handler(spell_id: str):
    """Spell effect: ATTACH_CRAB_DR — legacy alias kept for SURF_SPELLCRAFT."""
    return _make_attach_effect_handler(spell_id)


# ---------------------------------------------------------------------------
# Effect code -> factory mapping
# ---------------------------------------------------------------------------

#: Maps SPELL_DB "effect" codes to their factory functions.
EFFECT_FACTORIES = {
    "GAIN_GOLD": _make_gain_gold_handler,
    "BUFF_MINION": _make_buff_minion_handler,
    "BUFF_BOARD": _make_buff_board_handler,
    "BUFF_BOARD_TYPE": _make_buff_board_type_handler,
    "DISCOVER": _make_discover_handler,
    "DISCOVER_TIER_UP": _make_discover_tier_up_handler,
    "GET_RANDOM_UNIT": _make_get_random_unit_handler,
    "FREE_REFRESH": _make_free_refresh_handler,
    "BUFF_TAVERN": _make_buff_tavern_handler,
    "ATTACH_EFFECT": _make_attach_effect_handler,
    "ATTACH_CRAB_DR": _make_attach_crab_dr_handler,
    "BUFF_ALL_FRIENDLY": _make_buff_board_handler,       # alias: buff all board
    "BUFF_ALL_BY_TYPE": _make_buff_board_type_handler,   # alias: buff board by type
    "SET_STATS_20_20": _make_set_stats_20_20_handler,
}

#: Effect codes whose spells require a board target.
_TARGET_REQUIRED_EFFECTS: Set[str] = {
    "BUFF_MINION",
    "ATTACH_EFFECT",
    "ATTACH_CRAB_DR",
    "SET_STATS_20_20",
}


def build_spell_registry():
    """Auto-build SPELL_TRIGGER_REGISTRY and SPELLS_REQUIRE_TARGET from SPELL_DB.

    Returns:
        (registry, require_target) where registry maps spell_id -> List[TriggerDef]
        and require_target is a set of spell_ids that need a board target.
    """
    from .configs import SPELL_DB

    registry: Dict[str, List[TriggerDef]] = {}
    require_target: Set[str] = set()

    for spell_id, data in SPELL_DB.items():
        effect_code = data.get("effect", "")
        factory = EFFECT_FACTORIES.get(effect_code)
        if factory is None:
            continue  # unknown effect; leave unregistered (cast will fail gracefully)

        handler = factory(spell_id)
        registry[spell_id] = [
            TriggerDef(
                event_type=EventType.SPELL_CAST,
                condition=lambda ctx, e, uid: True,
                effect=handler,
                name=f"Spell: {data['name']}",
            )
        ]

        if effect_code in _TARGET_REQUIRED_EFFECTS:
            require_target.add(spell_id)

    return registry, require_target


# ---------------------------------------------------------------------------
# Module-level registries — auto-generated from SPELL_DB.
# These are the authoritative references imported by tavern.py.
# ---------------------------------------------------------------------------

SPELL_TRIGGER_REGISTRY, SPELLS_REQUIRE_TARGET = build_spell_registry()
