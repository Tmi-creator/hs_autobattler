from __future__ import annotations

from typing import Dict, List, Union

from .enums import UnitType, CardIDs, SpellIDs, EffectIDs, MechanicType
from .event_system import EntityRef, Event, EventType, TriggerDef


def _is_self_play(ctx, event: Event, trigger_uid: int) -> bool:
    return event.source is not None and event.source.uid == trigger_uid


def _played_unit(ctx, event: Event):
    return ctx.resolve_unit(event.source)


def _is_friendly_death_exclude_self(ctx, event: Event, trigger_uid: int) -> bool:
    if event.event_type != EventType.MINION_DIED:
        return False

    dead_pos = event.source_pos or (event.snapshot.pos if event.snapshot else None)
    if not dead_pos:
        return False

    owner_pos = ctx.resolve_pos(EntityRef(trigger_uid))
    if not owner_pos:
        return False

    return (dead_pos.side == owner_pos.side) and (event.source.uid != trigger_uid)


def make_avenge_trigger(n: int, effect_fn, name: str = "Avenge"):
    """
    Make TriggerDef for Avenge mechanic (X).
    n: How many allies should die
    effect_fn: function that should play
    """

    def avenge_wrapper(ctx, event: Event, trigger_uid: int):
        avenger = ctx.resolve_unit(EntityRef(trigger_uid))
        if not avenger or not avenger.is_alive:
            return

        # correct limit because gold triggers twice
        limit = n * 2 if avenger.is_golden else n
        avenger.avenge_counter += 1
        if avenger.avenge_counter >= limit:
            avenger.avenge_counter = 0
            repeats = 2 if avenger.is_golden else 1  # double the effect
            for _ in range(repeats):
                effect_fn(ctx, event, trigger_uid)

    return TriggerDef(
        event_type=EventType.MINION_DIED,
        condition=_is_friendly_death_exclude_self,
        effect=avenge_wrapper,
        name=f"{name} (Avenge {n})",
        priority=-10
    )


def _gain_coin(ctx, event: Event, trigger_uid: int) -> None:
    pos = ctx.resolve_pos(EntityRef(trigger_uid))
    if not pos:
        return
    ctx.gain_gold(pos.side, 1)


def _summon_tabbycat(ctx, event: Event, trigger_uid: int) -> None:
    pos = ctx.resolve_pos(EntityRef(trigger_uid))
    if not pos:
        return
    ctx.summon(pos.side, CardIDs.TABBYCAT, pos.slot + 1)


def _summon_scallywag_token(ctx, event: Event, trigger_uid: int) -> None:
    pos = event.source_pos or (event.snapshot.pos if event.snapshot else None)
    if pos:
        ctx.summon(pos.side, CardIDs.PIRATE_TOKEN, pos.slot)


def _summon_imprisoner_token(ctx, event: Event, trigger_uid: int) -> None:
    pos = event.source_pos or (event.snapshot.pos if event.snapshot else None)
    if pos:
        ctx.summon(pos.side, CardIDs.IMP_TOKEN, pos.slot)


def _summon_crab_token(ctx, event: Event, trigger_uid: int) -> None:
    pos = event.source_pos or (event.snapshot.pos if event.snapshot else None)
    if pos:
        ctx.summon(pos.side, CardIDs.CRAB_TOKEN, pos.slot)


def _wrath_weaver_buff(ctx, event: Event, trigger_uid: int) -> None:
    played = _played_unit(ctx, event)
    if not played or UnitType.DEMON not in played.type:
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


def _swampstriker_buff(ctx, event: Event, trigger_uid: int) -> None:
    unit = _played_unit(ctx, event)
    if not unit:
        return
    if UnitType.MURLOC not in unit.type:
        return
    if _is_self_play(ctx, event, trigger_uid):
        return
    pos = ctx.resolve_pos(EntityRef(trigger_uid))
    if not pos:
        return
    ctx.buff_perm(EntityRef(trigger_uid), 1, 0)


def _minted_corsair_coin(ctx, event: Event, trigger_uid: int) -> None:
    pos = ctx.resolve_pos(EntityRef(trigger_uid))
    if not pos:
        return
    ctx.add_spell_to_hand(pos.side, SpellIDs.TAVERN_COIN)


TRIGGER_REGISTRY: Dict[Union[CardIDs, EffectIDs], List[TriggerDef]] = {
    CardIDs.SHELL_COLLECTOR: [
        TriggerDef(
            event_type=EventType.MINION_PLAYED,
            condition=_is_self_play,
            effect=_gain_coin,
            name="Shell Collector Battlecry",
            priority=10,
        )
    ],
    CardIDs.ALLEYCAT: [
        TriggerDef(
            event_type=EventType.MINION_PLAYED,
            condition=_is_self_play,
            effect=_summon_tabbycat,
            name="Alleycat Battlecry",
            priority=10,
        )
    ],
    CardIDs.SCALLYWAG: [
        TriggerDef(
            event_type=EventType.MINION_DIED,
            condition=_is_self_play,
            effect=_summon_scallywag_token,
            name="Scallywag Deathrattle",
        )
    ],
    CardIDs.IMPRISONER: [
        TriggerDef(
            event_type=EventType.MINION_DIED,
            condition=_is_self_play,
            effect=_summon_imprisoner_token,
            name="Imprisoner Deathrattle",
        )
    ],
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
    EffectIDs.CRAB_DEATHRATTLE: [
        TriggerDef(
            event_type=EventType.MINION_DIED,
            condition=lambda ctx, event, trigger_uid: event.source is not None and event.source.uid == trigger_uid,
            effect=_summon_crab_token,
            name="Attached Crab Deathrattle",
        )
    ],
}


def _summon_golden_tabbycat(ctx, event, trigger_uid: int) -> None:
    pos = ctx.resolve_pos(EntityRef(trigger_uid))
    if not pos:
        return
    ctx.summon(pos.side, CardIDs.TABBYCAT, pos.slot + 1, is_golden=True)


GOLDEN_TRIGGER_REGISTRY = {
    CardIDs.ALLEYCAT: [
        TriggerDef(
            event_type=EventType.MINION_PLAYED,
            condition=_is_self_play,
            effect=_summon_golden_tabbycat,
            name="Golden Alleycat Battlecry",
            priority=10,
        )
    ],
}


def _apply_elemental_buff(ctx, event: Event, trigger_uid: int) -> None:
    unit = ctx.resolve_unit(event.source)
    if not unit:
        return
    if UnitType.ELEMENTAL not in unit.type:
        return
    pos = ctx.resolve_pos(event.source)
    if not pos:
        return
    player = ctx.players_by_uid.get(pos.side)
    if not player:
        return
    buff_atk, buff_hp = player.mechanics.get_stat(MechanicType.ELEMENTAL_BUFF)
    if buff_atk > 0 or buff_hp > 0:
        ctx.buff_perm(event.source, buff_atk, buff_hp)


SYSTEM_TRIGGER_REGISTRY = {
    EventType.MINION_ADDED_TO_SHOP: [
        TriggerDef(
            event_type=EventType.MINION_ADDED_TO_SHOP,
            condition=lambda ctx, e, ref: True,
            effect=_apply_elemental_buff,
            name="Global Elemental Buff"
        )
    ]
}
