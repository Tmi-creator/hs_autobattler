from __future__ import annotations

from typing import Dict, List

from .event_system import EntityRef, Event, EventType, TriggerDef


def _spell_gain_gold(ctx, event: Event, trigger_uid: int) -> None:
    if not event.source_pos:
        return
    ctx.gain_gold(event.source_pos.side, 1)


def _spell_buff_minion(ctx, event: Event, trigger_uid: int) -> None:
    if not event.target:
        return
    ctx.buff(EntityRef(event.target.uid), 2, 2)


SPELL_TRIGGER_REGISTRY: Dict[str, List[TriggerDef]] = {
    "S001": [
        TriggerDef(
            event_type=EventType.SPELL_CAST,
            condition=lambda ctx, event, ref: True,
            effect=_spell_gain_gold,
            name="Tavern Coin",
        )
    ],
    "S002": [
        TriggerDef(
            event_type=EventType.SPELL_CAST,
            condition=lambda ctx, event, ref: True,
            effect=_spell_buff_minion,
            name="Heroic Charm",
        )
    ],
}

SPELLS_REQUIRE_TARGET = {"S002"}
