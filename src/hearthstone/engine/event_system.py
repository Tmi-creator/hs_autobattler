from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING, Callable, Deque, Dict, List, Optional, Set

if TYPE_CHECKING:
    from .entities import Player

from .auras import recalculate_board_auras
from .entities import HandCard, Spell, Unit
from .enums import Tags, UnitType


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
    MINION_ADDED_TO_SHOP = auto()
    DIVINE_SHIELD_LOST = auto()
    OVERKILL = auto()
    TAVERN_REFRESHED = auto()
    HERO_DAMAGED = auto()


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
    types: List[UnitType]
    tags: Set[Tags]


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
    priority: int = 0


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
        card_pool: Optional[object] = None,
    ):
        self.players_by_uid = players_by_uid
        self._uid_provider = uid_provider
        self._event_queue = event_queue
        self.card_pool = card_pool  # CardPool, None during combat
        self._uid_to_pos: Dict[int, PosRef] = {}
        for player_id, _ in players_by_uid.items():
            self._reindex_side(player_id)

    def resolve_unit(self, ref: Optional[EntityRef]) -> Optional[Unit]:
        if not ref:
            return None
        pos = self._uid_to_pos.get(ref.uid)
        if not pos:
            return None
        player = self.players_by_uid.get(pos.side)
        if not player:
            return None
        if pos.zone == Zone.BOARD:
            if pos.slot < 0 or pos.slot >= len(player.board):
                return None
            return player.board[pos.slot]
        if pos.zone == Zone.SHOP:
            if pos.slot < 0 or pos.slot >= len(player.store):
                return None
            item = player.store[pos.slot]
            return item.unit
        if pos.zone == Zone.HAND:
            if pos.slot < 0 or pos.slot >= len(player.hand):
                return None
            hand_item = player.hand[pos.slot]
            return hand_item.unit
        return None

    def iter_board_units(self, side: int) -> list[tuple[int, Unit]]:
        player = self.players_by_uid.get(side)
        if not player:
            return []
        return list(enumerate(player.board))

    def get_adjacent(self, side: int, uid: int) -> list[tuple[int, Unit]]:
        """Return live adjacent units [(slot, Unit), ...] for the unit with given uid."""
        player = self.players_by_uid.get(side)
        if not player:
            return []
        idx = None
        for i, u in enumerate(player.board):
            if u.uid == uid:
                idx = i
                break
        if idx is None:
            return []
        result = []
        if idx > 0:
            result.append((idx - 1, player.board[idx - 1]))
        if idx < len(player.board) - 1:
            result.append((idx + 1, player.board[idx + 1]))
        return result

    def get_leftmost(self, side: int) -> Optional[tuple[int, Unit]]:
        """Return (0, unit) for the leftmost board unit, or None if empty."""
        player = self.players_by_uid.get(side)
        if not player or not player.board:
            return None
        return (0, player.board[0])

    def get_rightmost(self, side: int) -> Optional[tuple[int, Unit]]:
        """Return (last_idx, unit) for the rightmost board unit, or None if empty."""
        player = self.players_by_uid.get(side)
        if not player or not player.board:
            return None
        idx = len(player.board) - 1
        return (idx, player.board[idx])

    def resolve_pos(self, ref: Optional[EntityRef]) -> Optional[PosRef]:
        if not ref:
            return None
        return self._uid_to_pos.get(ref.uid)

    def _clear_side_index(self, side: int) -> None:
        stale_uids = [uid for uid, pos in self._uid_to_pos.items() if pos.side == side]
        for uid in stale_uids:
            self._uid_to_pos.pop(uid, None)

    def _reindex_side(self, side: int) -> None:
        player = self.players_by_uid.get(side)
        if not player:
            return
        self._clear_side_index(side)
        # 1. BOARD
        for idx, unit in enumerate(player.board):
            self._uid_to_pos[unit.uid] = PosRef(side=side, zone=Zone.BOARD, slot=idx)

        # 2. HAND
        for idx, card in enumerate(player.hand):
            self._uid_to_pos[card.uid] = PosRef(side=side, zone=Zone.HAND, slot=idx)

        # 3. SHOP
        for idx, item in enumerate(player.store):
            if item.unit:
                self._uid_to_pos[item.unit.uid] = PosRef(side=side, zone=Zone.SHOP, slot=idx)

    def _reindex_all(self) -> None:
        for player_id, _ in self.players_by_uid.items():
            self._reindex_side(player_id)

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
            self.emit_event(Event(
                event_type=EventType.HERO_DAMAGED,
                source_pos=PosRef(side=side, zone=Zone.HERO, slot=0),
                value=amount,
            ))

    def heal_hero(self, side: int, amount: int) -> None:
        player = self.players_by_uid.get(side)
        if player:
            player.health += amount

    def buff_perm(self, target_ref: EntityRef, atk: int, hp: int) -> None:
        unit = self.resolve_unit(target_ref)
        if not unit:
            return
        unit.perm_atk_add += atk
        unit.perm_hp_add += hp
        unit.recalc_stats()

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

    def iter_store_units(self, side: int) -> list[tuple[int, Unit]]:
        results: list[tuple[int, Unit]] = []
        player = self.players_by_uid.get(side)
        if not player:
            return []

        for idx, item in enumerate(player.store):
            if item.unit:
                results.append((idx, item.unit))

        return results

    def attach_effect_perm(self, target_ref: EntityRef, effect_id: str, count: int = 1) -> None:
        unit = self.resolve_unit(target_ref)
        if not unit:
            return
        unit.attached_perm[effect_id] = unit.attached_perm.get(effect_id, 0) + count

    def attach_effect_turn(self, target_ref: EntityRef, effect_id: str, count: int = 1) -> None:
        unit = self.resolve_unit(target_ref)
        if not unit:
            return
        unit.attached_turn[effect_id] = unit.attached_turn.get(effect_id, 0) + count

    def attach_effect_combat(self, target_ref: EntityRef, effect_id: str, count: int = 1) -> None:
        unit = self.resolve_unit(target_ref)
        if not unit:
            return
        unit.attached_combat[effect_id] = unit.attached_combat.get(effect_id, 0) + count

    def consume_random_store_unit(self, side: int) -> tuple[int, int] | None:
        """Remove a random unit from the store and return it to the pool.
        Returns (atk, hp) or None."""
        import random

        player = self.players_by_uid.get(side)
        if not player:
            return None
        store_units = [(i, item) for i, item in enumerate(player.store) if item.unit]
        if not store_units:
            return None
        idx, item = random.choice(store_units)
        consumed = item.unit
        if not consumed:
            return None
        atk, hp = consumed.cur_atk, consumed.cur_hp
        if self.card_pool:
            self.card_pool.return_cards([consumed.card_id])
        player.store.pop(idx)
        return atk, hp

    def add_unit_to_hand(self, side: int, card_id: str) -> bool:
        """Create a unit from DB and add it to the player's hand.
        Does NOT interact with pool — use draw_from_pool for pool-aware version."""
        player = self.players_by_uid.get(side)
        if not player or len(player.hand) >= 10:
            return False
        uid = self._uid_provider()
        new_unit = Unit.create_from_db(card_id, uid, side)
        player.hand.append(HandCard(uid=uid, unit=new_unit))
        return True

    def draw_from_pool(self, side: int, tier: int, count: int = 1) -> list[str]:
        """Draw random unit(s) from the shared pool into player's hand.
        Returns list of drawn card_ids (may be shorter than count if hand full)."""
        player = self.players_by_uid.get(side)
        if not player or not self.card_pool:
            return []
        drawn = self.card_pool.draw_cards(count, max_tier=tier)
        added = []
        for card_id in drawn:
            if len(player.hand) >= 10:
                break
            uid = self._uid_provider()
            new_unit = Unit.create_from_db(card_id, uid, side)
            player.hand.append(HandCard(uid=uid, unit=new_unit))
            added.append(card_id)
        not_added = drawn[len(added) :]
        if not_added:
            self.card_pool.return_cards(not_added)
        return added

    def make_golden(self, ref: EntityRef) -> bool:
        """Make a unit golden: double base stats, set is_golden flag."""
        unit = self.resolve_unit(ref)
        if not unit or unit.is_golden:
            return False
        unit.is_golden = True
        unit.base_atk *= 2
        unit.base_hp *= 2
        unit.recalc_stats()
        return True

    def summon(
        self, side: int, card_id: str, insert_index: int, is_golden: bool = False
    ) -> Optional[EntityRef]:  # noqa: E501
        player = self.players_by_uid.get(side)
        if not player:
            return None
        if len(player.board) >= 7:
            return None
        index = max(0, min(insert_index, len(player.board)))
        unit = Unit.create_from_db(card_id, self._uid_provider(), side, is_golden)
        player.board.insert(index, unit)
        self._reindex_side(side)
        summoned = EntityRef(uid=unit.uid)
        pos = self._uid_to_pos.get(unit.uid)
        recalculate_board_auras(player.board)
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
    def __init__(
        self,
        trigger_registry: Dict[str, List[TriggerDef]],
        golden_trigger_registry: Optional[Dict[str, List[TriggerDef]]] = None,
        executor: Optional[EffectExecutor] = None,
    ):
        self.trigger_registry = trigger_registry
        self.golden_trigger_registry = golden_trigger_registry or {}
        self.executor = executor or EffectExecutor()

    def process_event(
        self,
        event: Event,
        players_by_uid: Dict[int, Player],
        uid_provider: Callable[[], int],
        extra_triggers: Optional[List[TriggerInstance]] = None,
        card_pool: Optional[object] = None,
    ) -> None:
        queue: Deque[Event] = deque([event])
        ctx = EffectContext(players_by_uid, uid_provider, queue, card_pool)
        initial_event = event
        while queue:
            current_event = queue.popleft()
            triggers = self.collect_triggers(current_event, ctx)
            if extra_triggers and current_event == initial_event:
                triggers.extend(extra_triggers)
            for trigger in self.order_triggers(triggers, current_event, ctx):
                if trigger.trigger_def.condition(ctx, current_event, trigger.trigger_uid):
                    for _ in range(trigger.stacks):
                        self.executor.run(
                            trigger.trigger_def.effect, ctx, current_event, trigger.trigger_uid
                        )

    def collect_triggers(self, event: Event, ctx: EffectContext) -> List[TriggerInstance]:

        triggers: List[TriggerInstance] = []
        for _player_id, player in ctx.players_by_uid.items():
            for _slot, unit in enumerate(player.board):
                stacks_multiplier = 1
                if unit.is_golden:
                    if unit.card_id in self.golden_trigger_registry:
                        active_defs = self.golden_trigger_registry[unit.card_id]
                    else:
                        active_defs = self.trigger_registry.get(unit.card_id, [])
                        stacks_multiplier = 2
                else:
                    active_defs = self.trigger_registry.get(unit.card_id, [])

                for trigger_def in active_defs:
                    if trigger_def.event_type == event.event_type:
                        triggers.append(
                            TriggerInstance(
                                trigger_def=trigger_def,
                                trigger_uid=unit.uid,
                                stacks=stacks_multiplier,
                            )
                        )
                for attached in (unit.attached_perm, unit.attached_turn, unit.attached_combat):
                    for index, count in attached.items():
                        if count <= 0:
                            continue
                        trigger_defs = self.trigger_registry.get(index, [])
                        for trigger_def in trigger_defs:
                            if trigger_def.event_type == event.event_type:
                                triggers.append(
                                    TriggerInstance(
                                        trigger_def=trigger_def,
                                        trigger_uid=unit.uid,
                                        stacks=count,
                                    )
                                )

            # Also scan hand for START_OF_COMBAT and MINION_PLAYED triggers
            if event.event_type in (EventType.START_OF_COMBAT, EventType.MINION_PLAYED):
                for hc in player.hand:
                    if not hc.unit:
                        continue
                    unit = hc.unit
                    active_defs = self.trigger_registry.get(unit.card_id, [])
                    stacks_multiplier = 2 if unit.is_golden else 1
                    for trigger_def in active_defs:
                        if trigger_def.event_type == event.event_type:
                            triggers.append(
                                TriggerInstance(
                                    trigger_def=trigger_def,
                                    trigger_uid=unit.uid,
                                    stacks=stacks_multiplier,
                                )
                            )
        if event.event_type in SYSTEM_TRIGGER_REGISTRY:
            for trig_def in SYSTEM_TRIGGER_REGISTRY[event.event_type]:
                triggers.append(
                    TriggerInstance(
                        trigger_def=trig_def,
                        trigger_uid=0,
                        stacks=1,
                    )
                )

        # Multiplier auras (Brann/Titus/Drakkari): increase stacks on matching triggers
        if not hasattr(self, '_multiplier_cache'):
            from .card_def import ALL_CARDS
            self._multiplier_cache = {
                card.card_id: card.multiplier
                for card in ALL_CARDS if card.multiplier is not None
            }
        for _player_id, player in ctx.players_by_uid.items():
            for unit in player.board:
                mult_def = self._multiplier_cache.get(unit.card_id)
                if not mult_def:
                    continue
                if mult_def.event_type_name != event.event_type.name:
                    continue
                # Apply: increase stacks on same-side triggers that match
                for i, trigger in enumerate(triggers):
                    # Determine the side of the trigger's owning unit
                    trigger_pos = ctx.resolve_pos(EntityRef(trigger.trigger_uid))
                    trigger_side = trigger_pos.side if trigger_pos else -1
                    if trigger_side != -1 and trigger_side != _player_id:
                        continue  # only boost own side
                    if trigger.trigger_uid == unit.uid:
                        continue  # multiplier doesn't boost itself
                    if mult_def.self_only:
                        # Only boost triggers where the unit's event source is itself
                        if event.source and event.source.uid != trigger.trigger_uid:
                            continue
                    triggers[i] = TriggerInstance(
                        trigger_def=trigger.trigger_def,
                        trigger_uid=trigger.trigger_uid,
                        stacks=trigger.stacks + mult_def.extra_stacks,
                    )

        return triggers

    def order_triggers(
        self,
        triggers: List[TriggerInstance],
        event: Event,
        ctx: EffectContext,
    ) -> List[TriggerInstance]:
        active_side = None
        source_pos = event.source_pos or (event.snapshot.pos if event.snapshot else None)
        source_uid = None
        if event.source:
            source_uid = event.source.uid
        elif event.snapshot:
            source_uid = event.snapshot.uid

        if source_pos:
            active_side = source_pos.side
        elif event.source:
            pos = ctx.resolve_pos(event.source)
            active_side = pos.side if pos else None

        def sort_key(trigger: TriggerInstance) -> tuple[int, int, int, int, int]:
            trig_uid = trigger.trigger_uid

            pos = ctx.resolve_pos(EntityRef(trigger.trigger_uid))
            unit = ctx.resolve_unit(EntityRef(trigger.trigger_uid))
            unit_uid = unit.uid if unit else trig_uid

            is_source_trigger = (
                event.event_type == EventType.MINION_DIED
                and source_uid is not None
                and trig_uid == source_uid
            )
            # if its trigger of dead source, pos already gone from ctx (popped from board)
            # use snapshot/source_pos to not get slot=999
            if pos is None and is_source_trigger and source_pos is not None:
                pos = source_pos
            slot = pos.slot if pos else 999
            side = pos.side if pos else -1

            if side == -1:
                side_priority = 2
            elif active_side is None:
                side_priority = 0
            else:
                side_priority = 0 if side == active_side else 1
            group = 0 if is_source_trigger else 1

            return (
                group,
                -trigger.trigger_def.priority,
                side_priority,
                slot,
                unit_uid,
            )

        return sorted(triggers, key=sort_key)


# =====================================================================
# System triggers (global, not tied to any card)
# =====================================================================


def _apply_elemental_buff(ctx: EffectContext, event: Event, _trigger_uid: int) -> None:
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
    from .enums import MechanicType
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
