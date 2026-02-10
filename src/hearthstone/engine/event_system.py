from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum, auto
from typing import Callable, Deque, Dict, Iterable, List, Optional, Set

from .auras import recalculate_board_auras
from .enums import Tags, UnitType
from .entities import HandCard, Player, Spell, Unit


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
    ):
        self.players_by_uid = players_by_uid
        self._uid_provider = uid_provider
        self._event_queue = event_queue
        self._uid_to_pos: Dict[int, PosRef] = {}
        for player_id, player in players_by_uid.items():
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
            item = player.hand[pos.slot]
            return item.unit
        return None

    def iter_board_units(self, side: int) -> Iterable[tuple[int, Unit]]:
        player = self.players_by_uid.get(side)
        if not player:
            return []
        return list(enumerate(player.board))

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
        for player_id, player in self.players_by_uid.items():
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
        results = []
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

    def summon(self, side: int, card_id: str, insert_index: int, is_golden: bool = False) -> Optional[EntityRef]:
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
    def __init__(self,
                 trigger_registry: Dict[str, List[TriggerDef]],
                 golden_trigger_registry: Dict[str, List[TriggerDef]] = None,
                 executor: Optional[EffectExecutor] = None):
        self.trigger_registry = trigger_registry
        self.golden_trigger_registry = golden_trigger_registry or {}
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
                if trigger.trigger_def.condition(ctx, current_event, trigger.trigger_uid):
                    for _ in range(trigger.stacks):
                        self.executor.run(trigger.trigger_def.effect, ctx, current_event, trigger.trigger_uid)

    def collect_triggers(self, event: Event, ctx: EffectContext) -> List[TriggerInstance]:
        from .effects import SYSTEM_TRIGGER_REGISTRY
        triggers: List[TriggerInstance] = []
        for player_id, player in ctx.players_by_uid.items():
            for slot, unit in enumerate(player.board):
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
                                stacks=stacks_multiplier
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
        if event.event_type in SYSTEM_TRIGGER_REGISTRY:
            for trig_def in SYSTEM_TRIGGER_REGISTRY[event.event_type]:
                triggers.append(
                    TriggerInstance(trigger_def=trig_def,
                                    trigger_uid=0,
                                    stacks=1,
                                    )
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

        def sort_key(trigger: TriggerInstance) -> tuple:
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
