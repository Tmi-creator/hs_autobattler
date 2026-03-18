"""Tests for the event system: EffectContext, EventManager.

Covers: resolve_unit/pos in all zones, buff functions, summon,
gain_gold, damage_hero, add_spell_to_hand, attach_effect,
trigger collection (normal/golden/attached), trigger ordering.
"""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING, Callable, Dict

from hearthstone.engine.entities import HandCard, Player, Spell, StoreItem, Unit
from hearthstone.engine.enums import CardIDs, EffectIDs, SpellIDs
from hearthstone.engine.event_system import (
    EffectContext,
    EntityRef,
    Event,
    EventType,
    Zone,
)

if TYPE_CHECKING:
    from hearthstone.engine.game import Game


# ===================================================================
#  HELPERS
# ===================================================================


def _make_context(
    players: Dict[int, Player],
    uid_start: int = 50000,
) -> EffectContext:
    """Build EffectContext from a players dict."""
    counter = [uid_start]

    def uid_provider() -> int:
        counter[0] += 1
        return counter[0]

    return EffectContext(players, uid_provider, deque())


# ===================================================================
#  1. RESOLVE UNIT IN ALL ZONES
# ===================================================================


class TestEffectContextResolve:
    """resolve_unit / resolve_pos for BOARD, HAND, SHOP zones."""

    def test_resolve_unit_on_board(self) -> None:
        p = Player(uid=0, board=[], hand=[])
        unit = Unit.create_from_db(CardIDs.TABBYCAT, uid=1, owner_id=0)
        p.board.append(unit)
        ctx = _make_context({0: p})

        resolved = ctx.resolve_unit(EntityRef(uid=1))
        assert resolved is not None
        assert resolved.uid == 1

    def test_resolve_unit_in_hand(self) -> None:
        p = Player(uid=0, board=[], hand=[])
        unit = Unit.create_from_db(CardIDs.TABBYCAT, uid=1, owner_id=0)
        p.hand.append(HandCard(uid=1, unit=unit))
        ctx = _make_context({0: p})

        resolved = ctx.resolve_unit(EntityRef(uid=1))
        assert resolved is not None
        assert resolved.uid == 1

    def test_resolve_unit_in_store(self) -> None:
        p = Player(uid=0, board=[], hand=[])
        unit = Unit.create_from_db(CardIDs.TABBYCAT, uid=1, owner_id=0)
        p.store.append(StoreItem(unit=unit))
        ctx = _make_context({0: p})

        resolved = ctx.resolve_unit(EntityRef(uid=1))
        assert resolved is not None
        assert resolved.uid == 1

    def test_resolve_nonexistent_returns_none(self) -> None:
        p = Player(uid=0, board=[], hand=[])
        ctx = _make_context({0: p})
        assert ctx.resolve_unit(EntityRef(uid=999)) is None

    def test_resolve_none_ref_returns_none(self) -> None:
        p = Player(uid=0, board=[], hand=[])
        ctx = _make_context({0: p})
        assert ctx.resolve_unit(None) is None

    def test_resolve_pos_correct_zone(self) -> None:
        p = Player(uid=0, board=[], hand=[])
        unit = Unit.create_from_db(CardIDs.TABBYCAT, uid=1, owner_id=0)
        p.board.append(unit)
        ctx = _make_context({0: p})

        pos = ctx.resolve_pos(EntityRef(uid=1))
        assert pos is not None
        assert pos.side == 0
        assert pos.zone == Zone.BOARD
        assert pos.slot == 0

    def test_resolve_pos_second_slot(self) -> None:
        p = Player(uid=0, board=[], hand=[])
        u1 = Unit.create_from_db(CardIDs.TABBYCAT, uid=1, owner_id=0)
        u2 = Unit.create_from_db(CardIDs.TABBYCAT, uid=2, owner_id=0)
        p.board = [u1, u2]
        ctx = _make_context({0: p})

        pos = ctx.resolve_pos(EntityRef(uid=2))
        assert pos is not None
        assert pos.slot == 1

    def test_iter_board_units(self) -> None:
        p = Player(uid=0, board=[], hand=[])
        u1 = Unit.create_from_db(CardIDs.TABBYCAT, uid=1, owner_id=0)
        u2 = Unit.create_from_db(CardIDs.TABBYCAT, uid=2, owner_id=0)
        p.board = [u1, u2]
        ctx = _make_context({0: p})

        result = ctx.iter_board_units(0)
        assert len(result) == 2
        assert result[0] == (0, u1)
        assert result[1] == (1, u2)

    def test_iter_store_units(self) -> None:
        p = Player(uid=0, board=[], hand=[])
        unit = Unit.create_from_db(CardIDs.TABBYCAT, uid=1, owner_id=0)
        p.economy.store = [
            StoreItem(unit=unit),
            StoreItem(spell=Spell.create_from_db(SpellIDs.BANANA)),
        ]
        ctx = _make_context({0: p})

        result = ctx.iter_store_units(0)
        assert len(result) == 1  # Only the unit, not the spell
        assert result[0][1].uid == 1


# ===================================================================
#  2. BUFF FUNCTIONS
# ===================================================================


class TestEffectContextBuffs:
    """buff_perm, buff_turn, buff_combat."""

    def test_buff_perm_updates_stats(self) -> None:
        p = Player(uid=0, board=[], hand=[])
        unit = Unit.create_from_db(CardIDs.TABBYCAT, uid=1, owner_id=0)
        p.board.append(unit)
        ctx = _make_context({0: p})

        ctx.buff_perm(EntityRef(uid=1), 3, 5)

        assert unit.perm_atk_add == 3
        assert unit.perm_hp_add == 5
        assert unit.cur_atk == 1 + 3
        assert unit.cur_hp == 1 + 5

    def test_buff_turn_updates_stats(self) -> None:
        p = Player(uid=0, board=[], hand=[])
        unit = Unit.create_from_db(CardIDs.TABBYCAT, uid=1, owner_id=0)
        p.board.append(unit)
        ctx = _make_context({0: p})

        ctx.buff_turn(EntityRef(uid=1), 2, 3)

        assert unit.turn_atk_add == 2
        assert unit.turn_hp_add == 3

    def test_buff_combat_updates_stats(self) -> None:
        p = Player(uid=0, board=[], hand=[])
        unit = Unit.create_from_db(CardIDs.TABBYCAT, uid=1, owner_id=0)
        p.board.append(unit)
        ctx = _make_context({0: p})

        ctx.buff_combat(EntityRef(uid=1), 1, 1)

        assert unit.combat_atk_add == 1
        assert unit.combat_hp_add == 1

    def test_buff_nonexistent_no_crash(self) -> None:
        p = Player(uid=0, board=[], hand=[])
        ctx = _make_context({0: p})
        # Should not crash
        ctx.buff_perm(EntityRef(uid=999), 1, 1)
        ctx.buff_turn(EntityRef(uid=999), 1, 1)
        ctx.buff_combat(EntityRef(uid=999), 1, 1)


# ===================================================================
#  3. CONTEXT ACTIONS
# ===================================================================


class TestEffectContextActions:
    """gain_gold, damage_hero, add_spell_to_hand, summon."""

    def test_gain_gold(self) -> None:
        p = Player(uid=0, board=[], hand=[])
        p.gold = 5
        ctx = _make_context({0: p})
        ctx.gain_gold(0, 3)
        assert p.gold == 8

    def test_damage_hero(self) -> None:
        p = Player(uid=0, board=[], hand=[])
        p.health = 30
        ctx = _make_context({0: p})
        ctx.damage_hero(0, 5)
        assert p.health == 25

    def test_add_spell_to_hand(self) -> None:
        p = Player(uid=0, board=[], hand=[])
        ctx = _make_context({0: p})
        ctx.add_spell_to_hand(0, SpellIDs.TAVERN_COIN)

        assert len(p.hand) == 1
        assert p.hand[0].spell is not None
        assert p.hand[0].spell.card_id == SpellIDs.TAVERN_COIN

    def test_add_spell_full_hand_blocked(self) -> None:
        p = Player(uid=0, board=[], hand=[])
        for i in range(10):
            p.hand.append(HandCard(uid=i + 100))
        ctx = _make_context({0: p})

        ctx.add_spell_to_hand(0, SpellIDs.TAVERN_COIN)

        assert len(p.hand) == 10  # unchanged

    def test_summon_creates_unit(self) -> None:
        p = Player(uid=0, board=[], hand=[])
        ctx = _make_context({0: p})

        ref = ctx.summon(0, CardIDs.TABBYCAT, 0)

        assert ref is not None
        assert len(p.board) == 1
        assert p.board[0].card_id == CardIDs.TABBYCAT

    def test_summon_full_board_returns_none(self) -> None:
        p = Player(uid=0, board=[], hand=[])
        for i in range(7):
            p.board.append(Unit.create_from_db(CardIDs.TABBYCAT, uid=i + 1, owner_id=0))
        ctx = _make_context({0: p})

        ref = ctx.summon(0, CardIDs.TABBYCAT, 0)

        assert ref is None
        assert len(p.board) == 7

    def test_summon_at_specific_index(self) -> None:
        p = Player(uid=0, board=[], hand=[])
        u1 = Unit.create_from_db(CardIDs.TABBYCAT, uid=1, owner_id=0)
        u2 = Unit.create_from_db(CardIDs.TABBYCAT, uid=2, owner_id=0)
        p.board = [u1, u2]
        ctx = _make_context({0: p})

        ctx.summon(0, CardIDs.SCALLYWAG, 1)

        assert len(p.board) == 3
        assert p.board[1].card_id == CardIDs.SCALLYWAG

    def test_summon_emits_event(self) -> None:
        p = Player(uid=0, board=[], hand=[])
        queue: deque[Event] = deque()
        counter = [50000]

        def uid_provider() -> int:
            counter[0] += 1
            return counter[0]

        ctx = EffectContext({0: p}, uid_provider, queue)
        ctx.summon(0, CardIDs.TABBYCAT, 0)

        assert len(queue) > 0
        assert queue[0].event_type == EventType.MINION_SUMMONED


# ===================================================================
#  4. ATTACH EFFECTS
# ===================================================================


class TestEffectContextAttach:
    """attach_effect_perm/turn/combat."""

    def test_attach_perm(self) -> None:
        p = Player(uid=0, board=[], hand=[])
        unit = Unit.create_from_db(CardIDs.TABBYCAT, uid=1, owner_id=0)
        p.board.append(unit)
        ctx = _make_context({0: p})

        ctx.attach_effect_perm(EntityRef(uid=1), EffectIDs.CRAB_DEATHRATTLE, 2)

        assert unit.attached_perm[EffectIDs.CRAB_DEATHRATTLE] == 2

    def test_attach_turn(self) -> None:
        p = Player(uid=0, board=[], hand=[])
        unit = Unit.create_from_db(CardIDs.TABBYCAT, uid=1, owner_id=0)
        p.board.append(unit)
        ctx = _make_context({0: p})

        ctx.attach_effect_turn(EntityRef(uid=1), EffectIDs.CRAB_DEATHRATTLE)

        assert unit.attached_turn[EffectIDs.CRAB_DEATHRATTLE] == 1

    def test_attach_combat(self) -> None:
        p = Player(uid=0, board=[], hand=[])
        unit = Unit.create_from_db(CardIDs.TABBYCAT, uid=1, owner_id=0)
        p.board.append(unit)
        ctx = _make_context({0: p})

        ctx.attach_effect_combat(EntityRef(uid=1), "TEST_EFFECT")

        assert unit.attached_combat["TEST_EFFECT"] == 1

    def test_attach_stacks_additively(self) -> None:
        p = Player(uid=0, board=[], hand=[])
        unit = Unit.create_from_db(CardIDs.TABBYCAT, uid=1, owner_id=0)
        p.board.append(unit)
        ctx = _make_context({0: p})

        ctx.attach_effect_perm(EntityRef(uid=1), "E1", 1)
        ctx.attach_effect_perm(EntityRef(uid=1), "E1", 2)

        assert unit.attached_perm["E1"] == 3

    def test_attach_to_nonexistent_no_crash(self) -> None:
        p = Player(uid=0, board=[], hand=[])
        ctx = _make_context({0: p})
        ctx.attach_effect_perm(EntityRef(uid=999), "E1")  # No crash


# ===================================================================
#  5. TRIGGER COLLECTION
# ===================================================================


class TestTriggerCollection:
    """EventManager.collect_triggers from board units and attached effects."""

    def test_collect_from_board_unit(
        self,
        empty_game: "Game",
        mock_unit: Callable[..., Unit],
    ) -> None:
        p = empty_game.players[0]
        shell = mock_unit(CardIDs.SHELL_COLLECTOR, owner_id=p.uid)
        p.board.append(shell)

        event = Event(event_type=EventType.MINION_PLAYED)
        ctx = _make_context({p.uid: p})

        triggers = empty_game.event_manager.collect_triggers(event, ctx)
        trigger_names = [t.trigger_def.name for t in triggers]

        assert "Shell Collector Battlecry" in trigger_names

    def test_golden_unit_without_golden_registry_doubles_stacks(
        self,
        empty_game: "Game",
        mock_unit: Callable[..., Unit],
    ) -> None:
        p = empty_game.players[0]
        weaver = mock_unit(CardIDs.WRATH_WEAVER, owner_id=p.uid, is_golden=True)
        p.board.append(weaver)

        event = Event(event_type=EventType.MINION_PLAYED)
        ctx = _make_context({p.uid: p})

        triggers = empty_game.event_manager.collect_triggers(event, ctx)
        weaver_trig = next(t for t in triggers if "Wrath Weaver" in t.trigger_def.name)

        assert weaver_trig.stacks == 2  # No golden_trigger_registry entry → 2x

    def test_golden_unit_with_golden_registry_uses_it(
        self,
        empty_game: "Game",
        mock_unit: Callable[..., Unit],
    ) -> None:
        p = empty_game.players[0]
        cat = mock_unit(CardIDs.ALLEYCAT, owner_id=p.uid, is_golden=True)
        p.board.append(cat)

        event = Event(event_type=EventType.MINION_PLAYED)
        ctx = _make_context({p.uid: p})

        triggers = empty_game.event_manager.collect_triggers(event, ctx)
        alley_trig = next(t for t in triggers if "Golden Alleycat" in t.trigger_def.name)

        assert alley_trig.stacks == 1  # Golden registry → stacks=1, uses golden effect

    def test_attached_effects_collected(
        self,
        empty_game: "Game",
        mock_unit: Callable[..., Unit],
    ) -> None:
        p = empty_game.players[0]
        unit = mock_unit(CardIDs.TABBYCAT, owner_id=p.uid)
        unit.attached_turn[EffectIDs.CRAB_DEATHRATTLE] = 2
        p.board.append(unit)

        event = Event(event_type=EventType.MINION_DIED)
        ctx = _make_context({p.uid: p})

        triggers = empty_game.event_manager.collect_triggers(event, ctx)
        crab_trigs = [t for t in triggers if "Crab Deathrattle" in t.trigger_def.name]

        assert len(crab_trigs) == 1
        assert crab_trigs[0].stacks == 2

    def test_zero_stack_attached_effect_skipped(
        self,
        empty_game: "Game",
        mock_unit: Callable[..., Unit],
    ) -> None:
        p = empty_game.players[0]
        unit = mock_unit(CardIDs.TABBYCAT, owner_id=p.uid)
        unit.attached_turn[EffectIDs.CRAB_DEATHRATTLE] = 0  # Zero stacks
        p.board.append(unit)

        event = Event(event_type=EventType.MINION_DIED)
        ctx = _make_context({p.uid: p})

        triggers = empty_game.event_manager.collect_triggers(event, ctx)
        crab_trigs = [t for t in triggers if "Crab Deathrattle" in t.trigger_def.name]

        assert len(crab_trigs) == 0  # Skipped because count <= 0


# ===================================================================
#  6. REINDEX
# ===================================================================


class TestReindex:
    """_reindex_side correctly updates uid-to-pos mappings."""

    def test_reindex_after_board_change(self) -> None:
        p = Player(uid=0, board=[], hand=[])
        u1 = Unit.create_from_db(CardIDs.TABBYCAT, uid=1, owner_id=0)
        u2 = Unit.create_from_db(CardIDs.TABBYCAT, uid=2, owner_id=0)
        p.board = [u1, u2]
        ctx = _make_context({0: p})

        # Remove u1 from board
        p.board.pop(0)
        ctx._reindex_side(0)

        pos = ctx.resolve_pos(EntityRef(uid=2))
        assert pos is not None
        assert pos.slot == 0  # u2 shifted to slot 0

    def test_reindex_clears_stale_entries(self) -> None:
        p = Player(uid=0, board=[], hand=[])
        u1 = Unit.create_from_db(CardIDs.TABBYCAT, uid=1, owner_id=0)
        p.board = [u1]
        ctx = _make_context({0: p})

        # Remove from board
        p.board.clear()
        ctx._reindex_side(0)

        assert ctx.resolve_unit(EntityRef(uid=1)) is None
