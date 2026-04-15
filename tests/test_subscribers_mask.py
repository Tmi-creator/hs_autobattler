"""
Tests for the C++ subscribers-mask optimization in CombatBoard.

The subscribers[event_type] bitmask is maintained incrementally in
insert_at/remove_at, and rebuilt from scratch by recalculate_subscribers.
These tests verify that:
  1. Both paths (parse+recalc vs repeated insert_at) produce identical masks.
  2. remove_at correctly shifts bits after the removed slot.
  3. End-to-end combat with deathrattle summons (which triggers insert_at
     mid-combat) still produces stable outcomes across seeds.
"""
import os
import sys

import pytest

sys.path.insert(0, "cpp/build")
os.add_dll_directory(r"C:\msys64\mingw64\bin")
import hs_engine_cpp as cpp_engine

cpp_engine.register_all_effects()

# Type / tag bit constants (mirror cpp/include/types.h)
BEAST = 1 << 0
DEMON = 1 << 2
MURLOC = 1 << 3
PIRATE = 1 << 4
MECH = 1 << 6
UNDEAD = 1 << 7

TAUNT = 1 << 1
DIVINE_SHIELD = 1 << 2

# Card IDs with known combat triggers (from cpp/include/generated_card_ids.h).
CORD_PULLER = 103          # deathrattle summon → MINION_DIED
HARMLESS_BONEHEAD = 107    # deathrattle summon × 2
TUSKED_CAMPER = 119        # rally → ATTACK_DECLARED
SWAMPSTRIKER = 118         # on_play → MINION_PLAYED
VANILLA_STATLINE = 0       # no triggers at all


def cu(card_id, atk, hp, types=0, tags=0, tier=1, golden=False):
    return (card_id, atk, hp, types, tags, tier, golden)


# ============================================================
# 1. Parse path vs insert_at path — must yield identical masks.
# ============================================================


@pytest.mark.skipif(
    not hasattr(cpp_engine, "debug_board_via_parse"),
    reason="Debug helpers not built into module",
)
class TestSubscribersShiftCorrectness:
    """Builds a board via two independent paths and asserts state matches."""

    @staticmethod
    def _both_paths(units):
        via_parse = cpp_engine.debug_board_via_parse(units)
        via_inserts = cpp_engine.debug_board_via_inserts(units)
        return via_parse, via_inserts

    def test_empty_board(self):
        a, b = self._both_paths([])
        assert a == b
        assert a[0] == [0] * 18  # no subscribers for any event
        assert a[1] == 0         # no taunts
        assert a[2] == 0         # no damage

    def test_single_vanilla_unit(self):
        units = [cu(VANILLA_STATLINE, 3, 3, BEAST, 0, 2)]
        a, b = self._both_paths(units)
        assert a == b
        assert a[0] == [0] * 18  # vanilla unit subscribes to nothing
        assert a[2] == 2          # damage = tier

    def test_single_deathrattle_unit(self):
        units = [cu(CORD_PULLER, 3, 1, PIRATE, 0, 1)]
        a, b = self._both_paths(units)
        assert a == b
        # Cord Puller has MINION_DIED trigger (event_type = 5), so subscribers[5] bit 0 is set
        assert a[0][5] == 1  # slot 0 subscribes to MINION_DIED

    def test_single_taunt_unit(self):
        units = [cu(VANILLA_STATLINE, 2, 5, BEAST, TAUNT, 2)]
        a, b = self._both_paths(units)
        assert a == b
        assert a[1] == 1  # taunt_mask bit 0 set
        assert a[0] == [0] * 18  # no triggers

    def test_full_board_mixed(self):
        """7 units with various triggers and taunts — the stress case."""
        units = [
            cu(CORD_PULLER, 3, 1, PIRATE, 0, 1),        # slot 0: MINION_DIED
            cu(VANILLA_STATLINE, 4, 5, BEAST, 0, 2),    # slot 1: nothing
            cu(TUSKED_CAMPER, 3, 3, BEAST, TAUNT, 1),   # slot 2: ATTACK_DECLARED + taunt
            cu(HARMLESS_BONEHEAD, 1, 4, UNDEAD, 0, 1),  # slot 3: MINION_DIED
            cu(VANILLA_STATLINE, 6, 6, DEMON, TAUNT, 3),# slot 4: taunt
            cu(SWAMPSTRIKER, 2, 3, MURLOC, 0, 1),       # slot 5: MINION_PLAYED
            cu(TUSKED_CAMPER, 3, 3, BEAST, 0, 1),       # slot 6: ATTACK_DECLARED
        ]
        a, b = self._both_paths(units)
        assert a == b, (
            "insert_at shift path diverged from parse+recalculate!\n"
            f"  parse:   {a}\n"
            f"  inserts: {b}"
        )
        # Sanity: taunt bits 2 and 4 set
        assert a[1] == (1 << 2) | (1 << 4)

    def test_reversed_order_stability(self):
        """Same units in reverse — mask values for a given slot depend on slot index, not UID."""
        a_forward = cpp_engine.debug_board_via_parse([
            cu(CORD_PULLER, 3, 1, PIRATE, 0, 1),
            cu(VANILLA_STATLINE, 4, 5, BEAST, 0, 2),
            cu(HARMLESS_BONEHEAD, 1, 4, UNDEAD, 0, 1),
        ])
        a_reversed = cpp_engine.debug_board_via_parse([
            cu(HARMLESS_BONEHEAD, 1, 4, UNDEAD, 0, 1),
            cu(VANILLA_STATLINE, 4, 5, BEAST, 0, 2),
            cu(CORD_PULLER, 3, 1, PIRATE, 0, 1),
        ])
        # Forward: cord in slot 0, bonehead in slot 2 → subscribers[MINION_DIED] = 0b101 = 5
        # Reverse: bonehead in slot 0, cord in slot 2 → same 0b101 = 5
        assert a_forward[0][5] == 0b101
        assert a_reversed[0][5] == 0b101


# ============================================================
# 2. remove_at shift correctness at each slot index.
# ============================================================


@pytest.mark.skipif(
    not hasattr(cpp_engine, "debug_board_remove"),
    reason="Debug helpers not built into module",
)
class TestRemoveAtShift:

    @staticmethod
    def _make_units():
        """Board with triggers at slots 0, 2, 5 (MINION_DIED)."""
        return [
            cu(CORD_PULLER, 3, 1, PIRATE, 0, 1),        # slot 0 — subscribes
            cu(VANILLA_STATLINE, 4, 5, BEAST, 0, 2),    # slot 1 — vanilla
            cu(HARMLESS_BONEHEAD, 1, 4, UNDEAD, 0, 1),  # slot 2 — subscribes
            cu(VANILLA_STATLINE, 6, 6, DEMON, 0, 3),    # slot 3 — vanilla
            cu(VANILLA_STATLINE, 2, 5, BEAST, TAUNT, 2),# slot 4 — taunt, no trigger
            cu(CORD_PULLER, 3, 1, PIRATE, 0, 1),        # slot 5 — subscribes
        ]

    def test_remove_first_shifts_all_down(self):
        units = self._make_units()
        # Before: MINION_DIED mask = 0b100101 (slots 0, 2, 5)
        # After remove(0): slots 2,5 → 1,4; mask = 0b010010 = 18
        subs, taunt, _ = cpp_engine.debug_board_remove(units, 0)
        assert subs[5] == (1 << 1) | (1 << 4)
        # Taunt was at old slot 4 → now at slot 3
        assert taunt == (1 << 3)

    def test_remove_middle_splits_shift(self):
        units = self._make_units()
        # remove(3): slot 4,5 shift to 3,4
        # MINION_DIED: slots 0, 2 stay; slot 5 → 4. Mask = 0b10101 = 21
        subs, taunt, _ = cpp_engine.debug_board_remove(units, 3)
        assert subs[5] == (1 << 0) | (1 << 2) | (1 << 4)
        # Taunt at old slot 4 → new slot 3
        assert taunt == (1 << 3)

    def test_remove_last_clears_top_bit(self):
        units = self._make_units()
        # remove(5): slot 5's MINION_DIED bit disappears, nothing below shifts
        subs, taunt, _ = cpp_engine.debug_board_remove(units, 5)
        assert subs[5] == (1 << 0) | (1 << 2)
        # Taunt at slot 4 unchanged (it's below the removed slot)
        assert taunt == (1 << 4)

    def test_remove_subscribed_unit_clears_its_bit(self):
        units = self._make_units()
        # remove(2) — bonehead — its MINION_DIED bit should be gone.
        # slots 3,4,5 shift to 2,3,4
        # MINION_DIED: slot 0 stays, slot 5 → 4. Mask = 0b10001 = 17
        subs, taunt, _ = cpp_engine.debug_board_remove(units, 2)
        assert subs[5] == (1 << 0) | (1 << 4)
        # Taunt at slot 4 → slot 3
        assert taunt == (1 << 3)


# ============================================================
# 3. End-to-end: combats with mid-fight insert_at still deterministic.
# ============================================================


class TestEndToEndWithSummons:
    """If subscribers shift is broken, deathrattle-heavy boards would mis-trigger
    and produce either crashes or different damage values. Verify stability."""

    def test_deathrattle_board_deterministic(self):
        """Same seed → same result. Runs a board with many deathrattle summons."""
        board = [
            cu(HARMLESS_BONEHEAD, 1, 4, UNDEAD, 0, 1),
            cu(CORD_PULLER, 3, 1, PIRATE, 0, 1),
            cu(VANILLA_STATLINE, 5, 5, BEAST, 0, 2),
        ]
        opp = [
            cu(VANILLA_STATLINE, 4, 4, DEMON, 0, 2),
            cu(VANILLA_STATLINE, 3, 3, MECH, DIVINE_SHIELD, 1),
            cu(VANILLA_STATLINE, 2, 6, BEAST, TAUNT, 2),
        ]
        r1 = cpp_engine.fast_combat(board, opp, seed=42,
                                    tavern_tier_0=2, tavern_tier_1=2)
        r2 = cpp_engine.fast_combat(board, opp, seed=42,
                                    tavern_tier_0=2, tavern_tier_1=2)
        assert r1 == r2

    def test_summons_do_not_crash(self):
        """Stress test — many seeds, board with deathrattles that summon
        multiple minions on death (insert_at called repeatedly mid-combat)."""
        board = [
            cu(HARMLESS_BONEHEAD, 1, 4, UNDEAD, 0, 1),
            cu(HARMLESS_BONEHEAD, 1, 4, UNDEAD, 0, 1),
            cu(CORD_PULLER, 3, 1, PIRATE, 0, 1),
        ]
        opp = [
            cu(VANILLA_STATLINE, 5, 5, BEAST, 0, 2),
            cu(VANILLA_STATLINE, 5, 5, BEAST, 0, 2),
            cu(VANILLA_STATLINE, 5, 5, BEAST, 0, 2),
        ]
        results = cpp_engine.fast_combat_batch(
            board, opp, base_seed=0, count=500,
            tavern_tier_0=2, tavern_tier_1=2,
        )
        assert len(results) == 500
        # Shouldn't crash, and at least some outcomes should vary by seed.
        unique_outcomes = {tuple(r) for r in results}
        assert len(unique_outcomes) > 1, "Expected RNG variance across seeds"
