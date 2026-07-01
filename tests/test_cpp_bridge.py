"""
Test the C++ bridge: verify that _unit_to_cpp conversion and
resolve_combat_fast() produce valid results.

Run:  python -m pytest tests/test_cpp_bridge.py -v
"""
import sys
import os
import random

import pytest

# Paths — same as bench_cpp_vs_python.py
sys.path.insert(0, "cpp/build")
sys.path.insert(0, "src")

# Try to add DLL dir on Windows (for MinGW)
if sys.platform == "win32":
    try:
        os.add_dll_directory(r"C:\msys64\mingw64\bin")
    except (OSError, AttributeError):
        pass

from hearthstone.engine.combat import CombatManager
from hearthstone.engine.cpp_bridge import (
    CARD_ID_MAP,
    TAG_TO_BIT,
    TYPE_TO_BIT,
    get_cpp_engine,
)
from hearthstone.engine.entities import Player, Unit
from hearthstone.engine.enums import BattleOutcome, CardIDs, Tags, UnitType


# Skip all tests if C++ engine is not compiled
cpp = get_cpp_engine()
pytestmark = pytest.mark.skipif(cpp is None, reason="C++ engine not built")


# ================================================================
# Test mapping correctness
# ================================================================
class TestMappings:
    def test_all_cardids_mapped(self):
        """Every CardIDs enum member should have a C++ mapping."""
        for card_id in CardIDs:
            assert card_id.value in CARD_ID_MAP, (
                f"{card_id.name} ({card_id.value}) missing from CARD_ID_MAP"
            )

    def test_all_types_mapped(self):
        """Every UnitType should map to a non-zero bit."""
        for ut in UnitType:
            assert ut in TYPE_TO_BIT, f"{ut.name} missing from TYPE_TO_BIT"
            assert TYPE_TO_BIT[ut] > 0

    def test_all_tags_mapped(self):
        """Every Tags enum member should map to a non-zero bit."""
        for tag in Tags:
            assert tag in TAG_TO_BIT, f"{tag.name} missing from TAG_TO_BIT"
            assert TAG_TO_BIT[tag] > 0

    def test_type_bits_unique(self):
        """All type bits must be powers of 2 (no collisions)."""
        bits = list(TYPE_TO_BIT.values())
        assert len(bits) == len(set(bits))
        for b in bits:
            assert b & (b - 1) == 0, f"Not a power of 2: {b}"

    def test_tag_bits_unique(self):
        """All tag bits must be powers of 2 (no collisions)."""
        bits = list(TAG_TO_BIT.values())
        assert len(bits) == len(set(bits))
        for b in bits:
            assert b & (b - 1) == 0, f"Not a power of 2: {b}"


# ================================================================
# Test _unit_to_cpp conversion
# ================================================================
class TestUnitConversion:
    def test_basic_unit(self):
        """Convert a simple unit to C++ tuple."""
        u = Unit.create_from_db(CardIDs.SCALLYWAG, uid=1, owner_id=0)
        t = CombatManager._unit_to_cpp(u)
        card_id, atk, hp, types, tags, tier, golden = t
        assert card_id == 103  # Scallywag
        assert atk == 3
        assert hp == 1
        assert types == TYPE_TO_BIT[UnitType.PIRATE]
        assert tier == 1
        assert golden is False

    def test_golden_unit(self):
        """Golden unit should have doubled stats and golden=True."""
        u = Unit.create_from_db(CardIDs.SCALLYWAG, uid=1, owner_id=0, is_golden=True)
        t = CombatManager._unit_to_cpp(u)
        card_id, atk, hp, types, tags, tier, golden = t
        assert card_id == 103
        assert atk == 6  # doubled
        assert hp == 2   # doubled
        assert golden is True

    def test_unit_with_tags(self):
        """Unit with TAUNT + DIVINE_SHIELD should have correct tag bits."""
        u = Unit.create_from_db(CardIDs.ANNOY_O_TRON, uid=1, owner_id=0)
        t = CombatManager._unit_to_cpp(u)
        _, _, _, _, tags, _, _ = t
        assert tags & TAG_TO_BIT[Tags.TAUNT]
        assert tags & TAG_TO_BIT[Tags.DIVINE_SHIELD]


# ================================================================
# Test resolve_combat_fast
# ================================================================
class TestFastCombat:
    def _make_player(self, units, uid=0, tavern_tier=1):
        """Helper: create a Player with given units."""
        p = Player(uid=uid, board=units, hand=[], health=40)
        p.tavern_tier = tavern_tier
        return p

    def test_basic_1v1(self):
        """1v1 combat should produce WIN, LOSE, or DRAW."""
        u0 = Unit.create_from_db(CardIDs.SCALLYWAG, uid=1, owner_id=0)
        u1 = Unit.create_from_db(CardIDs.ALLEYCAT, uid=2, owner_id=1)
        p0 = self._make_player([u0], uid=0)
        p1 = self._make_player([u1], uid=1)
        cm = CombatManager()
        result, damage = cm.resolve_combat_fast(p0, p1)
        assert isinstance(result, BattleOutcome)
        assert result != BattleOutcome.NO_END

    def test_empty_boards_draw(self):
        """Empty vs Empty = DRAW."""
        p0 = self._make_player([], uid=0)
        p1 = self._make_player([], uid=1)
        cm = CombatManager()
        result, damage = cm.resolve_combat_fast(p0, p1)
        assert result == BattleOutcome.DRAW
        assert damage == 0

    def test_statistical_consistency(self):
        """Run N combats, ensure both engines produce similar distributions."""
        N = 200
        cpp_wins = 0
        py_wins = 0

        for seed in range(N):
            random.seed(seed)
            u0 = Unit.create_from_db(CardIDs.IMPRISONER, uid=100, owner_id=0)
            u1 = Unit.create_from_db(CardIDs.ANNOY_O_TRON, uid=200, owner_id=1)
            p0_cpp = self._make_player([u0], uid=0, tavern_tier=2)
            p1_cpp = self._make_player([u1], uid=1, tavern_tier=2)

            cm = CombatManager()
            result, _ = cm.resolve_combat_fast(p0_cpp, p1_cpp)
            if result == BattleOutcome.WIN:
                cpp_wins += 1

        for seed in range(N):
            random.seed(seed)
            u0 = Unit.create_from_db(CardIDs.IMPRISONER, uid=100, owner_id=0)
            u1 = Unit.create_from_db(CardIDs.ANNOY_O_TRON, uid=200, owner_id=1)
            p0_py = self._make_player([u0], uid=0, tavern_tier=2)
            p1_py = self._make_player([u1], uid=1, tavern_tier=2)

            cm = CombatManager()
            result, _ = cm.resolve_combat(p0_py, p1_py)
            if result == BattleOutcome.WIN:
                py_wins += 1

        cpp_wr = cpp_wins / N
        py_wr = py_wins / N
        # Different RNGs, but results should be within ~15%
        assert abs(cpp_wr - py_wr) < 0.20, (
            f"Win rates diverge too much: C++ {cpp_wr:.2f} vs Python {py_wr:.2f}"
        )
