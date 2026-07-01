"""
Direct C++ Combat Tests.
Validates core combat mechanics directly on the compiled C++ engine bindings.
"""

import sys
import os
import pytest

sys.path.insert(0, "cpp/build")
sys.path.insert(0, "src")
if sys.platform == "win32":
    try:
        os.add_dll_directory(r"C:\msys64\mingw64\bin")
    except (OSError, AttributeError):
        pass

import hs_engine_cpp as engine
engine.register_all_effects()

from hearthstone.engine.enums import CardIDs, Tags, UnitType
from hearthstone.engine.cpp_bridge import CARD_ID_MAP, TAG_TO_BIT, TYPE_TO_BIT

# Map enums to bits
TAUNT = TAG_TO_BIT[Tags.TAUNT]
DIVINE_SHIELD = TAG_TO_BIT[Tags.DIVINE_SHIELD]
WINDFURY = TAG_TO_BIT[Tags.WINDFURY]
REBORN = TAG_TO_BIT[Tags.REBORN]
CLEAVE = TAG_TO_BIT[Tags.CLEAVE]
VENOMOUS = TAG_TO_BIT[Tags.VENOMOUS]
POISONOUS = TAG_TO_BIT[Tags.POISONOUS]

BEAST = TYPE_TO_BIT[UnitType.BEAST]
MECH = TYPE_TO_BIT[UnitType.MECH]
MURLOC = TYPE_TO_BIT[UnitType.MURLOC]
DRAGON = TYPE_TO_BIT[UnitType.DRAGON]
ELEMENTAL = TYPE_TO_BIT[UnitType.ELEMENTAL]

# Helper to build a C++ combat tuple
def make_unit(card_id, atk, hp, types=0, tags=0, tier=1, golden=False):
    cpp_id = CARD_ID_MAP.get(card_id, 0) if isinstance(card_id, CardIDs) else int(card_id)
    return (cpp_id, atk, hp, types, tags, tier, golden)

# Battle Outcome constants
DRAW = 1
WIN = 2
LOSE = 3


# ============================================================
# Test 1: Windfury (Attacking twice)
# ============================================================
def test_cpp_windfury():
    # Side 0: 4/10 Windfury minion (attacks first)
    # Side 1: Two 2/2 minions with no tags
    # With windfury:
    # 1. 4/10 attacks first 2/2 -> 2/2 dies, windfury minion goes to 4/8
    # 2. 4/8 attacks second 2/2 -> 2/2 dies, windfury minion goes to 4/6
    # Both enemies dead -> WIN, 1 surviving minion
    side0 = [make_unit("0", 4, 10, tags=WINDFURY)]
    side1 = [make_unit("0", 2, 2), make_unit("0", 2, 2)]
    
    outcome, damage = engine.fast_combat(side0, side1, seed=1)
    assert outcome == WIN


# ============================================================
# Test 2: Reborn
# ============================================================
def test_cpp_reborn():
    # Side 0: 2/1 Reborn minion
    # Side 1: 3/2 vanilla minion
    # Combat progression:
    # 1. They trade -> both die.
    # 2. Reborn triggers -> Side 0 summons 2/1 (with 1 HP and no Reborn).
    # 3. Side 1 is empty, Side 0 has 2/1 minion -> WIN
    side0 = [make_unit("0", 2, 1, tags=REBORN)]
    side1 = [make_unit("0", 3, 2)]
    
    outcome, damage = engine.fast_combat(side0, side1, seed=1)
    assert outcome == WIN
    assert damage > 0


# ============================================================
# Test 3: Divine Shield
# ============================================================
def test_cpp_divine_shield():
    # Side 0: 2/1 with Divine Shield
    # Side 1: 10/10 vanilla minion
    # Progression:
    # 1. 2/1 DS hits 10/10:
    #    - 10/10 takes 2 damage -> becomes 10/8
    #    - 2/1 DS takes 10 damage but loses Divine Shield -> survives as 2/1
    # 2. 10/8 hits 2/1:
    #    - 2/1 dies
    #    - 10/8 takes 2 damage -> becomes 10/6
    # Side 1 wins -> LOSE for side 0
    side0 = [make_unit("0", 2, 1, tags=DIVINE_SHIELD)]
    side1 = [make_unit("0", 10, 10)]
    
    outcome, damage = engine.fast_combat(side0, side1, seed=1)
    assert outcome == LOSE


# ============================================================
# Test 4: Cleave
# ============================================================
def test_cpp_cleave():
    # Side 0: 4/10 Cleave minion
    # Side 1: Three 2/2 minions (slots 0, 1, 2)
    # Cleave minion attacks the middle minion (slot 1):
    # - Middle minion takes 4 damage -> dies
    # - Left (slot 0) minion takes 4 damage -> dies
    # - Right (slot 2) minion takes 4 damage -> dies
    # All enemies die in one attack -> WIN
    side0 = [make_unit("0", 4, 10, tags=CLEAVE)]
    side1 = [make_unit("0", 2, 2), make_unit("0", 2, 2), make_unit("0", 2, 2)]
    
    # We choose a seed that guarantees middle minion is attacked, or since there is taunt,
    # let's put taunt on the middle one to guarantee cleave hits all three!
    side1_taunt = [
        make_unit("0", 2, 2),
        make_unit("0", 2, 2, tags=TAUNT),
        make_unit("0", 2, 2),
    ]
    outcome, damage = engine.fast_combat(side0, side1_taunt, seed=42)
    assert outcome == WIN


# ============================================================
# Test 5: Venomous
# ============================================================
def test_cpp_venomous():
    # Side 0: 1/1 Venomous minion
    # Side 1: 100/100 vanilla minion
    # Progression:
    # 1. 1/1 Venomous hits 100/100 -> deals 1 damage.
    # 2. Venomous trigger kills the 100/100 instantly.
    # 3. Both die -> DRAW
    side0 = [make_unit("0", 1, 1, tags=VENOMOUS)]
    side1 = [make_unit("0", 100, 100)]
    
    outcome, damage = engine.fast_combat(side0, side1, seed=1)
    assert outcome == DRAW


# ============================================================
# Test 6: Equivalence of batch methods (Tuples, NumPy, Flat list)
# ============================================================
def test_cpp_batch_equivalence():
    # Complex boards
    side0_py = [
        make_unit("101", 3, 4, types=MECH, tags=DIVINE_SHIELD | TAUNT),
        make_unit("103", 2, 1, tags=REBORN),
        make_unit("207", 2, 4, types=BEAST),
    ]
    side1_py = [
        make_unit("105", 5, 5, tags=CLEAVE),
        make_unit("117", 4, 3, tags=WINDFURY),
    ]
    
    base_seed = 42
    count = 100
    
    # 1. Tuples
    results_tup = engine.fast_combat_batch(side0_py, side1_py, base_seed, count)
    
    # 2. NumPy
    import numpy as np
    side0_np = np.array(side0_py, dtype=np.int32)
    side1_np = np.array(side1_py, dtype=np.int32)
    results_np = engine.fast_combat_batch_np(side0_np, side1_np, base_seed, count)
    
    # 3. Flat
    side0_flat = []
    for u in side0_py:
        side0_flat.extend(u)
    side1_flat = []
    for u in side1_py:
        side1_flat.extend(u)
    results_flat = engine.fast_combat_batch_flat(side0_flat, side1_flat, base_seed, count)
    
    # Assert exact 1:1 match
    assert len(results_tup) == count
    assert results_tup == results_np, "NumPy results mismatch!"
    assert results_tup == results_flat, "Flat results mismatch!"


# ============================================================
# Test 7: Amber Guardian (StartOfCombatBuffRandomFriendlyTypeAndDS)
# ============================================================
def test_cpp_amber_guardian():
    # Side 0: Amber Guardian (3/2, Dragon) + Sleepy Supporter (3/5, Dragon)
    # Side 1: 1/1 vanilla minion
    # SoC: Amber Guardian buffs Sleepy Supporter +2/+2 and Divine Shield -> 5/7 with DS.
    # 1/1 attacks and pops DS. 5/7 attacks and kills 1/1.
    # Side 0 survives -> WIN.
    side0 = [
        make_unit(CardIDs.AMBER_GUARDIAN, 3, 2, types=DRAGON),
        make_unit(CardIDs.SLEEPY_SUPPORTER, 3, 5, types=DRAGON),
    ]
    side1 = [make_unit("0", 1, 1)]
    outcome, damage = engine.fast_combat(side0, side1, seed=1)
    assert outcome == WIN


# ============================================================
# Test 8: Roaring Recruiter (OnFriendlyAttackBuffSelf)
# ============================================================
def test_cpp_roaring_recruiter():
    # Side 0: Roaring Recruiter (2/8, Dragon) + Sleepy Supporter (3/5, Dragon, Taunt)
    # Side 1: 3/3 vanilla minion
    # Sleepy Supporter attacks the 3/3:
    # - OnFriendlyAttackBuffSelf: Roaring Recruiter buffs Sleepy Supporter (attacker) +3/+1 -> becomes 6/6.
    # - 6/6 hits 3/3 -> 3/3 dies, Sleepy Supporter survives as 6/3.
    # Side 0 survives -> WIN.
    side0 = [
        make_unit(CardIDs.ROARING_RECRUITER, 2, 8, types=DRAGON),
        make_unit(CardIDs.SLEEPY_SUPPORTER, 3, 5, types=DRAGON, tags=TAUNT),
    ]
    side1 = [make_unit("0", 3, 3)]
    outcome, damage = engine.fast_combat(side0, side1, seed=1)
    assert outcome == WIN


# ============================================================
# Test 9: Twilight Watcher (OnFriendlyAttackBuffTriggerSelf)
# ============================================================
def test_cpp_twilight_watcher():
    # Side 0: Twilight Watcher (3/7, Dragon) + Sleepy Supporter (3/5, Dragon, attacks first)
    # Side 1: 3/3 vanilla minion
    # Sleepy Supporter attacks 3/3:
    # - OnFriendlyAttackBuffTriggerSelf: Twilight Watcher (self) is buffed +1/+3 -> becomes 4/10.
    # - Sleepy Supporter trades with 3/3 -> both die or Sleepy Supporter survives as 3/2.
    # - Twilight Watcher survives as 4/10.
    # Side 0 survives -> WIN.
    side0 = [
        make_unit(CardIDs.TWILIGHT_WATCHER, 3, 7, types=DRAGON),
        make_unit(CardIDs.SLEEPY_SUPPORTER, 3, 5, types=DRAGON),
    ]
    side1 = [make_unit("0", 3, 3)]
    outcome, damage = engine.fast_combat(side0, side1, seed=1)
    assert outcome == WIN


# ============================================================
# Test 10: Bonker (RallyBuffAllOthersByType)
# ============================================================
def test_cpp_bonker():
    # Side 0: Bonker (2/7, attacks first) + Sleepy Supporter (3/5)
    # Side 1: 2/2 vanilla minion
    # Bonker attacks 2/2:
    # - RallyBuffAllOthersByType: Bonker buffs Sleepy Supporter +2/+2 -> becomes 5/7.
    # - Bonker survives as 2/5.
    # - Sleepy Supporter survives as 5/7.
    # Side 0 survives -> WIN.
    side0 = [
        make_unit(CardIDs.BONKER, 2, 7),
        make_unit(CardIDs.SLEEPY_SUPPORTER, 3, 5),
    ]
    side1 = [make_unit("0", 2, 2)]
    outcome, damage = engine.fast_combat(side0, side1, seed=1)
    assert outcome == WIN


# ============================================================
# Test 11: Ultraviolet Ascendant (StartOfCombatBuffFriendlyTypeScaling)
# ============================================================
def test_cpp_ultraviolet_ascendant():
    # Side 0: Ultraviolet Ascendant (6/3, Elemental) + Waveling (6/1, Elemental)
    # Side 1: 1/1 vanilla minion
    # SoC: Ultraviolet Ascendant buffs other friendly Elementals +3/+2 -> Waveling becomes 9/3.
    # Waveling survives and wins.
    # Side 0 survives -> WIN.
    side0 = [
        make_unit(CardIDs.ULTRAVIOLET_ASCENDANT, 6, 3, types=ELEMENTAL),
        make_unit(CardIDs.WAVELING, 6, 1, types=ELEMENTAL),
    ]
    side1 = [make_unit("0", 1, 1)]
    outcome, damage = engine.fast_combat(side0, side1, seed=1)
    assert outcome == WIN

