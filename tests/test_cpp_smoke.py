"""
Quick smoke test for the C++ engine.
Tests basic combat scenarios to verify correctness.
"""

import sys, os

sys.path.insert(0, "cpp/build")
os.add_dll_directory(r"C:\msys64\mingw64\bin")

import hs_engine_cpp as engine

# Register all effects (must call once)
engine.register_all_effects()

# ============================================================
# Constants from types.h
# ============================================================
# UnitTypes
NONE = 0
BEAST = 1 << 0
DRAGON = 1 << 1
DEMON = 1 << 2
MURLOC = 1 << 3
PIRATE = 1 << 4
MECH = 1 << 6

# Tags
NO_TAGS = 0
TAUNT = 1 << 1
DIVINE_SHIELD = 1 << 2
WINDFURY = 1 << 3
POISONOUS = 1 << 4
REBORN = 1 << 5
CLEAVE = 1 << 7

# CardIDs
ALLEYCAT = 102
SCALLYWAG = 103
ANNOY_O_TRON = 105
IMPRISONER = 108
SPAWN_OF_NZOTH = 206
KABOOM_BOT = 207

# BattleOutcome
NO_END = 0
DRAW = 1
WIN = 2
LOSE = 3


def make_unit(card_id, atk, hp, types=NONE, tags=NO_TAGS, tier=1, golden=False):
    return (card_id, atk, hp, types, tags, tier, golden)


# ============================================================
# Test 1: Simple 1v1 — higher stats win
# ============================================================
def test_simple_1v1():
    side0 = [make_unit(0, 10, 10)]
    side1 = [make_unit(0, 3, 3)]
    outcome, damage = engine.fast_combat(side0, side1, seed=42)
    assert outcome == WIN, f"Expected WIN, got {outcome}"
    print(f"  Test 1 OK: 10/10 vs 3/3 -> WIN, damage={damage}")


# ============================================================
# Test 2: Mirror 1/1 vs 1/1 — should be a draw
# ============================================================
def test_mirror_1v1():
    side0 = [make_unit(0, 1, 1)]
    side1 = [make_unit(0, 1, 1)]
    outcome, damage = engine.fast_combat(side0, side1, seed=42)
    assert outcome == DRAW, f"Expected DRAW, got {outcome}"
    print(f"  Test 2 OK: 1/1 vs 1/1 -> DRAW")


# ============================================================
# Test 3: Taunt targeting
# Side 0: [2/3, 0/1, 0/1] (3 units -> attacks first)
# Side 1: [100/1 taunt, 1/1]
# With taunt: 2/3 MUST hit 100/1 taunt -> kills it, takes 100 counter -> dies.
#   0/1s can't attack. 1/1 kills them. Always LOSE.
# Without taunt: ~50% chance 2/3 hits 1/1 -> kills it, takes 1 -> survives.
#   Then 100/1 hits 2/2 -> 2/2 dies -> but 100/1 took 2 counter -> also dies.
#   0/1s survive alone -> WIN.
# ============================================================
def test_taunt():
    losses = 0
    for seed in range(100):
        side0 = [make_unit(0, 2, 3), make_unit(0, 0, 1), make_unit(0, 0, 1)]
        side1 = [
            make_unit(0, 100, 1, tags=TAUNT),  # taunt: 100 counter-damage
            make_unit(0, 1, 1),                 # weak, no taunt
        ]
        outcome, _ = engine.fast_combat(side0, side1, seed=seed)
        if outcome == LOSE:
            losses += 1
    assert losses == 100, f"Expected 100 losses (taunt forces into 100/1), got {losses}"
    print("  Test 3 OK: Taunt verified over 100 seeds -> always LOSE")


# ============================================================
# Test 4: Divine Shield survives one hit
# ============================================================
def test_divine_shield():
    side0 = [make_unit(0, 1, 1, tags=DIVINE_SHIELD)]
    side1 = [make_unit(0, 1, 1)]
    # DS unit takes first hit, loses shield, then attacks and kills enemy 1/1.
    # Next turn: DS unit (now 1/1 no shield) vs nothing -> WIN
    outcome, damage = engine.fast_combat(side0, side1, seed=42)
    assert outcome == WIN, f"Expected WIN, got {outcome}"
    print(f"  Test 4 OK: Divine Shield -> WIN")


# ============================================================
# Test 5a: Scallywag 3/1 vs 1/3 — both die, DR token survives -> WIN
# ============================================================
def test_scallywag_vs_weak():
    # Regardless of who attacks first:
    # 3/1 and 1/3 trade -> both die (3 dmg kills 1/3, 1 dmg kills 3/1)
    # Scallywag DR fires -> 1/1 pirate token spawns
    # Enemy is dead -> token is alone -> WIN
    results = engine.fast_combat_batch(
        [make_unit(SCALLYWAG, 3, 1, types=PIRATE)], [make_unit(0, 1, 3)], base_seed=0, count=100
    )
    wins = sum(1 for r in results if r[0] == WIN)
    assert wins == 100, f"Expected 100 wins, got {wins}"
    print(f"  Test 5a OK: Scallywag 3/1 vs 1/3 -> always WIN (DR token survives)")


# ============================================================
# Test 5b: Scallywag 3/1 vs 1/5 — enemy survives token
# ============================================================
def test_scallywag_vs_strong():
    # 3/1 attacks 1/5: enemy goes to 1/2, scallywag dies.
    # DR: 1/1 pirate token (immediate attack) -> enemy 1/1
    # Enemy attacks token -> both die -> DRAW? or token dies, enemy survives at 1/1?
    # Actually: token attacks first (immediate), then enemy counter-attacks.
    # Token 1/1 hits enemy 1/2 -> enemy takes 1 -> 1/1. Token takes 1 counter -> dies.
    # Enemy 1/1 survives -> LOSE for side0.
    outcome, damage = engine.fast_combat(
        [make_unit(SCALLYWAG, 3, 1, types=PIRATE)], [make_unit(0, 1, 5)], seed=42
    )
    assert outcome == LOSE, f"Expected LOSE, got {outcome}"
    print(f"  Test 5b OK: Scallywag 3/1 vs 1/5 -> LOSE (enemy survives at 1hp)")


# ============================================================
# Test 6: Batch — run 1000 combats
# ============================================================
def test_batch():
    side0 = [make_unit(0, 3, 3)]
    side1 = [make_unit(0, 3, 3)]
    results = engine.fast_combat_batch(side0, side1, base_seed=0, count=1000)
    outcomes = [r[0] for r in results]
    draws = outcomes.count(DRAW)
    # 3/3 vs 3/3: both die simultaneously -> should always be DRAW
    assert draws == 1000, f"Expected 1000 draws, got {draws}"
    print(f"  Test 6 OK: 1000x 3/3 vs 3/3 -> 100% draws")


# ============================================================
# Test 7: Batch stats — 5/3 vs 3/5
# ============================================================
def test_batch_stats():
    side0 = [make_unit(0, 5, 3)]
    side1 = [make_unit(0, 3, 5)]
    results = engine.fast_combat_batch(side0, side1, base_seed=0, count=10000)
    outcomes = [r[0] for r in results]
    wins = outcomes.count(WIN)
    losses = outcomes.count(LOSE)
    draws = outcomes.count(DRAW)
    print(f"  Test 7: 5/3 vs 3/5 over 10K: W={wins} L={losses} D={draws}")
    # Both should die: 5/3 takes 3 dmg -> 0hp, 3/5 takes 5 dmg -> 0hp
    # Both die same turn -> DRAW
    assert draws == 10000, f"Expected all draws for 5/3 vs 3/5"
    print(f"  Test 7 OK: all draws as expected")


# ============================================================
# Test 8: Performance — 10K full-board combats
# ============================================================
def test_perf():
    import time

    side0 = [make_unit(0, 3, 3) for _ in range(7)]
    side1 = [make_unit(0, 3, 3) for _ in range(7)]

    start = time.perf_counter()
    results = engine.fast_combat_batch(side0, side1, base_seed=0, count=10000)
    elapsed = time.perf_counter() - start

    print(
        f"  Test 8: 10K combats (7v7 vanilla) in {elapsed:.3f}s = {10000 / elapsed:.0f} combats/sec"
    )


# ============================================================
# Run all
# ============================================================
if __name__ == "__main__":
    print("=== C++ Engine Smoke Tests ===")
    test_simple_1v1()
    test_mirror_1v1()
    test_taunt()
    test_divine_shield()
    test_scallywag_vs_weak()
    test_scallywag_vs_strong()
    test_batch()
    test_batch_stats()
    test_perf()
    print("\nAll tests passed!")
