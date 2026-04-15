"""C++-only combat benchmark (for py-spy profiling)."""
import sys
import os
import time

sys.path.insert(0, "cpp/build")
os.add_dll_directory(r"C:\msys64\mingw64\bin")
import hs_engine_cpp as cpp_engine
cpp_engine.register_all_effects()

BEAST   = 1 << 0
DEMON   = 1 << 2
MURLOC  = 1 << 3
PIRATE  = 1 << 4
MECH    = 1 << 6

TAUNT         = 1 << 1
DIVINE_SHIELD = 1 << 2
WINDFURY      = 1 << 3

C = {
    "SCALLYWAG":        103,
    "ANNOY_O_TRON":     105,
    "IMPRISONER":       108,
    "DIRE_WOLF":        109,
    "SPAWN_OF_NZOTH":   206,
    "KABOOM_BOT":       207,
    "MURLOC_WARLEADER": 202,
    "SOUTHSEA_CAPTAIN": 204,
}

def cu(card_id, atk, hp, types=0, tags=0, tier=1, golden=False):
    return (card_id, atk, hp, types, tags, tier, golden)

BOARD_A = [
    cu(C["SCALLYWAG"], 3, 1, PIRATE),
    cu(0, 4, 5, BEAST, 0, 2),
    cu(C["IMPRISONER"], 3, 3, DEMON, TAUNT),
    cu(C["KABOOM_BOT"], 2, 2, MECH, 0, 2),
    cu(C["SPAWN_OF_NZOTH"], 2, 2, 0, 0, 2),
    cu(0, 5, 4, 0, 0, 2),
    cu(0, 3, 6, MURLOC, WINDFURY),
]

BOARD_B = [
    cu(C["DIRE_WOLF"], 1, 2, BEAST),
    cu(C["MURLOC_WARLEADER"], 3, 3, MURLOC, 0, 2),
    cu(0, 2, 3, MURLOC),
    cu(C["ANNOY_O_TRON"], 1, 2, MECH, TAUNT | DIVINE_SHIELD),
    cu(C["SOUTHSEA_CAPTAIN"], 3, 3, PIRATE, 0, 2),
    cu(0, 4, 3, PIRATE),
    cu(0, 2, 5, BEAST, TAUNT, 2),
]

BOARD_C = [
    cu(0, 6, 8, DEMON, 0, 3),
    cu(0, 5, 5, MECH, DIVINE_SHIELD, 2),
    cu(0, 8, 4, BEAST, WINDFURY, 3),
    cu(0, 3, 10, 0, TAUNT, 3),
    cu(C["KABOOM_BOT"], 2, 2, MECH, 0, 2),
    cu(0, 4, 6, PIRATE, 0, 2),
    cu(C["SCALLYWAG"], 3, 1, PIRATE),
]

MATCHUPS = [
    (BOARD_A, BOARD_B),
    (BOARD_B, BOARD_C),
    (BOARD_C, BOARD_A),
]

NUM_COMBATS = int(os.environ.get("N", "200000"))

if __name__ == "__main__":
    # Warmup
    for b0, b1 in MATCHUPS:
        cpp_engine.fast_combat_batch(b0, b1, base_seed=0, count=100,
                                     tavern_tier_0=3, tavern_tier_1=3)

    if hasattr(cpp_engine, "prof_reset"):
        cpp_engine.prof_reset()

    total = 0
    start = time.perf_counter()
    for b0, b1 in MATCHUPS:
        cpp_engine.fast_combat_batch(b0, b1, base_seed=0, count=NUM_COMBATS,
                                     tavern_tier_0=3, tavern_tier_1=3)
        total += NUM_COMBATS
    elapsed = time.perf_counter() - start
    rate = total / elapsed
    print(f"C++: {total} combats in {elapsed:.3f}s = {rate:,.0f} combats/sec")

    if hasattr(cpp_engine, "prof_dump"):
        rows = cpp_engine.prof_dump()
        # First row is RESOLVE_COMBAT — use as 100% baseline
        base_cycles = rows[0][1] or 1
        print(f"\n{'section':<20} {'%':>7} {'Mcyc':>10} {'calls':>10} {'cyc/call':>10}")
        print("-" * 62)
        for name, cycles, calls in rows:
            pct = 100.0 * cycles / base_cycles
            mcyc = cycles / 1e6
            cpc = (cycles / calls) if calls else 0
            print(f"{name:<20} {pct:>6.1f}% {mcyc:>10.1f} {calls:>10} {cpc:>10.0f}")
