"""
Performance benchmark: Python engine vs C++ engine
Uses real cards with effects, deathrattles, auras, divine shields.
"""
import sys
import os
import time
import random

# ============================================================
# Setup C++ engine
# ============================================================
sys.path.insert(0, "cpp/build")
os.add_dll_directory(r"C:\msys64\mingw64\bin")
import hs_engine_cpp as cpp_engine
cpp_engine.register_all_effects()

# ============================================================
# Setup Python engine
# ============================================================
sys.path.insert(0, "src")
from hearthstone.engine.combat import CombatManager
from hearthstone.engine.entities import Player, Unit
from hearthstone.engine.enums import CardIDs, Tags, UnitType

# ============================================================
# C++ constants (mirror types.h)
# ============================================================
BEAST   = 1 << 0
DEMON   = 1 << 2
MURLOC  = 1 << 3
PIRATE  = 1 << 4
MECH    = 1 << 6

TAUNT         = 1 << 1
DIVINE_SHIELD = 1 << 2
WINDFURY      = 1 << 3

# CardIDs for C++
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
    """make C++ unit tuple"""
    return (card_id, atk, hp, types, tags, tier, golden)


# ============================================================
# Boards — real game scenarios, 7v7
# ============================================================
# Board A: Deathrattle (Scallywag, Imprisoner, Kaboom Bot, Spawn of N'Zoth)
BOARD_A = [
    cu(C["SCALLYWAG"], 3, 1, PIRATE),
    cu(0, 4, 5, BEAST, 0, 2),
    cu(C["IMPRISONER"], 3, 3, DEMON, TAUNT),
    cu(C["KABOOM_BOT"], 2, 2, MECH, 0, 2),
    cu(C["SPAWN_OF_NZOTH"], 2, 2, 0, 0, 2),
    cu(0, 5, 4, 0, 0, 2),
    cu(0, 3, 6, MURLOC, WINDFURY),
]

# Board B: Aura + DS comp
BOARD_B = [
    cu(C["DIRE_WOLF"], 1, 2, BEAST),
    cu(C["MURLOC_WARLEADER"], 3, 3, MURLOC, 0, 2),
    cu(0, 2, 3, MURLOC),
    cu(C["ANNOY_O_TRON"], 1, 2, MECH, TAUNT | DIVINE_SHIELD),
    cu(C["SOUTHSEA_CAPTAIN"], 3, 3, PIRATE, 0, 2),
    cu(0, 4, 3, PIRATE),
    cu(0, 2, 5, BEAST, TAUNT, 2),
]

# Board C: Mixed big stats + DR
BOARD_C = [
    cu(0, 6, 8, DEMON, 0, 3),
    cu(0, 5, 5, MECH, DIVINE_SHIELD, 2),
    cu(0, 8, 4, BEAST, WINDFURY, 3),
    cu(0, 3, 10, 0, TAUNT, 3),
    cu(C["KABOOM_BOT"], 2, 2, MECH, 0, 2),
    cu(0, 4, 6, PIRATE, 0, 2),
    cu(C["SCALLYWAG"], 3, 1, PIRATE),
]


# ============================================================
# Python board builder using create_from_db for cards with effects
# ============================================================
# Map C++ card_ids to Python card_ids
PY_CARDS = {
    C["SCALLYWAG"]:        CardIDs.SCALLYWAG,
    C["ANNOY_O_TRON"]:     CardIDs.ANNOY_O_TRON,
    C["IMPRISONER"]:       CardIDs.IMPRISONER,
    C["DIRE_WOLF"]:        CardIDs.DIRE_WOLF_ALPHA,
    C["SPAWN_OF_NZOTH"]:   CardIDs.SPAWN_OF_NZOTH,
    C["KABOOM_BOT"]:       CardIDs.KABOOM_BOT,
    C["MURLOC_WARLEADER"]: CardIDs.MURLOC_WARLEADER,
    C["SOUTHSEA_CAPTAIN"]: CardIDs.SOUTHSEA_CAPTAIN,
}

TYPE_MAP = {
    BEAST: [UnitType.BEAST],
    DEMON: [UnitType.DEMON],
    MURLOC: [UnitType.MURLOC],
    PIRATE: [UnitType.PIRATE],
    MECH: [UnitType.MECH],
}

TAG_MAP = {
    TAUNT: Tags.TAUNT,
    DIVINE_SHIELD: Tags.DIVINE_SHIELD,
    WINDFURY: Tags.WINDFURY,
}


def cpp_board_to_python(cpp_board, uid_start):
    """Convert a C++ board spec into a list of Python Unit objects."""
    uid = uid_start
    units = []
    for card_id, atk, hp, types, tags, tier, golden in cpp_board:
        if card_id != 0 and card_id in PY_CARDS:
            u = Unit.create_from_db(PY_CARDS[card_id], uid, 0, golden)
        else:
            py_types = TYPE_MAP.get(types, [UnitType.NEUTRAL])
            py_tags = set()
            for bit, tag in TAG_MAP.items():
                if tags & bit:
                    py_tags.add(tag)
            u = Unit(
                uid=uid, card_id="generic", owner_id=0,
                base_hp=hp, base_atk=atk,
                max_hp=hp, max_atk=atk,
                cur_hp=hp, cur_atk=atk,
                tier=tier, types=py_types, tags=py_tags,
                is_golden=golden,
            )
        uid += 1
        units.append(u)
    return units, uid


NUM_COMBATS = 5000
MATCHUPS = [
    (BOARD_A, BOARD_B),
    (BOARD_B, BOARD_C),
    (BOARD_C, BOARD_A),
]


# ============================================================
# C++ benchmark
# ============================================================
def bench_cpp():
    start = time.perf_counter()
    total = 0
    for b0, b1 in MATCHUPS:
        cpp_engine.fast_combat_batch(b0, b1, base_seed=0, count=NUM_COMBATS,
                                     tavern_tier_0=3, tavern_tier_1=3)
        total += NUM_COMBATS
    elapsed = time.perf_counter() - start
    return total, elapsed


# ============================================================
# Python benchmark
# ============================================================
def bench_python():
    total = 0
    start = time.perf_counter()
    for mi, (b0_spec, b1_spec) in enumerate(MATCHUPS):
        for i in range(NUM_COMBATS):
            random.seed(i + mi * NUM_COMBATS)
            board0, uid = cpp_board_to_python(b0_spec, 10000)
            board1, _ = cpp_board_to_python(b1_spec, uid)
            p0 = Player(uid=0, board=board0, hand=[], health=40)
            p0.tavern_tier = 3
            p1 = Player(uid=1, board=board1, hand=[], health=40)
            p1.tavern_tier = 3
            cm = CombatManager()
            cm.resolve_combat(p0, p1)
            total += 1
    elapsed = time.perf_counter() - start
    return total, elapsed


# ============================================================
# Run
# ============================================================
if __name__ == "__main__":
    n = NUM_COMBATS * len(MATCHUPS)
    print(f"=== Performance Benchmark ({n} total combats, {len(MATCHUPS)} matchups) ===\n")

    print(f"Running C++ engine...")
    cpp_total, cpp_time = bench_cpp()
    cpp_rate = cpp_total / cpp_time
    print(f"  C++:    {cpp_total:>6} combats in {cpp_time:.3f}s = {cpp_rate:>8,.0f} combats/sec\n")

    print(f"Running Python engine...")
    py_total, py_time = bench_python()
    py_rate = py_total / py_time
    print(f"  Python: {py_total:>6} combats in {py_time:.3f}s = {py_rate:>8,.0f} combats/sec\n")

    speedup = cpp_rate / py_rate
    print(f"=== SPEEDUP: C++ is {speedup:.1f}x faster than Python ===")
