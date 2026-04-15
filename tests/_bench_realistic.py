"""
Realistic late-game combat benchmark.

Сценарии смоделированы под реальные lvl 5-7 доски Battlegrounds:
  - Полные 7×7 борды
  - Золотые юниты (2× stacks на триггерах)
  - Реальные synergy-цепочки (deathrattle chains, rally, on-death buffs)
  - Poison / divine shield / cleave / windfury / reborn
  - Долгие бои (30-50+ атак с суммонами токенов)
"""
import os
import sys
import time

sys.path.insert(0, "cpp/build")
os.add_dll_directory(r"C:\msys64\mingw64\bin")
import hs_engine_cpp as cpp_engine

cpp_engine.register_all_effects()

# ============================================================
# Type / Tag bit constants (mirror cpp/include/types.h)
# ============================================================
BEAST = 1 << 0
DRAGON = 1 << 1
DEMON = 1 << 2
MURLOC = 1 << 3
PIRATE = 1 << 4
ELEMENTAL = 1 << 5
MECH = 1 << 6
UNDEAD = 1 << 7
NAGA = 1 << 8
QUILBOAR = 1 << 9

IMMEDIATE_ATTACK = 1 << 0
TAUNT = 1 << 1
DIVINE_SHIELD = 1 << 2
WINDFURY = 1 << 3
POISONOUS = 1 << 4
REBORN = 1 << 5
VENOMOUS = 1 << 6
CLEAVE = 1 << 7

# Cards with known combat triggers (from generated_card_ids.h + register_all_effects)
CORD_PULLER = 103          # DR: summon microbot
HARMLESS_BONEHEAD = 107    # DR: summon 2 skeletons
MANASABER = 108            # DR: summon 2 cublings with taunt
MISFIT_DRAGONLING = 110    # SoC: +tier/+tier
ROT_HIDE_GNOLL = 116       # on friendly death: +1/0
SWAMPSTRIKER = 118         # on murloc play: +1/0
TUSKED_CAMPER = 119        # rally: +1/1
TWILIGHT_HATCHLING = 120   # DR: summon whelp w/ immediate attack
WRATH_WEAVER = 121         # on demon play: +2/1
SEWER_RAT = 203            # DR: summon turtle taunt
MECHAGNOME_INTERPRETER = 205   # on mech play: +2/1
CADAVER_CARETAKER = 305    # DR: summon 3 skeletons
HANDLESS_FORSAKEN = 307    # DR: summon reborn hand
PEGGY_STURDYBONE = 326     # on pirate play: +2/1
DEVOUT_HELLCALLER = 404    # on friendly death: +1/2
HEROIC_UNDERDOG = 409      # rally: +1/0
SLY_RAPTOR = 420           # DR: summon skeleton
MONSTROUS_MACAW = 431      # rally: +1/1
ICHORON_THE_PROTECTOR = 438    # on elemental play: +0/1
NOMI_KITCHEN_NIGHTMARE = 512   # on elemental play: +2/2
RAZORFEN_VINEWEAVER = 514  # rally: +1/1
SANGUINE_REFINER = 613     # rally: +1/1
PRIMITIVE_PAINTER = 620    # on murloc play: +1/2
THE_LAST_ONE_STANDING = 703    # rally: +12/12


def cu(card_id, atk, hp, types=0, tags=0, tier=1, golden=False):
    return (card_id, atk, hp, types, tags, tier, golden)


# ============================================================
# Matchup 1: Deathrattle Undead chain vs Demons w/ heal deaths
# Реалистичный tier 5-6 board с длинной цепочкой смертей
# ============================================================
UNDEAD_DEATHRATTLE = [
    cu(HARMLESS_BONEHEAD, 20, 20, UNDEAD, 0, 1, golden=True),        # 2x skeletons on death
    cu(CADAVER_CARETAKER, 30, 30, UNDEAD, 0, 3),                     # 3x skeletons on death
    cu(HANDLESS_FORSAKEN, 25, 20, UNDEAD, TAUNT, 1),                 # reborn hand
    cu(SLY_RAPTOR, 18, 18, UNDEAD, DIVINE_SHIELD, 1),                # skeleton on death
    cu(ROT_HIDE_GNOLL, 15, 30, UNDEAD, TAUNT | REBORN, 1),           # +1/0 per friendly death
    cu(DEVOUT_HELLCALLER, 22, 25, DEMON, 0, 4),                      # +1/+2 per friendly death
    cu(HARMLESS_BONEHEAD, 20, 20, UNDEAD, REBORN, 1),                # more skeletons
]

DEMONS_BIG = [
    cu(0, 40, 40, DEMON, DIVINE_SHIELD | TAUNT, 4),
    cu(0, 35, 35, DEMON, POISONOUS, 4),
    cu(0, 28, 50, DEMON, TAUNT | REBORN, 5),
    cu(0, 50, 20, DEMON, WINDFURY, 5),
    cu(0, 30, 30, DEMON, DIVINE_SHIELD, 4),
    cu(0, 25, 40, DEMON, CLEAVE, 4),
    cu(0, 45, 45, DEMON, 0, 6),
]

# ============================================================
# Matchup 2: Poison murlocs vs big beast wall
# Классическая ситуация "яд против статов"
# ============================================================
POISON_MURLOCS = [
    cu(0, 8, 12, MURLOC, POISONOUS | DIVINE_SHIELD, 3),
    cu(0, 6, 10, MURLOC, POISONOUS, 2),
    cu(0, 10, 8, MURLOC, POISONOUS | WINDFURY, 4),
    cu(0, 5, 15, MURLOC, POISONOUS | TAUNT, 3),
    cu(SWAMPSTRIKER, 12, 15, MURLOC, 0, 1, golden=True),
    cu(PRIMITIVE_PAINTER, 20, 30, MURLOC, POISONOUS, 6),
    cu(0, 7, 20, MURLOC, POISONOUS | DIVINE_SHIELD, 5),
]

BIG_BEASTS = [
    cu(0, 60, 80, BEAST, TAUNT | REBORN, 6),
    cu(0, 50, 60, BEAST, CLEAVE | WINDFURY, 5),
    cu(0, 45, 45, BEAST, DIVINE_SHIELD, 5),
    cu(MANASABER, 70, 40, BEAST, 0, 1, golden=True),     # summon cublings
    cu(0, 80, 30, BEAST, WINDFURY, 6),
    cu(0, 40, 70, BEAST, TAUNT, 5),
    cu(0, 55, 55, BEAST, CLEAVE, 6),
]

# ============================================================
# Matchup 3: Mechs with divine shields vs rally beasts
# Rally-эффекты срабатывают на ATTACK_DECLARED каждого юнита
# ============================================================
DS_MECHS = [
    cu(0, 40, 50, MECH, DIVINE_SHIELD | TAUNT, 5),
    cu(0, 35, 35, MECH, DIVINE_SHIELD | CLEAVE, 5),
    cu(MECHAGNOME_INTERPRETER, 30, 40, MECH, DIVINE_SHIELD, 2, golden=True),
    cu(0, 50, 50, MECH, DIVINE_SHIELD | WINDFURY, 6),
    cu(0, 45, 45, MECH, DIVINE_SHIELD, 5),
    cu(0, 30, 30, MECH, DIVINE_SHIELD | POISONOUS, 4),
    cu(0, 55, 40, MECH, DIVINE_SHIELD | REBORN, 6),
]

RALLY_BEASTS = [
    cu(TUSKED_CAMPER, 20, 25, BEAST, 0, 1, golden=True),
    cu(HEROIC_UNDERDOG, 22, 22, 0, 0, 4, golden=True),
    cu(MONSTROUS_MACAW, 25, 25, BEAST, 0, 4, golden=True),
    cu(SANGUINE_REFINER, 30, 30, 0, 0, 6, golden=True),
    cu(RAZORFEN_VINEWEAVER, 28, 28, QUILBOAR, 0, 5, golden=True),
    cu(THE_LAST_ONE_STANDING, 40, 40, 0, 0, 7),
    cu(0, 35, 35, BEAST, CLEAVE, 5),
]

# ============================================================
# Matchup 4: Dragons vs Reborn wall
# Twilight Hatchling — immediate attack on death, бесконечные re-triggers
# ============================================================
DRAGONS_HATCH = [
    cu(TWILIGHT_HATCHLING, 15, 12, DRAGON, 0, 1, golden=True),
    cu(TWILIGHT_HATCHLING, 15, 12, DRAGON, REBORN, 1),
    cu(MISFIT_DRAGONLING, 25, 30, DRAGON, 0, 1, golden=True),
    cu(0, 40, 35, DRAGON, DIVINE_SHIELD, 5),
    cu(0, 55, 30, DRAGON, WINDFURY | CLEAVE, 6),
    cu(0, 35, 50, DRAGON, TAUNT, 5),
    cu(TWILIGHT_HATCHLING, 15, 12, DRAGON, 0, 1),
]

REBORN_WALL = [
    cu(0, 30, 30, 0, TAUNT | REBORN, 4),
    cu(0, 25, 40, 0, TAUNT | REBORN, 4),
    cu(0, 20, 50, 0, TAUNT | REBORN, 5),
    cu(0, 35, 35, 0, REBORN | DIVINE_SHIELD, 5),
    cu(ROT_HIDE_GNOLL, 25, 35, UNDEAD, REBORN, 1, golden=True),
    cu(HANDLESS_FORSAKEN, 20, 25, UNDEAD, TAUNT | REBORN, 1, golden=True),
    cu(0, 30, 45, 0, TAUNT | REBORN | DIVINE_SHIELD, 6),
]

# ============================================================
# Matchup 5: Pure stats slugfest — long combat with many attacks
# Всё таунты, никто не умирает быстро, 50+ атак
# ============================================================
TANK_WALL_A = [
    cu(0, 30, 80, 0, TAUNT, 4),
    cu(0, 35, 75, 0, TAUNT, 5),
    cu(0, 25, 100, 0, TAUNT, 5),
    cu(0, 40, 70, 0, TAUNT, 5),
    cu(0, 30, 90, 0, TAUNT, 6),
    cu(0, 45, 60, 0, TAUNT, 6),
    cu(0, 50, 80, 0, TAUNT, 6),
]

TANK_WALL_B = [
    cu(0, 35, 70, 0, TAUNT, 5),
    cu(0, 30, 85, 0, TAUNT, 5),
    cu(0, 40, 75, 0, TAUNT, 6),
    cu(0, 25, 95, 0, TAUNT, 5),
    cu(0, 45, 65, 0, TAUNT, 6),
    cu(0, 30, 90, 0, TAUNT, 6),
    cu(0, 38, 78, 0, TAUNT, 6),
]

MATCHUPS = [
    ("Undead DR chain vs Big Demons",     UNDEAD_DEATHRATTLE, DEMONS_BIG),
    ("Poison Murlocs vs Big Beasts",      POISON_MURLOCS,     BIG_BEASTS),
    ("DS Mechs vs Rally Beasts",          DS_MECHS,           RALLY_BEASTS),
    ("Dragons Hatchlings vs Reborn Wall", DRAGONS_HATCH,      REBORN_WALL),
    ("Tank Wall Slugfest",                TANK_WALL_A,        TANK_WALL_B),
]

NUM_COMBATS = int(os.environ.get("N", "20000"))


def run_bench():
    # Warmup
    for _, b0, b1 in MATCHUPS:
        cpp_engine.fast_combat_batch(b0, b1, base_seed=0, count=100,
                                     tavern_tier_0=6, tavern_tier_1=6)

    if hasattr(cpp_engine, "prof_reset"):
        cpp_engine.prof_reset()

    print(f"{'matchup':<40} {'combats/sec':>12} {'us/combat':>10}")
    print("-" * 66)

    total_combats = 0
    total_time = 0.0
    per_matchup = []

    for name, b0, b1 in MATCHUPS:
        start = time.perf_counter()
        cpp_engine.fast_combat_batch(b0, b1, base_seed=0, count=NUM_COMBATS,
                                     tavern_tier_0=6, tavern_tier_1=6)
        elapsed = time.perf_counter() - start
        rate = NUM_COMBATS / elapsed
        us_per = 1e6 * elapsed / NUM_COMBATS
        print(f"{name:<40} {rate:>12,.0f} {us_per:>9.2f}")
        per_matchup.append((name, rate, us_per))
        total_combats += NUM_COMBATS
        total_time += elapsed

    print("-" * 66)
    total_rate = total_combats / total_time
    total_us = 1e6 * total_time / total_combats
    print(f"{'TOTAL':<40} {total_rate:>12,.0f} {total_us:>9.2f}")

    if hasattr(cpp_engine, "prof_dump"):
        rows = cpp_engine.prof_dump()
        base_cycles = rows[0][1] or 1
        print(f"\n{'section':<20} {'%':>7} {'Mcyc':>10} {'calls':>10} {'cyc/call':>10}")
        print("-" * 62)
        for name, cycles, calls in rows:
            pct = 100.0 * cycles / base_cycles
            mcyc = cycles / 1e6
            cpc = (cycles / calls) if calls else 0
            print(f"{name:<20} {pct:>6.1f}% {mcyc:>10.1f} {calls:>10} {cpc:>10.0f}")


if __name__ == "__main__":
    run_bench()
