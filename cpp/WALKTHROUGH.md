# Walkthrough: Hearthstone Battlegrounds C++ Engine Migration

## Goal

Port the combat simulation engine from Python to C++ via pybind11 for 20x+ throughput improvement in RL training rollouts.

## What Was Built

### Core C++ Engine (`cpp/`)

#### Data Layer (`include/`)

- [types.h](file:///c:/Users/Timur/PycharmProjects/hs_autobattler/cpp/include/types.h) — POD structs: `Unit`, `CombatBoard`, `CombatState`, `Event`, `BattleResult`. All fixed-size, zero-heap, trivially copyable. `CombatState` = 3416 bytes.
- [entities.h](file:///c:/Users/Timur/PycharmProjects/hs_autobattler/cpp/include/entities.h) — `CardID`, `UnitTypes`, `Tags` constants. Bitset-based type/tag system.
- [rng.h](file:///c:/Users/Timur/PycharmProjects/hs_autobattler/cpp/include/rng.h) — SplitMix64 RNG (deterministic, fast, no heap).

#### Logic Layer (`src/`)

- [event_system.cpp](file:///c:/Users/Timur/PycharmProjects/hs_autobattler/cpp/src/event_system.cpp) — Event processing pipeline: `process_event()` → `collect_triggers()` → `order_triggers()` → execute. Shared helper `collect_unit_triggers()` eliminates duplication between board-scan and death-trigger collection.
- [effects.cpp](file:///c:/Users/Timur/PycharmProjects/hs_autobattler/cpp/src/effects.cpp) — 10 card effects ported (Alleycat, Scallywag, Annoy-o-Tron, Imprisoner, Dire Wolf Alpha, Spawn of N'Zoth, Kaboom Bot, Deflect-o-Bot, etc.). Helper functions: `buff_perm()`, `buff_combat()`, `deal_damage_to_unit()`, `summon_unit()`. 5 generic conditions.
- [auras.cpp](file:///c:/Users/Timur/PycharmProjects/hs_autobattler/cpp/src/auras.cpp) — **Data-driven** aura system. Each aura = one struct in a table (`AuraDef`), one generic `apply_aura()` function. Adding a new aura = one line. Supports `NEIGHBOURS` and `TYPE_OTHERS` modes.
- [combat.cpp](file:///c:/Users/Timur/PycharmProjects/hs_autobattler/cpp/src/combat.cpp) — Full combat resolution loop: attack selection, cleave, divine shield, poison/venom, windfury, immediate attack (Scallywag tokens), death cleanup with trigger pre-collection, reborn.

#### Python Bindings (`bindings/`)

- [pybind_module.cpp](file:///c:/Users/Timur/PycharmProjects/hs_autobattler/cpp/bindings/pybind_module.cpp) — Exposes `fast_combat()` (single) and `fast_combat_batch()` (N combats, boards parsed once, GIL released during loop, memcpy template per combat).

---

### Tests (`tests/`)

- [test_cpp_smoke.py](file:///c:/Users/Timur/PycharmProjects/hs_autobattler/tests/test_cpp_smoke.py) — 9 tests: basic 1v1, draw, taunt targeting (statistical, 100 seeds), divine shield, Scallywag DR (two scenarios), batch determinism, performance.
- [bench_cpp_vs_python.py](file:///c:/Users/Timur/PycharmProjects/hs_autobattler/tests/bench_cpp_vs_python.py) — Real-card benchmark (3 matchups × 5K combats with DR, auras, DS, windfury).

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| POD-only structs, no `std::vector` | Cache-friendly, trivially copyable, `memcpy`-able for batch |
| Function pointers, no virtual dispatch | Zero overhead per trigger call |
| Data-driven auras | New aura = 1 line in table, no new functions |
| Generic conditions + type check in effect body | Scales to 100+ cards without explosion of condition functions |
| `collect_unit_triggers()` shared helper | Eliminates copy-paste between `collect_triggers` and death trigger pre-collection |
| SplitMix64 RNG | Fast, deterministic, no heap. Not compatible with Python's `random` module (different sequences from same seed) |
| GIL release in `fast_combat_batch` | Pure C++ loop, no Python object access during combat |

## Performance Results

```
C++:     15,000 combats in 2.0s  =  7,412 combats/sec
Python:  15,000 combats in 41.9s =    358 combats/sec

SPEEDUP: 20.7x
```

Boards: 7v7 with Scallywag, Imprisoner, Kaboom Bot, Spawn of N'Zoth, Dire Wolf Alpha, Murloc Warleader, Southsea Captain, Annoy-o-Tron, divine shields, windfury.

## Remaining Work

- **Add cards** — As new cards are added to the Python engine, port their effects to `effects.cpp` and auras to `g_aura_table`
- **RL integration** — Replace `CombatManager.resolve_combat()` with `fast_combat()` in training pipeline
- **Statistical parity check** — Run 100K+ combats on both engines, compare win rate distributions (not 1:1 due to different RNG)
