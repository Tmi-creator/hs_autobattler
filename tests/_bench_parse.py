"""
Parse/marshal bench — сравниваем:
  (A) fast_combat_batch(board, board, count=N) — parse один раз, N комбатов
  (B) N × fast_combat(board, board, seed) — parse на каждом вызове

Разность скоростей = стоимость parse + tuple→C++ для одного комбата.
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, "cpp/build")
os.add_dll_directory(r"C:\msys64\mingw64\bin")
import hs_engine_cpp as cpp_engine

cpp_engine.register_all_effects()

BEAST = 1 << 0
DEMON = 1 << 2
PIRATE = 1 << 4
MECH = 1 << 6
UNDEAD = 1 << 7

TAUNT = 1 << 1
DIVINE_SHIELD = 1 << 2

def cu(card_id, atk, hp, types=0, tags=0, tier=1, golden=False):
    return (card_id, atk, hp, types, tags, tier, golden)

# Realistic tier 5-6 board
BOARD_A = [
    cu(107, 20, 20, UNDEAD, 0, 1, golden=True),
    cu(305, 30, 30, UNDEAD, 0, 3),
    cu(307, 25, 20, UNDEAD, TAUNT, 1),
    cu(420, 18, 18, UNDEAD, DIVINE_SHIELD, 1),
    cu(116, 15, 30, UNDEAD, TAUNT, 1),
    cu(404, 22, 25, DEMON, 0, 4),
    cu(107, 20, 20, UNDEAD, 0, 1),
]

BOARD_B = [
    cu(0, 40, 40, DEMON, DIVINE_SHIELD | TAUNT, 4),
    cu(0, 35, 35, DEMON, 0, 4),
    cu(0, 28, 50, DEMON, TAUNT, 5),
    cu(0, 50, 20, DEMON, 0, 5),
    cu(0, 30, 30, DEMON, DIVINE_SHIELD, 4),
    cu(0, 25, 40, DEMON, 0, 4),
    cu(0, 45, 45, DEMON, 0, 6),
]

N = 20000

# Numpy form: shape (N_units, 7) int32 — layout matches UnitNpCol in pybind_module.cpp
def to_np(board):
    return np.array(board, dtype=np.int32)

BOARD_A_NP = to_np(BOARD_A)
BOARD_B_NP = to_np(BOARD_B)

# Warmup
for _ in range(100):
    cpp_engine.fast_combat(BOARD_A, BOARD_B, seed=0, tavern_tier_0=6, tavern_tier_1=6)
    cpp_engine.fast_combat_np(BOARD_A_NP, BOARD_B_NP, seed=0, tavern_tier_0=6, tavern_tier_1=6)
cpp_engine.fast_combat_batch(BOARD_A, BOARD_B, base_seed=0, count=1000, tavern_tier_0=6, tavern_tier_1=6)

# (A) batch: parse once
start = time.perf_counter()
cpp_engine.fast_combat_batch(BOARD_A, BOARD_B, base_seed=0, count=N,
                             tavern_tier_0=6, tavern_tier_1=6)
batch_time = time.perf_counter() - start
batch_rate = N / batch_time
batch_us = 1e6 * batch_time / N

# (B) single-shot with list-of-tuples: parse every call (old path)
start = time.perf_counter()
for i in range(N):
    cpp_engine.fast_combat(BOARD_A, BOARD_B, seed=i, tavern_tier_0=6, tavern_tier_1=6)
single_time = time.perf_counter() - start
single_rate = N / single_time
single_us = 1e6 * single_time / N

# (C) single-shot with numpy: new fast path
start = time.perf_counter()
for i in range(N):
    cpp_engine.fast_combat_np(BOARD_A_NP, BOARD_B_NP, seed=i, tavern_tier_0=6, tavern_tier_1=6)
np_time = time.perf_counter() - start
np_rate = N / np_time
np_us = 1e6 * np_time / N

print(f"{'mode':<35} {'combats/sec':>12} {'us/combat':>10}")
print("-" * 59)
print(f"{'(A) batch (parse once)':<35} {batch_rate:>12,.0f} {batch_us:>9.2f}")
print(f"{'(B) single, list-of-tuples':<35} {single_rate:>12,.0f} {single_us:>9.2f}")
print(f"{'(C) single, numpy int32':<35} {np_rate:>12,.0f} {np_us:>9.2f}")
print()
list_parse = single_us - batch_us
np_parse = np_us - batch_us
print(f"Parse overhead (list):  {list_parse:.2f} us per call ({list_parse / single_us * 100:.0f}% of total)")
print(f"Parse overhead (numpy): {np_parse:.2f} us per call ({np_parse / np_us * 100:.0f}% of total)")
print(f"Speedup numpy vs list:  {single_time / np_time:.2f}x")
