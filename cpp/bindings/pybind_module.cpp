// pybind_module.cpp — pybind11 entry point
// Exposes resolve_combat, register_all_effects, fast_combat, fast_combat_batch

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "types.h"
#include "rng.h"
#include "entities.h"
#include "event_system.h"
#include <cstring>
#include <vector>

namespace py = pybind11;

// ============================================================
// Helper: parse Python list of tuples into a CombatBoard (once)
// ============================================================
static void parse_board(CombatBoard& board, py::list& units, int32_t& next_uid) {
    board.count = 0;
    for (size_t i = 0; i < units.size() && i < GameConst::MAX_BOARD; ++i) {
        py::tuple t = units[i].cast<py::tuple>();
        Unit& u = board.units[board.count++];
        u = Unit{};  // zero-init
        u.card_id   = t[0].cast<int16_t>();
        u.atk_base  = t[1].cast<int16_t>();
        u.hp_base   = t[2].cast<int16_t>();
        u.types     = t[3].cast<TypeBitset>();
        u.tags      = t[4].cast<TagBitset>();
        u.tier      = t[5].cast<int8_t>();
        u.is_golden = t[6].cast<bool>();
        u.uid       = next_uid++;
    }
}

// ============================================================
// fast_combat — single combat, returns (outcome, damage)
// ============================================================
static std::pair<int, int> fast_combat(
    py::list side0, py::list side1,
    uint64_t seed,
    int8_t tavern_tier_0 = 1, int8_t tavern_tier_1 = 1
) {
    CombatState state{};
    rng_seed(state.rng, seed);
    parse_board(state.boards[0], side0, state.next_uid);
    parse_board(state.boards[1], side1, state.next_uid);
    state.boards[0].tavern_tier = tavern_tier_0;
    state.boards[1].tavern_tier = tavern_tier_1;

    BattleResult result = resolve_combat(state);
    return {static_cast<int>(result.outcome), static_cast<int>(result.damage)};
}

// ============================================================
// fast_combat_batch — run N combats, parse boards ONCE
// Uses memcpy template + releases GIL during combat loop
// Returns flat list of (outcome, damage) pairs
// ============================================================
static py::list fast_combat_batch(
    py::list side0, py::list side1,
    uint64_t base_seed, int count,
    int8_t tavern_tier_0 = 1, int8_t tavern_tier_1 = 1
) {
    // 1. Parse boards ONCE (with GIL held)
    CombatState template_state{};
    parse_board(template_state.boards[0], side0, template_state.next_uid);
    parse_board(template_state.boards[1], side1, template_state.next_uid);
    template_state.boards[0].tavern_tier = tavern_tier_0;
    template_state.boards[1].tavern_tier = tavern_tier_1;

    // 2. Pre-allocate results
    struct Result { int outcome; int damage; };
    std::vector<Result> results(count);

    // 3. Release GIL, run pure C++ loop
    {
        py::gil_scoped_release release;
        for (int i = 0; i < count; ++i) {
            // Fast copy via memcpy (CombatState is POD)
            CombatState state;
            std::memcpy(&state, &template_state, sizeof(CombatState));

            // Re-seed RNG per combat
            rng_seed(state.rng, base_seed + i);

            // Reset UIDs (each combat gets fresh UIDs for spawned tokens)
            state.next_uid = template_state.next_uid;

            // Reset attacker indices
            state.attacker_idx[0] = 0;
            state.attacker_idx[1] = 0;

            BattleResult r = resolve_combat(state);
            results[i] = {static_cast<int>(r.outcome), static_cast<int>(r.damage)};
        }
    }

    // 4. Build Python results (GIL re-acquired)
    py::list py_results;
    for (int i = 0; i < count; ++i) {
        py_results.append(py::make_tuple(results[i].outcome, results[i].damage));
    }
    return py_results;
}

PYBIND11_MODULE(hs_engine_cpp, m) {
    m.doc() = "Hearthstone Battlegrounds C++ engine core";

    m.def("get_state_size", []() { return static_cast<int>(sizeof(CombatState)); },
          "Returns sizeof(CombatState) in bytes");

    m.def("register_all_effects", &register_all_effects,
          "Register all card effects (call once at startup)");

    m.def("fast_combat", &fast_combat,
          "Run one combat. Each unit = (card_id, atk, hp, types, tags, tier, is_golden)",
          py::arg("side0"), py::arg("side1"), py::arg("seed"),
          py::arg("tavern_tier_0") = 1, py::arg("tavern_tier_1") = 1);

    m.def("fast_combat_batch", &fast_combat_batch,
          "Run N combats with seeds [base_seed, base_seed+N). Boards parsed once.",
          py::arg("side0"), py::arg("side1"),
          py::arg("base_seed"), py::arg("count"),
          py::arg("tavern_tier_0") = 1, py::arg("tavern_tier_1") = 1);
}
