// pybind_module.cpp — pybind11 entry point
// Exposes resolve_combat, register_all_effects, fast_combat, fast_combat_batch

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "types.h"
#include "rng.h"
#include "entities.h"
#include "event_system.h"
#include "profiler.h"
#include <cstring>
#include <vector>

namespace py = pybind11;

// ============================================================
// Layout для numpy-based parse (fast_combat_np / fast_combat_batch_np).
// Python передаёт array shape (N_units, UNIT_NP_COLS) dtype=int32.
// Порядок колонок фиксирован и обязан совпадать со стороны Python.
// ============================================================
static constexpr int UNIT_NP_COLS = 7;
enum UnitNpCol {
    UNP_CARD_ID  = 0,
    UNP_ATK      = 1,
    UNP_HP       = 2,
    UNP_TYPES    = 3,
    UNP_TAGS     = 4,
    UNP_TIER     = 5,
    UNP_GOLDEN   = 6,
};

// Парсит numpy доску (N, 7) int32 в CombatBoard напрямую через raw pointer.
// Не делает type-checked cast'ов — один bulk read + field assignment.
// Стоит ~10× меньше чем parse_board с .cast<>() на каждое поле.
static void parse_board_np(CombatBoard& board, const int32_t* data, int n_units, int32_t& next_uid) {
    board.count = 0;
    const int n = (n_units > GameConst::MAX_BOARD) ? GameConst::MAX_BOARD : n_units;
    for (int i = 0; i < n; ++i) {
        const int32_t* row = data + i * UNIT_NP_COLS;
        Unit& u = board.units[board.count++];
        u = Unit{};
        u.card_id   = static_cast<int16_t>(row[UNP_CARD_ID]);
        u.atk_base  = static_cast<int16_t>(row[UNP_ATK]);
        u.hp_base   = static_cast<int16_t>(row[UNP_HP]);
        u.types     = static_cast<TypeBitset>(row[UNP_TYPES]);
        u.tags      = static_cast<TagBitset>(row[UNP_TAGS]);
        u.tier      = static_cast<int8_t>(row[UNP_TIER]);
        u.is_golden = row[UNP_GOLDEN] != 0;
        u.uid       = next_uid++;
    }
}

// Validates a numpy board array and returns raw pointer. Throws on invalid shape.
// Использует прямые accessor'ы py::array_t (.ndim/.shape/.data) вместо .request() —
// .request() аллоцирует buffer_info struct, а для hot path нам нужны только
// 3 числа и указатель.
static const int32_t* board_np_ptr(const py::array_t<int32_t>& arr, int& n_units) {
    if (arr.ndim() != 2 || arr.shape(1) != UNIT_NP_COLS) {
        throw py::value_error("board must be shape (N, 7) int32");
    }
    n_units = static_cast<int>(arr.shape(0));
    return arr.data();
}

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
// fast_combat_np — single combat, numpy inputs.
// Boards передаются как py::array_t<int32_t> shape (N, 7). На hot path
// вместо 98 type-checked .cast<>() мы делаем один memcpy-like pointer access.
// На realistic workload это срезает parse-overhead с ~8us до ~1us per call.
// ============================================================
static std::pair<int, int> fast_combat_np(
    py::array_t<int32_t> side0, py::array_t<int32_t> side1,
    uint64_t seed,
    int8_t tavern_tier_0 = 1, int8_t tavern_tier_1 = 1
) {
    int n0 = 0, n1 = 0;
    const int32_t* data0 = board_np_ptr(side0, n0);
    const int32_t* data1 = board_np_ptr(side1, n1);

    // Намеренно НЕ используем `CombatState state{}` — это ~3.3KB zero-init на
    // каждый вызов (~500ns). Вместо этого инициализируем явно только критичные
    // поля, а slot-маски (subscribers, taunt, dead, aura) обнуляются в
    // recalculate_subscribers() в прологе resolve_combat.
    CombatState state;
    state.next_uid = GameConst::INITIAL_UID;
    state.attacker_idx[0] = 0;
    state.attacker_idx[1] = 0;
    state.has_pending_deaths = false;
    rng_seed(state.rng, seed);

    state.boards[0].count = 0;
    state.boards[0].tavern_tier = tavern_tier_0;
    state.boards[1].count = 0;
    state.boards[1].tavern_tier = tavern_tier_1;

    parse_board_np(state.boards[0], data0, n0, state.next_uid);
    parse_board_np(state.boards[1], data1, n1, state.next_uid);

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

// ============================================================
// fast_combat_batch_np — run N combats, numpy inputs, parse once
// Releases GIL during combat loop and uses raw pointer parsing.
// Returns flat list of (outcome, damage) pairs
// ============================================================
static py::list fast_combat_batch_np(
    py::array_t<int32_t> side0, py::array_t<int32_t> side1,
    uint64_t base_seed, int count,
    int8_t tavern_tier_0 = 1, int8_t tavern_tier_1 = 1
) {
    int n0 = 0, n1 = 0;
    const int32_t* data0 = board_np_ptr(side0, n0);
    const int32_t* data1 = board_np_ptr(side1, n1);

    // 1. Parse boards ONCE (with GIL held)
    CombatState template_state;
    template_state.next_uid = GameConst::INITIAL_UID;
    template_state.attacker_idx[0] = 0;
    template_state.attacker_idx[1] = 0;
    template_state.has_pending_deaths = false;

    template_state.boards[0].count = 0;
    template_state.boards[0].tavern_tier = tavern_tier_0;
    template_state.boards[1].count = 0;
    template_state.boards[1].tavern_tier = tavern_tier_1;

    parse_board_np(template_state.boards[0], data0, n0, template_state.next_uid);
    parse_board_np(template_state.boards[1], data1, n1, template_state.next_uid);

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

// ============================================================
// fast_combat_batch_flat — run N combats, flat std::vector<int32_t> inputs, parse once
// Releases GIL during combat loop and uses flat indexing.
// Returns flat list of (outcome, damage) pairs
// ============================================================
static void parse_board_flat(CombatBoard& board, const std::vector<int32_t>& flat_data, int32_t& next_uid) {
    board.count = 0;
    int n_units = static_cast<int>(flat_data.size()) / UNIT_NP_COLS;
    int n = (n_units > GameConst::MAX_BOARD) ? GameConst::MAX_BOARD : n_units;
    for (int i = 0; i < n; ++i) {
        int idx = i * UNIT_NP_COLS;
        Unit& u = board.units[board.count++];
        u = Unit{};
        u.card_id   = static_cast<int16_t>(flat_data[idx + UNP_CARD_ID]);
        u.atk_base  = static_cast<int16_t>(flat_data[idx + UNP_ATK]);
        u.hp_base   = static_cast<int16_t>(flat_data[idx + UNP_HP]);
        u.types     = static_cast<TypeBitset>(flat_data[idx + UNP_TYPES]);
        u.tags      = static_cast<TagBitset>(flat_data[idx + UNP_TAGS]);
        u.tier      = static_cast<int8_t>(flat_data[idx + UNP_TIER]);
        u.is_golden = flat_data[idx + UNP_GOLDEN] != 0;
        u.uid       = next_uid++;
    }
}

static py::list fast_combat_batch_flat(
    const std::vector<int32_t>& side0, const std::vector<int32_t>& side1,
    uint64_t base_seed, int count,
    int8_t tavern_tier_0 = 1, int8_t tavern_tier_1 = 1
) {
    // 1. Parse boards ONCE (with GIL held)
    CombatState template_state;
    template_state.next_uid = GameConst::INITIAL_UID;
    template_state.attacker_idx[0] = 0;
    template_state.attacker_idx[1] = 0;
    template_state.has_pending_deaths = false;

    template_state.boards[0].count = 0;
    template_state.boards[0].tavern_tier = tavern_tier_0;
    template_state.boards[1].count = 0;
    template_state.boards[1].tavern_tier = tavern_tier_1;

    parse_board_flat(template_state.boards[0], side0, template_state.next_uid);
    parse_board_flat(template_state.boards[1], side1, template_state.next_uid);

    // 2. Pre-allocate results
    struct Result { int outcome; int damage; };
    std::vector<Result> results(count);

    // 3. Release GIL, run pure C++ loop
    {
        py::gil_scoped_release release;
        for (int i = 0; i < count; ++i) {
            CombatState state;
            std::memcpy(&state, &template_state, sizeof(CombatState));

            rng_seed(state.rng, base_seed + i);
            state.next_uid = template_state.next_uid;
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

    m.def("fast_combat_np", &fast_combat_np,
          "Run one combat. Boards are numpy int32 arrays of shape (N, 7): "
          "[card_id, atk, hp, types, tags, tier, is_golden]. "
          "~8x faster parse than fast_combat(list-of-tuples) — use this for MCTS/RL.",
          py::arg("side0"), py::arg("side1"), py::arg("seed"),
          py::arg("tavern_tier_0") = 1, py::arg("tavern_tier_1") = 1);

    m.def("fast_combat_batch", &fast_combat_batch,
          "Run N combats with seeds [base_seed, base_seed+N). Boards parsed once.",
          py::arg("side0"), py::arg("side1"),
          py::arg("base_seed"), py::arg("count"),
          py::arg("tavern_tier_0") = 1, py::arg("tavern_tier_1") = 1);

    m.def("fast_combat_batch_np", &fast_combat_batch_np,
          "Run N combats with seeds [base_seed, base_seed+N). Boards parsed once via numpy.",
          py::arg("side0"), py::arg("side1"),
          py::arg("base_seed"), py::arg("count"),
          py::arg("tavern_tier_0") = 1, py::arg("tavern_tier_1") = 1);

    m.def("fast_combat_batch_flat", &fast_combat_batch_flat,
          "Run N combats with seeds [base_seed, base_seed+N). Boards parsed once via flat vector.",
          py::arg("side0"), py::arg("side1"),
          py::arg("base_seed"), py::arg("count"),
          py::arg("tavern_tier_0") = 1, py::arg("tavern_tier_1") = 1);



    // ==========================================================
    // Debug helpers — inspect subscribers/taunt_mask state.
    // Used by unit tests to verify shift-correctness of insert_at/remove_at.
    // ==========================================================

    // Build a board by parse_board path (direct write + recalculate_subscribers)
    // and return (subscribers: [int×EVENT_TYPE_COUNT], taunt_mask, damage).
    m.def("debug_board_via_parse", [](py::list units) {
        CombatBoard board;
        int32_t next_uid = 10000;
        parse_board(board, units, next_uid);
        recalculate_subscribers(board);

        py::list subs;
        for (int e = 0; e < static_cast<int>(EventType::EVENT_TYPE_COUNT); ++e) {
            subs.append(static_cast<int>(board.subscribers[e]));
        }
        return py::make_tuple(subs, static_cast<int>(board.taunt_mask),
                              static_cast<int>(board.damage));
    }, "Build board via parse_board + recalculate_subscribers, return state");

    // Build the same board by repeated insert_at() at various positions,
    // exercising the shift logic. Return same tuple as debug_board_via_parse.
    // Each unit is inserted at position 0 first, then subsequent inserts go
    // at the end (count) — this exercises both the shift-up (insert at 0 when
    // board is non-empty) and append (insert at end) paths.
    m.def("debug_board_via_inserts", [](py::list units) {
        CombatBoard board;
        int32_t next_uid = 10000;

        // Build in reverse: insert each unit at position 0 so it pushes existing ones right.
        // The resulting order matches parse_board order.
        for (int i = static_cast<int>(units.size()) - 1; i >= 0; --i) {
            py::tuple t = units[static_cast<size_t>(i)].cast<py::tuple>();
            Unit u{};
            u.card_id   = t[0].cast<int16_t>();
            u.atk_base  = t[1].cast<int16_t>();
            u.hp_base   = t[2].cast<int16_t>();
            u.types     = t[3].cast<TypeBitset>();
            u.tags      = t[4].cast<TagBitset>();
            u.tier      = t[5].cast<int8_t>();
            u.is_golden = t[6].cast<bool>();
            u.uid       = next_uid++;
            board.insert_at(0, u);
        }

        py::list subs;
        for (int e = 0; e < static_cast<int>(EventType::EVENT_TYPE_COUNT); ++e) {
            subs.append(static_cast<int>(board.subscribers[e]));
        }
        return py::make_tuple(subs, static_cast<int>(board.taunt_mask),
                              static_cast<int>(board.damage));
    }, "Build board via repeated insert_at(0), return state");

    // Exercise remove_at: build board, remove unit at given slot, return state.
    m.def("debug_board_remove", [](py::list units, int remove_idx) {
        CombatBoard board;
        int32_t next_uid = 10000;
        parse_board(board, units, next_uid);
        recalculate_subscribers(board);
        board.remove_at(remove_idx);

        py::list subs;
        for (int e = 0; e < static_cast<int>(EventType::EVENT_TYPE_COUNT); ++e) {
            subs.append(static_cast<int>(board.subscribers[e]));
        }
        return py::make_tuple(subs, static_cast<int>(board.taunt_mask),
                              static_cast<int>(board.damage));
    }, "Build board, remove_at(idx), return state");

    m.def("prof_reset", []() { g_prof.reset(); },
          "Reset TSC profiler counters");

    m.def("prof_dump", []() {
        py::list out;
        const char* names[] = {
            "RESOLVE_COMBAT", "PROCESS_EVENT", "COLLECT_TRIGGERS", "SORT_TRIGGERS",
            "PERFORM_ATTACK", "CLEANUP_DEAD", "CLEANUP_DEAD_WORK", "FIND_TARGET", "RECALC_AURAS"
        };
        for (int i = 0; i < (int)ProfSection::COUNT; ++i) {
            out.append(py::make_tuple(names[i], g_prof.cycles[i], g_prof.calls[i]));
        }
        return out;
    }, "Dump TSC profiler data: list of (name, total_cycles, call_count)");
}
