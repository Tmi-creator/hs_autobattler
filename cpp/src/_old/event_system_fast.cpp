// event_system.cpp — BFS event processing, trigger collection and sorting
// Mirrors Python: EventManager.process_event(), collect_triggers(), order_triggers()

#include "event_system.h"
#include <algorithm>
#include <tuple>

// ============================================================
// Global effect table — registered once at startup, sorted for binary search
// ============================================================
static EffectTableEntry g_effect_table[GameConst::MAX_EFFECT_ENTRIES];
static int g_num_entries = 0;
static bool g_table_sorted = false;

// ============================================================
// System triggers — global triggers not bound to any unit
// Indexed by EventType for O(1) lookup
// Python equivalent: SYSTEM_TRIGGER_REGISTRY
// ============================================================
struct SystemTriggerList {
    TriggerDef defs[GameConst::MAX_SYSTEM_TRIGGERS];
    int count = 0;
};

static SystemTriggerList g_system_triggers[static_cast<int>(EventType::EVENT_TYPE_COUNT)];

void register_system_trigger(const TriggerDef &def) {
    int idx = static_cast<int>(def.event_type);
    assert(idx >= 0 && idx < static_cast<int>(EventType::EVENT_TYPE_COUNT));
    auto &list = g_system_triggers[idx];
    assert(list.count < GameConst::MAX_SYSTEM_TRIGGERS && "Too many system triggers for this event type!");
    list.defs[list.count++] = def;
}

void register_effect_entry(
    int16_t id,
    const TriggerDef *defs, int count,
    const TriggerDef *golden_defs, int golden_count
) {
    assert(g_num_entries < GameConst::MAX_EFFECT_ENTRIES && "Effect table full!");
    assert(count <= GameConst::MAX_TRIGGERS_PER_CARD && "Too many triggers for one card!");
    assert(golden_count <= GameConst::MAX_TRIGGERS_PER_CARD && "Too many golden triggers!");
    auto &entry = g_effect_table[g_num_entries++];
    entry.id = id;
    entry.num_triggers = count;
    for (int i = 0; i < count; ++i) {
        entry.triggers[i] = defs[i];
    }
    entry.has_golden_override = (golden_defs != nullptr && golden_count > 0);
    entry.num_golden_triggers = golden_count;
    for (int i = 0; i < golden_count; ++i) {
        entry.golden_triggers[i] = golden_defs[i];
    }
    g_table_sorted = false; // invalidate sort
}

// Called after all register_effect_entry() calls.
// Sorts the table by id for O(log n) binary search.
void finalize_effect_table() {
    std::sort(g_effect_table, g_effect_table + g_num_entries,
              [](const EffectTableEntry &a, const EffectTableEntry &b) {
                  return a.id < b.id;
              });
    g_table_sorted = true;
}

// O(log n) binary search on sorted table.
// Must call finalize_effect_table() after all registrations.
const EffectTableEntry *find_effect_entry(int16_t id) {
    assert(g_table_sorted && "Call finalize_effect_table() after registering all effects!");
    // std::lower_bound on sorted array
    int lo = 0, hi = g_num_entries;
    while (lo < hi) {
        int mid = lo + (hi - lo) / 2;
        if (g_effect_table[mid].id < id) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    if (lo < g_num_entries && g_effect_table[lo].id == id) {
        return &g_effect_table[lo];
    }
    return nullptr;
}

// ============================================================
// find_unit_pos — find a unit by UID in CombatState
// Returns false if not found (unit dead or not on board)
// ============================================================
bool find_unit_pos(const CombatState &state, int32_t uid, int8_t &out_side, int8_t &out_slot) {
    for (int s = 0; s < 2; ++s) {
        const auto &board = state.boards[s];
        for (int i = 0; i < board.count; ++i) {
            if (board.units[i].uid == uid) {
                out_side = static_cast<int8_t>(s);
                out_slot = static_cast<int8_t>(i);
                return true;
            }
        }
    }
    out_side = -1;
    out_slot = -1;
    return false;
}

// ============================================================
// collect_unit_triggers — collect triggers for a SINGLE unit
// Handles card_id lookup, golden logic, attached effects.
// Shared by collect_triggers (board scan) and combat's collect_death_triggers.
// ============================================================
int collect_unit_triggers(
    const Unit &unit, EventType event_type,
    int8_t side, int8_t slot,
    TriggerInstance *out_triggers, int max_out
) {
    int count = 0;

    // 1. Card-level triggers
    const EffectTableEntry *entry = find_effect_entry(unit.card_id);
    if (entry) {
        const TriggerDef *active_defs;
        int num_defs;
        int stacks_mult;

        if (unit.is_golden && entry->has_golden_override) {
            active_defs = entry->golden_triggers;
            num_defs = entry->num_golden_triggers;
            stacks_mult = 1;
        } else if (unit.is_golden) {
            active_defs = entry->triggers;
            num_defs = entry->num_triggers;
            stacks_mult = 2;
        } else {
            active_defs = entry->triggers;
            num_defs = entry->num_triggers;
            stacks_mult = 1;
        }

        for (int t = 0; t < num_defs; ++t) {
            if (active_defs[t].event_type == event_type) {
                assert(count < max_out && "Too many triggers for unit!");
                out_triggers[count++] = {
                    &active_defs[t],
                    unit.uid,
                    side, slot,
                    static_cast<int16_t>(stacks_mult)
                };
            }
        }
    }

    // 2. Attached effects (3 scopes)
    auto check_attached = [&](const std::array<AttachedEffect, GameConst::MAX_ATTACHED> &arr, uint8_t num) {
        for (int a = 0; a < num; ++a) {
            if (arr[a].count <= 0) continue;
            const EffectTableEntry *att_entry = find_effect_entry(arr[a].effect_id);
            if (!att_entry) continue;
            for (int t = 0; t < att_entry->num_triggers; ++t) {
                if (att_entry->triggers[t].event_type == event_type) {
                    assert(count < max_out && "Too many triggers for unit!");
                    out_triggers[count++] = {
                        &att_entry->triggers[t],
                        unit.uid,
                        side, slot,
                        arr[a].count
                    };
                }
            }
        }
    };
    check_attached(unit.attached_perm, unit.num_perm);
    check_attached(unit.attached_turn, unit.num_turn);
    check_attached(unit.attached_combat, unit.num_combat);

    return count;
}

// ============================================================
// collect_triggers — mirrors Python EventManager.collect_triggers()
// Scans all units on both boards + extra triggers + system triggers
// ============================================================
int collect_triggers(
    const CombatState &state,
    const Event &event,
    const TriggerInstance *extra_triggers,
    int num_extra,
    TriggerInstance *out_triggers
) {
    int count = 0;

    for (int s = 0; s < 2; ++s) {
        const auto &board = state.boards[s];
        for (int i = 0; i < board.count; ++i) {
            if (board.units[i].is_empty()) continue;
            int added = collect_unit_triggers(
                board.units[i], event.event_type,
                static_cast<int8_t>(s), static_cast<int8_t>(i),
                out_triggers + count, GameConst::MAX_TRIGGERS_PER_EVENT - count
            );
            count += added;
        }
    }

    // Extra triggers (pre-collected death triggers from cleanup_dead)
    for (int i = 0; i < num_extra; ++i) {
        assert(count < GameConst::MAX_TRIGGERS_PER_EVENT && "Too many triggers!");
        out_triggers[count++] = extra_triggers[i];
    }

    // System triggers (global, not bound to any unit)
    int evt_idx = static_cast<int>(event.event_type);
    if (evt_idx >= 0 && evt_idx < static_cast<int>(EventType::EVENT_TYPE_COUNT)) {
        const auto &sys = g_system_triggers[evt_idx];
        for (int i = 0; i < sys.count; ++i) {
            assert(count < GameConst::MAX_TRIGGERS_PER_EVENT && "Too many triggers!");
            out_triggers[count++] = {
                &sys.defs[i],
                0, // trigger_uid = 0 (no owning unit)
                -1, -1, // no position
                1 // stacks = 1
            };
        }
    }

    return count;
}

// ============================================================
// sort_triggers — mirrors Python EventManager.order_triggers()
//
// Sort key (ascending tuple comparison):
//   1. group: 0 = source's own trigger (dead unit's deathrattle fires first)
//             1 = everyone else
//   2. -priority: higher priority number = fires earlier
//   3. side_priority: triggers on the side that CAUSED the event fire first
//      0 = same side as event source → fires first
//      1 = opponent side → fires second
//      2 = unknown side → fires last
//   4. slot: left-to-right board position
//   5. uid: tie-breaker (order of creation)
// ============================================================
void sort_triggers(
    TriggerInstance *triggers,
    int count,
    const Event &event
) {
    if (count <= 1) return;

    int8_t active_side = event.source_side;
    if (active_side < 0 && event.snapshot.valid) {
        active_side = event.snapshot.side;
    }

    int32_t source_uid = event.source_uid;
    if (source_uid == 0 && event.snapshot.valid) {
        source_uid = event.snapshot.uid;
    }

    // Pre-compute sort keys to avoid redundant computation during std::sort
    uint64_t keys[GameConst::MAX_TRIGGERS_PER_EVENT];

    for (int i = 0; i < count; ++i) {
        const auto &t = triggers[i];

        bool is_source_trigger = (
            event.event_type == EventType::MINION_DIED &&
            source_uid != 0 &&
            t.trigger_uid == source_uid
        );

        int8_t side = t.side;
        int8_t slot = t.slot;

        // Dead unit's trigger: position already gone from board, use snapshot
        if (side < 0 && is_source_trigger && event.snapshot.valid) {
            side = event.snapshot.side;
            slot = event.snapshot.slot;
        }

        int side_priority;
        if (side < 0) {
            side_priority = 2;
        } else if (active_side < 0) {
            side_priority = 0;
        } else {
            side_priority = side != active_side;
        }
        keys[i] = is_source_trigger ? 0 : 1;
        keys[i] <<= 8;
        keys[i] += (1 << 8) - 1 - t.def->priority;
        keys[i] <<= 2;
        keys[i] += side_priority;
        keys[i] <<= 4;
        keys[i] += slot < 0 ? 15 : slot;
        keys[i] <<= 48;
        keys[i] += t.trigger_uid;
    }

    // Sort indices by pre-computed keys, then reorder triggers
    int16_t indices[GameConst::MAX_TRIGGERS_PER_EVENT];
    for (int i = 0; i < count; ++i) indices[i] = i;

    std::sort(indices, indices + count, [&](int16_t a, int16_t b) {
        return keys[a] < keys[b];
    });

    TriggerInstance sorted[GameConst::MAX_TRIGGERS_PER_EVENT];
    for (int i = 0; i < count; ++i) sorted[i] = triggers[indices[i]];
    for (int i = 0; i < count; ++i) triggers[i] = sorted[i];
}

// ============================================================
// process_event — BFS event processing
// Mirrors Python EventManager.process_event()
//
// 1. Push initial event into queue
// 2. While queue not empty:
//    a. Pop event
//    b. Collect triggers (card_id + attached + extra)
//    c. Sort triggers
//    d. For each trigger: check condition → execute effect × stacks
//    e. Effects may push new events into the same queue → BFS order
// ============================================================

thread_local EventQueue queue;

void process_event(
    CombatState &state,
    Event initial_event,
    const TriggerInstance *extra_triggers,
    int num_extra
) {
    queue.reset();
    queue.push(initial_event);

    bool is_initial = true;
    TriggerInstance triggers[GameConst::MAX_TRIGGERS_PER_EVENT];
    while (!queue.empty()) {
        Event &current = queue.pop();

        int trigger_count = collect_triggers(
            state, current,
            is_initial ? extra_triggers : nullptr,
            is_initial ? num_extra : 0,
            triggers
        );

        // Sort triggers by game rules
        sort_triggers(triggers, trigger_count, current);

        // Execute triggers
        for (int i = 0; i < trigger_count; ++i) {
            const auto &trig = triggers[i];
            if (!trig.def || !trig.def->condition || !trig.def->effect) continue;

            // Check condition
            if (!trig.def->condition(state, current, trig.trigger_uid)) continue;

            // Fire effect × stacks
            for (int s = 0; s < trig.stacks; ++s) {
                trig.def->effect(state, queue, current, trig.trigger_uid);
            }
        }

        is_initial = false;
    }
}
