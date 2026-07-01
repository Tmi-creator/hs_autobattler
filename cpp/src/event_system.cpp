// event_system.cpp — BFS event processing, trigger collection and sorting
// Mirrors Python: EventManager.process_event(), collect_triggers(), order_triggers()

#include "event_system.h"
#include "profiler.h"
#include <algorithm>
#include <tuple>

// ============================================================
// Global effect table + direct-index lookup.
// Таблица g_effect_table хранит сами записи, g_effect_index[id] даёт O(1)
// доступ по card_id / effect_id. Размер плоского массива — GameConst::EFFECT_INDEX_SIZE.
// ============================================================
static EffectTableEntry g_effect_table[GameConst::MAX_EFFECT_ENTRIES];
static int g_num_entries = 0;
static bool g_table_finalized = false;
static const EffectTableEntry* g_effect_index[GameConst::EFFECT_INDEX_SIZE] = {};

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

// Битовая маска event types, на которые зарегистрирован хотя бы один system trigger.
// Обновляется только в register_system_trigger (на старте). Читается в has_any_subscribers.
uint32_t g_system_event_mask = 0;

void register_system_trigger(const TriggerDef &def) {
    int idx = static_cast<int>(def.event_type);
    assert(idx >= 0 && idx < static_cast<int>(EventType::EVENT_TYPE_COUNT));
    auto &list = g_system_triggers[idx];
    assert(list.count < GameConst::MAX_SYSTEM_TRIGGERS && "Too many system triggers for this event type!");
    list.defs[list.count++] = def;
    g_system_event_mask |= (1u << idx);
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
    g_table_finalized = false; // invalidate index
}

// Called after all register_effect_entry() calls. Строит плоский индекс
// g_effect_index[id] -> EffectTableEntry* для O(1) лукапа.
void finalize_effect_table() {
    for (int i = 0; i < GameConst::EFFECT_INDEX_SIZE; ++i) g_effect_index[i] = nullptr;
    for (int i = 0; i < g_num_entries; ++i) {
        const int16_t id = g_effect_table[i].id;
        assert(id >= 0 && id < GameConst::EFFECT_INDEX_SIZE &&
               "Effect id out of index range — увеличь GameConst::EFFECT_INDEX_SIZE");
        g_effect_index[id] = &g_effect_table[i];
    }
    g_table_finalized = true;
}

// O(1) direct-index lookup. Один bounds-check + один load из плоского массива.
const EffectTableEntry *find_effect_entry(int16_t id) {
    assert(g_table_finalized && "Call finalize_effect_table() after registering all effects!");
    const unsigned u = static_cast<unsigned>(id);
    if (u >= GameConst::EFFECT_INDEX_SIZE) return nullptr;
    return g_effect_index[u];
}

// ============================================================
// compute_unit_event_mask — считает маску "на какие события подписан юнит".
// Объявлена в entities.h (forward-decl), реализация здесь потому что
// нуждается в find_effect_entry из глобальной таблицы эффектов.
// Вызывается из CombatBoard::insert_at и из recalculate_subscribers.
// ============================================================
uint32_t compute_unit_event_mask(const Unit& unit) {
    uint32_t m = 0;

    // 1. Card-level triggers. Golden с overrideом читает golden_triggers, иначе — обычные.
    const EffectTableEntry* entry = find_effect_entry(unit.card_id);
    if (entry) {
        const TriggerDef* defs;
        int num;
        if (unit.is_golden && entry->has_golden_override) {
            defs = entry->golden_triggers;
            num  = entry->num_golden_triggers;
        } else {
            defs = entry->triggers;
            num  = entry->num_triggers;
        }
        for (int i = 0; i < num; ++i) {
            m |= (1u << static_cast<int>(defs[i].event_type));
        }
    }

    // 2. Attached effects (3 scopes). Правило `count <= 0` — пропуск, как в collect_unit_triggers.
    auto add_attached = [&](const std::array<AttachedEffect, GameConst::MAX_ATTACHED>& arr, uint8_t num) {
        for (int a = 0; a < num; ++a) {
            if (arr[a].count <= 0) continue;
            const EffectTableEntry* att = find_effect_entry(arr[a].effect_id);
            if (!att) continue;
            for (int i = 0; i < att->num_triggers; ++i) {
                m |= (1u << static_cast<int>(att->triggers[i].event_type));
            }
        }
    };
    add_attached(unit.attached_perm,   unit.num_perm);
    add_attached(unit.attached_turn,   unit.num_turn);
    add_attached(unit.attached_combat, unit.num_combat);

    return m;
}

// ============================================================
// recalculate_subscribers — полный ребилд всех slot-масок + damage.
// Нужен после path'ов, которые пишут в units[] напрямую (parse_board в pybind).
// В insert_at/remove_at поддержание инкрементальное, этот ребилд не нужен.
// ============================================================
void recalculate_subscribers(CombatBoard& board) {
    for (int e = 0; e < static_cast<int>(EventType::EVENT_TYPE_COUNT); ++e) {
        board.subscribers[e] = 0;
    }
    board.taunt_mask = 0;
    board.aura_source_mask = 0;
    board.dead_slot_mask = 0;
    board.damage = 0;

    for (int i = 0; i < board.count; ++i) {
        const Unit& u = board.units[i];
        const uint8_t bit = static_cast<uint8_t>(1u << i);

        // subscribers
        uint32_t um = compute_unit_event_mask(u);
        while (um) {
            int e = __builtin_ctz(um);
            um &= um - 1;
            board.subscribers[e] |= static_cast<uint16_t>(bit);
        }

        if (u.has_tag(Tags::TAUNT))    board.taunt_mask       |= bit;
        if (is_aura_source(u.card_id)) board.aura_source_mask |= bit;

        // damage (sum of tiers — used for battle result)
        board.damage += static_cast<uint8_t>(u.tier);
    }
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
    ProfScope _ps(ProfSection::COLLECT_TRIGGERS);
    int count = 0;
    const int evt_idx = static_cast<int>(event.event_type);

    // Вместо слепого скана 7×2=14 слотов итерируем только по подписчикам
    // через ctz по предрасчитанной битовой маске. Обычная маска содержит 0-2 бит,
    // а не 7, и для событий без подписчиков (большинство) цикл не выполняется вообще.
    for (int s = 0; s < 2; ++s) {
        const auto& board = state.boards[s];
        uint16_t mask = board.subscribers[evt_idx];
        while (mask) {
            const int i = __builtin_ctz(mask);
            mask &= static_cast<uint16_t>(mask - 1);
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

// Сортирует массив индексов по соответствующим uint64 ключам возрастанию.
// Используется в sort_triggers для упорядочивания срабатывания триггеров внутри
// одного event. Типичный count = 3-5, максимум MAX_TRIGGERS_PER_EVENT (64), поэтому
// insertion sort здесь быстрее std::sort: у него нет setup-overhead на partition,
// предсказуемые бранчи, и на совсем маленьких n компилятор разворачивает внутренний
// while в линейные mov'ы. Алгоритм стабильный — важно для тай-брейков по uid.
static void insertion_sort_by_key(int16_t* indices, int count, const uint64_t* keys) {
    for (int i = 1; i < count; ++i) {
        const int16_t idx = indices[i];
        const uint64_t key = keys[idx];
        int j = i;
        while (j > 0 && keys[indices[j - 1]] > key) {
            indices[j] = indices[j - 1];
            --j;
        }
        indices[j] = idx;
    }
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
    ProfScope _ps(ProfSection::SORT_TRIGGERS);

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
        /*
        sort by:
        is source trigger?
        -priority
        side_priority
        slot
        uid(tiebreaker)
         */
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

    insertion_sort_by_key(indices, count, keys);

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
    const Event &initial_event,
    const TriggerInstance *extra_triggers,
    int num_extra
) {
    ProfScope _ps(ProfSection::PROCESS_EVENT);
    // Safety-net early exit: если нет ни extra triggers (death path), ни подписчиков
    // на этот event type — process_event делать нечего. Callsite в перф-критичных
    // путях должен был проверить это сам через has_any_subscribers() и не строить
    // Event вообще, но если не проверил — спасаемся здесь.
    if (num_extra == 0 && !has_any_subscribers(state, initial_event.event_type)) {
        return;
    }

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
            if (!trig.def->condition(state, current, trig.trigger_uid, trig.side, trig.slot)) continue;

            // Fire effect × stacks
            for (int s = 0; s < trig.stacks; ++s) {
                trig.def->effect(state, queue, current, trig.trigger_uid, trig.side, trig.slot);
            }
        }

        is_initial = false;
    }
}
