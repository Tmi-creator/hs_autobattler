#pragma once
// event_system.h — Event, EventQueue, trigger collection and BFS processing
// Mirrors Python event_system.py + combat.py death handling

#include <array>
#include <cassert>
#include <cstdint>
#include "types.h"
#include "entities.h"

// ============================================================
// MinionSnapshot — snapshot of a unit at the moment of death
// Needed because the unit is removed from board before triggers fire
// ============================================================
struct MinionSnapshot {
    int32_t uid = 0;
    int16_t card_id = CardID::INVALID;
    int8_t  side = -1;
    int8_t  slot = -1;
    int16_t atk = 0;
    int16_t hp = 0;
    TypeBitset types = UnitTypes::NONE;
    TagBitset  tags  = Tags::NONE;
    bool valid = false;
};

// ============================================================
// Event — single event in the BFS queue
// All references are by UID (not pointers!), resolved at trigger time
// ============================================================
struct Event {
    EventType event_type = EventType::EVENT_TYPE_COUNT; // invalid sentinel
    int32_t source_uid = 0;   // UID of source unit (0 = no source)
    int32_t target_uid = 0;   // UID of target unit (0 = no target)
    int8_t  source_side = -1; // Side (0 or 1) of source
    int8_t  source_slot = -1; // Board slot of source at event time
    int8_t  target_side = -1;
    int8_t  target_slot = -1;
    int16_t value = 0;        // Damage amount, buff value, etc.
    MinionSnapshot snapshot;  // For MINION_DIED: snapshot of the dead unit
};

// ============================================================
// EventQueue — BFS queue on the stack (no heap allocation)
// Lives within ONE process_event() call
// ============================================================
struct EventQueue {
    std::array<Event, GameConst::MAX_EVENT_QUEUE> data;
    uint16_t head = 0;
    uint16_t tail = 0;

    void push(const Event& e) {
        assert(tail < GameConst::MAX_EVENT_QUEUE && "EventQueue overflow!");
        data[tail++] = e;
    }
    Event& pop() {
        assert(!empty() && "EventQueue underflow!");
        return data[head++];
    }
    bool empty() const { return head >= tail; }
    void reset() { head = tail = 0; }
};

// ============================================================
// Effect function pointer type
// Args: CombatState&, EventQueue&, Event&, trigger_uid
// No heap, no closures, no virtuals — just a function pointer
// ============================================================
using EffectFn = void(*)(CombatState&, EventQueue&, const Event&, int32_t trigger_uid, int8_t side, int8_t slot);

// ============================================================
// Condition function pointer type
// Returns true if the trigger should fire
// ============================================================
using ConditionFn = bool(*)(const CombatState&, const Event&, int32_t trigger_uid, int8_t side, int8_t slot);

// ============================================================
// TriggerDef — static definition of a card's trigger
// Stored in the global effect table, never changes at runtime
// ============================================================
struct TriggerDef {
    EventType   event_type;
    ConditionFn condition;
    EffectFn    effect;
    int8_t      priority = 0;  // Higher = fires earlier (Battlecry=10, Avenge=-10)
};

// ============================================================
// TriggerInstance — a trigger ready to fire for this specific event
// Collected fresh for each event during collect_triggers()
// ============================================================
struct TriggerInstance {
    const TriggerDef* def = nullptr;
    int32_t trigger_uid = 0;  // UID of the unit that owns this trigger
    int8_t  side = -1;        // Side of the trigger owner
    int8_t  slot = -1;        // Slot of the trigger owner
    int16_t stacks = 1;       // Times to fire (golden=2, attached=count)
};

// ============================================================
// EffectTableEntry — maps card_id/effect_id → array of TriggerDefs
// Registered at startup, immutable at runtime
// ============================================================
struct EffectTableEntry {
    int16_t id = CardID::INVALID;  // card_id or effect_id
    int num_triggers = 0;
    int num_golden_triggers = 0;
    TriggerDef triggers[GameConst::MAX_TRIGGERS_PER_CARD];
    TriggerDef golden_triggers[GameConst::MAX_TRIGGERS_PER_CARD]; // golden overrides
    bool has_golden_override = false;
};

// Forward declarations — implemented in event_system.cpp
int collect_triggers(
    const CombatState& state,
    const Event& event,
    const TriggerInstance* extra_triggers,
    int num_extra,
    TriggerInstance* out_triggers  // output buffer
);

void sort_triggers(
    TriggerInstance* triggers,
    int count,
    const Event& event
);

void process_event(
    CombatState& state,
    Event initial_event,
    const TriggerInstance* extra_triggers = nullptr,
    int num_extra = 0
);

// ============================================================
// Effect registration — called once at startup
// ============================================================
void register_all_effects();

// Must be called after all register_effect_entry() calls.
// Sorts the table for O(log n) binary search lookup.
void finalize_effect_table();

// Register a card's triggers (normal + optional golden override)
void register_effect_entry(
    int16_t id,
    const TriggerDef* defs, int count,
    const TriggerDef* golden_defs = nullptr, int golden_count = 0
);

// Register a global system trigger (not attached to any unit)
// E.g. "when elemental added to shop → buff"
void register_system_trigger(const TriggerDef& def);

const EffectTableEntry* find_effect_entry(int16_t id);

// Helper: collect triggers for a SINGLE unit matching an EventType
// Used by both collect_triggers (live board scan) and collect_death_triggers (pre-removal)
int collect_unit_triggers(
    const Unit& unit, EventType event_type,
    int8_t side, int8_t slot,
    TriggerInstance* out_triggers, int max_out
);

// ============================================================
// Auras — wipe & reapply
// ============================================================
void recalculate_board_auras(CombatBoard& board);

// ============================================================
// Combat — main resolution loop
// ============================================================
BattleResult resolve_combat(CombatState& state);
