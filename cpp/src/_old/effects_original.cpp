// effects.cpp — Card effect functions + registration
// Ports Python effects.py TRIGGER_REGISTRY, GOLDEN_TRIGGER_REGISTRY, SYSTEM_TRIGGER_REGISTRY
//
// Each effect is a plain function: void(CombatState&, EventQueue&, const Event&, int32_t trigger_uid)
// Each condition is:              bool(const CombatState&, const Event&, int32_t trigger_uid)

#include "event_system.h"
#include <cstring> // memcpy for summon

// ============================================================
// Helper: resolve unit by UID → returns pointer (nullptr if not found)
// ============================================================
static Unit* resolve_unit(CombatState& state, int32_t uid) {
    for (int s = 0; s < 2; ++s) {
        auto& board = state.boards[s];
        for (int i = 0; i < board.count; ++i) {
            if (board.units[i].uid == uid) return &board.units[i];
        }
    }
    return nullptr;
}

// ============================================================
// Helper: summon a unit on a side at a given slot
// Returns UID of summoned unit, or 0 if board full
// Emits MINION_SUMMONED event into queue
// ============================================================
static int32_t summon_unit(
    CombatState& state, EventQueue& queue,
    int8_t side, int8_t slot, int16_t card_id,
    int16_t atk, int16_t hp, TypeBitset types, TagBitset tags,
    int8_t tier, bool is_golden
) {
    auto& board = state.boards[side];
    if (board.count >= GameConst::MAX_BOARD) return 0;

    Unit unit{};
    unit.card_id = card_id;
    unit.uid = state.next_uid++;
    unit.types = types;
    unit.tags = tags;
    unit.tier = tier;
    unit.atk_base = atk;
    unit.hp_base = hp;
    unit.is_golden = is_golden;

    // Clamp slot to valid range
    if (slot < 0) slot = 0;
    if (slot > board.count) slot = board.count;

    board.insert_at(slot, unit);

    // Emit MINION_SUMMONED
    Event e{};
    e.event_type = EventType::MINION_SUMMONED;
    e.source_uid = unit.uid;
    e.source_side = side;
    e.source_slot = slot;
    queue.push(e);

    return unit.uid;
}

// ============================================================
// Helper: buff unit stats
// ============================================================
static void buff_perm(CombatState& state, int32_t uid, int16_t atk, int16_t hp) {
    Unit* unit = resolve_unit(state, uid);
    if (!unit) return;
    unit->perm_atk += atk;
    unit->perm_hp += hp;
}

static void buff_combat(CombatState& state, int32_t uid, int16_t atk, int16_t hp) {
    Unit* unit = resolve_unit(state, uid);
    if (!unit) return;
    unit->combat_atk += atk;
    unit->combat_hp += hp;
}

// ============================================================
// Helper: deal damage to a specific unit (handles DS + events)
// ============================================================
static void deal_damage_to_unit(
    CombatState& state, EventQueue& queue,
    int32_t source_uid, Unit& target, int8_t target_side, int8_t target_slot,
    int16_t damage
) {
    if (target.has_tag(Tags::DIVINE_SHIELD)) {
        target.remove_tag(Tags::DIVINE_SHIELD);
        Event e{};
        e.event_type = EventType::DIVINE_SHIELD_LOST;
        e.source_uid = target.uid;
        e.source_side = target_side;
        e.source_slot = target_slot;
        queue.push(e);
    } else {
        target.damage_taken += damage;
        Event e{};
        e.event_type = EventType::MINION_DAMAGED;
        e.source_uid = source_uid;
        e.target_uid = target.uid;
        e.target_side = target_side;
        e.target_slot = static_cast<int8_t>(target_slot);
        e.value = damage;
        queue.push(e);
    }
}

// ============================================================
// Helper: get event source position (from event or snapshot)
// Returns false if no position available
// ============================================================
static bool get_source_pos(const Event& event, int8_t& side, int8_t& slot) {
    if (event.source_side >= 0) {
        side = event.source_side;
        slot = event.source_slot;
        return true;
    }
    if (event.snapshot.valid) {
        side = event.snapshot.side;
        slot = event.snapshot.slot;
        return true;
    }
    return false;
}

// ================================================================
// CONDITIONS
// ================================================================

// "trigger_uid is the source of this event" (self-play / self-death)
static bool cond_is_self(const CombatState&, const Event& event, int32_t trigger_uid) {
    return event.source_uid == trigger_uid;
}

// Always true
static bool cond_always(const CombatState&, const Event&, int32_t) {
    return true;
}

// Self-death: the dying unit is this trigger's owner
static bool cond_self_death(const CombatState&, const Event& event, int32_t trigger_uid) {
    return event.source_uid == trigger_uid;
}

// Friendly death (for avenge-like triggers): same side, not self
static bool cond_friendly_death(const CombatState& state, const Event& event, int32_t trigger_uid) {
    int8_t dead_side = -1, dead_slot = -1;
    if (event.source_side >= 0) {
        dead_side = event.source_side;
    } else if (event.snapshot.valid) {
        dead_side = event.snapshot.side;
    }
    if (dead_side < 0) return false;

    // Find trigger owner's side
    int8_t owner_side = -1, owner_slot = -1;
    find_unit_pos(state, trigger_uid, owner_side, owner_slot);
    if (owner_side < 0) return false;

    return (dead_side == owner_side) && (event.source_uid != trigger_uid);
}

// Same-side event, not self (reusable: Deflect-o-Bot, Mechano-Egg, etc.)
static bool cond_same_side_not_self(const CombatState& state, const Event& event, int32_t trigger_uid) {
    if (event.source_uid == trigger_uid) return false;
    int8_t src_side = event.source_side;
    if (src_side < 0) return false;

    int8_t owner_side = -1, owner_slot = -1;
    find_unit_pos(state, trigger_uid, owner_side, owner_slot);
    return (owner_side >= 0 && src_side == owner_side);
}

// ================================================================
// EFFECT FUNCTIONS
// ================================================================

// --- Alleycat: summon a 1/1 Tabbycat ---
static void effect_summon_tabbycat(CombatState& state, EventQueue& queue, const Event& event, int32_t trigger_uid) {
    int8_t side = -1, slot = -1;
    find_unit_pos(state, trigger_uid, side, slot);
    if (side < 0) return;
    summon_unit(state, queue, side, slot + 1, CardID::TABBYCAT,
        1, 1, UnitTypes::BEAST, Tags::NONE, 1, false);
}

// --- Golden Alleycat: summon a golden 1/1 Tabbycat ---
static void effect_summon_golden_tabbycat(CombatState& state, EventQueue& queue, const Event& event, int32_t trigger_uid) {
    int8_t side = -1, slot = -1;
    find_unit_pos(state, trigger_uid, side, slot);
    if (side < 0) return;
    summon_unit(state, queue, side, slot + 1, CardID::TABBYCAT,
        1, 1, UnitTypes::BEAST, Tags::NONE, 1, true);
}

// --- Scallywag DR: summon 1/1 Pirate Token with IMMEDIATE_ATTACK ---
static void effect_summon_scallywag_token(CombatState& state, EventQueue& queue, const Event& event, int32_t trigger_uid) {
    int8_t side = -1, slot = -1;
    if (!get_source_pos(event, side, slot)) return;
    summon_unit(state, queue, side, slot, CardID::PIRATE_TOKEN,
        1, 1, UnitTypes::PIRATE, Tags::IMMEDIATE_ATTACK, 1, false);
}

// --- Imprisoner DR: summon 1/1 Imp Token ---
static void effect_summon_imp_token(CombatState& state, EventQueue& queue, const Event& event, int32_t trigger_uid) {
    int8_t side = -1, slot = -1;
    if (!get_source_pos(event, side, slot)) return;
    summon_unit(state, queue, side, slot, CardID::IMP_TOKEN,
        1, 1, UnitTypes::DEMON, Tags::NONE, 1, false);
}

// --- Crab Deathrattle (attached): summon 1/1 Crab Token ---
static void effect_summon_crab_token(CombatState& state, EventQueue& queue, const Event& event, int32_t trigger_uid) {
    int8_t side = -1, slot = -1;
    if (!get_source_pos(event, side, slot)) return;
    summon_unit(state, queue, side, slot, CardID::CRAB_TOKEN,
        1, 1, UnitTypes::BEAST, Tags::NONE, 1, false);
}

// --- Wrath Weaver: when ANOTHER demon is played, +2/+1 and take 1 hero damage ---
static void effect_wrath_weaver(CombatState& state, EventQueue& queue, const Event& event, int32_t trigger_uid) {
    // Check: played unit must be a demon
    Unit* played = resolve_unit(state, event.source_uid);
    if (!played) return;
    if (!(played->types & UnitTypes::DEMON)) return;
    // Must not be self
    if (event.source_uid == trigger_uid) return;

    buff_perm(state, trigger_uid, 2, 1);
    // Hero damage: tavern-phase only, no-op in CombatState.
}

// --- Swampstriker: when another Murloc is played, +1 atk ---
static void effect_swampstriker(CombatState& state, EventQueue& queue, const Event& event, int32_t trigger_uid) {
    Unit* played = resolve_unit(state, event.source_uid);
    if (!played) return;
    if (!(played->types & UnitTypes::MURLOC)) return;
    if (event.source_uid == trigger_uid) return;

    buff_perm(state, trigger_uid, 1, 0);
}

// --- Spawn of N'Zoth DR: +1/+1 to all friendly minions ---
static void effect_spawn_of_nzoth_dr(CombatState& state, EventQueue& queue, const Event& event, int32_t trigger_uid) {
    int8_t side = -1, slot = -1;
    if (!get_source_pos(event, side, slot)) return;

    auto& board = state.boards[side];
    for (int i = 0; i < board.count; ++i) {
        buff_combat(state, board.units[i].uid, 1, 1);
    }
}

// --- Kaboom Bot DR: deal 4 damage to a random enemy ---
static void effect_kaboom_bot_dr(CombatState& state, EventQueue& queue, const Event& event, int32_t trigger_uid) {
    int8_t side = -1, slot = -1;
    if (!get_source_pos(event, side, slot)) return;

    int8_t enemy_side = 1 - side;
    auto& enemy_board = state.boards[enemy_side];
    if (enemy_board.count == 0) return;

    // Random target
    int target_idx = rng_index(state.rng, enemy_board.count);
    deal_damage_to_unit(
        state, queue, trigger_uid,
        enemy_board.units[target_idx], enemy_side, static_cast<int8_t>(target_idx),
        4
    );
}

// --- Deflect-o-Bot: +2 atk and gain DS when Mech summoned on your side ---
static void effect_deflect_o_bot(CombatState& state, EventQueue& queue, const Event& event, int32_t trigger_uid) {
    // Type check: summoned unit must be a Mech
    Unit* summoned = resolve_unit(state, event.source_uid);
    if (!summoned || !(summoned->types & UnitTypes::MECH)) return;

    Unit* deflecto = resolve_unit(state, trigger_uid);
    if (!deflecto || !deflecto->is_alive()) return;

    buff_combat(state, trigger_uid, 2, 0);
    deflecto->set_tag(Tags::DIVINE_SHIELD);
}

// ================================================================
// REGISTRATION — maps card_id → triggers
// Called once at startup
// ================================================================
void register_all_effects() {
    // --- Shell Collector: Battlecry → gain coin (tavern phase only) ---
    {
        TriggerDef def{EventType::MINION_PLAYED, cond_is_self, nullptr, 10};
        // No effect in combat phase — shell collector is tavern-only.
        // We register it for parity but the effect is a no-op during combat.
        // TODO: implement when tavern phase is ported
    }

    // --- Alleycat: Battlecry → summon Tabbycat ---
    {
        TriggerDef normal{EventType::MINION_PLAYED, cond_is_self, effect_summon_tabbycat, 10};
        TriggerDef golden{EventType::MINION_PLAYED, cond_is_self, effect_summon_golden_tabbycat, 10};
        register_effect_entry(CardID::ALLEYCAT, &normal, 1, &golden, 1);
    }

    // --- Scallywag: DR → summon pirate token ---
    {
        TriggerDef def{EventType::MINION_DIED, cond_self_death, effect_summon_scallywag_token};
        register_effect_entry(CardID::SCALLYWAG, &def, 1);
    }

    // --- Imprisoner: DR → summon imp token ---
    {
        TriggerDef def{EventType::MINION_DIED, cond_self_death, effect_summon_imp_token};
        register_effect_entry(CardID::IMPRISONER, &def, 1);
    }

    // --- Wrath Weaver: when demon played → +2/+1 ---
    {
        TriggerDef def{EventType::MINION_PLAYED, cond_always, effect_wrath_weaver};
        register_effect_entry(CardID::WRATH_WEAVER, &def, 1);
    }

    // --- Swampstriker: when murloc played → +1 atk ---
    {
        TriggerDef def{EventType::MINION_PLAYED, cond_always, effect_swampstriker};
        register_effect_entry(CardID::SWAMPSTRIKER, &def, 1);
    }

    // --- Crab Deathrattle (attached effect) ---
    {
        TriggerDef def{EventType::MINION_DIED, cond_self_death, effect_summon_crab_token};
        register_effect_entry(EffectID::CRAB_DEATHRATTLE, &def, 1);
    }

    // --- Spawn of N'Zoth: DR → +1/+1 to all friendlies ---
    {
        TriggerDef def{EventType::MINION_DIED, cond_self_death, effect_spawn_of_nzoth_dr};
        register_effect_entry(CardID::SPAWN_OF_NZOTH, &def, 1);
    }

    // --- Kaboom Bot: DR → deal 4 to random enemy ---
    {
        TriggerDef def{EventType::MINION_DIED, cond_self_death, effect_kaboom_bot_dr};
        register_effect_entry(CardID::KABOOM_BOT, &def, 1);
    }

    // --- Deflect-o-Bot: when Mech summoned on your side → +2 atk + DS ---
    {
        TriggerDef def{EventType::MINION_SUMMONED, cond_same_side_not_self, effect_deflect_o_bot};
        register_effect_entry(CardID::DEFLECT_O_BOT, &def, 1);
    }

    // --- SYSTEM TRIGGERS ---
    // Elemental buff when added to shop — tavern phase only, no-op in combat
    // TODO: implement when tavern phase is ported

    // Sort the table for binary search
    finalize_effect_table();
}
