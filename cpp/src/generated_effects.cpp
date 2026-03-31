// generated_effects.cpp — AUTO-GENERATED from card_def.py
// DO NOT EDIT MANUALLY — run: python scripts/generate_cpp_effects.py
//
// Only combat-relevant effects are generated here.
// Tavern-phase effects (BC, sell) run in Python only.

#include "event_system.h"

// ── Helpers (from original effects.cpp) ──

static Unit* resolve_unit(CombatState& state, int32_t uid) {
    for (int s = 0; s < 2; ++s) {
        auto& board = state.boards[s];
        for (int i = 0; i < board.count; ++i) {
            if (board.units[i].uid == uid) return &board.units[i];
        }
    }
    return nullptr;
}

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
    if (slot < 0) slot = 0;
    if (slot > board.count) slot = board.count;
    board.insert_at(slot, unit);
    Event e{};
    e.event_type = EventType::MINION_SUMMONED;
    e.source_uid = unit.uid;
    e.source_side = side;
    e.source_slot = slot;
    queue.push(e);
    return unit.uid;
}

static void buff_perm(CombatState& state, int32_t uid, int16_t atk, int16_t hp) {
    Unit* u = resolve_unit(state, uid);
    if (u) { u->perm_atk += atk; u->perm_hp += hp; }
}

static void buff_combat(CombatState& state, int32_t uid, int16_t atk, int16_t hp) {
    Unit* u = resolve_unit(state, uid);
    if (u) { u->combat_atk += atk; u->combat_hp += hp; }
}

static bool get_source_pos(const Event& event, int8_t& side, int8_t& slot) {
    if (event.source_side >= 0) { side = event.source_side; slot = event.source_slot; return true; }
    if (event.snapshot.valid) { side = event.snapshot.side; slot = event.snapshot.slot; return true; }
    return false;
}

// ── Conditions ──

static bool cond_is_self(const CombatState&, const Event& e, int32_t uid) {
    return e.source_uid == uid;
}

static bool cond_self_death(const CombatState&, const Event& e, int32_t uid) {
    return e.source_uid == uid;
}

static bool cond_always(const CombatState&, const Event&, int32_t) {
    return true;
}

static bool cond_friendly_death(const CombatState& state, const Event& event, int32_t trigger_uid) {
    int8_t dead_side = event.source_side >= 0 ? event.source_side : (event.snapshot.valid ? event.snapshot.side : -1);
    if (dead_side < 0) return false;
    int8_t owner_side = -1, owner_slot = -1;
    find_unit_pos(state, trigger_uid, owner_side, owner_slot);
    if (owner_side < 0) return false;
    return (dead_side == owner_side) && (event.source_uid != trigger_uid);
}

static bool cond_friendly_soc(const CombatState& state, const Event&, int32_t trigger_uid) {
    int8_t side = -1, slot = -1;
    find_unit_pos(state, trigger_uid, side, slot);
    return side >= 0;
}

// ── Generated effect functions ──

static void effect_dr_cord_puller(CombatState& state, EventQueue& queue, const Event& event, int32_t) {
    int8_t side = -1, slot = -1;
    if (!get_source_pos(event, side, slot)) return;
    summon_unit(state, queue, side, slot, 901, 1, 1, UnitTypes::MECH, Tags::NONE, 1, false);
}

static void effect_dr_harmless_bonehead(CombatState& state, EventQueue& queue, const Event& event, int32_t) {
    int8_t side = -1, slot = -1;
    if (!get_source_pos(event, side, slot)) return;
    summon_unit(state, queue, side, slot, 902, 1, 1, UnitTypes::UNDEAD, Tags::NONE, 1, false);
    summon_unit(state, queue, side, slot, 902, 1, 1, UnitTypes::UNDEAD, Tags::NONE, 1, false);
}

static void effect_dr_manasaber(CombatState& state, EventQueue& queue, const Event& event, int32_t) {
    int8_t side = -1, slot = -1;
    if (!get_source_pos(event, side, slot)) return;
    summon_unit(state, queue, side, slot, 903, 0, 1, UnitTypes::BEAST, Tags::TAUNT, 1, false);
    summon_unit(state, queue, side, slot, 903, 0, 1, UnitTypes::BEAST, Tags::TAUNT, 1, false);
}

static void effect_soc_misfit_dragonling(CombatState& state, EventQueue&, const Event&, int32_t trigger_uid) {
    Unit* u = resolve_unit(state, trigger_uid);
    if (!u || !u->is_alive()) return;
    int8_t side = -1, slot = -1;
    find_unit_pos(state, trigger_uid, side, slot);
    if (side < 0) return;
    int16_t tier = state.boards[side].tavern_tier;
    buff_combat(state, trigger_uid, tier, tier);
}

static void effect_on_death_rot_hide_gnoll(CombatState& state, EventQueue&, const Event&, int32_t trigger_uid) {
    Unit* u = resolve_unit(state, trigger_uid);
    if (!u || !u->is_alive()) return;
    buff_combat(state, trigger_uid, 1, 0);
}

static void effect_on_play_swampstriker(CombatState& state, EventQueue&, const Event& event, int32_t trigger_uid) {
    Unit* played = resolve_unit(state, event.source_uid);
    if (!played || !(played->types & UnitTypes::MURLOC)) return;
    if (event.source_uid == trigger_uid) return;
    buff_perm(state, trigger_uid, 1, 0);
}

static void effect_rally_tusked_camper(CombatState& state, EventQueue&, const Event&, int32_t trigger_uid) {
    Unit* u = resolve_unit(state, trigger_uid);
    if (!u || !u->is_alive()) return;
    buff_combat(state, trigger_uid, 1, 1);
}

static void effect_dr_twilight_hatchling(CombatState& state, EventQueue& queue, const Event& event, int32_t) {
    int8_t side = -1, slot = -1;
    if (!get_source_pos(event, side, slot)) return;
    summon_unit(state, queue, side, slot, 904, 3, 3, UnitTypes::DRAGON, Tags::IMMEDIATE_ATTACK, 1, false);
}

static void effect_on_play_wrath_weaver(CombatState& state, EventQueue&, const Event& event, int32_t trigger_uid) {
    Unit* played = resolve_unit(state, event.source_uid);
    if (!played || !(played->types & UnitTypes::DEMON)) return;
    if (event.source_uid == trigger_uid) return;
    buff_perm(state, trigger_uid, 2, 1);
    // Hero damage (1) is tavern-only, no-op in combat
}

static void effect_dr_sewer_rat(CombatState& state, EventQueue& queue, const Event& event, int32_t) {
    int8_t side = -1, slot = -1;
    if (!get_source_pos(event, side, slot)) return;
    summon_unit(state, queue, side, slot, 906, 2, 3, UnitTypes::NONE, Tags::TAUNT, 2, false);
}

static void effect_on_play_mechagnome_interpreter(CombatState& state, EventQueue&, const Event& event, int32_t trigger_uid) {
    Unit* played = resolve_unit(state, event.source_uid);
    if (!played || !(played->types & UnitTypes::MECH)) return;
    if (event.source_uid == trigger_uid) return;
    buff_perm(state, trigger_uid, 2, 1);
}

static void effect_dr_cadaver_caretaker(CombatState& state, EventQueue& queue, const Event& event, int32_t) {
    int8_t side = -1, slot = -1;
    if (!get_source_pos(event, side, slot)) return;
    summon_unit(state, queue, side, slot, 902, 1, 1, UnitTypes::UNDEAD, Tags::NONE, 1, false);
    summon_unit(state, queue, side, slot, 902, 1, 1, UnitTypes::UNDEAD, Tags::NONE, 1, false);
    summon_unit(state, queue, side, slot, 902, 1, 1, UnitTypes::UNDEAD, Tags::NONE, 1, false);
}

static void effect_dr_handless_forsaken(CombatState& state, EventQueue& queue, const Event& event, int32_t) {
    int8_t side = -1, slot = -1;
    if (!get_source_pos(event, side, slot)) return;
    summon_unit(state, queue, side, slot, 908, 2, 1, UnitTypes::UNDEAD, Tags::REBORN, 1, false);
}

static void effect_on_play_peggy_sturdybone(CombatState& state, EventQueue&, const Event& event, int32_t trigger_uid) {
    Unit* played = resolve_unit(state, event.source_uid);
    if (!played || !(played->types & UnitTypes::PIRATE)) return;
    if (event.source_uid == trigger_uid) return;
    buff_perm(state, trigger_uid, 2, 1);
}

static void effect_on_death_devout_hellcaller(CombatState& state, EventQueue&, const Event&, int32_t trigger_uid) {
    Unit* u = resolve_unit(state, trigger_uid);
    if (!u || !u->is_alive()) return;
    buff_combat(state, trigger_uid, 1, 2);
}

static void effect_rally_heroic_underdog(CombatState& state, EventQueue&, const Event&, int32_t trigger_uid) {
    Unit* u = resolve_unit(state, trigger_uid);
    if (!u || !u->is_alive()) return;
    buff_combat(state, trigger_uid, 1, 0);
}

static void effect_dr_sly_raptor(CombatState& state, EventQueue& queue, const Event& event, int32_t) {
    int8_t side = -1, slot = -1;
    if (!get_source_pos(event, side, slot)) return;
    summon_unit(state, queue, side, slot, 902, 1, 1, UnitTypes::UNDEAD, Tags::NONE, 1, false);
}

static void effect_rally_monstrous_macaw(CombatState& state, EventQueue&, const Event&, int32_t trigger_uid) {
    Unit* u = resolve_unit(state, trigger_uid);
    if (!u || !u->is_alive()) return;
    buff_combat(state, trigger_uid, 1, 1);
}

static void effect_on_play_ichoron_the_protector(CombatState& state, EventQueue&, const Event& event, int32_t trigger_uid) {
    Unit* played = resolve_unit(state, event.source_uid);
    if (!played || !(played->types & UnitTypes::ELEMENTAL)) return;
    if (event.source_uid == trigger_uid) return;
    buff_perm(state, trigger_uid, 0, 1);
}

static void effect_on_play_nomi_kitchen_nightmare(CombatState& state, EventQueue&, const Event& event, int32_t trigger_uid) {
    Unit* played = resolve_unit(state, event.source_uid);
    if (!played || !(played->types & UnitTypes::ELEMENTAL)) return;
    if (event.source_uid == trigger_uid) return;
    buff_perm(state, trigger_uid, 2, 2);
}

static void effect_rally_razorfen_vineweaver(CombatState& state, EventQueue&, const Event&, int32_t trigger_uid) {
    Unit* u = resolve_unit(state, trigger_uid);
    if (!u || !u->is_alive()) return;
    buff_combat(state, trigger_uid, 1, 1);
}

static void effect_rally_sanguine_refiner(CombatState& state, EventQueue&, const Event&, int32_t trigger_uid) {
    Unit* u = resolve_unit(state, trigger_uid);
    if (!u || !u->is_alive()) return;
    buff_combat(state, trigger_uid, 1, 1);
}

static void effect_on_play_primitive_painter(CombatState& state, EventQueue&, const Event& event, int32_t trigger_uid) {
    Unit* played = resolve_unit(state, event.source_uid);
    if (!played || !(played->types & UnitTypes::MURLOC)) return;
    if (event.source_uid == trigger_uid) return;
    buff_perm(state, trigger_uid, 1, 2);
}

static void effect_rally_the_last_one_standing(CombatState& state, EventQueue&, const Event&, int32_t trigger_uid) {
    Unit* u = resolve_unit(state, trigger_uid);
    if (!u || !u->is_alive()) return;
    buff_combat(state, trigger_uid, 12, 12);
}

static void effect_dr_crab_attached(CombatState& state, EventQueue& queue, const Event& event, int32_t) {
    int8_t side = -1, slot = -1;
    if (!get_source_pos(event, side, slot)) return;
    summon_unit(state, queue, side, slot, 905, 3, 2, UnitTypes::BEAST, Tags::NONE, 1, false);
}

// ── Registration ──

void register_all_effects() {
    { TriggerDef def{EventType::MINION_DIED, cond_self_death, effect_dr_cord_puller}; register_effect_entry(103, &def, 1); }
    { TriggerDef def{EventType::MINION_DIED, cond_self_death, effect_dr_harmless_bonehead}; register_effect_entry(107, &def, 1); }
    { TriggerDef def{EventType::MINION_DIED, cond_self_death, effect_dr_manasaber}; register_effect_entry(108, &def, 1); }
    { TriggerDef def{EventType::START_OF_COMBAT, cond_friendly_soc, effect_soc_misfit_dragonling}; register_effect_entry(110, &def, 1); }
    { TriggerDef def{EventType::MINION_DIED, cond_friendly_death, effect_on_death_rot_hide_gnoll}; register_effect_entry(116, &def, 1); }
    { TriggerDef def{EventType::MINION_PLAYED, cond_always, effect_on_play_swampstriker}; register_effect_entry(118, &def, 1); }
    { TriggerDef def{EventType::ATTACK_DECLARED, cond_is_self, effect_rally_tusked_camper}; register_effect_entry(119, &def, 1); }
    { TriggerDef def{EventType::MINION_DIED, cond_self_death, effect_dr_twilight_hatchling}; register_effect_entry(120, &def, 1); }
    { TriggerDef def{EventType::MINION_PLAYED, cond_always, effect_on_play_wrath_weaver}; register_effect_entry(121, &def, 1); }
    { TriggerDef def{EventType::MINION_DIED, cond_self_death, effect_dr_sewer_rat}; register_effect_entry(203, &def, 1); }
    { TriggerDef def{EventType::MINION_PLAYED, cond_always, effect_on_play_mechagnome_interpreter}; register_effect_entry(205, &def, 1); }
    { TriggerDef def{EventType::MINION_DIED, cond_self_death, effect_dr_cadaver_caretaker}; register_effect_entry(305, &def, 1); }
    { TriggerDef def{EventType::MINION_DIED, cond_self_death, effect_dr_handless_forsaken}; register_effect_entry(307, &def, 1); }
    { TriggerDef def{EventType::MINION_PLAYED, cond_always, effect_on_play_peggy_sturdybone}; register_effect_entry(326, &def, 1); }
    { TriggerDef def{EventType::MINION_DIED, cond_friendly_death, effect_on_death_devout_hellcaller}; register_effect_entry(404, &def, 1); }
    { TriggerDef def{EventType::ATTACK_DECLARED, cond_is_self, effect_rally_heroic_underdog}; register_effect_entry(409, &def, 1); }
    { TriggerDef def{EventType::MINION_DIED, cond_self_death, effect_dr_sly_raptor}; register_effect_entry(420, &def, 1); }
    { TriggerDef def{EventType::ATTACK_DECLARED, cond_is_self, effect_rally_monstrous_macaw}; register_effect_entry(431, &def, 1); }
    { TriggerDef def{EventType::MINION_PLAYED, cond_always, effect_on_play_ichoron_the_protector}; register_effect_entry(438, &def, 1); }
    { TriggerDef def{EventType::MINION_PLAYED, cond_always, effect_on_play_nomi_kitchen_nightmare}; register_effect_entry(512, &def, 1); }
    { TriggerDef def{EventType::ATTACK_DECLARED, cond_is_self, effect_rally_razorfen_vineweaver}; register_effect_entry(514, &def, 1); }
    { TriggerDef def{EventType::ATTACK_DECLARED, cond_is_self, effect_rally_sanguine_refiner}; register_effect_entry(613, &def, 1); }
    { TriggerDef def{EventType::MINION_PLAYED, cond_always, effect_on_play_primitive_painter}; register_effect_entry(620, &def, 1); }
    { TriggerDef def{EventType::ATTACK_DECLARED, cond_is_self, effect_rally_the_last_one_standing}; register_effect_entry(703, &def, 1); }
    { TriggerDef def{EventType::MINION_DIED, cond_self_death, effect_dr_crab_attached}; register_effect_entry(EffectID::CRAB_DEATHRATTLE, &def, 1); }

    finalize_effect_table();
}
