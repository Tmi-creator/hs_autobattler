// generated_effects.cpp — AUTO-GENERATED from card_def.py
// DO NOT EDIT MANUALLY — run: python scripts/generate_cpp_effects.py
//
// Only combat-relevant effects are generated here.
// Tavern-phase effects (BC, sell) run in Python only.

#include "event_system.h"
#include "generated_card_db.h"

// ── Helpers ──

// Прямой O(1) lookup владельца триггера: side/slot пробрасываются из process_event.
// Сверка uid защищает от случая, когда слот переиспользован (death+reborn в той же позиции).
static Unit* trigger_owner(CombatState& state, int8_t side, int8_t slot, int32_t uid) {
    if (side < 0 || slot < 0) return nullptr;
    auto& board = state.boards[side];
    if (slot >= board.count) return nullptr;
    Unit& u = board.units[slot];
    return u.uid == uid ? &u : nullptr;
}

// Прямой lookup юнита-источника события через event.source_side/slot.
static Unit* event_source_unit(CombatState& state, const Event& event) {
    if (event.source_side < 0 || event.source_slot < 0) return nullptr;
    auto& board = state.boards[event.source_side];
    if (event.source_slot >= board.count) return nullptr;
    Unit& u = board.units[event.source_slot];
    return u.uid == event.source_uid ? &u : nullptr;
}

// Lookup юнита на обеих досках по UID, возвращает side/slot
static Unit* find_unit_by_uid(CombatState& state, int32_t uid, int8_t& out_side, int8_t& out_slot) {
    if (uid == 0) return nullptr;
    for (int s = 0; s < 2; ++s) {
        auto& board = state.boards[s];
        for (int i = 0; i < board.count; ++i) {
            if (board.units[i].uid == uid) {
                out_side = s;
                out_slot = i;
                return &board.units[i];
            }
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
    unit.avenge_counter = CardDB::avenge_threshold(card_id);
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

static bool get_source_pos(const Event& event, int8_t& side, int8_t& slot) {
    if (event.source_side >= 0) { side = event.source_side; slot = event.source_slot; return true; }
    if (event.snapshot.valid) { side = event.snapshot.side; slot = event.snapshot.slot; return true; }
    return false;
}

// ── Conditions ──

static bool cond_is_self(const CombatState&, const Event& e, int32_t uid, int8_t, int8_t) {
    return e.source_uid == uid;
}

static bool cond_self_death(const CombatState&, const Event& e, int32_t uid, int8_t, int8_t) {
    return e.source_uid == uid;
}

static bool cond_always(const CombatState&, const Event&, int32_t, int8_t, int8_t) {
    return true;
}

static bool cond_friendly_death(const CombatState&, const Event& event, int32_t trigger_uid, int8_t owner_side, int8_t) {
    int8_t dead_side = event.source_side >= 0
        ? event.source_side
        : (event.snapshot.valid ? event.snapshot.side : -1);
    if (dead_side < 0 || owner_side < 0) return false;
    return (dead_side == owner_side) && (event.source_uid != trigger_uid);
}

static bool cond_friendly_soc(const CombatState&, const Event&, int32_t, int8_t owner_side, int8_t) {
    return owner_side >= 0;
}

// ── Generated effect functions ──

static void effect_dr_cord_puller(CombatState& state, EventQueue& queue, const Event& event, int32_t, int8_t, int8_t) {
    int8_t s = -1, sl = -1;
    if (!get_source_pos(event, s, sl)) return;
    summon_unit(state, queue, s, sl, 901, 1, 1, UnitTypes::MECH, Tags::NONE, 1, false);
}

static void effect_dr_harmless_bonehead(CombatState& state, EventQueue& queue, const Event& event, int32_t, int8_t, int8_t) {
    int8_t s = -1, sl = -1;
    if (!get_source_pos(event, s, sl)) return;
    summon_unit(state, queue, s, sl, 902, 1, 1, UnitTypes::UNDEAD, Tags::NONE, 1, false);
    summon_unit(state, queue, s, sl, 902, 1, 1, UnitTypes::UNDEAD, Tags::NONE, 1, false);
}

static void effect_dr_manasaber(CombatState& state, EventQueue& queue, const Event& event, int32_t, int8_t, int8_t) {
    int8_t s = -1, sl = -1;
    if (!get_source_pos(event, s, sl)) return;
    summon_unit(state, queue, s, sl, 903, 0, 1, UnitTypes::BEAST, Tags::TAUNT, 1, false);
    summon_unit(state, queue, s, sl, 903, 0, 1, UnitTypes::BEAST, Tags::TAUNT, 1, false);
}

static void effect_soc_misfit_dragonling(CombatState& state, EventQueue&, const Event&, int32_t trigger_uid, int8_t side, int8_t slot) {
    Unit* u = trigger_owner(state, side, slot, trigger_uid);
    if (!u || !u->is_alive()) return;
    int16_t tier = state.boards[side].tavern_tier;
    u->combat_atk += tier;
    u->combat_hp += tier;
}

static void effect_on_death_rot_hide_gnoll(CombatState& state, EventQueue&, const Event&, int32_t trigger_uid, int8_t side, int8_t slot) {
    Unit* u = trigger_owner(state, side, slot, trigger_uid);
    if (!u || !u->is_alive()) return;
    u->combat_atk += 1;
}

static void effect_on_play_swampstriker(CombatState& state, EventQueue&, const Event& event, int32_t trigger_uid, int8_t side, int8_t slot) {
    if (event.source_uid == trigger_uid) return;
    Unit* played = event_source_unit(state, event);
    if (!played || !(played->types & UnitTypes::MURLOC)) return;
    Unit* u = trigger_owner(state, side, slot, trigger_uid);
    if (!u) return;
    u->perm_atk += 1;
}

static void effect_rally_tusked_camper(CombatState& state, EventQueue&, const Event&, int32_t trigger_uid, int8_t side, int8_t slot) {
    Unit* u = trigger_owner(state, side, slot, trigger_uid);
    if (!u || !u->is_alive()) return;
    u->combat_atk += 1;
    u->combat_hp += 1;
}

static void effect_dr_twilight_hatchling(CombatState& state, EventQueue& queue, const Event& event, int32_t, int8_t, int8_t) {
    int8_t s = -1, sl = -1;
    if (!get_source_pos(event, s, sl)) return;
    summon_unit(state, queue, s, sl, 904, 3, 3, UnitTypes::DRAGON, Tags::IMMEDIATE_ATTACK, 1, false);
}

static void effect_on_play_wrath_weaver(CombatState& state, EventQueue&, const Event& event, int32_t trigger_uid, int8_t side, int8_t slot) {
    if (event.source_uid == trigger_uid) return;
    Unit* played = event_source_unit(state, event);
    if (!played || !(played->types & UnitTypes::DEMON)) return;
    Unit* u = trigger_owner(state, side, slot, trigger_uid);
    if (!u) return;
    u->perm_atk += 2;
    u->perm_hp += 1;
    // Hero damage (1) is tavern-only, no-op in combat
}

static void effect_dr_sewer_rat(CombatState& state, EventQueue& queue, const Event& event, int32_t, int8_t, int8_t) {
    int8_t s = -1, sl = -1;
    if (!get_source_pos(event, s, sl)) return;
    summon_unit(state, queue, s, sl, 906, 2, 3, UnitTypes::NONE, Tags::TAUNT, 2, false);
}

static void effect_on_play_mechagnome_interpreter(CombatState& state, EventQueue&, const Event& event, int32_t trigger_uid, int8_t side, int8_t slot) {
    if (event.source_uid == trigger_uid) return;
    Unit* played = event_source_unit(state, event);
    if (!played || !(played->types & UnitTypes::MECH)) return;
    Unit* u = trigger_owner(state, side, slot, trigger_uid);
    if (!u) return;
    u->perm_atk += 2;
    u->perm_hp += 1;
}

static void effect_soc_humming_bird(CombatState& state, EventQueue&, const Event&, int32_t trigger_uid, int8_t side, int8_t slot) {
    Unit* u = trigger_owner(state, side, slot, trigger_uid);
    if (!u || !u->is_alive()) return;
    auto& board = state.boards[side];
    for (int i = 0; i < board.count; ++i) {
        Unit& target = board.units[i];
        if (target.is_alive() && (target.types & UnitTypes::BEAST)) {
            target.combat_atk += 1;
            target.combat_hp += 0;
        }
    }
}

static void effect_rally_sleepy_supporter(CombatState& state, EventQueue&, const Event&, int32_t trigger_uid, int8_t side, int8_t slot) {
    Unit* u = trigger_owner(state, side, slot, trigger_uid);
    if (!u || !u->is_alive()) return;
    auto& board = state.boards[side];
    Unit* candidates[GameConst::MAX_BOARD];
    int count = 0;
    for (int i = 0; i < board.count; ++i) {
        Unit& other = board.units[i];
        if (other.is_alive() && other.uid != trigger_uid && (other.types & UnitTypes::DRAGON)) {
            candidates[count++] = &other;
        }
    }
    if (count > 0) {
        int idx = rng_index(state.rng, count);
        candidates[idx]->combat_atk += 2;
        candidates[idx]->combat_hp += 3;
    }
}

static void effect_avenge_ghostly_ymirjar(CombatState& state, EventQueue& queue, const Event& event, int32_t trigger_uid, int8_t side, int8_t slot) {
    Unit* u = trigger_owner(state, side, slot, trigger_uid);
    if (!u || !u->is_alive()) return;
    u->avenge_counter--;
    if (u->avenge_counter <= 0) {
        u->avenge_counter = 4;
    }
}

static void effect_soc_irate_rooster(CombatState& state, EventQueue&, const Event&, int32_t trigger_uid, int8_t side, int8_t slot) {
    Unit* u = trigger_owner(state, side, slot, trigger_uid);
    if (!u || !u->is_alive()) return;
    auto& board = state.boards[side];
    for (int adj_idx : {slot - 1, slot + 1}) {
        if (adj_idx >= 0 && adj_idx < board.count) {
            Unit& target = board.units[adj_idx];
            if (target.is_alive()) {
                target.damage_taken += 1;
                if (target.get_hp() <= 0) {
                    board.dead_slot_mask |= (1 << adj_idx);
                    state.has_pending_deaths = true;
                }
                target.combat_atk += 4;
                target.combat_hp += 0;
            }
        }
    }
}

static void effect_avenge_bird_buddy(CombatState& state, EventQueue& queue, const Event& event, int32_t trigger_uid, int8_t side, int8_t slot) {
    Unit* u = trigger_owner(state, side, slot, trigger_uid);
    if (!u || !u->is_alive()) return;
    u->avenge_counter--;
    if (u->avenge_counter <= 0) {
        u->avenge_counter = 1;
        auto& board = state.boards[side];
        for (int i = 0; i < board.count; ++i) {
            Unit& target = board.units[i];
            if (target.is_alive() && (target.types & UnitTypes::BEAST)) {
                target.combat_atk += 1;
                target.combat_hp += 1;
            }
        }
    }
}

static void effect_avenge_budding_greenthumb(CombatState& state, EventQueue& queue, const Event& event, int32_t trigger_uid, int8_t side, int8_t slot) {
    Unit* u = trigger_owner(state, side, slot, trigger_uid);
    if (!u || !u->is_alive()) return;
    u->avenge_counter--;
    if (u->avenge_counter <= 0) {
        u->avenge_counter = 3;
        auto& board = state.boards[side];
        for (int adj_idx : {slot - 1, slot + 1}) {
            if (adj_idx >= 0 && adj_idx < board.count) {
                Unit& target = board.units[adj_idx];
                if (target.is_alive()) {
                    target.combat_atk += 2;
                    target.combat_hp += 2;
                }
            }
        }
    }
}

static void effect_dr_cadaver_caretaker(CombatState& state, EventQueue& queue, const Event& event, int32_t, int8_t, int8_t) {
    int8_t s = -1, sl = -1;
    if (!get_source_pos(event, s, sl)) return;
    summon_unit(state, queue, s, sl, 902, 1, 1, UnitTypes::UNDEAD, Tags::NONE, 1, false);
    summon_unit(state, queue, s, sl, 902, 1, 1, UnitTypes::UNDEAD, Tags::NONE, 1, false);
    summon_unit(state, queue, s, sl, 902, 1, 1, UnitTypes::UNDEAD, Tags::NONE, 1, false);
}

static void effect_dr_handless_forsaken(CombatState& state, EventQueue& queue, const Event& event, int32_t, int8_t, int8_t) {
    int8_t s = -1, sl = -1;
    if (!get_source_pos(event, s, sl)) return;
    summon_unit(state, queue, s, sl, 908, 2, 1, UnitTypes::UNDEAD, Tags::REBORN, 1, false);
}

static void effect_rally_rampager(CombatState& state, EventQueue&, const Event&, int32_t trigger_uid, int8_t side, int8_t slot) {
    Unit* u = trigger_owner(state, side, slot, trigger_uid);
    if (!u || !u->is_alive()) return;
    auto& board = state.boards[side];
    for (int i = 0; i < board.count; ++i) {
        Unit& target = board.units[i];
        if (target.is_alive() && target.uid != trigger_uid) {
            target.damage_taken += 1;
            if (target.get_hp() <= 0) {
                board.dead_slot_mask |= (1 << i);
                state.has_pending_deaths = true;
            }
        }
    }
}

static void effect_soc_amber_guardian(CombatState& state, EventQueue&, const Event&, int32_t trigger_uid, int8_t side, int8_t slot) {
    Unit* u = trigger_owner(state, side, slot, trigger_uid);
    if (!u || !u->is_alive()) return;
    auto& board = state.boards[side];
    Unit* candidates[GameConst::MAX_BOARD];
    int count = 0;
    for (int i = 0; i < board.count; ++i) {
        Unit& target = board.units[i];
        if (target.is_alive() && target.uid != trigger_uid && (target.types & UnitTypes::DRAGON)) {
            candidates[count++] = &target;
        }
    }
    if (count > 0) {
        int idx = rng_index(state.rng, count);
        candidates[idx]->combat_atk += 2;
        candidates[idx]->combat_hp += 2;
        candidates[idx]->set_tag(Tags::DIVINE_SHIELD);
    }
}

static void effect_trig_hardy_orca(CombatState& state, EventQueue&, const Event& event, int32_t trigger_uid, int8_t side, int8_t slot) {
    if (event.target_uid != trigger_uid) return;
    Unit* u = trigger_owner(state, side, slot, trigger_uid);
    if (!u || !u->is_alive()) return;
    auto& board = state.boards[side];
    for (int i = 0; i < board.count; ++i) {
        Unit& target = board.units[i];
        if (target.is_alive() && target.uid != trigger_uid) {
            target.combat_atk += 1;
            target.combat_hp += 1;
        }
    }
}

static void effect_trig_jelly_belly(CombatState& state, EventQueue&, const Event& event, int32_t trigger_uid, int8_t side, int8_t slot) {
    if (event.meta != 1) return;
    if (event.source_side != side) return;
    Unit* u = trigger_owner(state, side, slot, trigger_uid);
    if (!u || !u->is_alive()) return;
    u->combat_atk += 2;
    u->combat_hp += 3;
}

static void effect_dr_anubarak_nerubian_king(CombatState& state, EventQueue& queue, const Event& event, int32_t trigger_uid, int8_t side, int8_t slot) {
    auto& board = state.boards[side];
    for (int i = 0; i < board.count; ++i) {
        Unit& target = board.units[i];
        if (target.is_alive() && (target.types & UnitTypes::UNDEAD)) {
            target.combat_atk += 1;
            target.combat_hp += 0;
        }
    }
}

static void effect_trig_deflect_o_bot(CombatState& state, EventQueue&, const Event& event, int32_t trigger_uid, int8_t side, int8_t slot) {
    if (event.source_uid == trigger_uid) return;
    if (event.source_side != side) return;
    Unit* summoned = event_source_unit(state, event);
    if (!summoned || !(summoned->types & UnitTypes::MECH)) return;
    Unit* u = trigger_owner(state, side, slot, trigger_uid);
    if (!u || !u->is_alive()) return;
    u->combat_atk += 2;
    u->combat_hp += 0;
    u->set_tag(Tags::DIVINE_SHIELD);
}

static void effect_on_play_peggy_sturdybone(CombatState& state, EventQueue&, const Event& event, int32_t trigger_uid, int8_t side, int8_t slot) {
    if (event.source_uid == trigger_uid) return;
    Unit* played = event_source_unit(state, event);
    if (!played || !(played->types & UnitTypes::PIRATE)) return;
    Unit* u = trigger_owner(state, side, slot, trigger_uid);
    if (!u) return;
    u->perm_atk += 2;
    u->perm_hp += 1;
}

static void effect_trig_roaring_recruiter(CombatState& state, EventQueue&, const Event& event, int32_t trigger_uid, int8_t side, int8_t slot) {
    if (event.source_uid == trigger_uid) return;
    if (event.source_side != side) return;
    int8_t att_side = -1, att_slot = -1;
    Unit* attacker = find_unit_by_uid(state, event.source_uid, att_side, att_slot);
    if (!attacker || att_side != side || !(attacker->types & UnitTypes::DRAGON)) return;
    attacker->combat_atk += 3;
    attacker->combat_hp += 1;
}

static void effect_rally_bonker(CombatState& state, EventQueue&, const Event&, int32_t trigger_uid, int8_t side, int8_t slot) {
    Unit* u = trigger_owner(state, side, slot, trigger_uid);
    if (!u || !u->is_alive()) return;
    auto& board = state.boards[side];
    for (int i = 0; i < board.count; ++i) {
        Unit& target = board.units[i];
        if (target.is_alive() && target.uid != trigger_uid) {
            target.combat_atk += 2;
            target.combat_hp += 2;
        }
    }
}

static void effect_on_death_devout_hellcaller(CombatState& state, EventQueue&, const Event&, int32_t trigger_uid, int8_t side, int8_t slot) {
    Unit* u = trigger_owner(state, side, slot, trigger_uid);
    if (!u || !u->is_alive()) return;
    u->combat_atk += 1;
    u->combat_hp += 2;
}

static void effect_rally_heroic_underdog(CombatState& state, EventQueue&, const Event&, int32_t trigger_uid, int8_t side, int8_t slot) {
    Unit* u = trigger_owner(state, side, slot, trigger_uid);
    if (!u || !u->is_alive()) return;
    u->combat_atk += 1;
}

static void effect_soc_prized_promo_drake(CombatState& state, EventQueue&, const Event&, int32_t trigger_uid, int8_t side, int8_t slot) {
    Unit* u = trigger_owner(state, side, slot, trigger_uid);
    if (!u || !u->is_alive()) return;
    auto& board = state.boards[side];
    for (int i = 0; i < board.count; ++i) {
        Unit& target = board.units[i];
        if (target.is_alive() && (target.types & UnitTypes::DRAGON)) {
            target.combat_atk += 4;
            target.combat_hp += 4;
        }
    }
}

static void effect_dr_silent_enforcer(CombatState& state, EventQueue& queue, const Event& event, int32_t trigger_uid, int8_t side, int8_t slot) {
    for (int s = 0; s < 2; ++s) {
        auto& board = state.boards[s];
        for (int i = 0; i < board.count; ++i) {
            Unit& victim = board.units[i];
            if (victim.is_alive()) {
                if (victim.has_tag(Tags::DIVINE_SHIELD)) {
                    victim.remove_tag(Tags::DIVINE_SHIELD);
                    fire_unit_event(state, EventType::DIVINE_SHIELD_LOST, victim.uid, s, i);
                } else {
                    victim.damage_taken += 2;
                    if (victim.get_hp() <= 0) {
                        board.dead_slot_mask |= (1 << i);
                        state.has_pending_deaths = true;
                    }
                }
            }
        }
    }
}

static void effect_dr_sly_raptor(CombatState& state, EventQueue& queue, const Event& event, int32_t, int8_t, int8_t) {
    int8_t s = -1, sl = -1;
    if (!get_source_pos(event, s, sl)) return;
    summon_unit(state, queue, s, sl, 902, 1, 1, UnitTypes::UNDEAD, Tags::NONE, 1, false);
}

static void effect_soc_soulsplitter(CombatState& state, EventQueue&, const Event&, int32_t trigger_uid, int8_t side, int8_t slot) {
    Unit* u = trigger_owner(state, side, slot, trigger_uid);
    if (!u || !u->is_alive()) return;
    auto& board = state.boards[side];
    Unit* candidates[GameConst::MAX_BOARD];
    int count = 0;
    for (int i = 0; i < board.count; ++i) {
        Unit& target = board.units[i];
        if (target.is_alive() && target.uid != trigger_uid && (target.types & UnitTypes::UNDEAD) && !target.has_tag(Tags::REBORN)) {
            candidates[count++] = &target;
        }
    }
    if (count > 0) {
        int idx = rng_index(state.rng, count);
        candidates[idx]->set_tag(Tags::REBORN);
    }
}

static void effect_avenge_spirit_drake(CombatState& state, EventQueue& queue, const Event& event, int32_t trigger_uid, int8_t side, int8_t slot) {
    Unit* u = trigger_owner(state, side, slot, trigger_uid);
    if (!u || !u->is_alive()) return;
    u->avenge_counter--;
    if (u->avenge_counter <= 0) {
        u->avenge_counter = 3;
    }
}

static void effect_dr_tunnel_blaster(CombatState& state, EventQueue& queue, const Event& event, int32_t trigger_uid, int8_t side, int8_t slot) {
    for (int s = 0; s < 2; ++s) {
        auto& board = state.boards[s];
        for (int i = 0; i < board.count; ++i) {
            Unit& victim = board.units[i];
            if (victim.is_alive()) {
                if (victim.has_tag(Tags::DIVINE_SHIELD)) {
                    victim.remove_tag(Tags::DIVINE_SHIELD);
                    fire_unit_event(state, EventType::DIVINE_SHIELD_LOST, victim.uid, s, i);
                } else {
                    victim.damage_taken += 3;
                    if (victim.get_hp() <= 0) {
                        board.dead_slot_mask |= (1 << i);
                        state.has_pending_deaths = true;
                    }
                }
            }
        }
    }
}

static void effect_avenge_witchwing_nestmatron(CombatState& state, EventQueue& queue, const Event& event, int32_t trigger_uid, int8_t side, int8_t slot) {
    Unit* u = trigger_owner(state, side, slot, trigger_uid);
    if (!u || !u->is_alive()) return;
    u->avenge_counter--;
    if (u->avenge_counter <= 0) {
        u->avenge_counter = 3;
    }
}

static void effect_rally_monstrous_macaw(CombatState& state, EventQueue&, const Event&, int32_t trigger_uid, int8_t side, int8_t slot) {
    Unit* u = trigger_owner(state, side, slot, trigger_uid);
    if (!u || !u->is_alive()) return;
    u->combat_atk += 1;
    u->combat_hp += 1;
}

static void effect_dr_rylak_metalhead(CombatState& state, EventQueue& queue, const Event& event, int32_t trigger_uid, int8_t side, int8_t slot) {
    auto& board = state.boards[side];
    for (int i = 0; i < board.count; ++i) {
        Unit& target = board.units[i];
        if (target.is_alive()) {
            target.combat_atk += 1;
            target.combat_hp += 1;
        }
    }
}

static void effect_rally_sunken_advocate(CombatState& state, EventQueue&, const Event&, int32_t trigger_uid, int8_t side, int8_t slot) {
    Unit* u = trigger_owner(state, side, slot, trigger_uid);
    if (!u || !u->is_alive()) return;
    auto& board = state.boards[side];
    for (int i = 0; i < board.count; ++i) {
        Unit& target = board.units[i];
        if (target.is_alive() && target.uid != trigger_uid && (target.types & UnitTypes::NAGA)) {
            target.combat_atk += 1;
        }
    }
}

static void effect_trig_trigore_the_lasher(CombatState& state, EventQueue&, const Event& event, int32_t trigger_uid, int8_t side, int8_t slot) {
    if (event.target_uid == trigger_uid) return;
    int8_t tgt_side = -1, tgt_slot = -1;
    Unit* target = find_unit_by_uid(state, event.target_uid, tgt_side, tgt_slot);
    if (!target || tgt_side != side || !(target->types & UnitTypes::BEAST)) return;
    Unit* u = trigger_owner(state, side, slot, trigger_uid);
    if (!u || !u->is_alive()) return;
    u->combat_hp += 2;
}

static void effect_on_play_ichoron_the_protector(CombatState& state, EventQueue&, const Event& event, int32_t trigger_uid, int8_t side, int8_t slot) {
    if (event.source_uid == trigger_uid) return;
    Unit* played = event_source_unit(state, event);
    if (!played || !(played->types & UnitTypes::ELEMENTAL)) return;
    Unit* u = trigger_owner(state, side, slot, trigger_uid);
    if (!u) return;
    u->perm_hp += 1;
}

static void effect_avenge_champion_of_the_primus(CombatState& state, EventQueue& queue, const Event& event, int32_t trigger_uid, int8_t side, int8_t slot) {
    Unit* u = trigger_owner(state, side, slot, trigger_uid);
    if (!u || !u->is_alive()) return;
    u->avenge_counter--;
    if (u->avenge_counter <= 0) {
        u->avenge_counter = 2;
        auto& board = state.boards[side];
        for (int i = 0; i < board.count; ++i) {
            Unit& target = board.units[i];
            if (target.is_alive() && (target.types & UnitTypes::UNDEAD)) {
                target.combat_atk += 1;
                target.combat_hp += 0;
            }
        }
    }
}

static void effect_soc_corrupted_myrmidon(CombatState& state, EventQueue&, const Event&, int32_t trigger_uid, int8_t side, int8_t slot) {
    Unit* u = trigger_owner(state, side, slot, trigger_uid);
    if (!u || !u->is_alive()) return;
    u->combat_atk += 3;
    u->combat_hp += 3;
}

static void effect_dr_silithid_burrower(CombatState& state, EventQueue& queue, const Event& event, int32_t trigger_uid, int8_t side, int8_t slot) {
    auto& board = state.boards[side];
    for (int i = 0; i < board.count; ++i) {
        Unit& target = board.units[i];
        if (target.is_alive() && (target.types & UnitTypes::BEAST)) {
            target.combat_atk += 1;
            target.combat_hp += 1;
        }
    }
}

static void effect_avenge_silithid_burrower(CombatState& state, EventQueue& queue, const Event& event, int32_t trigger_uid, int8_t side, int8_t slot) {
    Unit* u = trigger_owner(state, side, slot, trigger_uid);
    if (!u || !u->is_alive()) return;
    u->avenge_counter--;
    if (u->avenge_counter <= 0) {
        u->avenge_counter = 1;
        u->combat_atk += 1;
        u->combat_hp += 1;
    }
}

static void effect_avenge_ghoul_of_the_feast(CombatState& state, EventQueue& queue, const Event& event, int32_t trigger_uid, int8_t side, int8_t slot) {
    Unit* u = trigger_owner(state, side, slot, trigger_uid);
    if (!u || !u->is_alive()) return;
    u->avenge_counter--;
    if (u->avenge_counter <= 0) {
        u->avenge_counter = 1;
        auto& board = state.boards[side];
        for (int i = 0; i < board.count; ++i) {
            Unit& target = board.units[i];
            if (target.is_alive()) {
                target.combat_atk += 2;
                target.combat_hp += 2;
            }
        }
    }
}

static void effect_trig_twilight_watcher(CombatState& state, EventQueue&, const Event& event, int32_t trigger_uid, int8_t side, int8_t slot) {
    if (event.source_uid == trigger_uid) return;
    if (event.source_side != side) return;
    int8_t att_side = -1, att_slot = -1;
    Unit* attacker = find_unit_by_uid(state, event.source_uid, att_side, att_slot);
    if (!attacker || att_side != side || !(attacker->types & UnitTypes::DRAGON)) return;
    Unit* u = trigger_owner(state, side, slot, trigger_uid);
    if (!u || !u->is_alive()) return;
    u->combat_atk += 1;
    u->combat_hp += 3;
}

static void effect_trig_unforgiving_treant(CombatState& state, EventQueue&, const Event& event, int32_t trigger_uid, int8_t side, int8_t slot) {
    if (event.target_uid != trigger_uid) return;
    Unit* u = trigger_owner(state, side, slot, trigger_uid);
    if (!u || !u->is_alive()) return;
    auto& board = state.boards[side];
    for (int i = 0; i < board.count; ++i) {
        Unit& target = board.units[i];
        if (target.is_alive() && target.uid != trigger_uid) {
            target.combat_atk += 2;
            target.combat_hp += 0;
        }
    }
}

static void effect_on_play_nomi_kitchen_nightmare(CombatState& state, EventQueue&, const Event& event, int32_t trigger_uid, int8_t side, int8_t slot) {
    if (event.source_uid == trigger_uid) return;
    Unit* played = event_source_unit(state, event);
    if (!played || !(played->types & UnitTypes::ELEMENTAL)) return;
    Unit* u = trigger_owner(state, side, slot, trigger_uid);
    if (!u) return;
    u->perm_atk += 2;
    u->perm_hp += 2;
}

static void effect_rally_bile_spitter(CombatState& state, EventQueue&, const Event&, int32_t trigger_uid, int8_t side, int8_t slot) {
    Unit* u = trigger_owner(state, side, slot, trigger_uid);
    if (!u || !u->is_alive()) return;
    auto& board = state.boards[side];
    Unit* candidates[GameConst::MAX_BOARD];
    int count = 0;
    for (int i = 0; i < board.count; ++i) {
        Unit& other = board.units[i];
        if (other.is_alive() && other.uid != trigger_uid && (other.types & UnitTypes::MURLOC)) {
            candidates[count++] = &other;
        }
    }
    if (count > 0) {
        int idx = rng_index(state.rng, count);
        candidates[idx]->combat_atk += 0;
        candidates[idx]->combat_hp += 0;
    }
}

static void effect_rally_razorfen_vineweaver(CombatState& state, EventQueue&, const Event&, int32_t trigger_uid, int8_t side, int8_t slot) {
    Unit* u = trigger_owner(state, side, slot, trigger_uid);
    if (!u || !u->is_alive()) return;
    u->combat_atk += 1;
    u->combat_hp += 1;
}

static void effect_dr_spiked_savior(CombatState& state, EventQueue& queue, const Event& event, int32_t trigger_uid, int8_t side, int8_t slot) {
    auto& board = state.boards[side];
    for (int i = 0; i < board.count; ++i) {
        Unit& target = board.units[i];
        if (target.is_alive()) {
            target.combat_atk += 1;
            target.combat_hp += 1;
            if (1 > 0) {
                target.damage_taken += 1;
                if (target.get_hp() <= 0) {
                    board.dead_slot_mask |= (1 << i);
                    state.has_pending_deaths = true;
                }
            }
        }
    }
}

static void effect_dr_leeroy_the_reckless(CombatState& state, EventQueue& queue, const Event& event, int32_t trigger_uid, int8_t side, int8_t slot) {
    int32_t killer_uid = event.meta;
    if (killer_uid != 0) {
        for (int s = 0; s < 2; ++s) {
            auto& board = state.boards[s];
            for (int i = 0; i < board.count; ++i) {
                Unit& target = board.units[i];
                if (target.uid == killer_uid && target.is_alive()) {
                    target.damage_taken += target.get_hp();
                    board.dead_slot_mask |= (1 << i);
                    state.has_pending_deaths = true;
                    return;
                }
            }
        }
    }
}

static void effect_avenge_stuntdrake(CombatState& state, EventQueue& queue, const Event& event, int32_t trigger_uid, int8_t side, int8_t slot) {
    Unit* u = trigger_owner(state, side, slot, trigger_uid);
    if (!u || !u->is_alive()) return;
    u->avenge_counter--;
    if (u->avenge_counter <= 0) {
        u->avenge_counter = 3;
        auto& board = state.boards[side];
        Unit* candidates[GameConst::MAX_BOARD];
        int count = 0;
        for (int i = 0; i < board.count; ++i) {
            Unit& target = board.units[i];
            if (target.is_alive() && (target.types & UnitTypes::DRAGON)) {
                candidates[count++] = &target;
            }
        }
        if (count > 0) {
            int idx = rng_index(state.rng, count);
            candidates[idx]->combat_atk += 14;
            candidates[idx]->combat_hp += 5;
        }
    }
}

static void effect_trig_iridescent_skyblazer(CombatState& state, EventQueue&, const Event& event, int32_t trigger_uid, int8_t side, int8_t slot) {
    int8_t tgt_side = -1, tgt_slot = -1;
    Unit* target = find_unit_by_uid(state, event.target_uid, tgt_side, tgt_slot);
    if (!target || tgt_side != side || !(target->types & UnitTypes::BEAST)) return;
    auto& board = state.boards[side];
    Unit* candidates[GameConst::MAX_BOARD];
    int count = 0;
    for (int i = 0; i < board.count; ++i) {
        Unit& other = board.units[i];
        if (other.is_alive() && other.uid != event.target_uid && (other.types & UnitTypes::BEAST)) {
            candidates[count++] = &other;
        }
    }
    if (count > 0) {
        int idx = rng_index(state.rng, count);
        candidates[idx]->combat_atk += 1;
        candidates[idx]->combat_hp += 1;
    }
}

static void effect_rally_niuzao(CombatState& state, EventQueue&, const Event&, int32_t trigger_uid, int8_t side, int8_t slot) {
    Unit* u = trigger_owner(state, side, slot, trigger_uid);
    if (!u || !u->is_alive()) return;
    auto& enemy_board = state.boards[1 - side];
    if (enemy_board.count > 0) {
        int idx = rng_index(state.rng, enemy_board.count);
        Unit& victim = enemy_board.units[idx];
        int16_t dmg = u->get_atk();
        if (dmg > 0) {
            if (victim.has_tag(Tags::DIVINE_SHIELD)) {
                victim.remove_tag(Tags::DIVINE_SHIELD);
                fire_unit_event(state, EventType::DIVINE_SHIELD_LOST, victim.uid, 1 - side, idx);
            } else {
                victim.damage_taken += dmg;
                if (victim.get_hp() <= 0) {
                    enemy_board.dead_slot_mask |= (1 << idx);
                    state.has_pending_deaths = true;
                }
            }
        }
    }
}

static void effect_dr_twilight_broodmother(CombatState& state, EventQueue& queue, const Event& event, int32_t trigger_uid, int8_t side, int8_t slot) {
    int8_t s = -1, sl = -1;
    if (!get_source_pos(event, s, sl)) return;
    summon_unit(state, queue, s, sl, 904, 3, 3, UnitTypes::DRAGON, Tags::TAUNT, 1, false);
    summon_unit(state, queue, s, sl, 904, 3, 3, UnitTypes::DRAGON, Tags::TAUNT, 1, false);
}

static void effect_soc_costume_enthusiast(CombatState& state, EventQueue&, const Event&, int32_t trigger_uid, int8_t side, int8_t slot) {
    Unit* u = trigger_owner(state, side, slot, trigger_uid);
    if (!u || !u->is_alive()) return;
    auto& board = state.boards[side];
    int16_t max_atk = 0;
    for (int i = 0; i < board.count; ++i) {
        Unit& target = board.units[i];
        if (target.is_alive() && target.uid != trigger_uid) {
            if (target.get_atk() > max_atk) max_atk = target.get_atk();
        }
    }
    if (max_atk > 0) u->combat_atk += max_atk;
}

static void effect_dr_goldrinn_the_great_wolf(CombatState& state, EventQueue& queue, const Event& event, int32_t trigger_uid, int8_t side, int8_t slot) {
    auto& board = state.boards[side];
    for (int i = 0; i < board.count; ++i) {
        Unit& target = board.units[i];
        if (target.is_alive() && (target.types & UnitTypes::BEAST)) {
            target.combat_atk += 8;
            target.combat_hp += 8;
        }
    }
}

static void effect_dr_ship_master_eudora(CombatState& state, EventQueue& queue, const Event& event, int32_t trigger_uid, int8_t side, int8_t slot) {
    auto& board = state.boards[side];
    for (int i = 0; i < board.count; ++i) {
        Unit& target = board.units[i];
        if (target.is_alive()) {
            target.combat_atk += 8;
            target.combat_hp += 8;
        }
    }
}

static void effect_soc_ultraviolet_ascendant(CombatState& state, EventQueue&, const Event&, int32_t trigger_uid, int8_t side, int8_t slot) {
    Unit* u = trigger_owner(state, side, slot, trigger_uid);
    if (!u || !u->is_alive()) return;
    auto& board = state.boards[side];
    for (int i = 0; i < board.count; ++i) {
        Unit& target = board.units[i];
        if (target.is_alive() && target.uid != trigger_uid && (target.types & UnitTypes::ELEMENTAL)) {
            target.combat_atk += 3;
            target.combat_hp += 2;
        }
    }
}

static void effect_soc_fire_forged_evoker(CombatState& state, EventQueue&, const Event&, int32_t trigger_uid, int8_t side, int8_t slot) {
    Unit* u = trigger_owner(state, side, slot, trigger_uid);
    if (!u || !u->is_alive()) return;
    auto& board = state.boards[side];
    for (int i = 0; i < board.count; ++i) {
        Unit& target = board.units[i];
        if (target.is_alive() && (target.types & UnitTypes::DRAGON)) {
            target.combat_atk += 2;
            target.combat_hp += 1;
        }
    }
}

static void effect_rally_sanguine_refiner(CombatState& state, EventQueue&, const Event&, int32_t trigger_uid, int8_t side, int8_t slot) {
    Unit* u = trigger_owner(state, side, slot, trigger_uid);
    if (!u || !u->is_alive()) return;
    u->combat_atk += 1;
    u->combat_hp += 1;
}

static void effect_rally_bloodsnout_warlord(CombatState& state, EventQueue&, const Event&, int32_t trigger_uid, int8_t side, int8_t slot) {
    Unit* u = trigger_owner(state, side, slot, trigger_uid);
    if (!u || !u->is_alive()) return;
    auto& board = state.boards[side];
    for (int i = 0; i < board.count; ++i) {
        Unit& target = board.units[i];
        if (target.is_alive() && target.uid != trigger_uid) {
            target.combat_atk += 3;
            target.combat_hp += 3;
        }
    }
}

static void effect_avenge_deathly_striker(CombatState& state, EventQueue& queue, const Event& event, int32_t trigger_uid, int8_t side, int8_t slot) {
    Unit* u = trigger_owner(state, side, slot, trigger_uid);
    if (!u || !u->is_alive()) return;
    u->avenge_counter--;
    if (u->avenge_counter <= 0) {
        u->avenge_counter = 4;
    }
}

static void effect_on_play_primitive_painter(CombatState& state, EventQueue&, const Event& event, int32_t trigger_uid, int8_t side, int8_t slot) {
    if (event.source_uid == trigger_uid) return;
    Unit* played = event_source_unit(state, event);
    if (!played || !(played->types & UnitTypes::MURLOC)) return;
    Unit* u = trigger_owner(state, side, slot, trigger_uid);
    if (!u) return;
    u->perm_atk += 1;
    u->perm_hp += 2;
}

static void effect_rally_the_last_one_standing(CombatState& state, EventQueue&, const Event&, int32_t trigger_uid, int8_t side, int8_t slot) {
    Unit* u = trigger_owner(state, side, slot, trigger_uid);
    if (!u || !u->is_alive()) return;
    u->combat_atk += 12;
    u->combat_hp += 12;
}

static void effect_soc_psychus(CombatState& state, EventQueue&, const Event&, int32_t trigger_uid, int8_t side, int8_t slot) {
    Unit* u = trigger_owner(state, side, slot, trigger_uid);
    if (!u || !u->is_alive()) return;
    auto& board = state.boards[side];
    int16_t best_atk = 0, best_hp = 0;
    for (int i = 0; i < board.count; ++i) {
        Unit& target = board.units[i];
        if (target.is_alive() && target.uid != trigger_uid) {
            if (target.get_atk() > best_atk) {
                best_atk = target.get_atk();
                best_hp = target.get_hp();
            }
        }
    }
    if (best_atk > 0) {
        int16_t gain_atk = std::max<int16_t>(0, best_atk - u->get_atk());
        int16_t gain_hp = std::max<int16_t>(0, best_hp - u->get_hp());
        u->combat_atk += gain_atk;
        u->combat_hp += gain_hp;
    }
}

static void effect_rally_obsidian_ravager(CombatState& state, EventQueue&, const Event&, int32_t trigger_uid, int8_t side, int8_t slot) {
    Unit* u = trigger_owner(state, side, slot, trigger_uid);
    if (!u || !u->is_alive()) return;
    auto& enemy_board = state.boards[1 - side];
    if (enemy_board.count > 0) {
        int idx = rng_index(state.rng, enemy_board.count);
        Unit& victim = enemy_board.units[idx];
        int16_t dmg = u->get_atk();
        if (dmg > 0) {
            if (victim.has_tag(Tags::DIVINE_SHIELD)) {
                victim.remove_tag(Tags::DIVINE_SHIELD);
                fire_unit_event(state, EventType::DIVINE_SHIELD_LOST, victim.uid, 1 - side, idx);
            } else {
                victim.damage_taken += dmg;
                if (victim.get_hp() <= 0) {
                    enemy_board.dead_slot_mask |= (1 << idx);
                    state.has_pending_deaths = true;
                }
            }
        }
    }
}

static void effect_dr_stitched_salvager(CombatState& state, EventQueue& queue, const Event& event, int32_t trigger_uid, int8_t side, int8_t slot) {
    auto& board = state.boards[side];
    for (int i = 0; i < board.count; ++i) {
        Unit& target = board.units[i];
        if (target.is_alive()) {
            target.combat_atk += 4;
            target.combat_hp += 4;
        }
    }
}

static void effect_dr_crab_attached(CombatState& state, EventQueue& queue, const Event& event, int32_t, int8_t, int8_t) {
    int8_t s = -1, sl = -1;
    if (!get_source_pos(event, s, sl)) return;
    summon_unit(state, queue, s, sl, 905, 3, 2, UnitTypes::BEAST, Tags::NONE, 1, false);
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
    { TriggerDef def{EventType::START_OF_COMBAT, cond_friendly_soc, effect_soc_humming_bird}; register_effect_entry(207, &def, 1); }
    { TriggerDef def{EventType::ATTACK_DECLARED, cond_is_self, effect_rally_sleepy_supporter}; register_effect_entry(213, &def, 1); }
    { TriggerDef def{EventType::MINION_DIED, cond_friendly_death, effect_avenge_ghostly_ymirjar}; register_effect_entry(218, &def, 1); }
    { TriggerDef def{EventType::START_OF_COMBAT, cond_friendly_soc, effect_soc_irate_rooster}; register_effect_entry(221, &def, 1); }
    { TriggerDef def{EventType::MINION_DIED, cond_friendly_death, effect_avenge_bird_buddy}; register_effect_entry(301, &def, 1); }
    { TriggerDef def{EventType::MINION_DIED, cond_friendly_death, effect_avenge_budding_greenthumb}; register_effect_entry(302, &def, 1); }
    { TriggerDef def{EventType::MINION_DIED, cond_self_death, effect_dr_cadaver_caretaker}; register_effect_entry(305, &def, 1); }
    { TriggerDef def{EventType::MINION_DIED, cond_self_death, effect_dr_handless_forsaken}; register_effect_entry(307, &def, 1); }
    { TriggerDef def{EventType::ATTACK_DECLARED, cond_is_self, effect_rally_rampager}; register_effect_entry(314, &def, 1); }
    { TriggerDef def{EventType::START_OF_COMBAT, cond_friendly_soc, effect_soc_amber_guardian}; register_effect_entry(317, &def, 1); }
    { TriggerDef def{EventType::MINION_DAMAGED, cond_always, effect_trig_hardy_orca}; register_effect_entry(318, &def, 1); }
    { TriggerDef def{EventType::MINION_SUMMONED, cond_always, effect_trig_jelly_belly}; register_effect_entry(320, &def, 1); }
    { TriggerDef def{EventType::MINION_DIED, cond_self_death, effect_dr_anubarak_nerubian_king}; register_effect_entry(321, &def, 1); }
    { TriggerDef def{EventType::MINION_SUMMONED, cond_always, effect_trig_deflect_o_bot}; register_effect_entry(325, &def, 1); }
    { TriggerDef def{EventType::MINION_PLAYED, cond_always, effect_on_play_peggy_sturdybone}; register_effect_entry(326, &def, 1); }
    { TriggerDef def{EventType::ATTACK_DECLARED, cond_always, effect_trig_roaring_recruiter}; register_effect_entry(328, &def, 1); }
    { TriggerDef def{EventType::ATTACK_DECLARED, cond_is_self, effect_rally_bonker}; register_effect_entry(403, &def, 1); }
    { TriggerDef def{EventType::MINION_DIED, cond_friendly_death, effect_on_death_devout_hellcaller}; register_effect_entry(404, &def, 1); }
    { TriggerDef def{EventType::ATTACK_DECLARED, cond_is_self, effect_rally_heroic_underdog}; register_effect_entry(409, &def, 1); }
    { TriggerDef def{EventType::START_OF_COMBAT, cond_friendly_soc, effect_soc_prized_promo_drake}; register_effect_entry(414, &def, 1); }
    { TriggerDef def{EventType::MINION_DIED, cond_self_death, effect_dr_silent_enforcer}; register_effect_entry(418, &def, 1); }
    { TriggerDef def{EventType::MINION_DIED, cond_self_death, effect_dr_sly_raptor}; register_effect_entry(420, &def, 1); }
    { TriggerDef def{EventType::START_OF_COMBAT, cond_friendly_soc, effect_soc_soulsplitter}; register_effect_entry(421, &def, 1); }
    { TriggerDef def{EventType::MINION_DIED, cond_friendly_death, effect_avenge_spirit_drake}; register_effect_entry(422, &def, 1); }
    { TriggerDef def{EventType::MINION_DIED, cond_self_death, effect_dr_tunnel_blaster}; register_effect_entry(424, &def, 1); }
    { TriggerDef def{EventType::MINION_DIED, cond_friendly_death, effect_avenge_witchwing_nestmatron}; register_effect_entry(426, &def, 1); }
    { TriggerDef def{EventType::ATTACK_DECLARED, cond_is_self, effect_rally_monstrous_macaw}; register_effect_entry(431, &def, 1); }
    { TriggerDef def{EventType::MINION_DIED, cond_self_death, effect_dr_rylak_metalhead}; register_effect_entry(433, &def, 1); }
    { TriggerDef def{EventType::ATTACK_DECLARED, cond_is_self, effect_rally_sunken_advocate}; register_effect_entry(434, &def, 1); }
    { TriggerDef def{EventType::MINION_DAMAGED, cond_always, effect_trig_trigore_the_lasher}; register_effect_entry(436, &def, 1); }
    { TriggerDef def{EventType::MINION_PLAYED, cond_always, effect_on_play_ichoron_the_protector}; register_effect_entry(438, &def, 1); }
    { TriggerDef def{EventType::MINION_DIED, cond_friendly_death, effect_avenge_champion_of_the_primus}; register_effect_entry(506, &def, 1); }
    { TriggerDef def{EventType::START_OF_COMBAT, cond_friendly_soc, effect_soc_corrupted_myrmidon}; register_effect_entry(507, &def, 1); }
    { TriggerDef def{EventType::MINION_DIED, cond_self_death, effect_dr_silithid_burrower}; register_effect_entry(508, &def, 1); }
    { TriggerDef def{EventType::MINION_DIED, cond_friendly_death, effect_avenge_silithid_burrower}; register_effect_entry(508, &def, 1); }
    { TriggerDef def{EventType::MINION_DIED, cond_friendly_death, effect_avenge_ghoul_of_the_feast}; register_effect_entry(509, &def, 1); }
    { TriggerDef def{EventType::ATTACK_DECLARED, cond_always, effect_trig_twilight_watcher}; register_effect_entry(510, &def, 1); }
    { TriggerDef def{EventType::MINION_DAMAGED, cond_always, effect_trig_unforgiving_treant}; register_effect_entry(511, &def, 1); }
    { TriggerDef def{EventType::MINION_PLAYED, cond_always, effect_on_play_nomi_kitchen_nightmare}; register_effect_entry(512, &def, 1); }
    { TriggerDef def{EventType::ATTACK_DECLARED, cond_is_self, effect_rally_bile_spitter}; register_effect_entry(513, &def, 1); }
    { TriggerDef def{EventType::ATTACK_DECLARED, cond_is_self, effect_rally_razorfen_vineweaver}; register_effect_entry(514, &def, 1); }
    { TriggerDef def{EventType::MINION_DIED, cond_self_death, effect_dr_spiked_savior}; register_effect_entry(518, &def, 1); }
    { TriggerDef def{EventType::MINION_DIED, cond_self_death, effect_dr_leeroy_the_reckless}; register_effect_entry(519, &def, 1); }
    { TriggerDef def{EventType::MINION_DIED, cond_friendly_death, effect_avenge_stuntdrake}; register_effect_entry(520, &def, 1); }
    { TriggerDef def{EventType::MINION_DAMAGED, cond_always, effect_trig_iridescent_skyblazer}; register_effect_entry(522, &def, 1); }
    { TriggerDef def{EventType::ATTACK_DECLARED, cond_is_self, effect_rally_niuzao}; register_effect_entry(523, &def, 1); }
    { TriggerDef def{EventType::MINION_DIED, cond_self_death, effect_dr_twilight_broodmother}; register_effect_entry(524, &def, 1); }
    { TriggerDef def{EventType::START_OF_COMBAT, cond_friendly_soc, effect_soc_costume_enthusiast}; register_effect_entry(525, &def, 1); }
    { TriggerDef def{EventType::MINION_DIED, cond_self_death, effect_dr_goldrinn_the_great_wolf}; register_effect_entry(601, &def, 1); }
    { TriggerDef def{EventType::MINION_DIED, cond_self_death, effect_dr_ship_master_eudora}; register_effect_entry(606, &def, 1); }
    { TriggerDef def{EventType::START_OF_COMBAT, cond_friendly_soc, effect_soc_ultraviolet_ascendant}; register_effect_entry(608, &def, 1); }
    { TriggerDef def{EventType::START_OF_COMBAT, cond_friendly_soc, effect_soc_fire_forged_evoker}; register_effect_entry(612, &def, 1); }
    { TriggerDef def{EventType::ATTACK_DECLARED, cond_is_self, effect_rally_sanguine_refiner}; register_effect_entry(613, &def, 1); }
    { TriggerDef def{EventType::ATTACK_DECLARED, cond_is_self, effect_rally_bloodsnout_warlord}; register_effect_entry(614, &def, 1); }
    { TriggerDef def{EventType::MINION_DIED, cond_friendly_death, effect_avenge_deathly_striker}; register_effect_entry(615, &def, 1); }
    { TriggerDef def{EventType::MINION_PLAYED, cond_always, effect_on_play_primitive_painter}; register_effect_entry(620, &def, 1); }
    { TriggerDef def{EventType::ATTACK_DECLARED, cond_is_self, effect_rally_the_last_one_standing}; register_effect_entry(703, &def, 1); }
    { TriggerDef def{EventType::START_OF_COMBAT, cond_friendly_soc, effect_soc_psychus}; register_effect_entry(705, &def, 1); }
    { TriggerDef def{EventType::ATTACK_DECLARED, cond_is_self, effect_rally_obsidian_ravager}; register_effect_entry(706, &def, 1); }
    { TriggerDef def{EventType::MINION_DIED, cond_self_death, effect_dr_stitched_salvager}; register_effect_entry(707, &def, 1); }
    { TriggerDef def{EventType::MINION_DIED, cond_self_death, effect_dr_crab_attached}; register_effect_entry(EffectID::CRAB_DEATHRATTLE, &def, 1); }

    finalize_effect_table();
}
