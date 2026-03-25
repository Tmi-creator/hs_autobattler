// combat.cpp — Main combat resolution loop
// Ports Python combat.py: CombatManager.resolve_combat()
//
// Flow:
// 1. Recalculate auras
// 2. Determine first attacker (larger board, or coin flip)
// 3. Fire START_OF_COMBAT
// 4. Main loop:
//    a. Check end of battle
//    b. Process IMMEDIATE_ATTACK queue
//    c. Find next attacker (skip 0-atk units)
//    d. Perform attack (damage, cleave, DS, poison)
//    e. Cleanup dead units (collect death triggers → process_event)
//    f. Switch sides
// 5. Fire END_OF_COMBAT

#include "event_system.h"

// ============================================================
// Helper: find a unit by UID on a specific side
// ============================================================
static int find_unit_idx(const CombatBoard& board, int32_t uid) {
    for (int i = 0; i < board.count; ++i) {
        if (board.units[i].uid == uid) return i;
    }
    return -1;
}

// ============================================================
// Helper: find random target (taunts first)
// ============================================================
static int find_target(const CombatBoard& board, RngState& rng) {
    // Collect taunt indices
    int taunts[GameConst::MAX_BOARD];
    int num_taunts = 0;
    for (int i = 0; i < board.count; ++i) {
        if (board.units[i].has_tag(Tags::TAUNT)) {
            taunts[num_taunts++] = i;
        }
    }
    if (num_taunts > 0) {
        return taunts[rng_index(rng, num_taunts)];
    }
    return rng_index(rng, board.count);
}

// ============================================================
// Helper: check if battle is over
// ============================================================
static BattleResult check_end(CombatState& state) {
    bool b0_alive = state.boards[0].count > 0;
    bool b1_alive = state.boards[1].count > 0;

    if (b0_alive && b1_alive) {
        return {BattleOutcome::NO_END, 0};
    }
    if (!b0_alive && !b1_alive) {
        return {BattleOutcome::DRAW, 0};
    }
    if (!b0_alive) {
        // Side 1 wins
        int16_t damage = state.boards[1].tavern_tier;
        for (int i = 0; i < state.boards[1].count; ++i) {
            damage += state.boards[1].units[i].tier;
        }
        int16_t neg_damage = -damage;
        return {BattleOutcome::LOSE, neg_damage};
    }
    // Side 0 wins
    int16_t damage = state.boards[0].tavern_tier;
    for (int i = 0; i < state.boards[0].count; ++i) {
        damage += state.boards[0].units[i].tier;
    }
    return {BattleOutcome::WIN, damage};
}


// ============================================================
// Handle reborn for a unit that just died
// ============================================================
static void handle_reborn(CombatState& state, EventQueue& queue, const Unit& dead, int8_t side, int8_t slot) {
    if (!dead.has_tag(Tags::REBORN)) return;

    auto& board = state.boards[side];
    if (board.count >= GameConst::MAX_BOARD) return;

    Unit reborn_unit{};
    reborn_unit.card_id = dead.card_id;
    reborn_unit.uid = state.next_uid++;
    reborn_unit.types = dead.types;
    reborn_unit.tags = dead.tags;
    reborn_unit.remove_tag(Tags::REBORN); // no double reborn
    reborn_unit.tier = dead.tier;
    reborn_unit.atk_base = dead.atk_base;
    reborn_unit.hp_base = dead.hp_base;
    reborn_unit.is_golden = dead.is_golden;
    // Reborn only keeps base tags from the card definition.
    // Combat-granted tags (e.g. DS from Deflect-o-Bot) are stripped.
    // TODO: when card DB exists, use base tags from DB instead of dead.tags
    reborn_unit.damage_taken = reborn_unit.hp_base - 1; // 1 HP

    int8_t insert_slot = slot;
    if (insert_slot > board.count) insert_slot = board.count;
    board.insert_at(insert_slot, reborn_unit);

    // Emit MINION_SUMMONED
    Event e{};
    e.event_type = EventType::MINION_SUMMONED;
    e.source_uid = reborn_unit.uid;
    e.source_side = side;
    e.source_slot = insert_slot;
    queue.push(e);
}

// ============================================================
// cleanup_dead — remove dead units, fire death triggers
// Mirrors Python: CombatManager.cleanup_dead()
// ============================================================
static void cleanup_dead(CombatState& state) {
    for (int p = 0; p < 2; ++p) {
        auto& board = state.boards[p];
        int i = 0;
        while (i < board.count) {
            Unit& unit = board.units[i];

            if (!unit.is_alive()) {
                // Snapshot before removal
                MinionSnapshot snap{};
                snap.uid = unit.uid;
                snap.card_id = unit.card_id;
                snap.side = static_cast<int8_t>(p);
                snap.slot = static_cast<int8_t>(i);
                snap.atk = unit.get_atk();
                snap.hp = unit.get_hp();
                snap.types = unit.types;
                snap.tags = unit.tags;
                snap.valid = true;

                // Pre-collect death triggers before removing from board
                TriggerInstance extra_triggers[GameConst::MAX_TRIGGERS_PER_EVENT];
                int num_extra = collect_unit_triggers(
                    unit, EventType::MINION_DIED,
                    -1, -1, // side/slot filled from snapshot during sort
                    extra_triggers, GameConst::MAX_TRIGGERS_PER_EVENT
                );

                bool has_reborn = unit.has_tag(Tags::REBORN);
                Unit dead_copy = unit; // save for reborn

                // Remove from board
                board.remove_at(i);
                recalculate_board_auras(board);

                // Adjust attack index
                if (i < state.attacker_idx[p]) {
                    state.attacker_idx[p]--;
                }

                // Build death event
                Event death_event{};
                death_event.event_type = EventType::MINION_DIED;
                death_event.source_uid = snap.uid;
                death_event.source_side = snap.side;
                death_event.source_slot = snap.slot;
                death_event.snapshot = snap;

                int before_count = board.count;

                // Process death triggers
                process_event(state, death_event, extra_triggers, num_extra);

                // Handle reborn after death triggers
                // Use a temporary queue for reborn summon event
                if (has_reborn) {
                    // Reborn summons at the death slot
                    EventQueue reborn_queue;
                    handle_reborn(state, reborn_queue, dead_copy, static_cast<int8_t>(p), static_cast<int8_t>(i));
                    // Process any events from reborn (MINION_SUMMONED)
                    while (!reborn_queue.empty()) {
                        Event& re = reborn_queue.pop();
                        process_event(state, re);
                    }
                }

                int units_added = board.count - before_count;

                // Adjust attack index for summoned units
                if (i < state.attacker_idx[p]) {
                    state.attacker_idx[p] += units_added;
                }
                i += units_added;
            } else {
                i++;
            }
        }
    }
    recalculate_board_auras(state.boards[0]);
    recalculate_board_auras(state.boards[1]);
}

// ============================================================
// perform_attack — single attack with damage, cleave, DS, poison
// Mirrors Python: CombatManager.perform_attack()
// ============================================================
static void perform_attack(CombatState& state, int attacker_side, int attacker_idx, int target_idx) {
    auto& atk_board = state.boards[attacker_side];
    auto& def_board = state.boards[1 - attacker_side];
    Unit& attacker = atk_board.units[attacker_idx];
    Unit& target = def_board.units[target_idx];

    int8_t a_side = static_cast<int8_t>(attacker_side);
    int8_t d_side = static_cast<int8_t>(1 - attacker_side);

    // ATTACK_DECLARED event
    {
        Event e{};
        e.event_type = EventType::ATTACK_DECLARED;
        e.source_uid = attacker.uid;
        e.target_uid = target.uid;
        e.source_side = a_side;
        e.source_slot = static_cast<int8_t>(attacker_idx);
        e.target_side = d_side;
        e.target_slot = static_cast<int8_t>(target_idx);
        process_event(state, e);
    }

    // Collect victims (main target + cleave neighbours)
    struct Victim { int idx; int8_t side; };
    Victim victims[3];
    int num_victims = 0;

    // Cleave: left neighbour
    if (attacker.has_tag(Tags::CLEAVE) && target_idx > 0) {
        victims[num_victims++] = {target_idx - 1, d_side};
    }
    // Main target
    victims[num_victims++] = {target_idx, d_side};
    // Cleave: right neighbour
    if (attacker.has_tag(Tags::CLEAVE) && target_idx < def_board.count - 1) {
        victims[num_victims++] = {target_idx + 1, d_side};
    }

    // Apply damage: attacker → victims
    auto apply_damage = [&](Unit& source, int8_t src_side, int src_idx,
                            Victim* targets, int num_targets) {
        int16_t dmg = source.get_atk();
        if (dmg <= 0) return;
        bool has_poison = source.has_tag(Tags::POISONOUS);
        bool has_venom = source.has_tag(Tags::VENOMOUS);
        bool venom_used = false;

        for (int v = 0; v < num_targets; ++v) {
            Unit& victim = state.boards[targets[v].side].units[targets[v].idx];
            if (!victim.is_alive()) continue;

            if (victim.has_tag(Tags::DIVINE_SHIELD)) {
                victim.remove_tag(Tags::DIVINE_SHIELD);
                Event e{};
                e.event_type = EventType::DIVINE_SHIELD_LOST;
                e.source_uid = victim.uid;
                e.source_side = targets[v].side;
                e.source_slot = static_cast<int8_t>(targets[v].idx);
                process_event(state, e);
            } else {
                int16_t hp_before = victim.get_hp();
                victim.damage_taken += dmg;
                if (has_poison || has_venom) {
                    if (victim.get_hp() > 0) {
                        // Poison/venom: set HP to 0
                        victim.damage_taken += victim.get_hp();
                    }
                    if (has_venom) venom_used = true;
                }

                int16_t actual = dmg;
                if (actual > 0 && actual > hp_before) {
                    Event e{};
                    e.event_type = EventType::OVERKILL;
                    e.source_uid = source.uid;
                    e.target_uid = victim.uid;
                    e.value = actual - hp_before;
                    process_event(state, e);
                }

                if (actual > 0) {
                    Event e{};
                    e.event_type = EventType::MINION_DAMAGED;
                    e.source_uid = source.uid;
                    e.target_uid = victim.uid;
                    e.value = actual;
                    process_event(state, e);

                    Event e2{};
                    e2.event_type = EventType::DAMAGE_DEALT;
                    e2.source_uid = source.uid;
                    e2.target_uid = victim.uid;
                    e2.value = actual;
                    process_event(state, e2);
                }
            }
        }
        if (venom_used) {
            source.remove_tag(Tags::VENOMOUS);
        }
    };

    // Attacker damages victims
    apply_damage(attacker, a_side, attacker_idx, victims, num_victims);

    // Target damages attacker (counter-attack)
    Victim atk_as_victim = {attacker_idx, a_side};
    apply_damage(target, d_side, target_idx, &atk_as_victim, 1);

    // AFTER_ATTACK event
    {
        Event e{};
        e.event_type = EventType::AFTER_ATTACK;
        e.source_uid = attacker.uid;
        e.target_uid = target.uid;
        e.source_side = a_side;
        e.source_slot = static_cast<int8_t>(attacker_idx);
        e.target_side = d_side;
        e.target_slot = static_cast<int8_t>(target_idx);
        process_event(state, e);
    }
}

// ============================================================
// resolve_combat — main combat loop
// ============================================================
BattleResult resolve_combat(CombatState& state) {
    // Init auras
    recalculate_board_auras(state.boards[0]);
    recalculate_board_auras(state.boards[1]);

    // Determine first attacker: larger board, or coin flip
    int attacker_player;
    if (state.boards[0].count > state.boards[1].count) {
        attacker_player = 0;
    } else if (state.boards[1].count > state.boards[0].count) {
        attacker_player = 1;
    } else {
        attacker_player = rng_index(state.rng, 2);
    }

    // START_OF_COMBAT
    {
        Event e{};
        e.event_type = EventType::START_OF_COMBAT;
        e.source_side = static_cast<int8_t>(attacker_player);
        e.source_slot = -1;
        process_event(state, e);
    }
    state.attacker_idx[0] = 0;
    state.attacker_idx[1] = 0;
    cleanup_dead(state);

    int can_attack[2] = {1, 1};

    while (true) {
        // Check end
        BattleResult result = check_end(state);
        if (result.outcome != BattleOutcome::NO_END) {
            // Fire END_OF_COMBAT
            Event e{};
            e.event_type = EventType::END_OF_COMBAT;
            process_event(state, e);
            return result;
        }

        if (can_attack[0] == 0 && can_attack[1] == 0) {
            return {BattleOutcome::DRAW, 0};
        }

        if (can_attack[attacker_player] == 0) {
            attacker_player = 1 - attacker_player;
            continue;
        }

        // 1. Immediate Attack batch
        while (true) {
            bool found_immediate = false;
            // Scan both sides, active player first
            int scan_order[2] = {attacker_player, 1 - attacker_player};
            for (int si = 0; si < 2; ++si) {
                int side = scan_order[si];
                auto& board = state.boards[side];
                for (int i = 0; i < board.count; ++i) {
                    if (board.units[i].is_alive() && board.units[i].has_tag(Tags::IMMEDIATE_ATTACK)) {
                        board.units[i].remove_tag(Tags::IMMEDIATE_ATTACK);
                        found_immediate = true;

                        // Find target on enemy side
                        int enemy = 1 - side;
                        if (state.boards[enemy].count == 0) continue;
                        int tgt = find_target(state.boards[enemy], state.rng);

                        perform_attack(state, side, i, tgt);
                        cleanup_dead(state);

                        BattleResult r = check_end(state);
                        if (r.outcome != BattleOutcome::NO_END) {
                            Event e{};
                            e.event_type = EventType::END_OF_COMBAT;
                            process_event(state, e);
                            return r;
                        }
                        // Re-scan from start
                        break;
                    }
                }
                if (found_immediate) break;
            }
            if (!found_immediate) break;
        }

        // 2. Normal attack
        auto& atk_board = state.boards[attacker_player];
        auto& def_board = state.boards[1 - attacker_player];

        if (state.attacker_idx[attacker_player] >= atk_board.count) {
            state.attacker_idx[attacker_player] = 0;
        }

        // Find next unit with atk > 0
        int atk_idx = state.attacker_idx[attacker_player];
        bool make_attack = false;
        for (int tries = 0; tries < atk_board.count; ++tries) {
            if (atk_board.units[atk_idx].get_atk() > 0) {
                make_attack = true;
                break;
            }
            atk_idx++;
            if (atk_idx >= atk_board.count) atk_idx = 0;
        }

        if (!make_attack) {
            can_attack[attacker_player] = 0;
            continue;
        }

        Unit& attacker_unit = atk_board.units[atk_idx];
        int num_attacks = 1;
        if (attacker_unit.has_tag(Tags::WINDFURY)) num_attacks += 1;

        for (int a = 0; a < num_attacks; ++a) {
            if (def_board.count == 0) break;
            int tgt = find_target(def_board, state.rng);

            perform_attack(state, attacker_player, atk_idx, tgt);
            cleanup_dead(state);

            // Re-check if attacker is still alive (might have died from counter-attack)
            if (atk_idx >= atk_board.count || atk_board.units[atk_idx].uid != attacker_unit.uid) {
                break;
            }
            if (!attacker_unit.is_alive()) break;

            BattleResult r = check_end(state);
            if (r.outcome != BattleOutcome::NO_END) {
                Event e{};
                e.event_type = EventType::END_OF_COMBAT;
                process_event(state, e);
                return r;
            }
        }

        // Advance attack index
        if (atk_idx < atk_board.count && atk_board.units[atk_idx].is_alive()) {
            state.attacker_idx[attacker_player] = atk_idx + 1;
        }

        // Switch sides
        attacker_player = 1 - attacker_player;
    }
}
