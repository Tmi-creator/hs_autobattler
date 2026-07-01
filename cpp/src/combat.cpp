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
#include "generated_card_db.h"
#include "profiler.h"
#include <immintrin.h>

// ============================================================
// Event-emit helpers.
//
// Паттерн "собрать Event, проверить has_any_subscribers, вызвать process_event"
// повторяется во всех hot-путях (apply_damage, perform_attack). Эти хелперы
// уносят boilerplate и гарантируют, что guard стоит ДО построения Event —
// процесс_event даже не запускается если на данный event_type никто не подписан.
//
// Guard внутри: если нет подписчиков → ранний return без единого store в Event.
// ============================================================

// Event с формой {source_uid, target_uid, value} — используется для урон-like
// событий: OVERKILL, MINION_DAMAGED, DAMAGE_DEALT.
static inline void fire_damage_event(CombatState& state, EventType t,
                                     int32_t src_uid, int32_t tgt_uid, int16_t value) {
    if (!has_any_subscribers(state, t)) return;
    Event e{};
    e.event_type = t;
    e.source_uid = src_uid;
    e.target_uid = tgt_uid;
    e.value = value;
    process_event(state, e);
}


// Event с полными позициями обеих сторон: ATTACK_DECLARED, AFTER_ATTACK.
static inline void fire_attack_event(CombatState& state, EventType t,
                                     int32_t src_uid, int8_t src_side, int8_t src_slot,
                                     int32_t tgt_uid, int8_t tgt_side, int8_t tgt_slot) {
    if (!has_any_subscribers(state, t)) return;
    Event e{};
    e.event_type = t;
    e.source_uid = src_uid;
    e.target_uid = tgt_uid;
    e.source_side = src_side;
    e.source_slot = src_slot;
    e.target_side = tgt_side;
    e.target_slot = tgt_slot;
    process_event(state, e);
}

// ============================================================
// Helper: find random target (taunts first)
// Я безумец и сэкономил такты процессора, теперь оно работает за 4 такта
// ============================================================
static int find_target(const CombatBoard &board, RngState &rng) {
    ProfScope _ps(ProfSection::FIND_TARGET);
    if (board.taunt_mask != 0) {
        // 1 такт: Считаем количество таунтов аппаратно
        int taunt_count = __builtin_popcount(board.taunt_mask);

        // Рандомим индекс нужного таунта от 0 до taunt_count - 1
        int rnd_idx = rng_index(rng, taunt_count);

        // 1-2 такта: Магия BMI2 PDEP (Parallel Bits Deposit)
        // Берем 1, сдвигаем на rnd_idx и "раскидываем" по единицам маски таунтов
        uint32_t isolated_bit = _pdep_u32(1 << rnd_idx, board.taunt_mask);

        // 1 такт: Находим позицию этого единственного бита
        return __builtin_ctz(isolated_bit);
    }

    // Если таунтов нет, просто рандомим среди живых (твоя старая логика)
    return rng_index(rng, board.count);
}

// ============================================================
// Helper: check if battle is over
// ============================================================
static BattleResult check_end(CombatState &state) {
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
        int16_t neg_damage = -(state.boards[1].damage + state.boards[1].tavern_tier);
        return {BattleOutcome::LOSE, neg_damage};
    }
    // Side 0 wins
    int16_t damage = state.boards[0].damage + state.boards[0].tavern_tier;
    return {BattleOutcome::WIN, damage};
}


// ============================================================
// Handle reborn for a unit that just died
// ============================================================
static void handle_reborn(CombatState &state, EventQueue &queue, const Unit &dead, int8_t side, int8_t slot) {
    if (!dead.has_tag(Tags::REBORN)) return;

    auto &board = state.boards[side];
    if (board.count >= GameConst::MAX_BOARD) return;

    Unit reborn_unit{};
    reborn_unit.card_id = dead.card_id;
    reborn_unit.uid = state.next_uid++;
    reborn_unit.types = dead.types;
    // Базовые теги из card DB, а не dead.tags. Это срезает комбат-гранты
    // (DS от Deflect-o-Bot, magnetic-стек и т.п.) — reborn юнит должен вернуться
    // в "чистое" состояние карты. REBORN снимаем — нет двойного перерождения.
    reborn_unit.tags = CardDB::base_tags(dead.card_id) & ~Tags::REBORN;
    reborn_unit.tier = dead.tier;
    reborn_unit.atk_base = dead.atk_base;
    reborn_unit.hp_base = dead.hp_base;
    reborn_unit.is_golden = dead.is_golden;
    reborn_unit.damage_taken = reborn_unit.hp_base - 1; // 1 HP

    int8_t insert_slot = slot;
    if (insert_slot > board.count) insert_slot = board.count;
    board.insert_at(insert_slot, reborn_unit);

    // Emit MINION_SUMMONED (meta = 1 indicates Reborn summon)
    Event e{};
    e.event_type = EventType::MINION_SUMMONED;
    e.source_uid = reborn_unit.uid;
    e.source_side = side;
    e.source_slot = insert_slot;
    e.meta = 1;
    queue.push(e);
}

// ============================================================
// cleanup_dead — remove dead units, fire death triggers
// Mirrors Python: CombatManager.cleanup_dead()
// ============================================================
static void cleanup_dead(CombatState &state) {
    ProfScope _ps(ProfSection::CLEANUP_DEAD);
    // Fast path: если никто не умирал с прошлой уборки, обе доски не трогаем.
    // apply_damage выставляет флаг когда hp уходит в 0. Мы сбросим его ниже.
    if (!state.has_pending_deaths) return;
    state.has_pending_deaths = false;
    ProfScope _pw(ProfSection::CLEANUP_DEAD_WORK);

    for (int p = 0; p < 2; ++p) {
        auto &board = state.boards[p];
        // Итерация через битмаску: O(num_dead) вместо O(count).
        // При удалении слот выбывает вместе с bit'ом в dead_slot_mask (remove_at
        // шифтит маски). При reborn insert_at вставляет юнита, его бит в маске
        // не ставится, биты других шифтятся вверх. В итоге следующий ctz находит
        // следующую смерть или выходит из цикла.
        while (board.dead_slot_mask != 0) {
            const int i = __builtin_ctz(board.dead_slot_mask);
            Unit &unit = board.units[i];

            // Snapshot (используется для sort_triggers и для эффектов, читающих
            // позицию мёртвого юнита через event.snapshot).
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

            // Lazy reborn copy: полная копия Unit (~200 байт с attached-массивами)
            // нужна только если reborn действительно сработает. Для большинства
            // смертей флаг выключен — копия не нужна.
            const bool has_reborn = unit.has_tag(Tags::REBORN);
            Unit dead_copy;
            if (has_reborn) dead_copy = unit;

            // Remove from board (remove_at автоматически шифтит dead_slot_mask,
            // taunt_mask и subscribers, очищая бит i).
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
            death_event.meta = unit.killer_uid;


            int before_count = board.count;

            // Process death triggers
            process_event(state, death_event, extra_triggers, num_extra);

            // Handle reborn after death triggers
            if (has_reborn) {
                EventQueue reborn_queue;
                handle_reborn(state, reborn_queue, dead_copy, static_cast<int8_t>(p), static_cast<int8_t>(i));
                while (!reborn_queue.empty()) {
                    Event &re = reborn_queue.pop();
                    process_event(state, re);
                }
            }

            int units_added = board.count - before_count;
            if (i < state.attacker_idx[p]) {
                state.attacker_idx[p] += units_added;
            }
        }
    }
    recalculate_board_auras(state.boards[0]);
    recalculate_board_auras(state.boards[1]);
}


struct Victim {
    int idx;
    int8_t side;
};

// Наносит урон от source по массиву victims. Используется и для основной атаки
// (attacker -> target + cleave neighbours), и для контр-атаки (target -> attacker).
// Для каждой жертвы: сначала снимается DS (и эмитится DIVINE_SHIELD_LOST),
// иначе damage_taken += dmg, затем poison/venom добивают до 0 HP, а после —
// OVERKILL (если перебор) и MINION_DAMAGED + DAMAGE_DEALT. VENOMOUS сгорает
// после первого успешного применения (одноразовый), POISONOUS остаётся.
// Раньше было лямбдой внутри perform_attack с [&]-захватом — вынесено в static
// ради гарантированного инлайна и того, чтобы функция была видна в профайлере.
static void apply_damage(Unit &source, Victim *targets, int num_targets, CombatState &state) {
    int16_t dmg = source.get_atk();
    if (dmg <= 0) return;
    bool has_poison = source.has_tag(Tags::POISONOUS);
    bool has_venom = source.has_tag(Tags::VENOMOUS);
    bool venom_used = false;

    for (int v = 0; v < num_targets; ++v) {
        Unit &victim = state.boards[targets[v].side].units[targets[v].idx];
        if (!victim.is_alive()) continue;

        if (victim.has_tag(Tags::DIVINE_SHIELD)) {
            // DS поглощает удар полностью: снимаем тэг и эмитим DIVINE_SHIELD_LOST.
            victim.remove_tag(Tags::DIVINE_SHIELD);
            fire_unit_event(state, EventType::DIVINE_SHIELD_LOST,
                            victim.uid, targets[v].side,
                            static_cast<int8_t>(targets[v].idx));
            continue;
        }

        // Обычное попадание: нанести урон, проверить poison/venom, эмитить
        // overkill+damage события.
        const int16_t hp_before = victim.get_hp();
        victim.damage_taken += dmg;

        if (has_poison || has_venom) {
            if (victim.get_hp() > 0) {
                victim.damage_taken += victim.get_hp();  // добить до 0
            }
            if (has_venom) venom_used = true;
        }

        // Если hp ушло в ноль — помечаем слот как труп. cleanup_dead потом
        // подберёт его через dead_slot_mask за O(1) без скана борда.
        if (victim.get_hp() <= 0) {
            victim.killer_uid = source.uid;
            state.has_pending_deaths = true;
            state.boards[targets[v].side].dead_slot_mask |=
                static_cast<uint8_t>(1u << targets[v].idx);
        }


        // Damage events. fire_damage_event проверит has_any_subscribers внутри
        // и пропустит построение Event если никто не слушает.
        if (dmg > hp_before) {
            fire_damage_event(state, EventType::OVERKILL, source.uid, victim.uid,
                              static_cast<int16_t>(dmg - hp_before));
        }
        fire_damage_event(state, EventType::MINION_DAMAGED, source.uid, victim.uid, dmg);
        fire_damage_event(state, EventType::DAMAGE_DEALT,   source.uid, victim.uid, dmg);
    }
    if (venom_used) {
        source.remove_tag(Tags::VENOMOUS);
    }
}

// ============================================================
// perform_attack — single attack with damage, cleave, DS, poison
// Mirrors Python: CombatManager.perform_attack()
// ============================================================
static void perform_attack(CombatState &state, int attacker_side, int attacker_idx, int target_idx) {
    ProfScope _ps(ProfSection::PERFORM_ATTACK);
    auto &atk_board = state.boards[attacker_side];
    auto &def_board = state.boards[1 - attacker_side];
    Unit &attacker = atk_board.units[attacker_idx];
    Unit &target = def_board.units[target_idx];

    const int8_t a_side = static_cast<int8_t>(attacker_side);
    const int8_t d_side = static_cast<int8_t>(1 - attacker_side);
    const int8_t a_slot = static_cast<int8_t>(attacker_idx);
    const int8_t t_slot = static_cast<int8_t>(target_idx);

    fire_attack_event(state, EventType::ATTACK_DECLARED,
                      attacker.uid, a_side, a_slot,
                      target.uid,   d_side, t_slot);

    // Собираем жертв: основная цель + соседи при CLEAVE.
    Victim victims[3];
    int num_victims = 0;
    if (attacker.has_tag(Tags::CLEAVE) && target_idx > 0) {
        victims[num_victims++] = {target_idx - 1, d_side};
    }
    victims[num_victims++] = {target_idx, d_side};
    if (attacker.has_tag(Tags::CLEAVE) && target_idx < def_board.count - 1) {
        victims[num_victims++] = {target_idx + 1, d_side};
    }

    // Attacker → victims, затем counter-attack target → attacker.
    apply_damage(attacker, victims, num_victims, state);
    Victim atk_as_victim = {attacker_idx, a_side};
    apply_damage(target, &atk_as_victim, 1, state);

    fire_attack_event(state, EventType::AFTER_ATTACK,
                      attacker.uid, a_side, a_slot,
                      target.uid,   d_side, t_slot);
}

// ============================================================
// resolve_combat — main combat loop
// ============================================================
BattleResult resolve_combat(CombatState &state) {
    ProfScope _ps(ProfSection::RESOLVE_COMBAT);
    // parse_board в pybind пишет в units[] минуя insert_at — subscribers/taunt_mask
    // после него невалидные. Ребилдим один раз в начале боя.
    recalculate_subscribers(state.boards[0]);
    recalculate_subscribers(state.boards[1]);
    {
        ProfScope _pa(ProfSection::RECALC_AURAS);
        recalculate_board_auras(state.boards[0]);
        recalculate_board_auras(state.boards[1]);
    }

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
                auto &board = state.boards[side];
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
        auto &atk_board = state.boards[attacker_player];
        auto &def_board = state.boards[1 - attacker_player];

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

        Unit &attacker_unit = atk_board.units[atk_idx];
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
