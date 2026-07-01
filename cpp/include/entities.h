#pragma once
// entities.h — POD game state structures
// ALL structs must be trivially copyable (no heap, no pointers, no std::vector)

#include <array>
#include <cassert>
#include <cstdint>
#include <type_traits>
#include "types.h"
#include "rng.h"

// ============================================================
// Attached Effect: effect_id + stack count
// ============================================================
struct AttachedEffect {
    int16_t effect_id = 0;
    int16_t count = 0;
};

// ============================================================
// Unit — single minion on board
// ============================================================
struct alignas(8) Unit {
    int16_t  card_id   = CardID::INVALID;
    int32_t  uid       = 0;
    TypeBitset types   = 0;     // 0 = typeless unit (some exist in BG)
    TagBitset  tags    = 0;     // 0 = no keywords
    bool     is_golden = false;
    int8_t   tier      = 1;

    // Base stats (from card definition)
    int16_t atk_base = 0;
    int16_t hp_base  = 0;

    // 4-scope buff system (matches Python entities.py)
    int16_t perm_atk = 0, perm_hp = 0;       // Permanent (Blood Gem, Nomi)
    int16_t turn_atk = 0, turn_hp = 0;       // Reset each tavern turn
    int16_t combat_atk = 0, combat_hp = 0;   // Reset each combat
    int16_t aura_atk = 0, aura_hp = 0;       // Wipe & Reapply

    // Урон, нанесённый юниту. HP = base + buffs - damage_taken.
    // Отдельное поле нужно потому что heal уменьшает damage_taken, а не баффает hp.
    int16_t damage_taken = 0;

    // Avenge counter (for avenge-mechanic cards)
    int8_t avenge_counter = 0;

    // Killer UID (tracks who killed this unit)
    int32_t killer_uid = 0;


    // Attached effects (3 scopes, fixed-size)
    uint8_t num_perm = 0, num_turn = 0, num_combat = 0;
    std::array<AttachedEffect, GameConst::MAX_ATTACHED> attached_perm{};
    std::array<AttachedEffect, GameConst::MAX_ATTACHED> attached_turn{};
    std::array<AttachedEffect, GameConst::MAX_ATTACHED> attached_combat{};

    // ---------- Computed stats ----------
    int16_t get_atk() const {
        return atk_base + perm_atk + turn_atk + combat_atk + aura_atk;
    }

    int16_t get_hp() const {
        return hp_base + perm_hp + turn_hp + combat_hp + aura_hp - damage_taken;
    }

    bool is_alive() const { return card_id != CardID::INVALID && get_hp() > 0; }
    bool is_empty() const { return card_id == CardID::INVALID; }

    // ---------- Tag helpers ----------
    bool has_tag(TagBitset t) const { return (tags & t) != 0; }
    void set_tag(TagBitset t)      { tags |= t; }
    void remove_tag(TagBitset t)   { tags &= ~t; }

    // ---------- Type helpers ----------
    // Проверяет наличие конкретного типа. Не передавай сюда 0!
    bool has_type(TypeBitset t) const {
        assert(t != 0 && "has_type(0) is meaningless — use has_any_type() instead");
        return (types & t) != 0;
    }

    // Есть ли у юнита хотя бы один тип (false = typeless/neutral-like)
    bool has_any_type() const { return types != 0; }

    // ---------- Scope resets ----------
    void reset_turn_buffs()   { turn_atk = turn_hp = 0; num_turn = 0; }
    void reset_combat_buffs() { combat_atk = combat_hp = 0; num_combat = 0; }
    void reset_aura_buffs()   { aura_atk = aura_hp = 0; }

    // ---------- Attached effect management ----------
    // Crashes on overflow (assert) — better to see the bug than silently lose effects
    static void add_attached(
        std::array<AttachedEffect, GameConst::MAX_ATTACHED>& arr,
        uint8_t& count,
        int16_t effect_id,
        int16_t cnt
    ) {
        // First: try to find existing effect_id and increment count
        for (int i = 0; i < count; ++i) {
            if (arr[i].effect_id == effect_id) {
                arr[i].count += cnt;
                return;
            }
        }
        // New effect — crash if full (indicates a design problem, not normal gameplay)
        assert(count < GameConst::MAX_ATTACHED && "Attached effects overflow! Increase MAX_ATTACHED");
        arr[count++] = {effect_id, cnt};
    }

    // Clear unit to empty state
    void clear() { *this = Unit{}; }
};


// Forward declaration — implemented in event_system.cpp.
// Возвращает 32-битную маску: бит i установлен ⇔ юнит имеет триггер на EventType(i).
// Считается через find_effect_entry(card_id) + обход attached-эффектов всех 3 scope.
uint32_t compute_unit_event_mask(const Unit& unit);

// Forward declaration — implemented in auras.cpp.
// true, если карта является источником ауры (по card_id есть запись в g_aura_table).
// Используется в insert_at для поддержания aura_source_mask и в recalculate_subscribers.
bool is_aura_source(int16_t card_id);

// ============================================================
// Slot-bitmask shift helpers.
//
// CombatBoard держит несколько битмасок, каждая из которых индексируется слотом
// борда: taunt_mask, dead_slot_mask, subscribers[e]. Когда юнит удаляется/вставляется,
// все эти маски должны синхронно сдвигаться вместе с массивом units[].
// Логика сдвига одинаковая для всех масок — вынесена в шаблон чтобы не плодить
// копипасту и чтобы типы масок (uint8_t / uint16_t) были явные.
//
// Инвариант: бит i в любой маске соответствует units[i].
// ============================================================

// При удалении слота idx: биты [0..idx-1] не трогаются, бит idx выпадает,
// биты [idx+1..] сдвигаются на одну позицию вниз.
template <typename Mask>
static inline Mask slot_mask_shift_on_remove(Mask mask, int idx) {
    const Mask keep_low = static_cast<Mask>((1u << idx) - 1);
    const Mask shift_hi = static_cast<Mask>(~((1u << (idx + 1)) - 1));
    return static_cast<Mask>((mask & keep_low) | ((mask & shift_hi) >> 1));
}

// При вставке нового слота в позицию idx: биты [0..idx-1] не трогаются,
// биты [idx..] сдвигаются на одну позицию вверх. Бит idx остаётся свободным —
// выставить его должен caller, если юнит подписан на это событие.
template <typename Mask>
static inline Mask slot_mask_shift_on_insert(Mask mask, int idx) {
    const Mask keep_low = static_cast<Mask>((1u << idx) - 1);
    const Mask shift_hi = static_cast<Mask>(~keep_low);
    return static_cast<Mask>((mask & keep_low) | ((mask & shift_hi) << 1));
}

// ============================================================
// Combat Board — one player's side during combat.
// Группирует юнитов + метаданные одной стороны.
// boards[0] и boards[1] в CombatState — две стороны боя.
//
// Slot-битмаски (taunt_mask, dead_slot_mask, subscribers[e]) дают O(1) ответы
// на "где таунт?", "где труп?", "кто подписан на event e?" — вместо линейного
// скана units[]. Все они поддерживаются incremental в insert_at / remove_at.
// ============================================================
struct CombatBoard {
    uint8_t taunt_mask = 0;
    uint8_t damage = 0;
    uint8_t count = 0;
    // Слоты с юнитами hp <= 0, ждущие уборки. Apply_damage выставляет бит,
    // cleanup_dead итерирует через __builtin_ctz вместо скана всех 7 слотов.
    uint8_t dead_slot_mask = 0;
    // Слоты с юнитами-источниками аур (is_aura_source(card_id)). Если маска == 0,
    // recalculate_board_auras делает мгновенный early-exit. Поддерживается
    // incremental в insert_at/remove_at + полный ребилд в recalculate_subscribers.
    uint8_t aura_source_mask = 0;
    // subscribers[e] — кто из слотов имеет триггер на EventType(e).
    // Полностью перестраивается recalculate_subscribers() в начале combat
    // (после parse_board, который пишет в units[] минуя insert_at).
    uint16_t subscribers[static_cast<int>(EventType::EVENT_TYPE_COUNT)] = {};
    std::array<Unit, GameConst::MAX_BOARD> units{};
    int8_t  tavern_tier = 1;
    int8_t  deathrattle_multiplier = 1; // Заготовка под Baron Rivendare (пока всегда 1)

    // Remove unit at index, shift remaining left
    void remove_at(int idx) {
        assert(idx >= 0 && idx < count && "remove_at: index out of bounds");
        damage -= units[idx].tier;
        for (int i = idx; i < count - 1; ++i) {
            units[i] = units[i + 1];
        }
        units[count - 1].clear();

        // Сдвигаем все slot-маски: бит idx выпадает, старшие биты опускаются на 1.
        for (int e = 0; e < static_cast<int>(EventType::EVENT_TYPE_COUNT); ++e) {
            subscribers[e] = slot_mask_shift_on_remove(subscribers[e], idx);
        }
        taunt_mask       = slot_mask_shift_on_remove(taunt_mask, idx);
        dead_slot_mask   = slot_mask_shift_on_remove(dead_slot_mask, idx);
        aura_source_mask = slot_mask_shift_on_remove(aura_source_mask, idx);

        count--;
    }

    // Insert unit at index, shift remaining right
    void insert_at(int idx, const Unit& unit) {
        assert(count < GameConst::MAX_BOARD && "insert_at: board is full");
        if (idx > count) idx = count;
        for (int i = count; i > idx; --i) {
            units[i] = units[i - 1];
        }
        units[idx] = unit;
        damage += unit.tier;

        // Сдвигаем все slot-маски наверх, освобождая бит idx под нового юнита.
        for (int e = 0; e < static_cast<int>(EventType::EVENT_TYPE_COUNT); ++e) {
            subscribers[e] = slot_mask_shift_on_insert(subscribers[e], idx);
        }
        taunt_mask       = slot_mask_shift_on_insert(taunt_mask, idx);
        dead_slot_mask   = slot_mask_shift_on_insert(dead_slot_mask, idx);
        aura_source_mask = slot_mask_shift_on_insert(aura_source_mask, idx);

        // Выставляем биты нового юнита в соответствующих масках.
        // compute_unit_event_mask даёт 32-битную маску "на какие event types подписан".
        const uint16_t new_bit = static_cast<uint16_t>(1u << idx);
        uint32_t event_mask = compute_unit_event_mask(unit);
        while (event_mask) {
            const int e = __builtin_ctz(event_mask);
            event_mask &= event_mask - 1;
            subscribers[e] |= new_bit;
        }
        if (unit.has_tag(Tags::TAUNT)) {
            taunt_mask |= static_cast<uint8_t>(new_bit);
        }
        if (is_aura_source(unit.card_id)) {
            aura_source_mask |= static_cast<uint8_t>(new_bit);
        }
        // Новый юнит живой → dead_slot_mask не трогаем.

        count++;
    }
};


// ============================================================
// Combat State — everything needed to simulate a single fight
// ============================================================
struct CombatState {
    CombatBoard boards[2];
    RngState rng;
    int32_t next_uid = GameConst::INITIAL_UID;
    int8_t  attacker_idx[2] = {0, 0}; // Next attacker position per side
    // Флаг "есть юниты с hp <= 0, которых надо убрать в cleanup_dead".
    // Ставится в apply_damage когда наносимый урон укладывает victim в 0 hp.
    // Читается в cleanup_dead как fast-path early exit: если false — никто не умирал
    // со времени прошлой уборки, обходить доски смысла нет.
    bool has_pending_deaths = false;
};

// Guarantee memcpy clone works
static_assert(std::is_trivially_copyable_v<CombatState>,
    "CombatState must be trivially copyable for memcpy clone!");

// ============================================================
// Battle Result
// ============================================================
struct BattleResult {
    BattleOutcome outcome = BattleOutcome::NO_END;
    int16_t damage = 0;
};
