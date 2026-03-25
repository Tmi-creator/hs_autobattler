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


// ============================================================
// Combat Board — one player's side during combat
// Группирует юнитов + метаданные одной стороны.
// boards[0] и boards[1] в CombatState — две стороны боя.
// ============================================================
struct CombatBoard {
    uint8_t count = 0;
    std::array<Unit, GameConst::MAX_BOARD> units{};
    int8_t  tavern_tier = 1;
    int8_t  deathrattle_multiplier = 1; // Заготовка под Baron Rivendare (пока всегда 1)

    // Remove unit at index, shift remaining left
    void remove_at(int idx) {
        assert(idx >= 0 && idx < count && "remove_at: index out of bounds");
        for (int i = idx; i < count - 1; ++i) {
            units[i] = units[i + 1];
        }
        units[count - 1].clear();
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
        count++;
    }
};


// ============================================================
// Combat State — everything needed to simulate a single fight
// ============================================================
struct CombatState {
    CombatBoard boards[2];
    RngState rng;
    int32_t next_uid = 10000;
    int8_t  attacker_idx[2] = {0, 0}; // Next attacker position per side
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
