#pragma once
// types.h — Enums and bitsets matching Python enums.py / event_system.py
// All values must correspond 1:1 to the Python side for serialization parity

#include <cstdint>

// ============================================================
// Unit Types — bitset for multi-type units (e.g. ALL = every type)
// Maps to Python: enums.py -> class UnitType(Enum)
// ============================================================
using TypeBitset = uint16_t;

namespace UnitTypes {
    constexpr TypeBitset NONE      = 0;  // Empty bitset = no types (typeless unit)
    constexpr TypeBitset BEAST     = 1 << 0;
    constexpr TypeBitset DRAGON    = 1 << 1;
    constexpr TypeBitset DEMON     = 1 << 2;
    constexpr TypeBitset MURLOC    = 1 << 3;
    constexpr TypeBitset PIRATE    = 1 << 4;
    constexpr TypeBitset ELEMENTAL = 1 << 5;
    constexpr TypeBitset MECH      = 1 << 6;
    constexpr TypeBitset UNDEAD    = 1 << 7;
    constexpr TypeBitset NAGA      = 1 << 8;
    constexpr TypeBitset QUILBOAR  = 1 << 9;
    constexpr TypeBitset NEUTRAL   = 1 << 10;
    constexpr TypeBitset ALL       = (1 << 11) - 1;  // All 11 types
    constexpr int COUNT            = 11;
}

// ============================================================
// Tags — bitset for unit keywords
// 0 = no tags set (valid initial state for a unit with no keywords)
// Maps to Python: enums.py -> class Tags(Enum)
// ============================================================
using TagBitset = uint32_t;

namespace Tags {
    constexpr TagBitset NONE             = 0;  // Empty bitset = no keywords
    constexpr TagBitset IMMEDIATE_ATTACK = 1 << 0;
    constexpr TagBitset TAUNT            = 1 << 1;
    constexpr TagBitset DIVINE_SHIELD    = 1 << 2;
    constexpr TagBitset WINDFURY         = 1 << 3;
    constexpr TagBitset POISONOUS        = 1 << 4;
    constexpr TagBitset REBORN           = 1 << 5;
    constexpr TagBitset VENOMOUS         = 1 << 6;
    constexpr TagBitset CLEAVE           = 1 << 7;
    constexpr TagBitset STEALTH          = 1 << 8;
    constexpr TagBitset MAGNETIC         = 1 << 9;
}

// ============================================================
// Battle Outcome
// Maps to Python: enums.py -> class BattleOutcome(Enum)
// ============================================================
enum class BattleOutcome : int8_t {
    NO_END = 0,
    DRAW   = 1,
    WIN    = 2,   // Player 0 wins
    LOSE   = 3    // Player 1 wins
};

// ============================================================
// Event Types — explicit values for clarity
// Maps to Python: event_system.py -> class EventType(Enum)
// ============================================================
enum class EventType : uint8_t {
    MINION_PLAYED      = 1,
    MINION_BOUGHT      = 2,
    MINION_SOLD        = 3,
    MINION_SUMMONED    = 4,
    MINION_DIED        = 5,
    MINION_DAMAGED     = 6,
    DAMAGE_DEALT       = 7,
    ATTACK_DECLARED    = 8,
    AFTER_ATTACK       = 9,
    START_OF_COMBAT    = 10,
    END_OF_COMBAT      = 11,
    START_OF_TURN      = 12,
    END_OF_TURN        = 13,
    SPELL_CAST         = 14,
    MINION_ADDED_TO_SHOP = 15,
    DIVINE_SHIELD_LOST = 16,
    OVERKILL           = 17,
    EVENT_TYPE_COUNT   = 18  // sentinel for array sizing
};

// ============================================================
// Card IDs + Effect IDs — AUTO-GENERATED, do not edit manually
// Run: python scripts/generate_cpp_effects.py
// ============================================================
#include "generated_card_ids.h"

// Spell IDs (not generated yet — small static set)
namespace SpellID {
    constexpr int16_t TAVERN_COIN    = 1001;
    constexpr int16_t BANANA         = 1002;
    constexpr int16_t BLOOD_GEM      = 1003;
    constexpr int16_t POINTY_ARROW   = 1004;
    constexpr int16_t FORTIFY        = 1005;
    constexpr int16_t APPLE          = 1006;
    constexpr int16_t SURF_SPELLCRAFT = 1007;
    constexpr int16_t TRIPLET_REWARD = 1999;
}

// ============================================================
// Game constants
// ============================================================
namespace GameConst {
    constexpr int MAX_BOARD    = 7;
    constexpr int MAX_HAND     = 10;
    constexpr int MAX_STORE    = 7;
    constexpr int MAX_DISCOVER = 3;
    constexpr int MAX_ATTACHED = 16; // can try 4 or 8 to even faster combats, but may crash
    constexpr int MAX_CARDS    = 512;
    constexpr int COST_BUY     = 3;
    constexpr int COST_REROLL  = 1;

    // Event system
    constexpr int MAX_EVENT_QUEUE       = 512;  // BFS queue per process_event() call
    constexpr int MAX_EFFECT_ENTRIES    = 512;  // Global effect table size
    constexpr int MAX_TRIGGERS_PER_EVENT = 64;  // Max triggers for one event
    constexpr int MAX_TRIGGERS_PER_CARD  = 4;   // Max trigger defs per card/effect
    constexpr int MAX_SYSTEM_TRIGGERS    = 16;  // Max system triggers per event type

    // Размер плоского direct-index массива для find_effect_entry.
    // Плотнее чем реально используемые id'шники (максимум ~5001), но round-up
    // степень двойки даёт compiler дешёвый bounds check. Статика: 8192 × 8B = 64KB.
    constexpr int EFFECT_INDEX_SIZE = 8192;

    // Стартовое значение next_uid в свежем CombatState. Выше 9999 чтобы
    // не пересекаться с UID'ами, которые Python-сторона использует для своих
    // объектов до передачи в C++. Инкрементируется parse_board и summon_unit.
    constexpr int32_t INITIAL_UID = 10000;
}
