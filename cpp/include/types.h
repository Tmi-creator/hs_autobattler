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
// Card IDs — numeric mapping of Python string IDs
// Python: CardIDs.WRATH_WEAVER = "101" -> C++: 101
// Tokens: 900+ range to avoid collisions
// ============================================================
namespace CardID {
    constexpr int16_t INVALID         = -1;

    // --- Tier 1 ---
    constexpr int16_t WRATH_WEAVER    = 101;
    constexpr int16_t ALLEYCAT        = 102;
    constexpr int16_t SCALLYWAG       = 103;
    constexpr int16_t SWAMPSTRIKER    = 104;
    constexpr int16_t ANNOY_O_TRON    = 105;
    constexpr int16_t SHELL_COLLECTOR = 107;
    constexpr int16_t IMPRISONER      = 108;
    constexpr int16_t MINTED_CORSAIR  = 109;
    constexpr int16_t FLIGHTY_SCOUT   = 110;
    constexpr int16_t DIRE_WOLF_ALPHA = 111;

    // --- Tier 2 ---
    constexpr int16_t LEAPFROGGER     = 201;
    constexpr int16_t MOLTEN_ROCK     = 202;
    constexpr int16_t MURLOC_WARLEADER = 203;
    constexpr int16_t SOUTHSEA_CAPTAIN = 204;
    constexpr int16_t ANNOY_O_MODULE  = 205;
    constexpr int16_t SPAWN_OF_NZOTH  = 206;
    constexpr int16_t KABOOM_BOT      = 207;

    // --- Tier 3 ---
    constexpr int16_t DEFLECT_O_BOT   = 301;

    // --- Tokens (900+ range) ---
    constexpr int16_t TABBYCAT        = 901;
    constexpr int16_t PIRATE_TOKEN    = 902;
    constexpr int16_t IMP_TOKEN       = 903;
    constexpr int16_t CRAB_TOKEN      = 904;
}

// ============================================================
// Spell IDs
// Python: SpellIDs.TAVERN_COIN = "S001" -> C++: 1001
// ============================================================
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
// EffectIDs — categorized by type via ID range:
//   5000+ = attached/spell effects
//   6000+ = deathrattles (6XNN where X = tier, NN = card)
//   7000+ = battlecries / on-play triggers
//   8000+ = combat-phase triggers
// По ID сразу видно тип эффекта.
namespace EffectID {
    constexpr int16_t NONE                    = 0;

    // --- Attached / Spell effects (5000+) ---
    constexpr int16_t CRAB_DEATHRATTLE        = 5001;

    // --- Deathrattles (6000+) ---
    constexpr int16_t SCALLYWAG_DR            = 6103;
    constexpr int16_t IMPRISONER_DR           = 6108;
    constexpr int16_t LEAPFROGGER_DR          = 6201;
    constexpr int16_t LEAPFROGGER_DR_GOLDEN   = 6202;
    constexpr int16_t LEAPFROGGER_JUMP        = 6203;
    constexpr int16_t LEAPFROGGER_JUMP_GOLDEN = 6204;
    constexpr int16_t SPAWN_OF_NZOTH_DR       = 6206;
    constexpr int16_t KABOOM_BOT_DR           = 6207;

    // --- Battlecries / On-play (7000+) ---
    constexpr int16_t SHELL_COLLECTOR_BC      = 7107;
    constexpr int16_t ALLEYCAT_BC             = 7102;
    constexpr int16_t ALLEYCAT_BC_GOLDEN      = 7103;
    constexpr int16_t WRATH_WEAVER_TRIGGER    = 7101;
    constexpr int16_t SWAMPSTRIKER_TRIGGER    = 7104;
    constexpr int16_t MINTED_CORSAIR_SELL     = 7109;

    // --- Combat triggers (8000+) ---
    constexpr int16_t DEFLECT_O_BOT_TRIGGER   = 8301;
}

// ============================================================
// Game constants
// ============================================================
namespace GameConst {
    constexpr int MAX_BOARD    = 7;
    constexpr int MAX_HAND     = 10;
    constexpr int MAX_STORE    = 7;
    constexpr int MAX_DISCOVER = 3;
    constexpr int MAX_ATTACHED = 16;
    constexpr int MAX_CARDS    = 512;
    constexpr int COST_BUY     = 3;
    constexpr int COST_REROLL  = 1;

    // Event system
    constexpr int MAX_EVENT_QUEUE       = 512;  // BFS queue per process_event() call
    constexpr int MAX_EFFECT_ENTRIES    = 128;  // Global effect table size
    constexpr int MAX_TRIGGERS_PER_EVENT = 64;  // Max triggers for one event
    constexpr int MAX_TRIGGERS_PER_CARD  = 4;   // Max trigger defs per card/effect
    constexpr int MAX_SYSTEM_TRIGGERS    = 16;  // Max system triggers per event type
}
