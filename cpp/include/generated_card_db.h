#pragma once
// generated_card_db.h — AUTO-GENERATED from card_def.py
// DO NOT EDIT MANUALLY — run: python scripts/generate_cpp_effects.py
//
// Base card data used by combat code: tags that a card has INHERENTLY
// by its definition, before any combat/aura/magnetic modifications.
// Needed for reborn / clone effects which must restore the "clean" state.

#include <cstdint>
#include "types.h"

namespace CardDB {

// Base tags of the card by id. Returns Tags::NONE for unknown ids
// (vanilla tokens, tests, etc). Compiler folds the switch into a jump table.
inline constexpr TagBitset base_tags(int16_t card_id) {
    switch (card_id) {
        case 101: return Tags::TAUNT | Tags::DIVINE_SHIELD;
        case 102: return Tags::DIVINE_SHIELD;
        case 103: return Tags::DIVINE_SHIELD;
        case 104: return Tags::DIVINE_SHIELD | Tags::WINDFURY;
        case 114: return Tags::TAUNT | Tags::REBORN;
        case 118: return Tags::WINDFURY;
        case 210: return Tags::TAUNT;
        case 303: return Tags::TAUNT | Tags::DIVINE_SHIELD | Tags::MAGNETIC;
        case 304: return Tags::VENOMOUS;
        case 311: return Tags::DIVINE_SHIELD;
        case 317: return Tags::TAUNT;
        case 318: return Tags::TAUNT;
        case 322: return Tags::TAUNT | Tags::REBORN;
        case 325: return Tags::DIVINE_SHIELD;
        case 327: return Tags::DIVINE_SHIELD;
        case 331: return Tags::MAGNETIC;
        case 332: return Tags::DIVINE_SHIELD;
        case 337: return Tags::CLEAVE;
        case 401: return Tags::MAGNETIC;
        case 402: return Tags::CLEAVE;
        case 407: return Tags::DIVINE_SHIELD;
        case 408: return Tags::DIVINE_SHIELD;
        case 409: return Tags::STEALTH;
        case 410: return Tags::DIVINE_SHIELD;
        case 415: return Tags::REBORN | Tags::MAGNETIC;
        case 418: return Tags::TAUNT;
        case 419: return Tags::DIVINE_SHIELD | Tags::WINDFURY;
        case 421: return Tags::REBORN;
        case 424: return Tags::TAUNT;
        case 425: return Tags::REBORN;
        case 433: return Tags::TAUNT;
        case 438: return Tags::DIVINE_SHIELD;
        case 439: return Tags::DIVINE_SHIELD;
        case 504: return Tags::TAUNT;
        case 511: return Tags::TAUNT;
        case 513: return Tags::VENOMOUS;
        case 516: return Tags::TAUNT;
        case 518: return Tags::TAUNT | Tags::REBORN;
        case 525: return Tags::DIVINE_SHIELD;
        case 616: return Tags::DIVINE_SHIELD | Tags::WINDFURY;
        case 903: return Tags::TAUNT;
        case 906: return Tags::TAUNT;
        default: return Tags::NONE;
    }
}

// Avenge threshold lookup. Returns 0 if card has no Avenge effect.
inline constexpr int8_t avenge_threshold(int16_t card_id) {
    switch (card_id) {
        case 218: return 4;
        case 301: return 1;
        case 302: return 3;
        case 422: return 3;
        case 426: return 3;
        case 506: return 2;
        case 508: return 1;
        case 509: return 1;
        case 520: return 3;
        case 615: return 4;
        default: return 0;
    }
}

} // namespace CardDB
