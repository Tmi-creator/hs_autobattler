// auras.cpp — Data-driven aura system
// Every aura described by a struct, one generic apply function.
// No per-card functions needed.

#include "event_system.h"

// ============================================================
// Aura definition — describes an aura purely as data
// ============================================================
enum class AuraMode : uint8_t {
    NEIGHBOURS,  // Dire Wolf: buff adjacent units
    TYPE_OTHERS, // Warleader/Captain: buff all other units of a type
};

struct AuraDef {
    int16_t   card_id;
    AuraMode  mode;
    TypeBitset target_type;  // relevant for TYPE_OTHERS (0 = any)
    int16_t   atk;           // base bonus (golden = 2x)
    int16_t   hp;            // base bonus (golden = 2x)
};

// ============================================================
// Aura registry — just data, no functions
// ============================================================
// No aura cards in current patch — table empty.
// Add entries here when aura cards are implemented.
static const AuraDef g_aura_table[] = {
    {CardID::INVALID, AuraMode::NEIGHBOURS, UnitTypes::NONE, 0, 0}, // placeholder
};
static constexpr int g_num_auras = 0; // actual count (placeholder doesn't count)

// Используется в CombatBoard::insert_at для поддержания aura_source_mask —
// набора слотов, в которых сидит юнит-источник ауры. Линейный скан короткой
// таблицы (≤ 10 карт ожидается), вызывается только при вставке юнита.
bool is_aura_source(int16_t card_id) {
    for (int a = 0; a < g_num_auras; ++a) {
        if (g_aura_table[a].card_id == card_id) return true;
    }
    return false;
}

// ============================================================
// Generic aura applicator
// ============================================================
static void apply_aura(const AuraDef& def, const Unit& source, CombatBoard& board, int source_idx) {
    int16_t atk = def.atk * (source.is_golden ? 2 : 1);
    int16_t hp  = def.hp  * (source.is_golden ? 2 : 1);

    switch (def.mode) {
        case AuraMode::NEIGHBOURS:
            if (source_idx > 0) {
                board.units[source_idx - 1].aura_atk += atk;
                board.units[source_idx - 1].aura_hp  += hp;
            }
            if (source_idx < board.count - 1) {
                board.units[source_idx + 1].aura_atk += atk;
                board.units[source_idx + 1].aura_hp  += hp;
            }
            break;

        case AuraMode::TYPE_OTHERS:
            for (int i = 0; i < board.count; ++i) {
                if (i == source_idx) continue;
                if (def.target_type == UnitTypes::NONE || (board.units[i].types & def.target_type)) {
                    board.units[i].aura_atk += atk;
                    board.units[i].aura_hp  += hp;
                }
            }
            break;
    }
}

// ============================================================
// recalculate_board_auras — wipe & reapply for one board
// ============================================================
void recalculate_board_auras(CombatBoard& board) {
    // Fast path: нет ни одного аура-источника на борде — ни сбрасывать,
    // ни применять нечего. aura_source_mask поддерживается в insert_at/remove_at
    // и в recalculate_subscribers (для pybind parse_board path).
    if (board.aura_source_mask == 0) return;

    // 1. Reset all aura buffs
    for (int i = 0; i < board.count; ++i) {
        board.units[i].reset_aura_buffs();
    }

    // 2. Apply auras — итерируем только по слотам с источниками через ctz.
    uint8_t mask = board.aura_source_mask;
    while (mask) {
        const int i = __builtin_ctz(mask);
        mask &= static_cast<uint8_t>(mask - 1);
        for (int a = 0; a < g_num_auras; ++a) {
            if (g_aura_table[a].card_id == board.units[i].card_id) {
                apply_aura(g_aura_table[a], board.units[i], board, i);
            }
        }
    }

    // Note: attached auras not implemented yet (no current card uses them in combat)
}
