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
static const AuraDef g_aura_table[] = {
    // Dire Wolf Alpha: neighbours +1/+0 (golden +2/+0)
    {CardID::DIRE_WOLF_ALPHA,  AuraMode::NEIGHBOURS,  UnitTypes::NONE,   1, 0},
    // Murloc Warleader: other murlocs +2/+0 (golden +4/+0)
    {CardID::MURLOC_WARLEADER, AuraMode::TYPE_OTHERS,  UnitTypes::MURLOC, 2, 0},
    // Southsea Captain: other pirates +1/+1 (golden +2/+2)
    {CardID::SOUTHSEA_CAPTAIN, AuraMode::TYPE_OTHERS,  UnitTypes::PIRATE, 1, 1},
};
static constexpr int g_num_auras = sizeof(g_aura_table) / sizeof(g_aura_table[0]);

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
    // 1. Reset all aura buffs
    for (int i = 0; i < board.count; ++i) {
        board.units[i].reset_aura_buffs();
    }

    // 2. Apply auras — scan board, match card_id against table
    for (int i = 0; i < board.count; ++i) {
        for (int a = 0; a < g_num_auras; ++a) {
            if (g_aura_table[a].card_id == board.units[i].card_id) {
                apply_aura(g_aura_table[a], board.units[i], board, i);
            }
        }
    }

    // Note: attached auras not implemented yet (no current card uses them in combat)
}
