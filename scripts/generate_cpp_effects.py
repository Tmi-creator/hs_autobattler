#!/usr/bin/env python3
"""
generate_cpp_effects.py — Generate C++ types.h CardID namespace and
effects.cpp register_all_effects() from Python card_def.py.

Single source of truth: src/hearthstone/engine/card_def.py

Generates:
  cpp/include/generated_card_ids.h   — CardID constants + token IDs
  cpp/src/generated_effects.cpp      — effect functions + register_all_effects()

Usage:
  python scripts/generate_cpp_effects.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from hearthstone.engine.card_def import (  # noqa: E402
    ALL_CARDS,
    CardDef,
    DeathrattleSummon,
    DeathrattleSummonWithTag,
    OnFriendlyDeathBuff,
    OnFriendlyPlayType,
    OnFriendlyPlayTypeDamageHero,
    RallyBuff,
    StartOfCombatBuffSelfByTier,
)
from hearthstone.engine.enums import CardIDs, EffectIDs, Tags, UnitType  # noqa: E402

# ── Mapping Python enums → C++ constants ──────────────────────────────

TYPE_MAP: dict[UnitType, str] = {
    UnitType.BEAST: "UnitTypes::BEAST",
    UnitType.DRAGON: "UnitTypes::DRAGON",
    UnitType.DEMON: "UnitTypes::DEMON",
    UnitType.MURLOC: "UnitTypes::MURLOC",
    UnitType.PIRATE: "UnitTypes::PIRATE",
    UnitType.ELEMENTAL: "UnitTypes::ELEMENTAL",
    UnitType.MECH: "UnitTypes::MECH",
    UnitType.UNDEAD: "UnitTypes::UNDEAD",
    UnitType.NAGA: "UnitTypes::NAGA",
    UnitType.QUILBOAR: "UnitTypes::QUILBOAR",
    UnitType.NEUTRAL: "UnitTypes::NEUTRAL",
    UnitType.ALL: "UnitTypes::ALL",
}

TAG_MAP: dict[Tags, str] = {
    Tags.IMMEDIATE_ATTACK: "Tags::IMMEDIATE_ATTACK",
    Tags.TAUNT: "Tags::TAUNT",
    Tags.DIVINE_SHIELD: "Tags::DIVINE_SHIELD",
    Tags.WINDFURY: "Tags::WINDFURY",
    Tags.POISONOUS: "Tags::POISONOUS",
    Tags.REBORN: "Tags::REBORN",
    Tags.VENOMOUS: "Tags::VENOMOUS",
    Tags.CLEAVE: "Tags::CLEAVE",
    Tags.STEALTH: "Tags::STEALTH",
    Tags.MAGNETIC: "Tags::MAGNETIC",
}


def card_id_to_cpp_int(card_id: str) -> int:
    """Convert Python card_id string to C++ int16_t value."""
    val = card_id.value if isinstance(card_id, CardIDs) else card_id
    if val.startswith("t"):
        # Token: "t001" → 901, "t002" → 902, ...
        return 900 + int(val[1:])
    return int(val)


def card_id_to_cpp_name(card: CardDef) -> str:
    """Convert card name to C++ constant name."""
    import re
    name = card.name.upper().replace(" ", "_").replace("'", "").replace("-", "_")
    name = re.sub(r"[^A-Z0-9_]", "", name)  # strip any remaining non-alphanumeric
    return name


def types_to_cpp(types: list[UnitType]) -> str:
    if not types:
        return "UnitTypes::NONE"
    return " | ".join(TYPE_MAP[t] for t in types)


def tags_to_cpp(tags: set[Tags]) -> str:
    if not tags:
        return "Tags::NONE"
    return " | ".join(TAG_MAP[t] for t in sorted(tags, key=lambda t: t.value))


# ── Generate card IDs header ──────────────────────────────────────────


def generate_card_db_header() -> str:
    """Generate cpp/include/generated_card_db.h with base_tags() and avenge_threshold() lookups."""
    lines = [
        "#pragma once",
        "// generated_card_db.h — AUTO-GENERATED from card_def.py",
        "// DO NOT EDIT MANUALLY — run: python scripts/generate_cpp_effects.py",
        "//",
        "// Base card data used by combat code: tags that a card has INHERENTLY",
        "// by its definition, before any combat/aura/magnetic modifications.",
        "// Needed for reborn / clone effects which must restore the \"clean\" state.",
        "",
        "#include <cstdint>",
        '#include "types.h"',
        "",
        "namespace CardDB {",
        "",
        "// Base tags of the card by id. Returns Tags::NONE for unknown ids",
        "// (vanilla tokens, tests, etc). Compiler folds the switch into a jump table.",
        "inline constexpr TagBitset base_tags(int16_t card_id) {",
        "    switch (card_id) {",
    ]

    for card in ALL_CARDS:
        if not card.tags:
            continue
        cpp_id = card_id_to_cpp_int(card.card_id)
        tags_cpp = tags_to_cpp(card.tags)
        lines.append(f"        case {cpp_id}: return {tags_cpp};")

    lines.extend([
        "        default: return Tags::NONE;",
        "    }",
        "}",
        "",
        "// Avenge threshold lookup. Returns 0 if card has no Avenge effect.",
        "inline constexpr int8_t avenge_threshold(int16_t card_id) {",
        "    switch (card_id) {",
    ])

    for card in ALL_CARDS:
        threshold = 0
        for eff in card.effects:
            if hasattr(eff, "threshold"):
                threshold = eff.threshold
                break
        if threshold > 0:
            cpp_id = card_id_to_cpp_int(card.card_id)
            lines.append(f"        case {cpp_id}: return {threshold};")

    lines.extend([
        "        default: return 0;",
        "    }",
        "}",
        "",
        "} // namespace CardDB",
        "",
    ])

    return "\n".join(lines)



def generate_card_ids_header() -> str:
    lines = [
        "#pragma once",
        "// generated_card_ids.h — AUTO-GENERATED from card_def.py",
        "// DO NOT EDIT MANUALLY — run: python scripts/generate_cpp_effects.py",
        "",
        "#include <cstdint>",
        "",
        "namespace CardID {",
        "    constexpr int16_t INVALID = -1;",
        "",
    ]

    # Group by tier
    tiers: dict[int, list[CardDef]] = {}
    tokens: list[CardDef] = []
    for card in ALL_CARDS:
        if card.is_token:
            tokens.append(card)
        else:
            tiers.setdefault(card.tier, []).append(card)

    for tier in sorted(tiers.keys()):
        lines.append(f"    // --- Tier {tier} ---")
        for card in tiers[tier]:
            cpp_name = card_id_to_cpp_name(card)
            cpp_val = card_id_to_cpp_int(card.card_id)
            lines.append(f"    constexpr int16_t {cpp_name:<30} = {cpp_val};")
        lines.append("")

    lines.append("    // --- Tokens ---")
    for card in tokens:
        cpp_name = card_id_to_cpp_name(card)
        cpp_val = card_id_to_cpp_int(card.card_id)
        lines.append(f"    constexpr int16_t {cpp_name:<30} = {cpp_val};")

    lines.append("}")
    lines.append("")

    # EffectIDs
    lines.append("namespace EffectID {")
    lines.append("    constexpr int16_t NONE = 0;")
    for eid in EffectIDs:
        cpp_name = eid.name
        # Map string effect IDs to numeric: E_DR_CRAB32 → 5001
        lines.append(f"    constexpr int16_t {cpp_name:<30} = 5001;  // {eid.value}")
    lines.append("}")
    lines.append("")

    return "\n".join(lines)


# ── Generate effects.cpp ──────────────────────────────────────────────


def generate_effects_cpp() -> str:
    """Generate C++ effect functions and register_all_effects() for combat-relevant triggers."""

    lines = [
        "// generated_effects.cpp — AUTO-GENERATED from card_def.py",
        "// DO NOT EDIT MANUALLY — run: python scripts/generate_cpp_effects.py",
        "//",
        "// Only combat-relevant effects are generated here.",
        "// Tavern-phase effects (BC, sell) run in Python only.",
        "",
        '#include "event_system.h"',
        '#include "generated_card_db.h"',
        "",
        "// ── Helpers ──",
        "",
        "// Прямой O(1) lookup владельца триггера: side/slot пробрасываются из process_event.",
        "// Сверка uid защищает от случая, когда слот переиспользован (death+reborn в той же позиции).",
        "static Unit* trigger_owner(CombatState& state, int8_t side, int8_t slot, int32_t uid) {",
        "    if (side < 0 || slot < 0) return nullptr;",
        "    auto& board = state.boards[side];",
        "    if (slot >= board.count) return nullptr;",
        "    Unit& u = board.units[slot];",
        "    return u.uid == uid ? &u : nullptr;",
        "}",
        "",
        "// Прямой lookup юнита-источника события через event.source_side/slot.",
        "static Unit* event_source_unit(CombatState& state, const Event& event) {",
        "    if (event.source_side < 0 || event.source_slot < 0) return nullptr;",
        "    auto& board = state.boards[event.source_side];",
        "    if (event.source_slot >= board.count) return nullptr;",
        "    Unit& u = board.units[event.source_slot];",
        "    return u.uid == event.source_uid ? &u : nullptr;",
        "}",
        "",
        "// Lookup юнита на обеих досках по UID, возвращает side/slot",
        "static Unit* find_unit_by_uid(CombatState& state, int32_t uid, int8_t& out_side, int8_t& out_slot) {",
        "    if (uid == 0) return nullptr;",
        "    for (int s = 0; s < 2; ++s) {",
        "        auto& board = state.boards[s];",
        "        for (int i = 0; i < board.count; ++i) {",
        "            if (board.units[i].uid == uid) {",
        "                out_side = s;",
        "                out_slot = i;",
        "                return &board.units[i];",
        "            }",
        "        }",
        "    }",
        "    return nullptr;",
        "}",
        "",
        "static int32_t summon_unit(",
        "    CombatState& state, EventQueue& queue,",
        "    int8_t side, int8_t slot, int16_t card_id,",
        "    int16_t atk, int16_t hp, TypeBitset types, TagBitset tags,",
        "    int8_t tier, bool is_golden",
        ") {",
        "    auto& board = state.boards[side];",
        "    if (board.count >= GameConst::MAX_BOARD) return 0;",
        "    Unit unit{};",
        "    unit.card_id = card_id;",
        "    unit.uid = state.next_uid++;",
        "    unit.types = types;",
        "    unit.tags = tags;",
        "    unit.tier = tier;",
        "    unit.atk_base = atk;",
        "    unit.hp_base = hp;",
        "    unit.is_golden = is_golden;",
        "    unit.avenge_counter = CardDB::avenge_threshold(card_id);",
        "    if (slot < 0) slot = 0;",
        "    if (slot > board.count) slot = board.count;",
        "    board.insert_at(slot, unit);",
        "    Event e{};",
        "    e.event_type = EventType::MINION_SUMMONED;",
        "    e.source_uid = unit.uid;",
        "    e.source_side = side;",
        "    e.source_slot = slot;",
        "    queue.push(e);",
        "    return unit.uid;",
        "}",
        "",
        "static bool get_source_pos(const Event& event, int8_t& side, int8_t& slot) {",
        "    if (event.source_side >= 0) { side = event.source_side; slot = event.source_slot; return true; }",
        "    if (event.snapshot.valid) { side = event.snapshot.side; slot = event.snapshot.slot; return true; }",
        "    return false;",
        "}",
        "",
        "// ── Conditions ──",
        "",
        "static bool cond_is_self(const CombatState&, const Event& e, int32_t uid, int8_t, int8_t) {",
        "    return e.source_uid == uid;",
        "}",
        "",
        "static bool cond_self_death(const CombatState&, const Event& e, int32_t uid, int8_t, int8_t) {",
        "    return e.source_uid == uid;",
        "}",
        "",
        "static bool cond_always(const CombatState&, const Event&, int32_t, int8_t, int8_t) {",
        "    return true;",
        "}",
        "",
        "static bool cond_friendly_death(const CombatState&, const Event& event, int32_t trigger_uid, int8_t owner_side, int8_t) {",
        "    int8_t dead_side = event.source_side >= 0",
        "        ? event.source_side",
        "        : (event.snapshot.valid ? event.snapshot.side : -1);",
        "    if (dead_side < 0 || owner_side < 0) return false;",
        "    return (dead_side == owner_side) && (event.source_uid != trigger_uid);",
        "}",
        "",
        "static bool cond_friendly_soc(const CombatState&, const Event&, int32_t, int8_t owner_side, int8_t) {",
        "    return owner_side >= 0;",
        "}",
        "",
        "// ── Generated effect functions ──",
        "",
    ]

    registrations: list[str] = []

    for card in ALL_CARDS:
        for eff in card.effects:
            cpp_id = card_id_to_cpp_int(card.card_id)
            cpp_name_safe = card_id_to_cpp_name(card).lower()
            eff_type = type(eff).__name__

            # DeathrattleSummon: combat-relevant
            if eff_type == "DeathrattleSummon":
                token = next(c for c in ALL_CARDS if c.card_id == eff.token_id)
                token_cpp = card_id_to_cpp_int(eff.token_id)
                token_types = types_to_cpp(token.types)
                token_tags = tags_to_cpp(token.tags)

                fn_name = f"effect_dr_{cpp_name_safe}"
                lines.append(
                    f"static void {fn_name}(CombatState& state, EventQueue& queue, const Event& event, int32_t, int8_t, int8_t) {{"
                )
                lines.append(f"    int8_t s = -1, sl = -1;")
                lines.append(f"    if (!get_source_pos(event, s, sl)) return;")
                for _ in range(eff.count):
                    lines.append(
                        f"    summon_unit(state, queue, s, sl, {token_cpp}, {token.atk}, {token.hp}, {token_types}, {token_tags}, {token.tier}, false);"
                    )
                lines.append(f"}}")
                lines.append("")

                registrations.append(
                    f"    {{ TriggerDef def{{EventType::MINION_DIED, cond_self_death, {fn_name}}}; register_effect_entry({cpp_id}, &def, 1); }}"
                )

            # DeathrattleSummonWithTag: combat-relevant (Twilight Hatchling)
            elif eff_type == "DeathrattleSummonWithTag":
                token = next(c for c in ALL_CARDS if c.card_id == eff.token_id)
                token_cpp = card_id_to_cpp_int(eff.token_id)
                token_types = types_to_cpp(token.types)
                extra_tag = TAG_MAP[eff.tag]
                base_tags = tags_to_cpp(token.tags)
                combined = f"{base_tags} | {extra_tag}" if token.tags else extra_tag

                fn_name = f"effect_dr_{cpp_name_safe}"
                lines.append(
                    f"static void {fn_name}(CombatState& state, EventQueue& queue, const Event& event, int32_t, int8_t, int8_t) {{"
                )
                lines.append(f"    int8_t s = -1, sl = -1;")
                lines.append(f"    if (!get_source_pos(event, s, sl)) return;")
                for _ in range(eff.count):
                    lines.append(
                        f"    summon_unit(state, queue, s, sl, {token_cpp}, {token.atk}, {token.hp}, {token_types}, {combined}, {token.tier}, false);"
                    )
                lines.append(f"}}")
                lines.append("")

                registrations.append(
                    f"    {{ TriggerDef def{{EventType::MINION_DIED, cond_self_death, {fn_name}}}; register_effect_entry({cpp_id}, &def, 1); }}"
                )

            # OnFriendlyDeathBuff: combat-relevant (Rot Hide Gnoll)
            elif eff_type == "OnFriendlyDeathBuff":
                fn_name = f"effect_on_death_{cpp_name_safe}"
                lines.append(
                    f"static void {fn_name}(CombatState& state, EventQueue&, const Event&, int32_t trigger_uid, int8_t side, int8_t slot) {{"
                )
                lines.append(f"    Unit* u = trigger_owner(state, side, slot, trigger_uid);")
                lines.append(f"    if (!u || !u->is_alive()) return;")
                if eff.atk:
                    lines.append(f"    u->combat_atk += {eff.atk};")
                if eff.hp:
                    lines.append(f"    u->combat_hp += {eff.hp};")
                lines.append(f"}}")
                lines.append("")

                registrations.append(
                    f"    {{ TriggerDef def{{EventType::MINION_DIED, cond_friendly_death, {fn_name}}}; register_effect_entry({cpp_id}, &def, 1); }}"
                )

            # StartOfCombatBuffSelfByTier: combat-relevant (Misfit Dragonling)
            elif eff_type == "StartOfCombatBuffSelfByTier":
                fn_name = f"effect_soc_{cpp_name_safe}"
                lines.append(
                    f"static void {fn_name}(CombatState& state, EventQueue&, const Event&, int32_t trigger_uid, int8_t side, int8_t slot) {{"
                )
                lines.append(f"    Unit* u = trigger_owner(state, side, slot, trigger_uid);")
                lines.append(f"    if (!u || !u->is_alive()) return;")
                lines.append(f"    int16_t tier = state.boards[side].tavern_tier;")
                lines.append(f"    u->combat_atk += tier;")
                lines.append(f"    u->combat_hp += tier;")
                lines.append(f"}}")
                lines.append("")

                registrations.append(
                    f"    {{ TriggerDef def{{EventType::START_OF_COMBAT, cond_friendly_soc, {fn_name}}}; register_effect_entry({cpp_id}, &def, 1); }}"
                )

            # OnFriendlyPlayType: combat-relevant only for Swampstriker (murloc summon in combat)
            elif eff_type == "OnFriendlyPlayType":
                type_mask = TYPE_MAP[eff.trigger_type]
                fn_name = f"effect_on_play_{cpp_name_safe}"
                lines.append(
                    f"static void {fn_name}(CombatState& state, EventQueue&, const Event& event, int32_t trigger_uid, int8_t side, int8_t slot) {{"
                )
                lines.append(f"    if (event.source_uid == trigger_uid) return;")
                lines.append(f"    Unit* played = event_source_unit(state, event);")
                lines.append(f"    if (!played || !(played->types & {type_mask})) return;")
                lines.append(f"    Unit* u = trigger_owner(state, side, slot, trigger_uid);")
                lines.append(f"    if (!u) return;")
                if eff.atk:
                    lines.append(f"    u->perm_atk += {eff.atk};")
                if eff.hp:
                    lines.append(f"    u->perm_hp += {eff.hp};")
                lines.append(f"}}")
                lines.append("")

                registrations.append(
                    f"    {{ TriggerDef def{{EventType::MINION_PLAYED, cond_always, {fn_name}}}; register_effect_entry({cpp_id}, &def, 1); }}"
                )

            # OnFriendlyPlayTypeDamageHero: Wrath Weaver (hero dmg is no-op in combat)
            elif eff_type == "OnFriendlyPlayTypeDamageHero":
                type_mask = TYPE_MAP[eff.trigger_type]
                fn_name = f"effect_on_play_{cpp_name_safe}"
                lines.append(
                    f"static void {fn_name}(CombatState& state, EventQueue&, const Event& event, int32_t trigger_uid, int8_t side, int8_t slot) {{"
                )
                lines.append(f"    if (event.source_uid == trigger_uid) return;")
                lines.append(f"    Unit* played = event_source_unit(state, event);")
                lines.append(f"    if (!played || !(played->types & {type_mask})) return;")
                lines.append(f"    Unit* u = trigger_owner(state, side, slot, trigger_uid);")
                lines.append(f"    if (!u) return;")
                if eff.atk:
                    lines.append(f"    u->perm_atk += {eff.atk};")
                if eff.hp:
                    lines.append(f"    u->perm_hp += {eff.hp};")
                lines.append(f"    // Hero damage ({eff.hero_dmg}) is tavern-only, no-op in combat")
                lines.append(f"}}")
                lines.append("")

                registrations.append(
                    f"    {{ TriggerDef def{{EventType::MINION_PLAYED, cond_always, {fn_name}}}; register_effect_entry({cpp_id}, &def, 1); }}"
                )

            # RallyBuff: combat-relevant (ATTACK_DECLARED, self)
            elif eff_type == "RallyBuff":
                # In C++ combat, Blood Gem mechanic modifiers don't exist — use base 1/1
                buff_atk = eff.atk if not eff.use_blood_gem else 1
                buff_hp = eff.hp if not eff.use_blood_gem else 1
                fn_name = f"effect_rally_{cpp_name_safe}"
                lines.append(
                    f"static void {fn_name}(CombatState& state, EventQueue&, const Event&, int32_t trigger_uid, int8_t side, int8_t slot) {{"
                )
                lines.append(f"    Unit* u = trigger_owner(state, side, slot, trigger_uid);")
                lines.append(f"    if (!u || !u->is_alive()) return;")
                if buff_atk:
                    lines.append(f"    u->combat_atk += {buff_atk};")
                if buff_hp:
                    lines.append(f"    u->combat_hp += {buff_hp};")
                lines.append(f"}}")
                lines.append("")

                registrations.append(
                    f"    {{ TriggerDef def{{EventType::ATTACK_DECLARED, cond_is_self, {fn_name}}}; register_effect_entry({cpp_id}, &def, 1); }}"
                )

            # Avenge: combat-relevant
            elif eff_type in ("AvengeEffect", "AvengeBuffFriendlyTypeGlobal"):
                fn_name = f"effect_avenge_{cpp_name_safe}"
                lines.append(
                    f"static void {fn_name}(CombatState& state, EventQueue& queue, const Event& event, int32_t trigger_uid, int8_t side, int8_t slot) {{"
                )
                lines.append(f"    Unit* u = trigger_owner(state, side, slot, trigger_uid);")
                lines.append(f"    if (!u || !u->is_alive()) return;")
                lines.append(f"    u->avenge_counter--;")
                lines.append(f"    if (u->avenge_counter <= 0) {{")
                lines.append(f"        u->avenge_counter = {eff.threshold};")

                if eff_type == "AvengeBuffFriendlyTypeGlobal":
                    cpp_type = TYPE_MAP[eff.trigger_type]
                    lines.append(f"        auto& board = state.boards[side];")
                    lines.append(f"        for (int i = 0; i < board.count; ++i) {{")
                    lines.append(f"            Unit& target = board.units[i];")
                    lines.append(f"            if (target.is_alive() && (target.types & {cpp_type})) {{")
                    lines.append(f"                target.combat_atk += {eff.atk};")
                    lines.append(f"                target.combat_hp += {eff.hp};")
                    lines.append(f"            }}")
                    lines.append(f"        }}")
                else:
                    if eff.buff_target == "self":
                        lines.append(f"        u->combat_atk += {eff.buff_atk};")
                        lines.append(f"        u->combat_hp += {eff.buff_hp};")
                    elif eff.buff_target == "friendly_type":
                        lines.append(f"        auto& board = state.boards[side];")
                        lines.append(f"        for (int i = 0; i < board.count; ++i) {{")
                        lines.append(f"            Unit& target = board.units[i];")
                        if eff.target_type:
                            cpp_type = TYPE_MAP[eff.target_type]
                            lines.append(f"            if (target.is_alive() && (target.types & {cpp_type})) {{")
                        else:
                            lines.append(f"            if (target.is_alive()) {{")
                        lines.append(f"                target.combat_atk += {eff.buff_atk};")
                        lines.append(f"                target.combat_hp += {eff.buff_hp};")
                        lines.append(f"            }}")
                        lines.append(f"        }}")
                    elif eff.buff_target == "random_friendly_type":
                        lines.append(f"        auto& board = state.boards[side];")
                        lines.append(f"        Unit* candidates[GameConst::MAX_BOARD];")
                        lines.append(f"        int count = 0;")
                        lines.append(f"        for (int i = 0; i < board.count; ++i) {{")
                        lines.append(f"            Unit& target = board.units[i];")
                        if eff.target_type:
                            cpp_type = TYPE_MAP[eff.target_type]
                            lines.append(f"            if (target.is_alive() && (target.types & {cpp_type})) {{")
                        else:
                            lines.append(f"            if (target.is_alive()) {{")
                        lines.append(f"                candidates[count++] = &target;")
                        lines.append(f"            }}")
                        lines.append(f"        }}")
                        lines.append(f"        if (count > 0) {{")
                        lines.append(f"            int idx = rng_index(state.rng, count);")
                        lines.append(f"            candidates[idx]->combat_atk += {eff.buff_atk};")
                        lines.append(f"            candidates[idx]->combat_hp += {eff.buff_hp};")
                        lines.append(f"        }}")
                    elif eff.buff_target == "adjacent":
                        lines.append(f"        auto& board = state.boards[side];")
                        lines.append(f"        for (int adj_idx : {{slot - 1, slot + 1}}) {{")
                        lines.append(f"            if (adj_idx >= 0 && adj_idx < board.count) {{")
                        lines.append(f"                Unit& target = board.units[adj_idx];")
                        lines.append(f"                if (target.is_alive()) {{")
                        lines.append(f"                    target.combat_atk += {eff.buff_atk};")
                        lines.append(f"                    target.combat_hp += {eff.buff_hp};")
                        lines.append(f"                }}")
                        lines.append(f"            }}")
                        lines.append(f"        }}")

                lines.append(f"    }}")
                lines.append(f"}}")
                lines.append("")

                registrations.append(
                    f"    {{ TriggerDef def{{EventType::MINION_DIED, cond_friendly_death, {fn_name}}}; register_effect_entry({cpp_id}, &def, 1); }}"
                )

            # Start of Combat Effects
            elif eff_type in ("StartOfCombatBuffSelf", "StartOfCombatBuffFriendlyType", "StartOfCombatBuffAllFriendlyType", "StartOfCombatGiveFriendlyTypeReborn", "StartOfCombatBuffSelfByHighestAllyAtk", "StartOfCombatBuffSelfByHighestBoardAtk", "StartOfCombatDamageAndBuffAdjacent", "StartOfCombatBuffRandomFriendlyTypeAndDS", "StartOfCombatBuffFriendlyTypeScaling"):
                fn_name = f"effect_soc_{cpp_name_safe}"
                lines.append(
                    f"static void {fn_name}(CombatState& state, EventQueue&, const Event&, int32_t trigger_uid, int8_t side, int8_t slot) {{"
                )
                lines.append(f"    Unit* u = trigger_owner(state, side, slot, trigger_uid);")
                lines.append(f"    if (!u || !u->is_alive()) return;")

                if eff_type == "StartOfCombatBuffSelf":
                    lines.append(f"    u->combat_atk += {eff.atk};")
                    lines.append(f"    u->combat_hp += {eff.hp};")

                elif eff_type in ("StartOfCombatBuffFriendlyType", "StartOfCombatBuffAllFriendlyType"):
                    cpp_type = TYPE_MAP[eff.trigger_type]
                    lines.append(f"    auto& board = state.boards[side];")
                    lines.append(f"    for (int i = 0; i < board.count; ++i) {{")
                    lines.append(f"        Unit& target = board.units[i];")
                    lines.append(f"        if (target.is_alive() && (target.types & {cpp_type})) {{")
                    lines.append(f"            target.combat_atk += {eff.atk};")
                    lines.append(f"            target.combat_hp += {eff.hp};")
                    lines.append(f"        }}")
                    lines.append(f"    }}")

                elif eff_type == "StartOfCombatGiveFriendlyTypeReborn":
                    cpp_type = TYPE_MAP[eff.trigger_type]
                    lines.append(f"    auto& board = state.boards[side];")
                    lines.append(f"    Unit* candidates[GameConst::MAX_BOARD];")
                    lines.append(f"    int count = 0;")
                    lines.append(f"    for (int i = 0; i < board.count; ++i) {{")
                    lines.append(f"        Unit& target = board.units[i];")
                    lines.append(f"        if (target.is_alive() && target.uid != trigger_uid && (target.types & {cpp_type}) && !target.has_tag(Tags::REBORN)) {{")
                    lines.append(f"            candidates[count++] = &target;")
                    lines.append(f"        }}")
                    lines.append(f"    }}")
                    lines.append(f"    if (count > 0) {{")
                    lines.append(f"        int idx = rng_index(state.rng, count);")
                    lines.append(f"        candidates[idx]->set_tag(Tags::REBORN);")
                    lines.append(f"    }}")

                elif eff_type == "StartOfCombatBuffSelfByHighestAllyAtk":
                    lines.append(f"    auto& board = state.boards[side];")
                    lines.append(f"    int16_t max_atk = 0;")
                    lines.append(f"    for (int i = 0; i < board.count; ++i) {{")
                    lines.append(f"        Unit& target = board.units[i];")
                    lines.append(f"        if (target.is_alive() && target.uid != trigger_uid) {{")
                    lines.append(f"            if (target.get_atk() > max_atk) max_atk = target.get_atk();")
                    lines.append(f"        }}")
                    lines.append(f"    }}")
                    lines.append(f"    if (max_atk > 0) u->combat_atk += max_atk;")

                elif eff_type == "StartOfCombatBuffSelfByHighestBoardAtk":
                    lines.append(f"    auto& board = state.boards[side];")
                    lines.append(f"    int16_t best_atk = 0, best_hp = 0;")
                    lines.append(f"    for (int i = 0; i < board.count; ++i) {{")
                    lines.append(f"        Unit& target = board.units[i];")
                    lines.append(f"        if (target.is_alive() && target.uid != trigger_uid) {{")
                    lines.append(f"            if (target.get_atk() > best_atk) {{")
                    lines.append(f"                best_atk = target.get_atk();")
                    lines.append(f"                best_hp = target.get_hp();")
                    lines.append(f"            }}")
                    lines.append(f"        }}")
                    lines.append(f"    }}")
                    lines.append(f"    if (best_atk > 0) {{")
                    lines.append(f"        int16_t gain_atk = std::max<int16_t>(0, best_atk - u->get_atk());")
                    lines.append(f"        int16_t gain_hp = std::max<int16_t>(0, best_hp - u->get_hp());")
                    lines.append(f"        u->combat_atk += gain_atk;")
                    lines.append(f"        u->combat_hp += gain_hp;")
                    lines.append(f"    }}")

                elif eff_type == "StartOfCombatDamageAndBuffAdjacent":
                    lines.append(f"    auto& board = state.boards[side];")
                    lines.append(f"    for (int adj_idx : {{slot - 1, slot + 1}}) {{")
                    lines.append(f"        if (adj_idx >= 0 && adj_idx < board.count) {{")
                    lines.append(f"            Unit& target = board.units[adj_idx];")
                    lines.append(f"            if (target.is_alive()) {{")
                    lines.append(f"                target.damage_taken += {eff.damage};")
                    lines.append(f"                if (target.get_hp() <= 0) {{")
                    lines.append(f"                    board.dead_slot_mask |= (1 << adj_idx);")
                    lines.append(f"                    state.has_pending_deaths = true;")
                    lines.append(f"                }}")
                    lines.append(f"                target.combat_atk += {eff.atk};")
                    lines.append(f"                target.combat_hp += {eff.hp};")
                    lines.append(f"            }}")
                    lines.append(f"        }}")
                    lines.append(f"    }}")

                elif eff_type == "StartOfCombatBuffRandomFriendlyTypeAndDS":
                    cpp_type = TYPE_MAP[eff.trigger_type]
                    lines.append(f"    auto& board = state.boards[side];")
                    lines.append(f"    Unit* candidates[GameConst::MAX_BOARD];")
                    lines.append(f"    int count = 0;")
                    lines.append(f"    for (int i = 0; i < board.count; ++i) {{")
                    lines.append(f"        Unit& target = board.units[i];")
                    lines.append(f"        if (target.is_alive() && target.uid != trigger_uid && (target.types & {cpp_type})) {{")
                    lines.append(f"            candidates[count++] = &target;")
                    lines.append(f"        }}")
                    lines.append(f"    }}")
                    lines.append(f"    if (count > 0) {{")
                    lines.append(f"        int idx = rng_index(state.rng, count);")
                    lines.append(f"        candidates[idx]->combat_atk += {eff.atk};")
                    lines.append(f"        candidates[idx]->combat_hp += {eff.hp};")
                    lines.append(f"        candidates[idx]->set_tag(Tags::DIVINE_SHIELD);")
                    lines.append(f"    }}")

                elif eff_type == "StartOfCombatBuffFriendlyTypeScaling":
                    cpp_type = TYPE_MAP[eff.trigger_type]
                    lines.append(f"    auto& board = state.boards[side];")
                    lines.append(f"    for (int i = 0; i < board.count; ++i) {{")
                    lines.append(f"        Unit& target = board.units[i];")
                    lines.append(f"        if (target.is_alive() && target.uid != trigger_uid && (target.types & {cpp_type})) {{")
                    lines.append(f"            target.combat_atk += {eff.atk};")
                    lines.append(f"            target.combat_hp += {eff.hp};")
                    lines.append(f"        }}")
                    lines.append(f"    }}")

                lines.append(f"}}")
                lines.append("")

                registrations.append(
                    f"    {{ TriggerDef def{{EventType::START_OF_COMBAT, cond_friendly_soc, {fn_name}}}; register_effect_entry({cpp_id}, &def, 1); }}"
                )

            # Deathrattles
            elif eff_type in ("DeathrattleDamageAllMinions", "DeathrattleBuffAllFriendlies", "DeathrattleBuffAllFriendliesGlobal", "DeathrattleBuffFriendlyTypeGlobal", "DeathrattleDestroyKiller", "DeathrattleGiveFriendliesScaling", "DeathrattleSummonTauntToken"):
                fn_name = f"effect_dr_{cpp_name_safe}"
                lines.append(
                    f"static void {fn_name}(CombatState& state, EventQueue& queue, const Event& event, int32_t trigger_uid, int8_t side, int8_t slot) {{"
                )

                if eff_type == "DeathrattleDamageAllMinions":
                    lines.append(f"    for (int s = 0; s < 2; ++s) {{")
                    lines.append(f"        auto& board = state.boards[s];")
                    lines.append(f"        for (int i = 0; i < board.count; ++i) {{")
                    lines.append(f"            Unit& victim = board.units[i];")
                    lines.append(f"            if (victim.is_alive()) {{")
                    lines.append(f"                if (victim.has_tag(Tags::DIVINE_SHIELD)) {{")
                    lines.append(f"                    victim.remove_tag(Tags::DIVINE_SHIELD);")
                    lines.append(f"                    fire_unit_event(state, EventType::DIVINE_SHIELD_LOST, victim.uid, s, i);")
                    lines.append(f"                }} else {{")
                    lines.append(f"                    victim.damage_taken += {eff.damage};")
                    lines.append(f"                    if (victim.get_hp() <= 0) {{")
                    lines.append(f"                        board.dead_slot_mask |= (1 << i);")
                    lines.append(f"                        state.has_pending_deaths = true;")
                    lines.append(f"                    }}")
                    lines.append(f"                }}")
                    lines.append(f"            }}")
                    lines.append(f"        }}")
                    lines.append(f"    }}")

                elif eff_type == "DeathrattleBuffAllFriendlies":
                    lines.append(f"    auto& board = state.boards[side];")
                    lines.append(f"    for (int i = 0; i < board.count; ++i) {{")
                    lines.append(f"        Unit& target = board.units[i];")
                    lines.append(f"        if (target.is_alive()) {{")
                    lines.append(f"            target.combat_atk += {eff.atk};")
                    lines.append(f"            target.combat_hp += {eff.hp};")
                    lines.append(f"        }}")
                    lines.append(f"    }}")

                elif eff_type in ("DeathrattleBuffAllFriendliesGlobal", "DeathrattleBuffFriendlyTypeGlobal"):
                    cpp_type = TYPE_MAP[eff.trigger_type]
                    lines.append(f"    auto& board = state.boards[side];")
                    lines.append(f"    for (int i = 0; i < board.count; ++i) {{")
                    lines.append(f"        Unit& target = board.units[i];")
                    lines.append(f"        if (target.is_alive() && (target.types & {cpp_type})) {{")
                    lines.append(f"            target.combat_atk += {eff.atk};")
                    lines.append(f"            target.combat_hp += {eff.hp};")
                    lines.append(f"        }}")
                    lines.append(f"    }}")

                elif eff_type == "DeathrattleDestroyKiller":
                    lines.append(f"    int32_t killer_uid = event.meta;")
                    lines.append(f"    if (killer_uid != 0) {{")
                    lines.append(f"        for (int s = 0; s < 2; ++s) {{")
                    lines.append(f"            auto& board = state.boards[s];")
                    lines.append(f"            for (int i = 0; i < board.count; ++i) {{")
                    lines.append(f"                Unit& target = board.units[i];")
                    lines.append(f"                if (target.uid == killer_uid && target.is_alive()) {{")
                    lines.append(f"                    target.damage_taken += target.get_hp();")
                    lines.append(f"                    board.dead_slot_mask |= (1 << i);")
                    lines.append(f"                    state.has_pending_deaths = true;")
                    lines.append(f"                    return;")
                    lines.append(f"                }}")
                    lines.append(f"            }}")
                    lines.append(f"        }}")
                    lines.append(f"    }}")

                elif eff_type == "DeathrattleGiveFriendliesScaling":
                    lines.append(f"    auto& board = state.boards[side];")
                    lines.append(f"    for (int i = 0; i < board.count; ++i) {{")
                    lines.append(f"        Unit& target = board.units[i];")
                    lines.append(f"        if (target.is_alive()) {{")
                    lines.append(f"            target.combat_atk += {eff.buff_atk};")
                    lines.append(f"            target.combat_hp += {eff.buff_hp};")
                    lines.append(f"            if ({eff.self_damage} > 0) {{")
                    lines.append(f"                target.damage_taken += {eff.self_damage};")
                    lines.append(f"                if (target.get_hp() <= 0) {{")
                    lines.append(f"                    board.dead_slot_mask |= (1 << i);")
                    lines.append(f"                    state.has_pending_deaths = true;")
                    lines.append(f"                }}")
                    lines.append(f"            }}")
                    lines.append(f"        }}")
                    lines.append(f"    }}")

                elif eff_type == "DeathrattleSummonTauntToken":
                    token = next(c for c in ALL_CARDS if c.card_id == eff.token_id)
                    token_cpp = card_id_to_cpp_int(eff.token_id)
                    token_types = types_to_cpp(token.types)
                    token_tags = tags_to_cpp(token.tags | {Tags.TAUNT})
                    lines.append(f"    int8_t s = -1, sl = -1;")
                    lines.append(f"    if (!get_source_pos(event, s, sl)) return;")
                    for _ in range(eff.count):
                        lines.append(
                            f"    summon_unit(state, queue, s, sl, {token_cpp}, {token.atk}, {token.hp}, {token_types}, {token_tags}, {token.tier}, false);"
                        )

                lines.append(f"}}")
                lines.append("")

                registrations.append(
                    f"    {{ TriggerDef def{{EventType::MINION_DIED, cond_self_death, {fn_name}}}; register_effect_entry({cpp_id}, &def, 1); }}"
                )

            # Custom Triggers
            elif eff_type in ("OnFriendlySummonedTypeBuff", "OnSelfDamagedBuffBoard", "OnFriendlyBeastDamagedBuffSelf", "OnFriendlyBeastDamagedBuffOther", "OnFriendlyRebornBuffSelf", "OnFriendlyAttackBuffSelf", "OnFriendlyAttackBuffTriggerSelf"):
                fn_name = f"effect_trig_{cpp_name_safe}"
                lines.append(
                    f"static void {fn_name}(CombatState& state, EventQueue&, const Event& event, int32_t trigger_uid, int8_t side, int8_t slot) {{"
                )

                if eff_type == "OnFriendlySummonedTypeBuff":
                    if eff.exclude_self:
                        lines.append(f"    if (event.source_uid == trigger_uid) return;")
                    lines.append(f"    if (event.source_side != side) return;")
                    lines.append(f"    Unit* summoned = event_source_unit(state, event);")
                    cpp_type = TYPE_MAP[eff.trigger_type]
                    lines.append(f"    if (!summoned || !(summoned->types & {cpp_type})) return;")
                    lines.append(f"    Unit* u = trigger_owner(state, side, slot, trigger_uid);")
                    lines.append(f"    if (!u || !u->is_alive()) return;")
                    lines.append(f"    u->combat_atk += {eff.atk};")
                    lines.append(f"    u->combat_hp += {eff.hp};")
                    if eff.gain_divine_shield:
                        lines.append(f"    u->set_tag(Tags::DIVINE_SHIELD);")

                elif eff_type == "OnSelfDamagedBuffBoard":
                    lines.append(f"    if (event.target_uid != trigger_uid) return;")
                    lines.append(f"    Unit* u = trigger_owner(state, side, slot, trigger_uid);")
                    lines.append(f"    if (!u || !u->is_alive()) return;")
                    lines.append(f"    auto& board = state.boards[side];")
                    lines.append(f"    for (int i = 0; i < board.count; ++i) {{")
                    lines.append(f"        Unit& target = board.units[i];")
                    lines.append(f"        if (target.is_alive() && target.uid != trigger_uid) {{")
                    lines.append(f"            target.combat_atk += {eff.atk};")
                    lines.append(f"            target.combat_hp += {eff.hp};")
                    lines.append(f"        }}")
                    lines.append(f"    }}")

                elif eff_type == "OnFriendlyBeastDamagedBuffSelf":
                    lines.append(f"    if (event.target_uid == trigger_uid) return;")
                    lines.append(f"    int8_t tgt_side = -1, tgt_slot = -1;")
                    lines.append(f"    Unit* target = find_unit_by_uid(state, event.target_uid, tgt_side, tgt_slot);")
                    lines.append(f"    if (!target || tgt_side != side || !(target->types & UnitTypes::BEAST)) return;")
                    lines.append(f"    Unit* u = trigger_owner(state, side, slot, trigger_uid);")
                    lines.append(f"    if (!u || !u->is_alive()) return;")
                    lines.append(f"    u->combat_hp += {eff.hp};")

                elif eff_type == "OnFriendlyBeastDamagedBuffOther":
                    lines.append(f"    int8_t tgt_side = -1, tgt_slot = -1;")
                    lines.append(f"    Unit* target = find_unit_by_uid(state, event.target_uid, tgt_side, tgt_slot);")
                    lines.append(f"    if (!target || tgt_side != side || !(target->types & UnitTypes::BEAST)) return;")
                    lines.append(f"    auto& board = state.boards[side];")
                    lines.append(f"    Unit* candidates[GameConst::MAX_BOARD];")
                    lines.append(f"    int count = 0;")
                    lines.append(f"    for (int i = 0; i < board.count; ++i) {{")
                    lines.append(f"        Unit& other = board.units[i];")
                    lines.append(f"        if (other.is_alive() && other.uid != event.target_uid && (other.types & UnitTypes::BEAST)) {{")
                    lines.append(f"            candidates[count++] = &other;")
                    lines.append(f"        }}")
                    lines.append(f"    }}")
                    lines.append(f"    if (count > 0) {{")
                    lines.append(f"        int idx = rng_index(state.rng, count);")
                    lines.append(f"        candidates[idx]->combat_atk += {eff.atk};")
                    lines.append(f"        candidates[idx]->combat_hp += {eff.hp};")
                    lines.append(f"    }}")

                elif eff_type == "OnFriendlyRebornBuffSelf":
                    lines.append(f"    if (event.meta != 1) return;")
                    lines.append(f"    if (event.source_side != side) return;")
                    lines.append(f"    Unit* u = trigger_owner(state, side, slot, trigger_uid);")
                    lines.append(f"    if (!u || !u->is_alive()) return;")
                    lines.append(f"    u->combat_atk += {eff.atk};")
                    lines.append(f"    u->combat_hp += {eff.hp};")

                elif eff_type == "OnFriendlyAttackBuffSelf":
                    lines.append(f"    if (event.source_uid == trigger_uid) return;")
                    lines.append(f"    if (event.source_side != side) return;")
                    lines.append(f"    int8_t att_side = -1, att_slot = -1;")
                    lines.append(f"    Unit* attacker = find_unit_by_uid(state, event.source_uid, att_side, att_slot);")
                    cpp_type = TYPE_MAP[eff.trigger_type]
                    lines.append(f"    if (!attacker || att_side != side || !(attacker->types & {cpp_type})) return;")
                    lines.append(f"    attacker->combat_atk += {eff.atk};")
                    lines.append(f"    attacker->combat_hp += {eff.hp};")

                elif eff_type == "OnFriendlyAttackBuffTriggerSelf":
                    lines.append(f"    if (event.source_uid == trigger_uid) return;")
                    lines.append(f"    if (event.source_side != side) return;")
                    lines.append(f"    int8_t att_side = -1, att_slot = -1;")
                    lines.append(f"    Unit* attacker = find_unit_by_uid(state, event.source_uid, att_side, att_slot);")
                    cpp_type = TYPE_MAP[eff.trigger_type]
                    lines.append(f"    if (!attacker || att_side != side || !(attacker->types & {cpp_type})) return;")
                    lines.append(f"    Unit* u = trigger_owner(state, side, slot, trigger_uid);")
                    lines.append(f"    if (!u || !u->is_alive()) return;")
                    lines.append(f"    u->combat_atk += {eff.atk};")
                    lines.append(f"    u->combat_hp += {eff.hp};")

                lines.append(f"}}")
                lines.append("")

                if eff_type in ("OnFriendlySummonedTypeBuff", "OnFriendlyRebornBuffSelf"):
                    event_type_cpp = "EventType::MINION_SUMMONED"
                elif eff_type in ("OnFriendlyAttackBuffSelf", "OnFriendlyAttackBuffTriggerSelf"):
                    event_type_cpp = "EventType::ATTACK_DECLARED"
                else:
                    event_type_cpp = "EventType::MINION_DAMAGED"
                cond_fn_cpp = "cond_always"
                registrations.append(
                    f"    {{ TriggerDef def{{{event_type_cpp}, {cond_fn_cpp}, {fn_name}}}; register_effect_entry({cpp_id}, &def, 1); }}"
                )

            # Rally
            elif eff_type in ("RallyDealDamageEqualToAtk", "RallyBuffRandomFriendlyType", "RallyBuffFriendlyTypeAtk", "RallyDamageOwnBoard", "RallyBuffAllOthersByType"):
                fn_name = f"effect_rally_{cpp_name_safe}"
                lines.append(
                    f"static void {fn_name}(CombatState& state, EventQueue&, const Event&, int32_t trigger_uid, int8_t side, int8_t slot) {{"
                )
                lines.append(f"    Unit* u = trigger_owner(state, side, slot, trigger_uid);")
                lines.append(f"    if (!u || !u->is_alive()) return;")

                if eff_type == "RallyDealDamageEqualToAtk":
                    lines.append(f"    auto& enemy_board = state.boards[1 - side];")
                    lines.append(f"    if (enemy_board.count > 0) {{")
                    lines.append(f"        int idx = rng_index(state.rng, enemy_board.count);")
                    lines.append(f"        Unit& victim = enemy_board.units[idx];")
                    lines.append(f"        int16_t dmg = u->get_atk();")
                    lines.append(f"        if (dmg > 0) {{")
                    lines.append(f"            if (victim.has_tag(Tags::DIVINE_SHIELD)) {{")
                    lines.append(f"                victim.remove_tag(Tags::DIVINE_SHIELD);")
                    lines.append(f"                fire_unit_event(state, EventType::DIVINE_SHIELD_LOST, victim.uid, 1 - side, idx);")
                    lines.append(f"            }} else {{")
                    lines.append(f"                victim.damage_taken += dmg;")
                    lines.append(f"                if (victim.get_hp() <= 0) {{")
                    lines.append(f"                    enemy_board.dead_slot_mask |= (1 << idx);")
                    lines.append(f"                    state.has_pending_deaths = true;")
                    lines.append(f"                }}")
                    lines.append(f"            }}")
                    lines.append(f"        }}")
                    lines.append(f"    }}")

                elif eff_type == "RallyBuffRandomFriendlyType":
                    cpp_type = TYPE_MAP[eff.trigger_type]
                    lines.append(f"    auto& board = state.boards[side];")
                    lines.append(f"    Unit* candidates[GameConst::MAX_BOARD];")
                    lines.append(f"    int count = 0;")
                    lines.append(f"    for (int i = 0; i < board.count; ++i) {{")
                    lines.append(f"        Unit& other = board.units[i];")
                    lines.append(f"        if (other.is_alive() && other.uid != trigger_uid && (other.types & {cpp_type})) {{")
                    lines.append(f"            candidates[count++] = &other;")
                    lines.append(f"        }}")
                    lines.append(f"    }}")
                    lines.append(f"    if (count > 0) {{")
                    lines.append(f"        int idx = rng_index(state.rng, count);")
                    lines.append(f"        candidates[idx]->combat_atk += {eff.atk};")
                    lines.append(f"        candidates[idx]->combat_hp += {eff.hp};")
                    lines.append(f"    }}")

                elif eff_type == "RallyBuffFriendlyTypeAtk":
                    cpp_type = TYPE_MAP[eff.trigger_type]
                    lines.append(f"    auto& board = state.boards[side];")
                    lines.append(f"    for (int i = 0; i < board.count; ++i) {{")
                    lines.append(f"        Unit& target = board.units[i];")
                    lines.append(f"        if (target.is_alive() && target.uid != trigger_uid && (target.types & {cpp_type})) {{")
                    lines.append(f"            target.combat_atk += {eff.atk};")
                    lines.append(f"        }}")
                    lines.append(f"    }}")

                elif eff_type == "RallyDamageOwnBoard":
                    lines.append(f"    auto& board = state.boards[side];")
                    lines.append(f"    for (int i = 0; i < board.count; ++i) {{")
                    lines.append(f"        Unit& target = board.units[i];")
                    lines.append(f"        if (target.is_alive() && target.uid != trigger_uid) {{")
                    lines.append(f"            target.damage_taken += {eff.damage};")
                    lines.append(f"            if (target.get_hp() <= 0) {{")
                    lines.append(f"                board.dead_slot_mask |= (1 << i);")
                    lines.append(f"                state.has_pending_deaths = true;")
                    lines.append(f"            }}")
                    lines.append(f"        }}")
                    lines.append(f"    }}")

                elif eff_type == "RallyBuffAllOthersByType":
                    lines.append(f"    auto& board = state.boards[side];")
                    lines.append(f"    for (int i = 0; i < board.count; ++i) {{")
                    lines.append(f"        Unit& target = board.units[i];")
                    lines.append(f"        if (target.is_alive() && target.uid != trigger_uid) {{")
                    lines.append(f"            target.combat_atk += {eff.count};")
                    lines.append(f"            target.combat_hp += {eff.count};")
                    lines.append(f"        }}")
                    lines.append(f"    }}")

                lines.append(f"}}")
                lines.append("")

                registrations.append(
                    f"    {{ TriggerDef def{{EventType::ATTACK_DECLARED, cond_is_self, {fn_name}}}; register_effect_entry({cpp_id}, &def, 1); }}"
                )

    # EffectIDs.CRAB_DEATHRATTLE — attached effect, always needed
    crab = next(c for c in ALL_CARDS if c.card_id == CardIDs.CRAB_TOKEN)
    crab_cpp = card_id_to_cpp_int(CardIDs.CRAB_TOKEN)
    lines.append(
        f"static void effect_dr_crab_attached(CombatState& state, EventQueue& queue, const Event& event, int32_t, int8_t, int8_t) {{"
    )
    lines.append(f"    int8_t s = -1, sl = -1;")
    lines.append(f"    if (!get_source_pos(event, s, sl)) return;")
    lines.append(
        f"    summon_unit(state, queue, s, sl, {crab_cpp}, {crab.atk}, {crab.hp}, {types_to_cpp(crab.types)}, {tags_to_cpp(crab.tags)}, {crab.tier}, false);"
    )
    lines.append(f"}}")
    lines.append("")

    registrations.append(
        f"    {{ TriggerDef def{{EventType::MINION_DIED, cond_self_death, effect_dr_crab_attached}}; register_effect_entry(EffectID::CRAB_DEATHRATTLE, &def, 1); }}"
    )

    # register_all_effects()
    lines.append("// ── Registration ──")
    lines.append("")
    lines.append("void register_all_effects() {")
    for reg in registrations:
        lines.append(reg)
    lines.append("")
    lines.append("    finalize_effect_table();")
    lines.append("}")
    lines.append("")

    return "\n".join(lines)



# ── Main ──────────────────────────────────────────────────────────────


def main():
    cpp_dir = ROOT / "cpp"
    include_dir = cpp_dir / "include"
    src_dir = cpp_dir / "src"

    # Generate card IDs header
    header = generate_card_ids_header()
    header_path = include_dir / "generated_card_ids.h"
    header_path.write_text(header, encoding="utf-8")
    print(f"Generated {header_path} ({len(ALL_CARDS)} cards)")

    # Generate card DB header (base tags lookup)
    db_header = generate_card_db_header()
    db_path = include_dir / "generated_card_db.h"
    db_path.write_text(db_header, encoding="utf-8")
    db_entries = sum(1 for c in ALL_CARDS if c.tags)
    print(f"Generated {db_path} ({db_entries} cards with base tags)")

    # Generate effects
    effects = generate_effects_cpp()
    effects_path = src_dir / "generated_effects.cpp"
    effects_path.write_text(effects, encoding="utf-8")

    # Count registrations
    reg_count = effects.count("register_effect_entry")
    print(f"Generated {effects_path} ({reg_count} effect registrations)")


if __name__ == "__main__":
    main()
