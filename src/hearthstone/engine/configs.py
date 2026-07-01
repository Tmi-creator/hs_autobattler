from __future__ import annotations

from typing import Any, Dict

from .card_def import build_card_db
from .enums import EffectIDs, MechanicType, SpellIDs, Tags

TIER_COPIES = {1: 16, 2: 15, 3: 13, 4: 11, 5: 9, 6: 7, 7: 3}

TAVERN_SLOTS = {1: 3, 2: 4, 3: 4, 4: 5, 5: 5, 6: 6}
TIER_UPGRADE_COSTS = {2: 5, 3: 7, 4: 8, 5: 9, 6: 10}

MECHANIC_DEFAULTS = {
    MechanicType.BLOOD_GEM: (1, 1),
    MechanicType.ELEMENTAL_BUFF: (0, 0),
}

COST_BUY = 3
COST_REROLL = 1
SPELLS_PER_ROLL = 1

# =====================================================================
# Card Database — generated from card_def.py (single source of truth)
# =====================================================================
CARD_DB: Dict[str, Any] = build_card_db()

# =====================================================================
# Spell Database
# =====================================================================
SPELL_DB: Dict[str, Any] = {
    SpellIDs.TAVERN_COIN: {
        "name": "Tavern Coin",
        "tier": 1,
        "cost": 1,
        "effect": "GAIN_GOLD",
        "params": {"gold": 1},
    },
    SpellIDs.BANANA: {
        "name": "Banana",
        "tier": 1,
        "cost": 3,
        "effect": "BUFF_MINION",
        "params": {"atk": 2, "hp": 2},
    },
    SpellIDs.BLOOD_GEM: {
        "name": "Blood Gem",
        "tier": 0,
        "cost": 0,
        "effect": "BUFF_MINION",
        "params": {"atk": 1, "hp": 1},
        "pool": False,
    },
    SpellIDs.POINTY_ARROW: {
        "name": "Pointy Arrow",
        "tier": 1,
        "cost": 1,
        "effect": "BUFF_MINION",
        "params": {"atk": 4, "hp": 0},
    },
    SpellIDs.FORTIFY: {
        "name": "Fortify",
        "tier": 1,
        "cost": 1,
        "effect": "BUFF_MINION",
        "params": {"atk": 0, "hp": 3, "tags": {Tags.TAUNT}},
    },
    SpellIDs.APPLE: {
        "name": "Apple",
        "tier": 1,
        "cost": 1,
        "effect": "BUFF_TAVERN",
        "params": {"atk": 1, "hp": 2},
    },
    SpellIDs.SURF_SPELLCRAFT: {
        "name": "Surf Spellcraft",
        "tier": 1,
        "cost": 0,
        "effect": "ATTACH_CRAB_DR",
        "params": {"effect_id": EffectIDs.CRAB_DEATHRATTLE, "count": 1},
        "is_temporary": True,
        "pool": False,
    },
    SpellIDs.SLIMY_SHIELD: {
        "name": "Slimy Shield",
        "tier": 2,
        "cost": 0,
        "effect": "BUFF_MINION",
        "params": {"atk": 1, "hp": 1, "tags": {Tags.TAUNT}},
        "pool": False,
    },
    SpellIDs.BLOOD_GEM_BARRAGE: {
        "name": "Blood Gem Barrage",
        "tier": 3,
        "cost": 0,
        "effect": "BUFF_ALL_FRIENDLY",
        "params": {"atk": 1, "hp": 1},
        "pool": False,
    },
    SpellIDs.GEM_CONFISCATION: {
        "name": "Gem Confiscation",
        "tier": 4,
        "cost": 0,
        "effect": "BUFF_MINION",
        "params": {"atk": 1, "hp": 1},
        "pool": False,
    },
    SpellIDs.TRIPLET_REWARD: {
        "name": "Triple Reward",
        "tier": 1,
        "cost": 0,
        "effect": "DISCOVER_TIER_UP",
        "is_temporary": False,
        "pool": False,
    },
    SpellIDs.HAUNTED_CARAPACE: {
        "name": "Haunted Carapace",
        "tier": 5,
        "cost": 0,
        "effect": "BUFF_ALL_FRIENDLY",
        "params": {"atk": 3, "hp": 1},
        "pool": False,
    },
    SpellIDs.STAFF_OF_ENRICHMENT: {
        "name": "Staff of Enrichment",
        "tier": 5,
        "cost": 0,
        "effect": "BUFF_TAVERN",
        "params": {"atk": 2, "hp": 2},
        "pool": False,
    },
    SpellIDs.SHINY_RING: {
        "name": "Shiny Ring",
        "tier": 5,
        "cost": 0,
        "effect": "BUFF_ALL_FRIENDLY",
        "params": {"atk": 1, "hp": 1},
        "pool": False,
    },
    SpellIDs.MOUNTING_AVALANCHE: {
        "name": "Mounting Avalanche",
        "tier": 6,
        "cost": 0,
        "effect": "BUFF_MINION",
        "params": {"atk": 2, "hp": 2},
        "pool": False,
    },
    SpellIDs.MISPLACED_TEA_SET: {
        "name": "Misplaced Tea Set",
        "tier": 6,
        "cost": 0,
        "effect": "BUFF_ALL_BY_TYPE",
        "params": {"atk": 2, "hp": 2},
        "pool": False,
    },
    SpellIDs.TOMB_TURNING: {
        "name": "Tomb Turning",
        "tier": 5,
        "cost": 0,
        "effect": "BUFF_MINION",
        "params": {"atk": 2, "hp": 2},
        "pool": False,
    },
}
