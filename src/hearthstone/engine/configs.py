from __future__ import annotations

from typing import Any, Dict

from .enums import CardIDs, EffectIDs, MechanicType, SpellIDs, Tags, UnitType

TIER_COPIES = {1: 16, 2: 15, 3: 13, 4: 11, 5: 9, 6: 7}

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
# Card Database — Tier 1 (patch 234747, 21 cards)
# =====================================================================
CARD_DB: Dict[str, Any] = {
    # --- TIER 1 ---
    CardIDs.ANNOY_O_TRON: {
        "name": "Annoy-o-Tron",
        "tier": 1,
        "atk": 1,
        "hp": 2,
        "type": [UnitType.MECH],
        "tags": {Tags.DIVINE_SHIELD, Tags.TAUNT},
    },
    CardIDs.AUREATE_LAUREATE: {
        "name": "Aureate Laureate",
        "tier": 1,
        "atk": 1,
        "hp": 1,
        "type": [UnitType.PIRATE],
        "tags": {Tags.DIVINE_SHIELD},
        # Battlecry: Make this minion Golden — shop-phase, TODO
    },
    CardIDs.CORD_PULLER: {
        "name": "Cord Puller",
        "tier": 1,
        "atk": 1,
        "hp": 1,
        "type": [UnitType.MECH],
        "tags": {Tags.DIVINE_SHIELD},
        "deathrattle": True,
    },
    CardIDs.CRACKLING_CYCLONE: {
        "name": "Crackling Cyclone",
        "tier": 1,
        "atk": 2,
        "hp": 1,
        "type": [UnitType.ELEMENTAL],
        "tags": {Tags.DIVINE_SHIELD, Tags.WINDFURY},
    },
    CardIDs.DUNE_DWELLER: {
        "name": "Dune Dweller",
        "tier": 1,
        "atk": 3,
        "hp": 2,
        "type": [UnitType.ELEMENTAL],
        # Battlecry: Give Elementals in Tavern +1/+1 this game — shop-phase
    },
    CardIDs.FLIGHTY_SCOUT: {
        "name": "Flighty Scout",
        "tier": 1,
        "atk": 3,
        "hp": 3,
        "type": [UnitType.MURLOC],
        # SoC: If in hand, summon copy — C++ combat only
    },
    CardIDs.HARMLESS_BONEHEAD: {
        "name": "Harmless Bonehead",
        "tier": 1,
        "atk": 1,
        "hp": 1,
        "type": [UnitType.UNDEAD],
        "deathrattle": True,
    },
    CardIDs.MANASABER: {
        "name": "Manasaber",
        "tier": 1,
        "atk": 4,
        "hp": 1,
        "type": [UnitType.BEAST],
        "deathrattle": True,
    },
    CardIDs.MINTED_CORSAIR: {
        "name": "Minted Corsair",
        "tier": 1,
        "atk": 1,
        "hp": 3,
        "type": [UnitType.PIRATE],
    },
    CardIDs.MISFIT_DRAGONLING: {
        "name": "Misfit Dragonling",
        "tier": 1,
        "atk": 2,
        "hp": 1,
        "type": [UnitType.DRAGON],
        # SoC: Gain stats equal to your Tier
    },
    CardIDs.OMINOUS_SEER: {
        "name": "Ominous Seer",
        "tier": 1,
        "atk": 2,
        "hp": 1,
        "type": [UnitType.DEMON, UnitType.NAGA],
        # Battlecry: Next tavern spell costs (1) less — shop-phase
    },
    CardIDs.PICKY_EATER: {
        "name": "Picky Eater",
        "tier": 1,
        "atk": 1,
        "hp": 1,
        "type": [UnitType.DEMON],
        # Battlecry: Consume random tavern minion for stats — shop-phase
    },
    CardIDs.RAZORFEN_GEOMANCER: {
        "name": "Razorfen Geomancer",
        "tier": 1,
        "atk": 2,
        "hp": 1,
        "type": [UnitType.QUILBOAR],
        # Battlecry: Get 2 Blood Gems — shop-phase
    },
    CardIDs.RISEN_RIDER: {
        "name": "Risen Rider",
        "tier": 1,
        "atk": 2,
        "hp": 1,
        "type": [UnitType.UNDEAD],
        "tags": {Tags.TAUNT, Tags.REBORN},
    },
    CardIDs.RIVER_SKIPPER: {
        "name": "River Skipper",
        "tier": 1,
        "atk": 1,
        "hp": 1,
        "type": [UnitType.MURLOC],
        # On sell: Get random T1 minion — shop-phase
    },
    CardIDs.ROT_HIDE_GNOLL: {
        "name": "Rot Hide Gnoll",
        "tier": 1,
        "atk": 1,
        "hp": 4,
        "type": [UnitType.UNDEAD],
        # +1 Atk per friendly death this combat — combat trigger
    },
    CardIDs.SURF_N_SURF: {
        "name": "Surf n' Surf",
        "tier": 1,
        "atk": 1,
        "hp": 1,
        "type": [UnitType.NAGA, UnitType.BEAST],
        # Spellcraft: DR summon 3/2 Crab — shop-phase spellcraft
    },
    CardIDs.SWAMPSTRIKER: {
        "name": "Swampstriker",
        "tier": 1,
        "atk": 1,
        "hp": 5,
        "type": [UnitType.MURLOC],
        "tags": {Tags.WINDFURY},
    },
    CardIDs.TUSKED_CAMPER: {
        "name": "Tusked Camper",
        "tier": 1,
        "atk": 2,
        "hp": 3,
        "type": [UnitType.QUILBOAR],
        # Rally: Blood Gem on self — C++ combat only
    },
    CardIDs.TWILIGHT_HATCHLING: {
        "name": "Twilight Hatchling",
        "tier": 1,
        "atk": 1,
        "hp": 1,
        "type": [UnitType.DRAGON],
        "deathrattle": True,
    },
    CardIDs.WRATH_WEAVER: {
        "name": "Wrath Weaver",
        "tier": 1,
        "atk": 1,
        "hp": 4,
        "type": [UnitType.DEMON],
    },
    # --- TOKENS ---
    CardIDs.MICROBOT: {
        "name": "Microbot",
        "tier": 1,
        "atk": 1,
        "hp": 1,
        "type": [UnitType.MECH],
        "is_token": True,
    },
    CardIDs.SKELETON: {
        "name": "Skeleton",
        "tier": 1,
        "atk": 1,
        "hp": 1,
        "type": [UnitType.UNDEAD],
        "is_token": True,
    },
    CardIDs.CUBLING: {
        "name": "Cubling",
        "tier": 1,
        "atk": 0,
        "hp": 1,
        "type": [UnitType.BEAST],
        "tags": {Tags.TAUNT},
        "is_token": True,
    },
    CardIDs.TWILIGHT_WHELP: {
        "name": "Twilight Whelp",
        "tier": 1,
        "atk": 3,
        "hp": 3,
        "type": [UnitType.DRAGON],
        "is_token": True,
    },
    CardIDs.CRAB_TOKEN: {
        "name": "Crab",
        "tier": 1,
        "atk": 3,
        "hp": 2,
        "type": [UnitType.BEAST],
        "is_token": True,
    },
}

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
        "effect": "BUFF_MINION",
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
    SpellIDs.TRIPLET_REWARD: {
        "name": "Triple Reward",
        "tier": 1,
        "cost": 0,
        "effect": "DISCOVER_TIER_UP",
        "is_temporary": False,
        "pool": False,
    },
}
