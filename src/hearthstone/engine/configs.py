from .enums import UnitType, Tags, MechanicType, CardIDs, SpellIDs, EffectIDs

TIER_COPIES = {
    1: 16,
    2: 15,
    3: 13,
    4: 11,
    5: 9,
    6: 7
}

TAVERN_SLOTS = {
    1: 3,
    2: 4,
    3: 4,
    4: 5,
    5: 5,
    6: 6
}
TIER_UPGRADE_COSTS = {
    2: 5,
    3: 7,
    4: 8,
    5: 9,
    6: 10
}

MECHANIC_DEFAULTS = {
    MechanicType.BLOOD_GEM: (1, 1),
    MechanicType.ELEMENTAL_BUFF: (0, 0),
}

COST_BUY = 3
COST_REROLL = 1
SPELLS_PER_ROLL = 1

# Database cards (MVP)
CARD_DB = {
    CardIDs.WRATH_WEAVER: {"name": "Wrath Weaver", "tier": 1, "atk": 1, "hp": 3, "type": [UnitType.DEMON]},
    CardIDs.SHELL_COLLECTOR: {"name": "Shell Collector", "tier": 1, "atk": 2, "hp": 1, "type": [UnitType.NAGA]},
    CardIDs.SWAMPSTRIKER: {"name": "Swampstriker", "tier": 1, "atk": 1, "hp": 5, "type": [UnitType.MURLOC],
                           "tags": {Tags.WINDFURY}},
    CardIDs.ANNOY_O_TRON: {
        "name": "Annoy-o-Tron",
        "tier": 1, "atk": 1, "hp": 2,
        "type": [UnitType.MECH],
        "tags": {Tags.DIVINE_SHIELD, Tags.TAUNT}
    },
    CardIDs.ALLEYCAT: {"name": "Alleycat", "tier": 1, "atk": 1, "hp": 1, "type": [UnitType.BEAST],
                       "token": CardIDs.TABBYCAT},
    CardIDs.SCALLYWAG: {"name": "Scallywag", "tier": 1, "atk": 3, "hp": 1, "type": [UnitType.PIRATE],
                        "deathrattle": True},
    CardIDs.IMPRISONER: {"name": "Imprisoner", "tier": 1, "atk": 3, "hp": 3, "type": [UnitType.DEMON],
                         "token": CardIDs.IMP_TOKEN,
                         "deathrattle": True, "tags": {Tags.TAUNT}},
    CardIDs.MINTED_CORSAIR: {"name": "Minted Corsair", "tier": 1, "atk": 1, "hp": 3, "type": [UnitType.PIRATE]},
    CardIDs.FLIGHTY_SCOUT: {"name": "Flighty Scout", "tier": 1, "atk": 3, "hp": 3, "type": [UnitType.MURLOC]},
    CardIDs.DIRE_WOLF_ALPHA: {"name": "Dire Wolf Alpha", "tier": 1, "atk": 1, "hp": 2, "type": [UnitType.BEAST]},

    # --- TIER 2 ---
    CardIDs.LEAPFROGGER: {"name": "Leapfrogger", "tier": 2, "atk": 4, "hp": 5, "type": [UnitType.BEAST],
                          "deathrattle": True},
    CardIDs.MOLTEN_ROCK: {"name": "Molten Rock", "tier": 2, "atk": 4, "hp": 7, "type": [UnitType.ELEMENTAL],
                          "tags": {Tags.TAUNT}},
    CardIDs.MURLOC_WARLEADER: {"name": "Murloc Warleader", "tier": 2, "atk": 3, "hp": 3, "type": [UnitType.MURLOC]},
    CardIDs.SOUTHSEA_CAPTAIN: {"name": "Southsea Captain", "tier": 2, "atk": 3, "hp": 3, "type": [UnitType.PIRATE]},
    CardIDs.ANNOY_O_MODULE: {
        "name": "Annoy-o-Module",
        "tier": 2,
        "atk": 2,
        "hp": 4,
        "type": [UnitType.MECH],
        "tags": {Tags.MAGNETIC, Tags.TAUNT, Tags.DIVINE_SHIELD},
        "is_token": False,
    },
    CardIDs.SPAWN_OF_NZOTH: {
        "name": "Spawn of N'Zoth",
        "tier": 2,
        "atk": 2,
        "hp": 2,
        "type": [UnitType.NEUTRAL],
        "deathrattle": True
    },
    CardIDs.KABOOM_BOT: {
        "name": "Kaboom Bot",
        "tier": 2,
        "atk": 2,
        "hp": 2,
        "type": [UnitType.MECH],
        "deathrattle": True
    },

    # --- TIER 3 ---
    CardIDs.DEFLECT_O_BOT: {
        "name": "Deflect-o-Bot",
        "tier": 3,
        "atk": 3,
        "hp": 2,
        "type": [UnitType.MECH],
        "tags": {Tags.DIVINE_SHIELD},
    },

    # --- TOKENS ---
    CardIDs.TABBYCAT: {"name": "Tabbycat", "tier": 1, "atk": 1, "hp": 1, "type": [UnitType.BEAST], "is_token": True},
    CardIDs.PIRATE_TOKEN: {"name": "Pirate", "tier": 1, "atk": 1, "hp": 1, "type": [UnitType.PIRATE],
                           "tags": [Tags.IMMEDIATE_ATTACK], "is_token": True},
    CardIDs.IMP_TOKEN: {"name": "Imp", "tier": 1, "atk": 1, "hp": 1, "type": [UnitType.DEMON], "is_token": True},
    CardIDs.CRAB_TOKEN: {"name": "Crab", "tier": 1, "atk": 3, "hp": 2, "type": [UnitType.BEAST], "is_token": True},
}

SPELL_DB = {
    SpellIDs.TAVERN_COIN: {"name": "Tavern Coin", "tier": 1, "cost": 1, "effect": "GAIN_GOLD", "params": {"gold": 1}},
    SpellIDs.BANANA: {"name": "Banana", "tier": 1, "cost": 3, "effect": "BUFF_MINION", "params": {"atk": 2, "hp": 2}},
    SpellIDs.BLOOD_GEM: {"name": "Blood Gem", "tier": 0, "cost": 0, "effect": "BUFF_MINION",
                         "params": {"atk": 1, "hp": 1}, "pool": False, },
    SpellIDs.POINTY_ARROW: {"name": "Pointy Arrow", "tier": 1, "cost": 1, "effect": "BUFF_MINION",
                            "params": {"atk": 4, "hp": 0}},
    SpellIDs.FORTIFY: {"name": "Fortify", "tier": 1, "cost": 1, "effect": "BUFF_MINION",
                       "params": {"atk": 0, "hp": 3, "tags": {Tags.TAUNT}}},
    SpellIDs.APPLE: {"name": "Apple", "tier": 1, "cost": 1, "effect": "BUFF_MINION", "params": {"atk": 1, "hp": 2}},
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
