from .enums import UnitType, Tags

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
    "BLOOD_GEM": (1, 1),
    "ELEMENTAL_BUFF": (0, 0),
}

COST_BUY = 3
COST_REROLL = 1
SPELLS_PER_ROLL = 1

# Мини-база данных (MVP)
CARD_DB = {
    # --- TIER 1 ---
    "101": {"name": "Wrath Weaver", "tier": 1, "atk": 1, "hp": 3, "type": [UnitType.DEMON]},
    "107": {"name": "Shell Collector", "tier": 1, "atk": 2, "hp": 1, "type": [UnitType.NAGA]},
    "104": {"name": "Swampstriker", "tier": 1, "atk": 1, "hp": 5, "type": [UnitType.MURLOC],
            "tags": [Tags.WINDFURY]},
    "105": {
        "name": "Annoy-o-Tron",
        "tier": 1, "atk": 1, "hp": 2,
        "type": [UnitType.MECH],
        "tags": [Tags.DIVINE_SHIELD, Tags.TAUNT]
    },
    "102": {"name": "Alleycat", "tier": 1, "atk": 1, "hp": 1, "type": [UnitType.BEAST], "token": "102t"},
    "103": {"name": "Scallywag", "tier": 1, "atk": 3, "hp": 1, "type": [UnitType.PIRATE], "deathrattle": True},
    "108": {"name": "Imprisoner", "tier": 1, "atk": 3, "hp": 3, "type": [UnitType.DEMON], "token": "108t",
            "deathrattle": True, "tags": [Tags.TAUNT]},
    "109": {"name": "Minted Corsair", "tier": 1, "atk": 1, "hp": 3, "type": [UnitType.PIRATE]},
    "110": {"name": "Flighty Scout", "tier": 1, "atk": 3, "hp": 3, "type": [UnitType.MURLOC]},

    # --- TIER 2 ---
    "201": {"name": "Leapfrogger", "tier": 2, "atk": 3, "hp": 3, "type": [UnitType.BEAST], "deathrattle": True},
    "202": {"name": "Molten Rock", "tier": 2, "atk": 2, "hp": 4, "type": [UnitType.ELEMENTAL], "tags": [Tags.TAUNT]},

    # --- TOKENS (призывные существа, нет в пуле) ---
    "102t": {"name": "Tabbycat", "tier": 1, "atk": 1, "hp": 1, "type": [UnitType.BEAST], "is_token": True},
    "103t": {"name": "Pirate", "tier": 1, "atk": 1, "hp": 1, "type": [UnitType.PIRATE], "is_token": True},
    "108t": {"name": "Imp", "tier": 1, "atk": 1, "hp": 1, "type": [UnitType.DEMON], "is_token": True},
    "001t": {"name": "Crab", "tier": 1, "atk": 3, "hp": 2, "type": [UnitType.BEAST], "is_token": True},
}

SPELL_DB = {
    "S001": {"name": "Tavern Coin", "tier": 1, "cost": 1, "effect": "GAIN_GOLD", "params": {"gold": 1}},
    "S002": {"name": "Banana", "tier": 1, "cost": 3, "effect": "BUFF_MINION", "params": {"atk": 2, "hp": 2}},
    "S003": {"name": "Blood Gem", "tier": 0, "cost": 0, "effect": "BUFF_MINION", "params": {"atk": 1, "hp": 1}},
    "S004": {"name": "Pointy Arrow", "tier": 1, "cost": 1, "effect": "BUFF_MINION", "params": {"atk": 4, "hp": 0}},
    "S005": {"name": "Fortify", "tier": 1, "cost": 1, "effect": "BUFF_MINION",
             "params": {"atk": 0, "hp": 3, "tags": [Tags.TAUNT]}},
    "S006": {"name": "Apple", "tier": 1, "cost": 1, "effect": "BUFF_MINION", "params": {"atk": 1, "hp": 2}},
    "S007": {"name": "Surf Spellcraft", "tier": 1, "cost": 0, "effect": "ATTACH_CRAB_DR",
             "params": {"effect_id": "E_DR_CRAB32", "count": 1}, "is_temporary": True},
}
