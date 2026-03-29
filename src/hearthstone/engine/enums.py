from __future__ import annotations

from enum import Enum, auto


class UnitType(Enum):
    BEAST = "Beast"
    DRAGON = "Dragon"
    DEMON = "Demon"
    MURLOC = "Murloc"
    PIRATE = "Pirate"
    ELEMENTAL = "Elemental"
    MECH = "Mech"
    UNDEAD = "Undead"
    NAGA = "Naga"
    QUILBOAR = "Quilboar"
    NEUTRAL = "Neutral"
    ALL = "All"


class Tags(Enum):
    IMMEDIATE_ATTACK = auto()
    TAUNT = auto()
    DIVINE_SHIELD = auto()
    WINDFURY = auto()
    POISONOUS = auto()
    REBORN = auto()
    VENOMOUS = auto()
    CLEAVE = auto()
    STEALTH = auto()
    MAGNETIC = auto()


class BattleOutcome(Enum):
    NO_END = 0
    DRAW = 1
    WIN = 2
    LOSE = 3


class MechanicType(Enum):
    BLOOD_GEM = "BLOOD_GEM"
    ELEMENTAL_BUFF = "ELEMENTAL_BUFF"


class CardIDs(str, Enum):
    # --- TIER 1 (21 cards, patch 234747) ---
    ANNOY_O_TRON = "101"
    AUREATE_LAUREATE = "102"
    CORD_PULLER = "103"
    CRACKLING_CYCLONE = "104"
    DUNE_DWELLER = "105"
    FLIGHTY_SCOUT = "106"
    HARMLESS_BONEHEAD = "107"
    MANASABER = "108"
    MINTED_CORSAIR = "109"
    MISFIT_DRAGONLING = "110"
    OMINOUS_SEER = "111"
    PICKY_EATER = "112"
    RAZORFEN_GEOMANCER = "113"
    RISEN_RIDER = "114"
    RIVER_SKIPPER = "115"
    ROT_HIDE_GNOLL = "116"
    SURF_N_SURF = "117"
    SWAMPSTRIKER = "118"
    TUSKED_CAMPER = "119"
    TWILIGHT_HATCHLING = "120"
    WRATH_WEAVER = "121"

    # --- TOKENS ---
    MICROBOT = "t001"
    SKELETON = "t002"
    CUBLING = "t003"
    TWILIGHT_WHELP = "t004"
    CRAB_TOKEN = "t005"


class SpellIDs(str, Enum):
    TAVERN_COIN = "S001"
    BANANA = "S002"
    BLOOD_GEM = "S003"
    POINTY_ARROW = "S004"
    FORTIFY = "S005"
    APPLE = "S006"
    SURF_SPELLCRAFT = "S007"

    TRIPLET_REWARD = "S999"


class EffectIDs(str, Enum):
    CRAB_DEATHRATTLE = "E_DR_CRAB32"
