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


class BattleOutcome(Enum):
    NO_END = 0
    DRAW = 1
    WIN = 2
    LOSE = 3


class MechanicType(Enum):
    BLOOD_GEM = "BLOOD_GEM"
    ELEMENTAL_BUFF = "ELEMENTAL_BUFF"


class CardIDs(str, Enum):
    # --- TIER 1 ---
    WRATH_WEAVER = "101"
    SHELL_COLLECTOR = "107"
    SWAMPSTRIKER = "104"
    ANNOY_O_TRON = "105"
    ALLEYCAT = "102"
    SCALLYWAG = "103"
    IMPRISONER = "108"
    MINTED_CORSAIR = "109"
    FLIGHTY_SCOUT = "110"
    DIRE_WOLF_ALPHA = "111"


    # --- TIER 2 ---
    LEAPFROGGER = "201"
    MOLTEN_ROCK = "202"
    MURLOC_WARLEADER = "203"
    SOUTHSEA_CAPTAIN = "204"

    # --- TOKENS ---
    TABBYCAT = "102t"
    PIRATE_TOKEN = "103t"
    IMP_TOKEN = "108t"
    CRAB_TOKEN = "001t"


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