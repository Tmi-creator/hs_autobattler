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
    TAUNT = auto()
    DIVINE_SHIELD = auto()
    WINDFURY = auto()
    POISONOUS = auto()
    REBORN = auto()
    VENOMOUS = auto()
    CLEAVE = auto()
    STEALTH = auto()
