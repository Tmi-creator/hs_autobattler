from dataclasses import dataclass, replace, field
from typing import Dict, List, Optional
from .enums import UnitType
from .configs import CARD_DB, SPELL_DB


@dataclass
class Unit:
    uid: int
    card_id: str
    owner_id: int
    max_hp: int  # initial hp(or after enchantments)
    max_atk: int  # initial atk(or after enchantments)
    tier: int
    cur_hp: int = 0  # current hp(in fight)
    cur_atk: int = 0  # current atk(in fight)
    type: List[UnitType] = field(default_factory=list)
    is_golden: bool = False
    has_taunt: bool = False
    has_divine_shield: bool = False
    has_windfury: bool = False
    has_poisonous: bool = False
    has_reborn: bool = False
    has_venomous: bool = False
    has_cleave: bool = False
    is_frozen: bool = False

    def restore_stats(self):
        self.cur_hp = self.max_hp
        self.cur_atk = self.max_atk

    @property
    def is_alive(self):
        return self.cur_hp > 0

    def combat_copy(self):
        return replace(
            self,
            type=list(self.type),
        )

    @staticmethod
    def create_from_db(card_id: str, uid: int, owner_id: int):
        """Фабричный метод: создает юнита по ID из базы"""
        data = CARD_DB.get(str(card_id))
        if not data:
            raise ValueError(f"Card {card_id} not found in DB")

        return Unit(
            uid=uid,
            card_id=card_id,
            owner_id=owner_id,
            max_hp=data['hp'],
            max_atk=data['atk'],
            cur_hp=data['hp'],
            cur_atk=data['atk'],
            tier=data['tier'],
            type=list(data.get('type', [])),

            has_taunt=data.get('taunt', False),
            has_divine_shield=data.get('divine_shield', False),
            has_windfury=data.get('windfury', False),
            has_poisonous=data.get('poisonous', False),
            has_reborn=data.get('reborn', False),
            has_venomous=data.get('venomous', False),
            has_cleave=data.get('cleave', False)
        )


@dataclass
class Spell:
    card_id: str
    name: str
    tier: int
    cost: int
    effect: str
    params: Dict[str, int] = field(default_factory=dict)
    is_temporary: bool = False

    @staticmethod
    def create_from_db(card_id: str):
        data = SPELL_DB.get(str(card_id))
        if not data:
            raise ValueError(f"Spell {card_id} not found in DB")
        return Spell(
            card_id=card_id,
            name=data["name"],
            tier=data["tier"],
            cost=data["cost"],
            effect=data["effect"],
            params=dict(data.get("params", {})),
            is_temporary=data.get("is_temporary", False),
        )


@dataclass
class StoreItem:
    unit: Optional[Unit] = None
    spell: Optional[Spell] = None
    is_frozen: bool = False

    @property
    def card_id(self):
        if self.unit:
            return self.unit.card_id
        if self.spell:
            return self.spell.card_id
        return ""


@dataclass
class HandCard:
    uid: int
    unit: Optional[Unit] = None
    spell: Optional[Spell] = None

    @property
    def cost(self):
        if self.unit:
            return 3
        elif self.spell:
            return self.spell.cost
        return 0

    @property
    def card_id(self):
        if self.unit:
            return self.unit.card_id
        return self.spell.card_id


@dataclass
class Player:
    uid: int
    board: List[Unit]
    hand: List[HandCard]
    store: List[StoreItem] = field(default_factory=list)
    tavern_tier: int = 1
    gold: int = 3
    gold_next_turn: int = 0
    spell_discount: int = 0
    health: int = 30
    up_cost: int = 5
    buff_elemental_hp: int = 0
    buff_elemental_atk: int = 0
    gem_atk: int = 0
    gem_hp: int = 0
