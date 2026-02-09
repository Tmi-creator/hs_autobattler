from dataclasses import dataclass, replace, field
from typing import Dict, List, Optional, Set, Tuple
from .enums import UnitType, Tags, MechanicType, CardIDs, SpellIDs
from .configs import CARD_DB, SPELL_DB, MECHANIC_DEFAULTS
from copy import deepcopy


@dataclass
class Unit:
    uid: int  # уникальный id
    card_id: str  # id карты
    owner_id: int  # id игрока
    tier: int

    base_hp: int
    base_atk: int
    max_hp: int
    max_atk: int

    cur_hp: int = 0
    cur_atk: int = 0
    perm_hp_add: int = 0  # баффы примененные навсегда
    perm_atk_add: int = 0
    turn_hp_add: int = 0  # баффы примененные только на этот ход
    turn_atk_add: int = 0
    combat_hp_add: int = 0  # баффы примененные только на этот бой
    combat_atk_add: int = 0
    aura_hp_add: int = 0  # баффы примененные аурой зависящей от расположения (вида соседние существа получают +1 атаки)
    aura_atk_add: int = 0

    avenge_counter: int = 0

    attached_perm: Dict[str, int] = field(default_factory=dict)
    attached_turn: Dict[str, int] = field(default_factory=dict)
    attached_combat: Dict[str, int] = field(default_factory=dict)
    types: List[UnitType] = field(default_factory=list)
    is_golden: bool = False
    is_frozen: bool = False
    tags: Set[Tags] = field(default_factory=set)

    @property
    def has_taunt(self):
        return Tags.TAUNT in self.tags

    @property
    def has_divine_shield(self):
        return Tags.DIVINE_SHIELD in self.tags

    @property
    def has_windfury(self):
        return Tags.WINDFURY in self.tags

    @property
    def has_poisonous(self):
        return Tags.POISONOUS in self.tags

    @property
    def has_reborn(self):
        return Tags.REBORN in self.tags

    @property
    def has_venomous(self):
        return Tags.VENOMOUS in self.tags

    @property
    def has_cleave(self):
        return Tags.CLEAVE in self.tags

    @property
    def has_stealth(self):
        return Tags.STEALTH in self.tags

    def recalc_stats(self) -> None:
        old_max_hp = self.max_hp
        old_cur_hp = self.cur_hp
        self.max_atk = (
                self.base_atk
                + self.perm_atk_add
                + self.turn_atk_add
                + self.combat_atk_add
                + self.aura_atk_add
        )
        self.max_hp = (
                self.base_hp
                + self.perm_hp_add
                + self.turn_hp_add
                + self.combat_hp_add
                + self.aura_hp_add
        )
        missing = max(old_max_hp - old_cur_hp, 0)
        new_cur_hp = self.max_hp - missing
        self.cur_hp = max(0, min(new_cur_hp, self.max_hp))
        self.cur_atk = self.max_atk

    def reset_aura_layer(self):
        self.aura_hp_add = 0
        self.aura_atk_add = 0

    def restore_stats(self):
        self.cur_hp = self.max_hp
        self.cur_atk = self.max_atk

    @property
    def is_alive(self):
        return self.cur_hp > 0

    def combat_copy(self):
        unit = replace(
            self,
            types=list(self.types),
            tags=set(self.tags),
            attached_perm=dict(self.attached_perm),
            attached_turn=dict(self.attached_turn),
            attached_combat=dict(),
            combat_hp_add=0,
            combat_atk_add=0,
            aura_hp_add=0,
            aura_atk_add=0,
            avenge_counter=0,
        )
        unit.recalc_stats()
        unit.restore_stats()
        return unit

    def reset_turn_layer(self) -> None:
        self.turn_hp_add = 0
        self.turn_atk_add = 0
        self.attached_turn = dict()
        self.recalc_stats()

    def reset_combat_layer(self) -> None:
        self.combat_hp_add = 0
        self.combat_atk_add = 0
        self.avenge_counter = 0
        self.attached_combat = dict()
        self.recalc_stats()

    @staticmethod
    def create_from_db(card_id: str, uid: int, owner_id: int, is_golden: bool = False):
        """Фабричный метод: создает юнита по ID из базы"""
        data = CARD_DB.get(card_id)

        if not data and isinstance(card_id, str):
            try:
                data = CARD_DB.get(CardIDs(card_id))
            except ValueError:
                pass

        if not data:
            raise ValueError(f"Card {card_id} not found in DB")

        mult = 2 if is_golden else 1

        base_hp = data['hp'] * mult
        base_atk = data['atk'] * mult

        unit = Unit(
            uid=uid,
            card_id=card_id,
            owner_id=owner_id,
            base_hp=base_hp,
            base_atk=base_atk,
            max_hp=data['hp'],
            max_atk=data['atk'],
            cur_hp=data['hp'],
            cur_atk=data['atk'],
            tier=data['tier'],
            types=list(data.get('type', [])),
            tags=set(data.get('tags', [])),
            is_golden=is_golden
        )
        unit.recalc_stats()
        unit.restore_stats()
        return unit


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
        data = SPELL_DB.get(card_id)

        if not data and isinstance(card_id, str):
            try:
                data = SPELL_DB.get(SpellIDs(card_id))
            except ValueError:
                pass

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
    def card_id(self):
        if self.unit:
            return self.unit.card_id
        if self.spell:
            return self.spell.card_id
        return "NO ID"


@dataclass
class EconomyState:
    """
    Вся экономика игрока здесь
    """
    gold: int = 3
    gold_next_turn: int = 0
    tavern_tier: int = 1
    spell_discount: int = 0
    up_cost: int = 5
    store: List[StoreItem] = field(default_factory=list)

    def new_turn(self, turn_number: int):
        self.gold = min(10, 3 + turn_number - 1) + self.gold_next_turn
        self.gold_next_turn = 0
        if self.up_cost > 0 and turn_number != 1:
            self.up_cost -= 1


@dataclass
class MechanicState:
    """Глобальные баффы и счетчики механик"""
    modifiers: Dict[MechanicType, Tuple[int, int]] = field(
        default_factory=lambda: MECHANIC_DEFAULTS.copy()
    )

    def modify_stat(self, key: MechanicType, atk_add: int, hp_add: int):
        """Универсальный метод баффа механики"""
        current_atk, current_hp = self.modifiers.get(key, (0, 0))
        self.modifiers[key] = (current_atk + atk_add, current_hp + hp_add)

    def get_stat(self, key: MechanicType) -> Tuple[int, int]:
        return self.modifiers.get(key, (0, 0))


@dataclass
class DiscoveryState:
    is_active: bool = False
    options: List[StoreItem] = field(default_factory=list)

    # Метаданные
    discover_tier: int = 1
    source: str = "Unknown"  # "Triplet", "HeroPower", "Primalfin"
    is_exact_tier: bool = False  # True для триплетов, False для остальных


@dataclass
class Player:
    uid: int
    board: List[Unit]
    hand: List[HandCard]
    economy: EconomyState = field(default_factory=EconomyState)
    mechanics: MechanicState = field(default_factory=MechanicState)
    health: int = 30

    discovery: DiscoveryState = field(default_factory=DiscoveryState)

    def combat_copy(self):
        return Player(
            uid=self.uid,
            board=[u.combat_copy() for u in self.board],
            hand=self.hand.copy(),
            economy=deepcopy(self.economy),
            mechanics=deepcopy(self.mechanics),
            health=self.health,
        )

    @property
    def is_discovering(self) -> bool:
        return self.discovery.is_active

    @property
    def gold(self) -> int:
        return self.economy.gold

    @gold.setter
    def gold(self, value: int):
        self.economy.gold = value

    @property
    def gold_next_turn(self) -> int:
        return self.economy.gold_next_turn

    @gold_next_turn.setter
    def gold_next_turn(self, value: int):
        self.economy.gold_next_turn = value

    @property
    def up_cost(self) -> int:
        return self.economy.up_cost

    @up_cost.setter
    def up_cost(self, value: int):
        self.economy.up_cost = value

    @property
    def tavern_tier(self) -> int:
        return self.economy.tavern_tier

    @tavern_tier.setter
    def tavern_tier(self, value: int):
        self.economy.tavern_tier = value

    @property
    def spell_discount(self) -> int:
        return self.economy.spell_discount

    @spell_discount.setter
    def spell_discount(self, value: int):
        self.economy.spell_discount = value

    @property
    def store(self) -> List[StoreItem]:
        return self.economy.store
