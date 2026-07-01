from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set

from .enums import CardIDs, EffectIDs, MechanicType, SpellIDs, Tags, UnitType

if TYPE_CHECKING:
    from .event_system import EffectContext, Event


def _event_system():
    """Lazy import to avoid circular import via configs -> entities -> event_system."""
    from . import event_system as _es  # noqa: PLC0415

    return _es


# ---------------------------------------------------------------------------
# Condition helpers (needed by factory functions below)
# ---------------------------------------------------------------------------


def _is_self_play(_ctx: EffectContext, event: Event, trigger_uid: int) -> bool:
    return event.source is not None and event.source.uid == trigger_uid


def _is_self_death(_ctx: EffectContext, event: Event, trigger_uid: int) -> bool:
    return event.source is not None and event.source.uid == trigger_uid


def _is_friendly_death_exclude_self(ctx: EffectContext, event: Event, trigger_uid: int) -> bool:
    es = _event_system()
    if event.event_type != es.EventType.MINION_DIED:
        return False
    dead_pos = event.source_pos or (event.snapshot.pos if event.snapshot else None)
    if not dead_pos:
        return False
    owner_pos = ctx.resolve_pos(es.EntityRef(trigger_uid))
    if not owner_pos:
        return False
    return (dead_pos.side == owner_pos.side) and (
        event.source is not None and event.source.uid != trigger_uid
    )


def _is_friendly_soc(ctx: EffectContext, _event: Event, trigger_uid: int) -> bool:
    """Condition: Start of Combat, and I'm on a board."""
    es = _event_system()
    pos = ctx.resolve_pos(es.EntityRef(trigger_uid))
    return pos is not None


# ---------------------------------------------------------------------------
# EffectDef base + subclasses
# ---------------------------------------------------------------------------


@dataclass
class EffectDef:
    """Base class for declarative effects."""

    pass


@dataclass
class DeathrattleSummon(EffectDef):
    """On death, summon token(s) at the dead unit's position."""

    token_id: str  # CardIDs value (string)
    count: int = 1


@dataclass
class DeathrattleSummonWithTag(EffectDef):
    """On death, summon token(s) and add a tag to each summoned unit."""

    token_id: str
    count: int = 1
    tag: Tags = Tags.IMMEDIATE_ATTACK


@dataclass
class BattlecryGainGold(EffectDef):
    """On self-play, gain gold immediately."""

    amount: int = 1


@dataclass
class BattlecryAddSpell(EffectDef):
    """On self-play, add spell(s) to hand."""

    spell_id: str  # SpellIDs value
    count: int = 1


@dataclass
class BattlecrySummonAtRight(EffectDef):
    """On self-play, summon a unit immediately to the right."""

    token_id: str
    count: int = 1


@dataclass
class BattlecryBuffSelf(EffectDef):
    atk: int = 0
    hp: int = 0


@dataclass
class BattlecrySpellDiscount(EffectDef):
    """On self-play, reduce spell cost by `amount` for the rest of the game."""

    amount: int = 1


@dataclass
class BattlecryModifyMechanic(EffectDef):
    """Modify a global mechanic stat (like Dune Dweller's elemental buff)."""

    mechanic: MechanicType
    atk: int = 0
    hp: int = 0


@dataclass
class ConsumeShopUnit(EffectDef):
    """Consume random shop unit, gain its stats."""

    pass


@dataclass
class BattlecryMakeGolden(EffectDef):
    """Make this minion golden on play."""

    pass


@dataclass
class SellForGold(EffectDef):
    """This minion sells for N gold instead of 1."""

    amount: int = 3


@dataclass
class RallyBuff(EffectDef):
    """Rally: when this unit attacks, buff itself (combat scope)."""

    atk: int = 0
    hp: int = 0
    use_blood_gem: bool = False  # if True, use player's Blood Gem values instead of atk/hp


@dataclass
class StartOfCombatFromHand(EffectDef):
    """SoC: if this is in your hand, summon a copy onto your board."""

    pass


@dataclass
class AvengeEffect(EffectDef):
    """Avenge(N): after N friendly deaths in combat, fire inner effect.
    The inner effect describes WHAT happens — buff self, buff type, buff adjacent, etc."""

    threshold: int
    # What to do when avenge fires:
    buff_atk: int = 0
    buff_hp: int = 0
    buff_scope: str = "combat"  # "combat" or "perm"
    buff_target: str = "self"  # "self", "friendly_type", "adjacent", "random_friendly_type"
    target_type: Optional[UnitType] = None  # for friendly_type / random_friendly_type


@dataclass
class MultiplierDef:
    """Not an EffectDef — metadata on CardDef for cards like Brann/Titus/Drakkari.
    When this unit is on board, matching triggers get extra stacks."""

    event_type_name: str  # EventType name to match (e.g. "MINION_PLAYED")
    self_only: bool = True  # True = only self-play triggers (battlecries), False = all
    extra_stacks: int = 1  # How many extra stacks (1 = double, 2 = triple)


@dataclass
class OnFriendlyPlayType(EffectDef):
    """When any friendly unit of specific type is played, buff self."""

    trigger_type: UnitType
    atk: int = 0
    hp: int = 0
    exclude_self: bool = True


@dataclass
class OnFriendlyPlayTypeDamageHero(EffectDef):
    """When friendly unit of specific type is played (excl. self), damage own hero and buff self."""

    trigger_type: UnitType
    hero_dmg: int = 1
    atk: int = 0
    hp: int = 0
    exclude_self: bool = True


@dataclass
class SellAddSpell(EffectDef):
    """On self-sell, add spell(s) to hand."""

    spell_id: str
    count: int = 1


@dataclass
class SellGetRandomUnit(EffectDef):
    """On self-sell, add a random T{tier} minion to hand."""

    tier: int = 1


@dataclass
class StartOfCombatBuffSelf(EffectDef):
    atk: int = 0
    hp: int = 0


@dataclass
class StartOfCombatBuffSelfByTier(EffectDef):
    """Gain +tier/+tier at start of combat."""

    pass


@dataclass
class OnFriendlyDeathBuff(EffectDef):
    """Gain stats when any friendly minion dies (excluding self), as a combat buff."""

    atk: int = 0
    hp: int = 0


@dataclass
class OnFriendlySummonedTypeBuff(EffectDef):
    """When a friendly unit of specific type is summoned (not played),
    gain +atk/+hp and optionally divine shield."""

    trigger_type: UnitType
    atk: int = 0
    hp: int = 0
    exclude_self: bool = True
    combat_buff: bool = False
    gain_divine_shield: bool = False


@dataclass
class DeathrattleBuffAllFriendlies(EffectDef):
    """On death, give all surviving friendly minions +atk/+hp (combat buff)."""

    atk: int = 0
    hp: int = 0


@dataclass
class DeathrattleRandomEnemyDamage(EffectDef):
    """On death, deal damage to a random enemy minion."""

    damage: int = 4


@dataclass
class CustomEffect(EffectDef):
    """For complex effects that cannot be described declaratively."""

    trigger_defs: list = field(default_factory=list)  # List[TriggerDef]


@dataclass
class EndOfTurnAddSpell(EffectDef):
    """End of turn: add spell(s) to hand."""

    spell_id: str
    count: int = 1


@dataclass
class EndOfTurnBuffAdjacent(EffectDef):
    """End of turn: buff adjacent units."""

    atk: int = 0
    hp: int = 0


@dataclass
class EndOfTurnBuffSelf(EffectDef):
    """End of turn: buff self."""

    atk: int = 0
    hp: int = 0


@dataclass
class EndOfTurnBuffBoard(EffectDef):
    """End of turn: buff all friendly board units."""

    atk: int = 0
    hp: int = 0


@dataclass
class EndOfTurnBuffBoardByType(EffectDef):
    """End of turn: buff all friendly board units of a type."""

    trigger_type: UnitType
    atk: int = 0
    hp: int = 0


@dataclass
class StartOfCombatBuffFriendlyType(EffectDef):
    """SoC: give all friendly units of type +atk/+hp (combat buff)."""

    trigger_type: UnitType
    atk: int = 0
    hp: int = 0


@dataclass
class BattlecryBuffAllByType(EffectDef):
    """BC: give all friendly units of type +atk/+hp (perm buff)."""

    trigger_type: UnitType
    atk: int = 0
    hp: int = 0


@dataclass
class OnFriendlyPlayTypeAddSpell(EffectDef):
    """When friendly of type is played, add spell to hand."""

    trigger_type: UnitType
    spell_id: str
    count: int = 1
    exclude_self: bool = True


@dataclass
class OnSummonedTypeBuffRandomOther(EffectDef):
    """When friendly of type is summoned, buff a random OTHER friendly of that type."""

    trigger_type: UnitType
    atk: int = 0
    hp: int = 0


@dataclass
class SellAddUnit(EffectDef):
    """On sell: add a specific unit to hand."""

    card_id: str


@dataclass
class RallyBuffRandomFriendlyType(EffectDef):
    """Rally: when this attacks, buff a random other friendly of type (combat buff)."""

    trigger_type: UnitType
    atk: int = 0
    hp: int = 0


@dataclass
class SellGetRandomUnitByType(EffectDef):
    """On sell: get a random unit of specific type from pool."""

    unit_type: UnitType


@dataclass
class ConsumeShopUnitForRandomFriendly(EffectDef):
    """BC: a random friendly of type consumes a shop unit for its stats."""

    trigger_type: UnitType


@dataclass
class OnTavernRefreshBuffRightmostShop(EffectDef):
    """After tavern refreshed: buff rightmost shop minion with atk/hp and optionally Reborn."""

    atk: int = 0
    hp: int = 0
    give_reborn: bool = False
    use_blood_gem: bool = False  # if True, apply blood gem count times instead


@dataclass
class SellBuffBoardScaling(EffectDef):
    """On sell: buff all board minions +atk/+hp, and increment scaling counter."""

    scaling_key: str
    atk_per: int = 0
    hp_per: int = 0


@dataclass
class StartOfCombatDamageAndBuffAdjacent(EffectDef):
    """SoC: deal damage to adjacent minions and give them ATK buff."""

    damage: int = 1
    atk: int = 0
    hp: int = 0


@dataclass
class OnHeroDamagedHealAndBuffSelf(EffectDef):
    """After hero takes damage: undo the damage and buff self +hp."""

    hp: int = 1


@dataclass
class EndOfTurnBuffAdjacentPerGolden(EffectDef):
    """EOT: buff adjacent +atk/+hp, repeat for each friendly golden minion."""

    atk: int = 0
    hp: int = 0


@dataclass
class SellDiscover(EffectDef):
    """On sell: discover a minion of base_tier (improves each turn via scaling_key)."""

    base_tier: int = 1
    scaling_key: str = ""


@dataclass
class DeathrattleAddSpell(EffectDef):
    """On death: add spell(s) to hand."""

    spell_id: str = ""
    count: int = 1


@dataclass
class RallyAddSpell(EffectDef):
    """Rally: when this unit attacks, add spell(s) to hand."""

    spell_id: str = ""
    count: int = 1


@dataclass
class EndOfTurnBuffSelfPerGolden(EffectDef):
    """EOT: buff self +atk/+hp for each friendly golden minion."""

    atk_per: int = 0
    hp_per: int = 0


@dataclass
class OnDivineShieldLostAddSpell(EffectDef):
    """After a friendly minion loses Divine Shield, add a spell to hand."""

    spell_id: str = ""
    count: int = 1


@dataclass
class BattlecryBuffAllByTypeIncludeHand(EffectDef):
    """BC: give all OTHER friendly units of type in hand AND board +atk/+hp."""

    trigger_type: UnitType = UnitType.NEUTRAL
    atk: int = 0
    hp: int = 0


@dataclass
class RallyDamageOwnBoard(EffectDef):
    """Rally: when this unit attacks, deal damage to all other friendly minions."""

    damage: int = 1


@dataclass
class DeathrattleModifyMechanic(EffectDef):
    """On death: permanently modify a global mechanic."""

    mechanic: MechanicType = MechanicType.BLOOD_GEM
    atk: int = 0
    hp: int = 0


@dataclass
class StartOfCombatBuffRandomFriendlyTypeAndDS(EffectDef):
    """SoC: give another friendly unit of type +atk/+hp and Divine Shield."""

    trigger_type: UnitType = UnitType.DRAGON
    atk: int = 0
    hp: int = 0


@dataclass
class OnSelfDamagedBuffBoard(EffectDef):
    """When this minion takes damage, buff all other friendly minions."""

    atk: int = 0
    hp: int = 0


@dataclass
class OnFriendlyRebornBuffSelf(EffectDef):
    """After a friendly minion triggers Reborn, buff self permanently."""

    atk: int = 0
    hp: int = 0


@dataclass
class DeathrattleBuffFriendlyTypeGlobal(EffectDef):
    """On death: permanently buff all friendly units of type +atk/+hp (even in hand)."""

    trigger_type: UnitType = UnitType.UNDEAD
    atk: int = 0
    hp: int = 0


@dataclass
class DeathrattleBuffShop(EffectDef):
    """On death: permanently buff all tavern shop minions +atk/+hp this game."""

    atk: int = 0
    hp: int = 0


@dataclass
class DeathrattleBuffHandRandom(EffectDef):
    """On death: give a random minion in hand +atk/+hp."""

    atk: int = 0
    hp: int = 0


@dataclass
class StartOfCombatGainGold(EffectDef):
    """Start of turn (shop phase): gain gold. Used for start-of-turn gold generators."""

    amount: int = 1


@dataclass
class OnFriendlyAttackBuffSelf(EffectDef):
    """When another friendly unit of type attacks, buff that unit permanently."""

    trigger_type: UnitType = UnitType.DRAGON
    atk: int = 0
    hp: int = 0


@dataclass
class OnSpellCastBuffSelf(EffectDef):
    """When a tavern spell is cast (played on a minion), gain +atk/+hp."""

    atk: int = 0
    hp: int = 0


@dataclass
class OnGainGoldBuffSelf(EffectDef):
    """After gaining gold (Tavern Coin), buff self +atk/+hp."""

    atk: int = 0
    hp: int = 0


@dataclass
class DeathrattleDamageAllMinions(EffectDef):
    """On death: deal damage to ALL minions on both sides."""

    damage: int = 3


@dataclass
class StartOfCombatBuffAllFriendlyType(EffectDef):
    """SoC: buff all friendly units of type +atk/+hp permanently."""

    trigger_type: UnitType = UnitType.DRAGON
    atk: int = 0
    hp: int = 0


@dataclass
class StartOfCombatGiveFriendlyTypeReborn(EffectDef):
    """SoC: give a random friendly unit of type Reborn."""

    trigger_type: UnitType = UnitType.UNDEAD


@dataclass
class AvengeAddSpell(EffectDef):
    """Avenge(N): add a spell to hand."""

    threshold: int = 3
    spell_id: str = ""
    count: int = 1


@dataclass
class BattlecryAddRandomUnit(EffectDef):
    """BC: add a random unit of specific type from pool to hand."""

    unit_type: Optional[UnitType] = None
    tier: Optional[int] = None


@dataclass
class BattlecryGainFreeRefreshes(EffectDef):
    """BC: gain N free refreshes immediately."""

    count: int = 2


@dataclass
class OnDivineShieldLostBuffUnit(EffectDef):
    """After a friendly minion loses Divine Shield, give it +atk/+hp permanently."""

    atk: int = 0
    hp: int = 0


@dataclass
class RallyBuffAllOthersByType(EffectDef):
    """Rally: buff all other friendly minions of type with 2 blood gems."""

    trigger_type: UnitType = UnitType.PIRATE
    count: int = 2  # number of blood gems to play


@dataclass
class EndOfTurnAddRandomSpell(EffectDef):
    """End of turn: add a random tavern spell from pool to hand."""

    pass


@dataclass
class OnFriendlyPlayTypeBuffSelfInHand(EffectDef):
    """While in hand, when a friendly unit of type is played, buff self."""

    trigger_type: UnitType = UnitType.MURLOC
    atk: int = 0
    hp: int = 0


@dataclass
class OnFriendlyAttackBuffTriggerSelf(EffectDef):
    """When another friendly unit of type attacks, buff self (the trigger unit) permanently."""

    trigger_type: UnitType = UnitType.DRAGON
    atk: int = 0
    hp: int = 0


@dataclass
class OnSpellCastBuffBoard(EffectDef):
    """When a tavern spell is cast, buff all friendly minions (or of a type)."""

    atk: int = 0
    hp: int = 0
    trigger_type: Optional[UnitType] = None  # None = all friendlies


@dataclass
class DeathrattleSummonTauntToken(EffectDef):
    """On death: summon N tokens and give them Taunt."""

    token_id: str = ""
    count: int = 1


@dataclass
class StartOfCombatBuffSelfByHighestAllyAtk(EffectDef):
    """SoC: set own attack to the highest friendly attack value."""

    pass


@dataclass
class StartOfCombatBuffSelfByHighestBoardAtk(EffectDef):
    """SoC: set own stats to match the highest-attack minion on board."""

    pass


@dataclass
class BattlecryMakeGoldenFriendlyByTier(EffectDef):
    """BC: make a friendly minion from tier <= max_tier golden."""

    max_tier: int = 4


@dataclass
class RallyBuffFriendlyTypeAtk(EffectDef):
    """Rally: give all other friendly units of type +atk permanently."""

    trigger_type: UnitType = UnitType.NAGA
    atk: int = 1


@dataclass
class OnFriendlyBeastDamagedBuffSelf(EffectDef):
    """When another friendly Beast takes damage, buff self +hp permanently."""

    hp: int = 2


@dataclass
class OnFriendlyBeastDamagedBuffOther(EffectDef):
    """When a friendly Beast takes damage, give a different friendly Beast +atk/+hp."""

    atk: int = 0
    hp: int = 0


@dataclass
class AvengeBuffFriendlyTypeGlobal(EffectDef):
    """Avenge(N): give all friendly units of type +atk permanently (even in hand)."""

    threshold: int = 2
    trigger_type: UnitType = UnitType.UNDEAD
    atk: int = 1
    hp: int = 0


@dataclass
class DeathrattleDestroyKiller(EffectDef):
    """On death: destroy the minion that killed this."""

    pass


@dataclass
class SellForGoldConditional(EffectDef):
    """Sells for extra gold if player lost last combat."""

    amount: int = 5


@dataclass
class DeathrattleBuffAllFriendliesGlobal(EffectDef):
    """On death: permanently buff all friendly beasts on board +atk/+hp."""

    trigger_type: UnitType = UnitType.BEAST
    atk: int = 8
    hp: int = 8


@dataclass
class EndOfTurnBuffFriendlyTypeNaga(EffectDef):
    """EoT: give all friendly Naga +atk/+hp (scales with diversity)."""

    atk: int = 2
    hp: int = 1


@dataclass
class OnFriendlyDemonDamageBuff(EffectDef):
    """After a friendly Demon deals damage, buff other friendlies +atk/+hp."""

    atk: int = 2
    hp: int = 1


@dataclass
class EndOfTurnConsumeTavernForDemon(EffectDef):
    """EoT: each friendly Demon consumes a tavern minion for its stats."""

    pass


@dataclass
class StartOfCombatBuffFriendlyTypeScaling(EffectDef):
    """SoC: give all friendlies of type +atk/+hp, scaling with elemental play count."""

    trigger_type: UnitType = UnitType.ELEMENTAL
    atk: int = 3
    hp: int = 2


@dataclass
class EndOfTurnTriggerAdjacentBattlecry(EffectDef):
    """EoT: trigger the battlecry of adjacent minions."""

    pass


@dataclass
class RallyDealDamageEqualToAtk(EffectDef):
    """Rally: deal damage equal to this minion's Attack to a random enemy."""

    pass


@dataclass
class DeathrattleGiveFriendliesScaling(EffectDef):
    """On death: give all friendlies +1/+1 and deal 1 damage to them."""

    buff_atk: int = 1
    buff_hp: int = 1
    self_damage: int = 1


# ---------------------------------------------------------------------------
# CardDef
# ---------------------------------------------------------------------------


@dataclass
class CardDef:
    card_id: str  # e.g. "101" or "t001"
    name: str
    tier: int
    atk: int
    hp: int
    types: list  # list[UnitType]
    tags: set = field(default_factory=set)
    is_token: bool = False
    deathrattle: bool = False  # metadata flag used by obs encoding
    effects: list = field(default_factory=list)  # list[EffectDef]
    multiplier: Optional[MultiplierDef] = None  # Brann/Titus/Drakkari

    @property
    def avenge_threshold(self) -> int:
        """Return avenge threshold if card has AvengeEffect, else 0."""
        for eff in self.effects:
            if isinstance(eff, AvengeEffect):
                return eff.threshold
        return 0


# ---------------------------------------------------------------------------
# ALL_CARDS — single source of truth for every card / token
# ---------------------------------------------------------------------------

def _coerce_type(val, expected_type):
    import enum
    from typing import get_args, get_origin, Union, List, Set, Dict
    if val is None:
        return None
    
    # Handle Union (e.g. Optional[X] / X | None)
    origin = get_origin(expected_type)
    if origin is Union:
        args = get_args(expected_type)
        for arg in args:
            if arg is not type(None):
                try:
                    return _coerce_type(val, arg)
                except Exception:
                    pass
        return val
        
    # Handle List/Set/Sequence
    if origin in (list, set, List, Set):
        arg = get_args(expected_type)[0]
        coerced = [_coerce_type(item, arg) for item in val]
        return set(coerced) if origin is set else coerced
        
    # Handle Dict
    if origin in (dict, Dict):
        args = get_args(expected_type)
        key_type, val_type = args[0], args[1]
        return {
            _coerce_type(k, key_type): _coerce_type(v, val_type)
            for k, v in val.items()
        }
        
    # Handle specific Enum first!
    if isinstance(expected_type, enum.EnumMeta):
        try:
            return expected_type[val]
        except KeyError:
            try:
                return expected_type(val)
            except ValueError:
                pass

    # Auto-convert string representation of card/spell IDs to Enums if they match
    # (used for fields annotated as str but holding CardIDs/SpellIDs in python memory)
    if isinstance(val, str):
        if val in CardIDs.__members__:
            return CardIDs[val]
        for member in CardIDs:
            if member.value == val:
                return member
        if val in SpellIDs.__members__:
            return SpellIDs[val]
        for member in SpellIDs:
            if member.value == val:
                return member
                
    # Default fallback
    try:
        return expected_type(val)
    except Exception:
        return val

def _get_all_subclasses(cls):
    subclasses = set(cls.__subclasses__())
    return subclasses.union([s for c in subclasses for s in _get_all_subclasses(c)])

def load_cards_from_json(path) -> List[CardDef]:
    import json
    from dataclasses import fields
    from typing import get_type_hints
    
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    effect_classes = {c.__name__: c for c in _get_all_subclasses(EffectDef)}
    # Cache resolved type hints for each effect class to resolve postponed annotations
    resolved_class_types = {}
    for cname, c in effect_classes.items():
        try:
            resolved_class_types[c] = get_type_hints(c)
        except Exception:
            resolved_class_types[c] = {}
            
    cards = []
    for item in data:
        effects = []
        for eff_data in item.get("effects", []):
            eff_data_copy = dict(eff_data)
            eff_type = eff_data_copy.pop("type")
            klass = effect_classes[eff_type]
            
            # Coerce fields of the effect dataclass
            constructor_args = {}
            resolved_types = resolved_class_types.get(klass, {})
            for f in fields(klass):
                if f.name in eff_data_copy:
                    expected_type = resolved_types.get(f.name, f.type)
                    constructor_args[f.name] = _coerce_type(eff_data_copy[f.name], expected_type)
            effects.append(klass(**constructor_args))
            
        # Parse multiplier if present
        multiplier = None
        if "multiplier" in item and item["multiplier"] is not None:
            m_data = item["multiplier"]
            multiplier = MultiplierDef(
                event_type_name=m_data["event_type_name"],
                self_only=m_data.get("self_only", True),
                extra_stacks=m_data.get("extra_stacks", 1)
            )

        # Construct CardDef
        card = CardDef(
            card_id=_coerce_type(item["card_id"], CardIDs),
            name=item["name"],
            tier=item["tier"],
            atk=item["atk"],
            hp=item["hp"],
            types=_coerce_type(item.get("types", []), List[UnitType]),
            tags=_coerce_type(item.get("tags", []), Set[Tags]),
            effects=effects,
            multiplier=multiplier,
            is_token=item.get("is_token", False),
            deathrattle=item.get("deathrattle", False)
        )
        cards.append(card)
    return cards

from pathlib import Path
ALL_CARDS: List[CardDef] = load_cards_from_json(Path(__file__).resolve().parents[3] / "data" / "cards.json")


# ---------------------------------------------------------------------------
# build_card_db  →  produces the same dict as the original hardcoded CARD_DB
# ---------------------------------------------------------------------------


def build_card_db() -> Dict[str, Any]:
    db: Dict[str, Any] = {}
    for card in ALL_CARDS:
        entry: Dict[str, Any] = {
            "name": card.name,
            "tier": card.tier,
            "atk": card.atk,
            "hp": card.hp,
            "type": card.types,
        }
        if card.tags:
            entry["tags"] = card.tags
        if card.is_token:
            entry["is_token"] = True
        if card.deathrattle:
            entry["deathrattle"] = True
        if card.avenge_threshold > 0:
            entry["avenge_threshold"] = card.avenge_threshold
        db[card.card_id] = entry
    return db


# ---------------------------------------------------------------------------
# Effect factory functions
# ---------------------------------------------------------------------------


def _make_dr_summon(token_id: str, count: int):
    """Deathrattle: summon `count` copies of `token_id` at the dead unit's slot."""

    def _effect(ctx: EffectContext, event: Event, _trigger_uid: int) -> None:
        pos = event.source_pos or (event.snapshot.pos if event.snapshot else None)
        if pos:
            for _ in range(count):
                ctx.summon(pos.side, token_id, pos.slot)

    return _effect


def _make_dr_summon_with_tag(token_id: str, count: int, tag: Tags):
    """Deathrattle: summon `count` copies of `token_id`, each with an extra tag."""

    def _effect(ctx: EffectContext, event: Event, _trigger_uid: int) -> None:
        pos = event.source_pos or (event.snapshot.pos if event.snapshot else None)
        if not pos:
            return
        for _ in range(count):
            ref = ctx.summon(pos.side, token_id, pos.slot)
            if ref:
                unit = ctx.resolve_unit(ref)
                if unit:
                    unit.tags.add(tag)

    return _effect


def _make_battlecry_summon_at_right(token_id: str):
    """Battlecry: summon a token immediately to the right of self."""

    def _effect(ctx: EffectContext, _event: Event, trigger_uid: int) -> None:
        es = _event_system()
        pos = ctx.resolve_pos(es.EntityRef(trigger_uid))
        if not pos:
            return
        ctx.summon(pos.side, token_id, pos.slot + 1)

    return _effect


def _make_battlecry_gain_gold(amount: int):
    """Battlecry: gain gold immediately."""

    def _effect(ctx: EffectContext, _event: Event, trigger_uid: int) -> None:
        es = _event_system()
        pos = ctx.resolve_pos(es.EntityRef(trigger_uid))
        if not pos:
            return
        ctx.gain_gold(pos.side, amount)

    return _effect


def _make_battlecry_add_spell(spell_id: str, count: int):
    """Battlecry: add spell(s) to hand."""

    def _effect(ctx: EffectContext, _event: Event, trigger_uid: int) -> None:
        es = _event_system()
        pos = ctx.resolve_pos(es.EntityRef(trigger_uid))
        if not pos:
            return
        for _ in range(count):
            ctx.add_spell_to_hand(pos.side, spell_id)

    return _effect


def _make_battlecry_spell_discount(amount: int):
    """Battlecry: next tavern spell costs `amount` less."""

    def _effect(ctx: EffectContext, _event: Event, trigger_uid: int) -> None:
        es = _event_system()
        pos = ctx.resolve_pos(es.EntityRef(trigger_uid))
        if not pos:
            return
        player = ctx.players_by_uid.get(pos.side)
        if not player:
            return
        player.spell_discount += amount

    return _effect


def _make_battlecry_modify_mechanic(mechanic: MechanicType, atk: int, hp: int):
    """Battlecry: modify a global mechanic stat (Dune Dweller)."""

    def _effect(ctx: EffectContext, _event: Event, trigger_uid: int) -> None:
        es = _event_system()
        pos = ctx.resolve_pos(es.EntityRef(trigger_uid))
        if not pos:
            return
        player = ctx.players_by_uid.get(pos.side)
        if not player:
            return
        player.mechanics.modify_stat(mechanic, atk, hp)

    return _effect


def _make_battlecry_consume_shop_unit():
    """Battlecry: consume random shop unit, gain its stats."""

    def _effect(ctx: EffectContext, _event: Event, trigger_uid: int) -> None:
        es = _event_system()
        pos = ctx.resolve_pos(es.EntityRef(trigger_uid))
        if not pos:
            return
        result = ctx.consume_random_store_unit(pos.side)
        if result:
            atk, hp = result
            ctx.buff_perm(es.EntityRef(trigger_uid), atk, hp)

    return _effect


def _make_sell_add_spell(spell_id: str, count: int):
    """On sell: add spell(s) to hand."""

    def _effect(ctx: EffectContext, _event: Event, trigger_uid: int) -> None:
        es = _event_system()
        pos = ctx.resolve_pos(es.EntityRef(trigger_uid))
        if not pos:
            return
        for _ in range(count):
            ctx.add_spell_to_hand(pos.side, spell_id)

    return _effect


def _make_battlecry_make_golden():
    """Battlecry: make this minion golden."""

    def _effect(ctx: EffectContext, _event: Event, trigger_uid: int) -> None:
        es = _event_system()
        ctx.make_golden(es.EntityRef(trigger_uid))

    return _effect


def _make_sell_for_gold(amount: int):
    """On sell: gain extra gold (total = amount instead of default 1)."""

    def _effect(ctx: EffectContext, _event: Event, trigger_uid: int) -> None:
        es = _event_system()
        pos = ctx.resolve_pos(es.EntityRef(trigger_uid))
        if not pos:
            return
        # Sell already gives 1 gold from tavern logic.
        # We give (amount - 1) extra to reach the target sell price.
        ctx.gain_gold(pos.side, amount - 1)

    return _effect


def _make_rally_buff(atk: int, hp: int, use_blood_gem: bool):
    """Rally: when this unit attacks, buff itself (combat scope)."""

    def _effect(ctx: EffectContext, _event: Event, trigger_uid: int) -> None:
        es = _event_system()
        if use_blood_gem:
            pos = ctx.resolve_pos(es.EntityRef(trigger_uid))
            if not pos:
                return
            player = ctx.players_by_uid.get(pos.side)
            if not player:
                return
            buff_atk, buff_hp = player.mechanics.get_stat(MechanicType.BLOOD_GEM)
        else:
            buff_atk, buff_hp = atk, hp
        ctx.buff_combat(es.EntityRef(trigger_uid), buff_atk, buff_hp)

    return _effect


def _make_soc_from_hand():
    """Start of Combat: if this unit is in hand, summon a copy onto board."""

    def _effect(ctx: EffectContext, _event: Event, trigger_uid: int) -> None:
        es = _event_system()
        pos = ctx.resolve_pos(es.EntityRef(trigger_uid))
        if not pos or pos.zone != es.Zone.HAND:
            return
        unit = ctx.resolve_unit(es.EntityRef(trigger_uid))
        if not unit:
            return
        player = ctx.players_by_uid.get(pos.side)
        if not player or len(player.board) >= 7:
            return
        ctx.summon(pos.side, unit.card_id, len(player.board), unit.is_golden)

    return _effect


def _make_sell_get_random_unit(tier: int):
    """On sell: draw a random T{tier} unit from the shared pool into hand."""

    def _effect(ctx: EffectContext, _event: Event, trigger_uid: int) -> None:
        es = _event_system()
        pos = ctx.resolve_pos(es.EntityRef(trigger_uid))
        if not pos:
            return
        ctx.draw_from_pool(pos.side, tier=tier, count=1)

    return _effect


def _make_start_of_combat_buff_self_by_tier():
    """Start of Combat: gain +tavern_tier/+tavern_tier (Misfit Dragonling)."""

    def _effect(ctx: EffectContext, _event: Event, trigger_uid: int) -> None:
        es = _event_system()
        unit = ctx.resolve_unit(es.EntityRef(trigger_uid))
        if not unit or not unit.is_alive:
            return
        pos = ctx.resolve_pos(es.EntityRef(trigger_uid))
        if not pos:
            return
        player = ctx.players_by_uid.get(pos.side)
        if not player:
            return
        tier = player.tavern_tier
        ctx.buff_combat(es.EntityRef(trigger_uid), tier, tier)

    return _effect


def _make_on_friendly_death_buff(atk: int, hp: int):
    """On any friendly death (excl. self), gain +atk/+hp as combat buff (Rot Hide Gnoll)."""

    def _effect(ctx: EffectContext, _event: Event, trigger_uid: int) -> None:
        es = _event_system()
        gnoll = ctx.resolve_unit(es.EntityRef(trigger_uid))
        if not gnoll or not gnoll.is_alive:
            return
        ctx.buff_combat(es.EntityRef(trigger_uid), atk, hp)

    return _effect


def _make_on_play_type_buff(trigger_type: UnitType, atk: int, hp: int, exclude_self: bool):
    """On any friendly play of a unit of `trigger_type`, buff self."""

    def _effect(ctx: EffectContext, event: Event, trigger_uid: int) -> None:
        es = _event_system()
        played = ctx.resolve_unit(event.source)
        if not played or trigger_type not in played.types:
            return
        if exclude_self and event.source and event.source.uid == trigger_uid:
            return
        pos = ctx.resolve_pos(es.EntityRef(trigger_uid))
        if not pos:
            return
        ctx.buff_perm(es.EntityRef(trigger_uid), atk, hp)

    return _effect


def _make_on_play_type_damage_hero(
    trigger_type: UnitType, hero_dmg: int, atk: int, hp: int, exclude_self: bool
):
    """Wrath Weaver pattern: on play of demon (not self), damage own hero and buff self."""

    def _effect(ctx: EffectContext, event: Event, trigger_uid: int) -> None:
        es = _event_system()
        played = ctx.resolve_unit(event.source)
        if not played or trigger_type not in played.types:
            return
        if exclude_self and event.source and event.source.uid == trigger_uid:
            return

        weaver = ctx.resolve_unit(es.EntityRef(trigger_uid))
        if not weaver:
            return

        pos = ctx.resolve_pos(es.EntityRef(trigger_uid))
        if not pos:
            return

        ctx.damage_hero(pos.side, hero_dmg)
        ctx.buff_perm(es.EntityRef(weaver.uid), atk, hp)

    return _effect


def _make_dr_buff_all_friendlies(atk: int, hp: int):
    """Deathrattle: buff all friendly minions (combat buff)."""

    def _effect(ctx: EffectContext, event: Event, _trigger_uid: int) -> None:
        es = _event_system()
        pos = event.source_pos or (event.snapshot.pos if event.snapshot else None)
        if not pos:
            return
        player = ctx.players_by_uid.get(pos.side)
        if not player:
            return
        for unit in player.board:
            ctx.buff_combat(es.EntityRef(unit.uid), atk, hp)

    return _effect


def _make_dr_random_enemy_damage(damage: int):
    """Deathrattle: deal `damage` to a random enemy minion."""

    def _effect(ctx: EffectContext, event: Event, trigger_uid: int) -> None:
        es = _event_system()
        source_pos = event.source_pos or (event.snapshot.pos if event.snapshot else None)
        if not source_pos:
            return

        enemy_side = 1 - source_pos.side
        enemy_player = ctx.players_by_uid.get(enemy_side)

        if not enemy_player or not enemy_player.board:
            return

        target = random.choice(enemy_player.board)

        if target.has_divine_shield:
            target.tags.discard(Tags.DIVINE_SHIELD)
            ctx.emit_event(
                es.Event(
                    event_type=es.EventType.DIVINE_SHIELD_LOST,
                    source=es.EntityRef(target.uid),
                    source_pos=es.PosRef(side=enemy_side, zone=es.Zone.BOARD, slot=-1),
                )
            )
        else:
            target.cur_hp -= damage
            ctx.emit_event(
                es.Event(
                    event_type=es.EventType.MINION_DAMAGED,
                    source=es.EntityRef(trigger_uid),
                    target=es.EntityRef(target.uid),
                    value=damage,
                )
            )

    return _effect


def _make_on_friendly_summoned_type_buff(
    trigger_type: UnitType,
    atk: int,
    hp: int,
    exclude_self: bool,
    combat_buff: bool,
    gain_divine_shield: bool,
):
    """Deflect-o-Bot pattern: on friendly mech summoned (not self),
    buff self and optionally grant divine shield."""

    def _effect(ctx: EffectContext, event: Event, trigger_uid: int) -> None:
        es = _event_system()
        summoned_unit = ctx.resolve_unit(event.source)
        if not summoned_unit:
            return
        if trigger_type not in summoned_unit.types:
            return
        if exclude_self and summoned_unit.uid == trigger_uid:
            return
        deflecto = ctx.resolve_unit(es.EntityRef(trigger_uid))
        if not deflecto or not deflecto.is_alive:
            return
        if combat_buff:
            ctx.buff_combat(es.EntityRef(trigger_uid), atk, hp)
        else:
            ctx.buff_perm(es.EntityRef(trigger_uid), atk, hp)
        if gain_divine_shield:
            deflecto.tags.add(Tags.DIVINE_SHIELD)

    return _effect


def _make_deflect_o_bot_condition():
    """Condition: a friendly unit was summoned by someone else on the same side."""

    def _condition(ctx: EffectContext, event: Event, uid: int) -> bool:
        es = _event_system()
        return bool(
            event.source_pos
            and ctx.resolve_pos(es.EntityRef(uid))
            and event.source_pos.side == ctx.resolve_pos(es.EntityRef(uid)).side  # type: ignore
            and event.source
            and event.source.uid != uid
        )

    return _condition


def _make_end_of_turn_add_spell(spell_id: str, count: int):
    """End of turn: add spell(s) to hand (unit must be on board)."""

    def _effect(ctx: EffectContext, _event: Event, trigger_uid: int) -> None:
        es = _event_system()
        pos = ctx.resolve_pos(es.EntityRef(trigger_uid))
        if not pos:
            return
        for _ in range(count):
            ctx.add_spell_to_hand(pos.side, spell_id)

    return _effect


def _make_eot_buff_adjacent(atk: int, hp: int):
    """End of turn: buff adjacent units (perm buff)."""

    def _effect(ctx: EffectContext, _event: Event, trigger_uid: int) -> None:
        es = _event_system()
        pos = ctx.resolve_pos(es.EntityRef(trigger_uid))
        if not pos:
            return
        for _slot, unit in ctx.get_adjacent(pos.side, trigger_uid):
            ctx.buff_perm(es.EntityRef(unit.uid), atk, hp)

    return _effect


def _make_eot_buff_self(atk: int, hp: int):
    """End of turn: buff self (perm buff)."""

    def _effect(ctx: EffectContext, _event: Event, trigger_uid: int) -> None:
        es = _event_system()
        pos = ctx.resolve_pos(es.EntityRef(trigger_uid))
        if not pos:
            return
        ctx.buff_perm(es.EntityRef(trigger_uid), atk, hp)

    return _effect


def _make_eot_buff_board(atk: int, hp: int):
    """End of turn: buff all friendly board units (perm buff)."""

    def _effect(ctx: EffectContext, _event: Event, trigger_uid: int) -> None:
        es = _event_system()
        pos = ctx.resolve_pos(es.EntityRef(trigger_uid))
        if not pos:
            return
        for _slot, unit in ctx.iter_board_units(pos.side):
            ctx.buff_perm(es.EntityRef(unit.uid), atk, hp)

    return _effect


def _make_eot_buff_board_by_type(trigger_type: UnitType, atk: int, hp: int):
    """End of turn: buff all friendly board units of a type (perm buff)."""

    def _effect(ctx: EffectContext, _event: Event, trigger_uid: int) -> None:
        es = _event_system()
        pos = ctx.resolve_pos(es.EntityRef(trigger_uid))
        if not pos:
            return
        for _slot, unit in ctx.iter_board_units(pos.side):
            if trigger_type in unit.types:
                ctx.buff_perm(es.EntityRef(unit.uid), atk, hp)

    return _effect


def _make_soc_buff_friendly_type(trigger_type: UnitType, atk: int, hp: int):
    """SoC: give all friendly units of matching type +atk/+hp as combat buff."""

    def _effect(ctx: EffectContext, _event: Event, trigger_uid: int) -> None:
        es = _event_system()
        pos = ctx.resolve_pos(es.EntityRef(trigger_uid))
        if not pos:
            return
        for _slot, unit in ctx.iter_board_units(pos.side):
            if trigger_type in unit.types:
                ctx.buff_combat(es.EntityRef(unit.uid), atk, hp)

    return _effect


def _make_bc_buff_all_by_type(trigger_type: UnitType, atk: int, hp: int):
    """BC: give all friendly units of matching type +atk/+hp as perm buff."""

    def _effect(ctx: EffectContext, _event: Event, trigger_uid: int) -> None:
        es = _event_system()
        pos = ctx.resolve_pos(es.EntityRef(trigger_uid))
        if not pos:
            return
        for _slot, unit in ctx.iter_board_units(pos.side):
            if trigger_type in unit.types:
                ctx.buff_perm(es.EntityRef(unit.uid), atk, hp)

    return _effect


def _make_on_play_type_add_spell(
    trigger_type: UnitType, spell_id: str, count: int, exclude_self: bool
):
    """On any friendly play of a unit of trigger_type, add spell to hand."""

    def _effect(ctx: EffectContext, event: Event, trigger_uid: int) -> None:
        es = _event_system()
        played = ctx.resolve_unit(event.source)
        if not played or trigger_type not in played.types:
            return
        if exclude_self and event.source and event.source.uid == trigger_uid:
            return
        pos = ctx.resolve_pos(es.EntityRef(trigger_uid))
        if not pos:
            return
        for _ in range(count):
            ctx.add_spell_to_hand(pos.side, spell_id)

    return _effect


def _make_on_summoned_buff_random_other(trigger_type: UnitType, atk: int, hp: int):
    """When friendly of trigger_type is summoned, buff a random OTHER friendly of that type."""

    def _effect(ctx: EffectContext, event: Event, trigger_uid: int) -> None:
        es = _event_system()
        summoned = ctx.resolve_unit(event.source)
        if not summoned or trigger_type not in summoned.types:
            return
        # Determine side of the summoned unit
        source_pos = event.source_pos
        if not source_pos:
            return
        # Collect all other friendly units of matching type on the same board
        candidates = [
            unit
            for _slot, unit in ctx.iter_board_units(source_pos.side)
            if trigger_type in unit.types and unit.uid != summoned.uid
        ]
        if not candidates:
            return
        target = random.choice(candidates)
        ctx.buff_perm(es.EntityRef(target.uid), atk, hp)

    return _effect


def _make_sell_add_unit_v2(card_id: str):
    """On sell: add a specific unit to hand, resolving side via source_pos."""

    def _effect(ctx: EffectContext, event: Event, trigger_uid: int) -> None:
        if event.source_pos:
            side = event.source_pos.side
        else:
            side = next(iter(ctx.players_by_uid), None)
            if side is None:
                return
        ctx.add_unit_to_hand(side, card_id)

    return _effect


def _make_rally_buff_random_type(trigger_type: UnitType, atk: int, hp: int):
    """Rally: when this unit attacks, buff a random OTHER friendly of trigger_type (combat buff)."""

    def _effect(ctx: EffectContext, _event: Event, trigger_uid: int) -> None:
        es = _event_system()
        pos = ctx.resolve_pos(es.EntityRef(trigger_uid))
        if not pos:
            return
        candidates = [
            unit
            for _slot, unit in ctx.iter_board_units(pos.side)
            if trigger_type in unit.types and unit.uid != trigger_uid
        ]
        if not candidates:
            return
        target = random.choice(candidates)
        ctx.buff_combat(es.EntityRef(target.uid), atk, hp)

    return _effect


def _make_sell_get_random_by_type(unit_type: UnitType):
    """On sell: get a random unit of specific type from pool into hand."""

    def _effect(ctx: EffectContext, event: Event, trigger_uid: int) -> None:
        from .configs import CARD_DB

        if event.source_pos:
            side = event.source_pos.side
        else:
            side = next(iter(ctx.players_by_uid), None)
            if side is None:
                return
        if not ctx.card_pool:
            return
        player = ctx.players_by_uid.get(side)
        if not player or len(player.hand) >= 10:
            return
        # Collect all candidates from pool tiers matching the type
        candidates: list[str] = []
        for tier_cards in ctx.card_pool.tiers.values():
            for cid in tier_cards:
                data = CARD_DB.get(cid)
                if data and unit_type in data.get("type", []):
                    candidates.append(cid)
        if not candidates:
            return
        chosen = random.choice(candidates)
        # Remove from pool
        for tier_cards in ctx.card_pool.tiers.values():
            if chosen in tier_cards:
                tier_cards.remove(chosen)
                break
        from .entities import HandCard, Unit

        uid = ctx._uid_provider()
        new_unit = Unit.create_from_db(chosen, uid, side)
        player.hand.append(HandCard(uid=uid, unit=new_unit))

    return _effect


def _make_consume_for_random_friendly(trigger_type: UnitType):
    """BC: consume a random shop unit, give its stats to a random friendly of trigger_type."""

    def _effect(ctx: EffectContext, _event: Event, trigger_uid: int) -> None:
        es = _event_system()
        pos = ctx.resolve_pos(es.EntityRef(trigger_uid))
        if not pos:
            return
        result = ctx.consume_random_store_unit(pos.side)
        if not result:
            return
        gained_atk, gained_hp = result
        candidates = [
            unit for _slot, unit in ctx.iter_board_units(pos.side) if trigger_type in unit.types
        ]
        if not candidates:
            return
        target = random.choice(candidates)
        ctx.buff_perm(es.EntityRef(target.uid), gained_atk, gained_hp)

    return _effect


def _make_on_tavern_refresh_buff_rightmost_shop(
    atk: int, hp: int, give_reborn: bool, use_blood_gem: bool
):
    """After tavern refreshed: buff rightmost shop minion."""

    def _effect(ctx: EffectContext, event: Event, trigger_uid: int) -> None:
        es = _event_system()
        # Identify the player's side from the event or the unit's position
        source_pos = event.source_pos
        if source_pos:
            side = source_pos.side
        else:
            pos = ctx.resolve_pos(es.EntityRef(trigger_uid))
            if not pos:
                return
            side = pos.side
        store_units = ctx.iter_store_units(side)
        if not store_units:
            return
        # rightmost = highest slot index
        _slot, target = store_units[-1]
        if use_blood_gem:
            player = ctx.players_by_uid.get(side)
            if not player:
                return
            from .enums import MechanicType

            buff_atk, buff_hp = player.mechanics.get_stat(MechanicType.BLOOD_GEM)
            ctx.buff_perm(es.EntityRef(target.uid), buff_atk, buff_hp)
            ctx.buff_perm(es.EntityRef(target.uid), buff_atk, buff_hp)  # plays 2 blood gems
        else:
            ctx.buff_perm(es.EntityRef(target.uid), atk, hp)
        if give_reborn:
            target.tags.add(Tags.REBORN)

    return _effect


def _make_on_tavern_refresh_buff_rightmost_shop_condition():
    """Condition: the refreshed player is the one with this unit on board."""

    def _condition(ctx: EffectContext, event: Event, trigger_uid: int) -> bool:
        es = _event_system()
        pos = ctx.resolve_pos(es.EntityRef(trigger_uid))
        if not pos:
            return False
        source_pos = event.source_pos
        if not source_pos:
            return False
        return pos.side == source_pos.side

    return _condition


def _make_sell_buff_board_scaling(scaling_key: str, atk_per: int, hp_per: int):
    """On sell: buff board by (base + scaling * count) and increment counter."""

    def _effect(ctx: EffectContext, event: Event, trigger_uid: int) -> None:
        if event.source_pos:
            side = event.source_pos.side
        else:
            side = next(iter(ctx.players_by_uid), None)
            if side is None:
                return
        player = ctx.players_by_uid.get(side)
        if not player:
            return
        count = player.mechanics.get_scaling(scaling_key)
        buff_atk = atk_per * (count + 1)
        buff_hp = hp_per * (count + 1)
        player.mechanics.increment_scaling(scaling_key)
        es = _event_system()
        for _slot, unit in ctx.iter_board_units(side):
            ctx.buff_perm(es.EntityRef(unit.uid), buff_atk, buff_hp)

    return _effect


def _make_soc_damage_and_buff_adjacent(damage: int, atk: int, hp: int):
    """SoC: deal damage to adjacent units and buff their ATK."""

    def _effect(ctx: EffectContext, _event: Event, trigger_uid: int) -> None:
        es = _event_system()
        pos = ctx.resolve_pos(es.EntityRef(trigger_uid))
        if not pos:
            return
        for _slot, unit in ctx.get_adjacent(pos.side, trigger_uid):
            unit.cur_hp -= damage
            ctx.buff_combat(es.EntityRef(unit.uid), atk, hp)

    return _effect


def _make_on_hero_damaged_heal_and_buff_self(hp: int):
    """After hero takes damage: undo damage, buff self +hp."""

    def _effect(ctx: EffectContext, event: Event, trigger_uid: int) -> None:
        es = _event_system()
        # Determine which side took damage
        source_pos = event.source_pos
        if not source_pos:
            return
        damaged_side = source_pos.side
        # Only fire if the unit is on the same side as the hero that took damage
        unit_pos = ctx.resolve_pos(es.EntityRef(trigger_uid))
        if not unit_pos or unit_pos.side != damaged_side:
            return
        unit = ctx.resolve_unit(es.EntityRef(trigger_uid))
        if not unit or not unit.is_alive:
            return
        # Heal the hero by the damage value
        damage_amount = event.value or 0
        ctx.heal_hero(damaged_side, damage_amount)
        # Buff self
        ctx.buff_perm(es.EntityRef(trigger_uid), 0, hp)

    return _effect


def _make_eot_buff_adjacent_per_golden(atk: int, hp: int):
    """EOT: buff adjacent +atk/+hp, once per friendly golden minion (minimum 1)."""

    def _effect(ctx: EffectContext, _event: Event, trigger_uid: int) -> None:
        es = _event_system()
        pos = ctx.resolve_pos(es.EntityRef(trigger_uid))
        if not pos:
            return
        player = ctx.players_by_uid.get(pos.side)
        if not player:
            return
        # Count golden minions on board
        golden_count = sum(1 for u in player.board if u.is_golden)
        repeats = max(1, golden_count)
        for _slot, unit in ctx.get_adjacent(pos.side, trigger_uid):
            for _ in range(repeats):
                ctx.buff_perm(es.EntityRef(unit.uid), atk, hp)

    return _effect


def _make_sell_discover(base_tier: int, scaling_key: str):
    """On sell: discover a minion of tier (base_tier + scaling_counter)."""

    def _effect(ctx: EffectContext, event: Event, trigger_uid: int) -> None:
        if event.source_pos:
            side = event.source_pos.side
        else:
            side = next(iter(ctx.players_by_uid), None)
            if side is None:
                return
        player = ctx.players_by_uid.get(side)
        if not player or not ctx.card_pool:
            return
        count = player.mechanics.get_scaling(scaling_key) if scaling_key else 0
        player.mechanics.increment_scaling(scaling_key) if scaling_key else None
        discover_tier = min(6, base_tier + count)
        # Set a pending discovery request on the player
        from .entities import DiscoveryRequest

        player.pending_discovery_request = DiscoveryRequest(
            tier=discover_tier,
            exact_tier=False,
            source="Patient Scout",
        )

    return _effect


def _make_dr_add_spell(spell_id: str, count: int):
    """Deathrattle: add spell(s) to hand."""

    def _effect(ctx: EffectContext, event: Event, _trigger_uid: int) -> None:
        es = _event_system()
        pos = event.source_pos or (event.snapshot.pos if event.snapshot else None)
        if not pos:
            return
        for _ in range(count):
            ctx.add_spell_to_hand(pos.side, spell_id)

    return _effect


def _make_rally_add_spell(spell_id: str, count: int):
    """Rally: when this unit attacks, add spell(s) to hand."""

    def _effect(ctx: EffectContext, _event: Event, trigger_uid: int) -> None:
        es = _event_system()
        pos = ctx.resolve_pos(es.EntityRef(trigger_uid))
        if not pos:
            return
        for _ in range(count):
            ctx.add_spell_to_hand(pos.side, spell_id)

    return _effect


def _make_eot_buff_self_per_golden(atk_per: int, hp_per: int):
    """EOT: buff self +atk_per/+hp_per for each friendly golden minion."""

    def _effect(ctx: EffectContext, _event: Event, trigger_uid: int) -> None:
        es = _event_system()
        pos = ctx.resolve_pos(es.EntityRef(trigger_uid))
        if not pos:
            return
        player = ctx.players_by_uid.get(pos.side)
        if not player:
            return
        golden_count = sum(1 for u in player.board if u.is_golden)
        if golden_count == 0:
            return
        ctx.buff_perm(es.EntityRef(trigger_uid), atk_per * golden_count, hp_per * golden_count)

    return _effect


def _make_on_divine_shield_lost_add_spell(spell_id: str, count: int):
    """After a friendly minion loses Divine Shield, add spell to hand."""

    def _effect(ctx: EffectContext, event: Event, trigger_uid: int) -> None:
        es = _event_system()
        # Check that the shield-losing unit is on the same side
        unit_pos = ctx.resolve_pos(es.EntityRef(trigger_uid))
        if not unit_pos:
            return
        source_pos = event.source_pos
        if not source_pos or source_pos.side != unit_pos.side:
            return
        for _ in range(count):
            ctx.add_spell_to_hand(unit_pos.side, spell_id)

    return _effect


def _make_bc_buff_all_by_type_include_hand(trigger_type: UnitType, atk: int, hp: int):
    """BC: give all OTHER friendly units of type in hand AND board +atk/+hp."""

    def _effect(ctx: EffectContext, _event: Event, trigger_uid: int) -> None:
        es = _event_system()
        pos = ctx.resolve_pos(es.EntityRef(trigger_uid))
        if not pos:
            return
        player = ctx.players_by_uid.get(pos.side)
        if not player:
            return
        targets = []
        for _slot, unit in ctx.iter_board_units(pos.side):
            if trigger_type in unit.types and unit.uid != trigger_uid:
                targets.append(unit)
        for hc in player.hand:
            if hc.unit and trigger_type in hc.unit.types and hc.unit.uid != trigger_uid:
                targets.append(hc.unit)
        for unit in targets:
            ctx.buff_perm(es.EntityRef(unit.uid), atk, hp)

    return _effect


def _make_rally_damage_own_board(damage: int):
    """Rally: deal damage to all other friendly minions."""

    def _effect(ctx: EffectContext, _event: Event, trigger_uid: int) -> None:
        es = _event_system()
        pos = ctx.resolve_pos(es.EntityRef(trigger_uid))
        if not pos:
            return
        for _slot, unit in ctx.iter_board_units(pos.side):
            if unit.uid != trigger_uid:
                unit.cur_hp -= damage

    return _effect


def _make_dr_modify_mechanic(mechanic: MechanicType, atk: int, hp: int):
    """Deathrattle: permanently modify a global mechanic stat."""

    def _effect(ctx: EffectContext, event: Event, _trigger_uid: int) -> None:
        pos = event.source_pos or (event.snapshot.pos if event.snapshot else None)
        if not pos:
            return
        player = ctx.players_by_uid.get(pos.side)
        if not player:
            return
        player.mechanics.modify_stat(mechanic, atk, hp)

    return _effect


def _make_soc_buff_random_friendly_type_and_ds(trigger_type: UnitType, atk: int, hp: int):
    """SoC: give another friendly unit of type +atk/+hp and Divine Shield."""

    def _effect(ctx: EffectContext, _event: Event, trigger_uid: int) -> None:
        es = _event_system()
        pos = ctx.resolve_pos(es.EntityRef(trigger_uid))
        if not pos:
            return
        candidates = [
            unit
            for _slot, unit in ctx.iter_board_units(pos.side)
            if trigger_type in unit.types and unit.uid != trigger_uid
        ]
        if not candidates:
            return
        target = random.choice(candidates)
        ctx.buff_combat(es.EntityRef(target.uid), atk, hp)
        target.tags.add(Tags.DIVINE_SHIELD)

    return _effect


def _make_on_self_damaged_buff_board(atk: int, hp: int):
    """When this minion takes damage, buff all other friendly minions."""

    def _effect(ctx: EffectContext, event: Event, trigger_uid: int) -> None:
        es = _event_system()
        # Check target is self
        if not event.target or event.target.uid != trigger_uid:
            return
        unit = ctx.resolve_unit(es.EntityRef(trigger_uid))
        if not unit or not unit.is_alive:
            return
        pos = ctx.resolve_pos(es.EntityRef(trigger_uid))
        if not pos:
            return
        for _slot, other in ctx.iter_board_units(pos.side):
            if other.uid != trigger_uid:
                ctx.buff_perm(es.EntityRef(other.uid), atk, hp)

    return _effect


def _make_felemental_bc():
    """BC: give all tavern minions +2/+1 this game (modifies ELEMENTAL_BUFF mechanic for shop buffing)."""

    def _effect(ctx: EffectContext, _event: Event, trigger_uid: int) -> None:
        es = _event_system()
        pos = ctx.resolve_pos(es.EntityRef(trigger_uid))
        if not pos:
            return
        # Buff all current shop units
        for _slot, unit in ctx.iter_store_units(pos.side):
            ctx.buff_perm(es.EntityRef(unit.uid), 2, 1)
        # Also buff future shop units via mechanic (reuse ELEMENTAL_BUFF for tavern)
        player = ctx.players_by_uid.get(pos.side)
        if player:
            from .enums import MechanicType

            player.mechanics.modify_stat(MechanicType.ELEMENTAL_BUFF, 2, 1)

    return _effect


def _make_on_friendly_reborn_buff_self(atk: int, hp: int):
    """After a friendly minion triggers Reborn, buff self permanently."""

    def _effect(ctx: EffectContext, event: Event, trigger_uid: int) -> None:
        es = _event_system()
        unit_pos = ctx.resolve_pos(es.EntityRef(trigger_uid))
        if not unit_pos:
            return
        # MINION_SUMMONED with meta=1 means it was a Reborn summon
        if event.meta != 1:
            return
        source_pos = event.source_pos
        if not source_pos or source_pos.side != unit_pos.side:
            return
        # Don't buff self if it's this unit that rebore (though possible)
        unit = ctx.resolve_unit(es.EntityRef(trigger_uid))
        if not unit or not unit.is_alive:
            return
        ctx.buff_perm(es.EntityRef(trigger_uid), atk, hp)

    return _effect


def _make_dr_buff_friendly_type_global(trigger_type: UnitType, atk: int, hp: int):
    """Deathrattle: permanently buff all friendly units of type in hand + board."""

    def _effect(ctx: EffectContext, event: Event, _trigger_uid: int) -> None:
        es = _event_system()
        pos = event.source_pos or (event.snapshot.pos if event.snapshot else None)
        if not pos:
            return
        player = ctx.players_by_uid.get(pos.side)
        if not player:
            return
        for unit in player.board:
            if trigger_type in unit.types:
                ctx.buff_perm(es.EntityRef(unit.uid), atk, hp)
        for hc in player.hand:
            if hc.unit and trigger_type in hc.unit.types:
                ctx.buff_perm(es.EntityRef(hc.unit.uid), atk, hp)

    return _effect


def _make_dr_buff_shop(atk: int, hp: int):
    """Deathrattle: permanently buff all shop minions."""

    def _effect(ctx: EffectContext, event: Event, _trigger_uid: int) -> None:
        es = _event_system()
        pos = event.source_pos or (event.snapshot.pos if event.snapshot else None)
        if not pos:
            return
        for _slot, unit in ctx.iter_store_units(pos.side):
            ctx.buff_perm(es.EntityRef(unit.uid), atk, hp)
        # Also persist the buff for future shop minions via ELEMENTAL_BUFF mechanic (hp only)
        player = ctx.players_by_uid.get(pos.side)
        if player:
            player.mechanics.modify_stat(MechanicType.ELEMENTAL_BUFF, atk, hp)

    return _effect


def _make_dr_buff_hand_random(atk: int, hp: int):
    """Deathrattle: give a random minion in hand +atk/+hp."""

    def _effect(ctx: EffectContext, event: Event, _trigger_uid: int) -> None:
        es = _event_system()
        pos = event.source_pos or (event.snapshot.pos if event.snapshot else None)
        if not pos:
            return
        player = ctx.players_by_uid.get(pos.side)
        if not player:
            return
        candidates = [hc.unit for hc in player.hand if hc.unit is not None]
        if not candidates:
            return
        target = random.choice(candidates)
        ctx.buff_perm(es.EntityRef(target.uid), atk, hp)

    return _effect


def _make_on_friendly_attack_buff_attacker(trigger_type: UnitType, atk: int, hp: int):
    """When another friendly unit of type attacks, buff that attacker permanently."""

    def _effect(ctx: EffectContext, event: Event, trigger_uid: int) -> None:
        es = _event_system()
        attacker = ctx.resolve_unit(event.source)
        if not attacker or trigger_type not in attacker.types:
            return
        if attacker.uid == trigger_uid:
            return
        # Check same side
        pos = ctx.resolve_pos(es.EntityRef(trigger_uid))
        if not pos:
            return
        att_pos = ctx.resolve_pos(es.EntityRef(attacker.uid))
        if not att_pos or att_pos.side != pos.side:
            return
        ctx.buff_perm(es.EntityRef(attacker.uid), atk, hp)

    return _effect


def _make_on_friendly_attack_buff_trigger(trigger_type: UnitType, atk: int, hp: int):
    """When another friendly unit of type attacks, buff the TRIGGER unit (self) permanently."""

    def _effect(ctx: EffectContext, event: Event, trigger_uid: int) -> None:
        es = _event_system()
        attacker = ctx.resolve_unit(event.source)
        if not attacker or trigger_type not in attacker.types:
            return
        if attacker.uid == trigger_uid:
            return
        pos = ctx.resolve_pos(es.EntityRef(trigger_uid))
        if not pos:
            return
        att_pos = ctx.resolve_pos(es.EntityRef(attacker.uid))
        if not att_pos or att_pos.side != pos.side:
            return
        ctx.buff_perm(es.EntityRef(trigger_uid), atk, hp)

    return _effect


def _make_on_friendly_play_type_buff_self_in_hand(trigger_type: UnitType, atk: int, hp: int):
    """While in hand, when a friendly unit of type is played, buff self."""

    def _effect(ctx: EffectContext, event: Event, trigger_uid: int) -> None:
        es = _event_system()
        played = ctx.resolve_unit(event.source)
        if not played or trigger_type not in played.types:
            return
        # Find the trigger unit in the player's hand
        for side, player in ctx.players_by_uid.items():
            for hc in player.hand:
                if hc.unit and hc.unit.uid == trigger_uid:
                    # Must be the same side as played unit
                    src_pos = event.source_pos
                    if src_pos and src_pos.side == side:
                        ctx.buff_perm(es.EntityRef(trigger_uid), atk, hp)
                    return

    return _effect


def _make_on_spell_cast_buff_self(atk: int, hp: int):
    """When a tavern spell is played, buff self +atk/+hp."""

    def _effect(ctx: EffectContext, event: Event, trigger_uid: int) -> None:
        es = _event_system()
        pos = ctx.resolve_pos(es.EntityRef(trigger_uid))
        if not pos:
            return
        # SPELL_PLAYED event source_pos.side must match owner
        source_pos = event.source_pos
        if not source_pos or source_pos.side != pos.side:
            return
        unit = ctx.resolve_unit(es.EntityRef(trigger_uid))
        if not unit or not unit.is_alive:
            return
        ctx.buff_perm(es.EntityRef(trigger_uid), atk, hp)

    return _effect


def _make_on_gain_gold_buff_self(atk: int, hp: int):
    """After gaining gold (tavern coin played), buff self +atk/+hp."""

    def _effect(ctx: EffectContext, event: Event, trigger_uid: int) -> None:
        es = _event_system()
        pos = ctx.resolve_pos(es.EntityRef(trigger_uid))
        if not pos:
            return
        source_pos = event.source_pos
        if not source_pos or source_pos.side != pos.side:
            return
        unit = ctx.resolve_unit(es.EntityRef(trigger_uid))
        if not unit or not unit.is_alive:
            return
        ctx.buff_perm(es.EntityRef(trigger_uid), atk, hp)

    return _effect


def _make_dr_damage_all_minions(damage: int):
    """Deathrattle: deal damage to ALL minions on both sides."""

    def _effect(ctx: EffectContext, event: Event, trigger_uid: int) -> None:
        es = _event_system()
        pos = event.source_pos or (event.snapshot.pos if event.snapshot else None)
        if not pos:
            return
        for side in list(ctx.players_by_uid.keys()):
            player = ctx.players_by_uid.get(side)
            if not player:
                continue
            for unit in list(player.board):
                if unit.has_divine_shield:
                    unit.tags.discard(Tags.DIVINE_SHIELD)
                    ctx.emit_event(
                        es.Event(
                            event_type=es.EventType.DIVINE_SHIELD_LOST,
                            source=es.EntityRef(unit.uid),
                            source_pos=es.PosRef(side=side, zone=es.Zone.BOARD, slot=-1),
                        )
                    )
                else:
                    unit.cur_hp -= damage

    return _effect


def _make_soc_buff_all_friendly_type(trigger_type: UnitType, atk: int, hp: int):
    """SoC: permanently buff all friendly units of type +atk/+hp."""

    def _effect(ctx: EffectContext, _event: Event, trigger_uid: int) -> None:
        es = _event_system()
        pos = ctx.resolve_pos(es.EntityRef(trigger_uid))
        if not pos:
            return
        for _slot, unit in ctx.iter_board_units(pos.side):
            if trigger_type in unit.types:
                ctx.buff_perm(es.EntityRef(unit.uid), atk, hp)

    return _effect


def _make_soc_give_friendly_type_reborn(trigger_type: UnitType):
    """SoC: give a random friendly unit of type Reborn."""

    def _effect(ctx: EffectContext, _event: Event, trigger_uid: int) -> None:
        es = _event_system()
        pos = ctx.resolve_pos(es.EntityRef(trigger_uid))
        if not pos:
            return
        candidates = [
            unit
            for _slot, unit in ctx.iter_board_units(pos.side)
            if trigger_type in unit.types
            and unit.uid != trigger_uid
            and Tags.REBORN not in unit.tags
        ]
        if not candidates:
            return
        target = random.choice(candidates)
        target.tags.add(Tags.REBORN)

    return _effect


def _make_bc_add_random_unit(unit_type: Optional[UnitType], tier: Optional[int]):
    """BC: add a random unit of type (or tier) from pool to hand."""

    def _effect(ctx: EffectContext, _event: Event, trigger_uid: int) -> None:
        from .configs import CARD_DB

        es = _event_system()
        pos = ctx.resolve_pos(es.EntityRef(trigger_uid))
        if not pos:
            return
        player = ctx.players_by_uid.get(pos.side)
        if not player or not ctx.card_pool:
            return
        candidates: list[str] = []
        for t, tier_cards in ctx.card_pool.tiers.items():
            if tier is not None and t != tier:
                continue
            for cid in tier_cards:
                data = CARD_DB.get(cid)
                if data and data.get("is_token"):
                    continue
                if unit_type is not None and unit_type not in data.get("type", []):
                    continue
                candidates.append(cid)
        if not candidates:
            return
        chosen = random.choice(candidates)
        for tier_cards in ctx.card_pool.tiers.values():
            if chosen in tier_cards:
                tier_cards.remove(chosen)
                break
        from .entities import Unit, HandCard

        uid = ctx._uid_provider()
        new_unit = Unit.create_from_db(chosen, uid, pos.side)
        player.hand.append(HandCard(uid=uid, unit=new_unit))

    return _effect


def _make_bc_gain_free_refreshes(count: int):
    """BC: gain N free refreshes immediately."""

    def _effect(ctx: EffectContext, _event: Event, trigger_uid: int) -> None:
        es = _event_system()
        pos = ctx.resolve_pos(es.EntityRef(trigger_uid))
        if not pos:
            return
        player = ctx.players_by_uid.get(pos.side)
        if not player:
            return
        player.free_refreshes = getattr(player, "free_refreshes", 0) + count

    return _effect


def _make_on_ds_lost_buff_unit(atk: int, hp: int):
    """After a friendly minion loses Divine Shield, give it +atk/+hp permanently."""

    def _effect(ctx: EffectContext, event: Event, trigger_uid: int) -> None:
        es = _event_system()
        unit_pos = ctx.resolve_pos(es.EntityRef(trigger_uid))
        if not unit_pos:
            return
        source_pos = event.source_pos
        if not source_pos or source_pos.side != unit_pos.side:
            return
        # Buff the unit that lost the shield
        if event.source:
            ctx.buff_perm(event.source, atk, hp)

    return _effect


def _make_rally_buff_all_others_blood_gems(count: int):
    """Rally: play `count` blood gems on every other friendly minion."""

    def _effect(ctx: EffectContext, _event: Event, trigger_uid: int) -> None:
        es = _event_system()
        pos = ctx.resolve_pos(es.EntityRef(trigger_uid))
        if not pos:
            return
        player = ctx.players_by_uid.get(pos.side)
        if not player:
            return
        gem_atk, gem_hp = player.mechanics.get_stat(MechanicType.BLOOD_GEM)
        for _slot, unit in ctx.iter_board_units(pos.side):
            if unit.uid != trigger_uid:
                for _ in range(count):
                    ctx.buff_perm(es.EntityRef(unit.uid), gem_atk, gem_hp)

    return _effect


def _make_eot_add_random_spell():
    """End of turn: add a random tavern spell to hand."""

    def _effect(ctx: EffectContext, _event: Event, trigger_uid: int) -> None:
        from .configs import SPELL_DB

        es = _event_system()
        pos = ctx.resolve_pos(es.EntityRef(trigger_uid))
        if not pos:
            return
        pool_spells = [
            sid
            for sid, data in SPELL_DB.items()
            if data.get("pool", True) and sid != SpellIDs.TRIPLET_REWARD
        ]
        if not pool_spells:
            pool_spells = [SpellIDs.TAVERN_COIN]
        chosen = random.choice(pool_spells)
        ctx.add_spell_to_hand(pos.side, chosen)

    return _effect


def _make_avenge_add_spell(spell_id: str, count: int):
    """Avenge fires: add spell to hand (used for Spirit Drake, etc.)."""

    def _effect(ctx: EffectContext, _event: Event, trigger_uid: int) -> None:
        es = _event_system()
        pos = ctx.resolve_pos(es.EntityRef(trigger_uid))
        if not pos:
            return
        for _ in range(count):
            ctx.add_spell_to_hand(pos.side, spell_id)

    return _effect


def _make_avenge_add_random_unit():
    """Avenge fires: add a random battlecry unit to hand (Witchwing Nestmatron)."""

    def _effect(ctx: EffectContext, _event: Event, trigger_uid: int) -> None:
        from .configs import CARD_DB

        es = _event_system()
        pos = ctx.resolve_pos(es.EntityRef(trigger_uid))
        if not pos:
            return
        player = ctx.players_by_uid.get(pos.side)
        if not player or not ctx.card_pool:
            return
        candidates: list[str] = []
        for tier_cards in ctx.card_pool.tiers.values():
            for cid in tier_cards:
                data = CARD_DB.get(cid)
                if data and not data.get("is_token"):
                    candidates.append(cid)
        if not candidates:
            return
        chosen = random.choice(candidates)
        for tier_cards in ctx.card_pool.tiers.values():
            if chosen in tier_cards:
                tier_cards.remove(chosen)
                break
        from .entities import Unit, HandCard

        uid = ctx._uid_provider()
        new_unit = Unit.create_from_db(chosen, uid, pos.side)
        player.hand.append(HandCard(uid=uid, unit=new_unit))

    return _effect


# ---------------------------------------------------------------------------
# New factory functions for T4-T7 EffectDef types
# ---------------------------------------------------------------------------


def _make_on_spell_cast_buff_board(atk: int, hp: int, trigger_type: Optional[UnitType]):
    """When a spell is cast, buff all friendly minions (or of a type)."""

    def _effect(ctx: EffectContext, event: Event, trigger_uid: int) -> None:
        es = _event_system()
        unit_pos = ctx.resolve_pos(es.EntityRef(trigger_uid))
        if not unit_pos:
            return
        # Only fire for same-side spells
        source_pos = event.source_pos
        if source_pos and source_pos.side != unit_pos.side:
            return
        for _slot, unit in ctx.iter_board_units(unit_pos.side):
            if trigger_type is None or trigger_type in unit.types:
                ctx.buff_perm(es.EntityRef(unit.uid), atk, hp)

    return _effect


def _make_dr_summon_taunt_token(token_id: str, count: int):
    """Deathrattle: summon N tokens and give them Taunt."""

    def _effect(ctx: EffectContext, event: Event, _trigger_uid: int) -> None:
        pos = event.source_pos or (event.snapshot.pos if event.snapshot else None)
        if not pos:
            return
        for _ in range(count):
            ref = ctx.summon(pos.side, token_id, pos.slot)
            if ref:
                unit = ctx.resolve_unit(ref)
                if unit:
                    unit.tags.add(Tags.TAUNT)

    return _effect


def _make_soc_buff_self_by_highest_ally_atk():
    """SoC: set own attack to the highest friendly attack value."""

    def _effect(ctx: EffectContext, _event: Event, trigger_uid: int) -> None:
        es = _event_system()
        pos = ctx.resolve_pos(es.EntityRef(trigger_uid))
        if not pos:
            return
        max_atk = max(
            (
                unit.cur_atk
                for _slot, unit in ctx.iter_board_units(pos.side)
                if unit.uid != trigger_uid
            ),
            default=0,
        )
        if max_atk <= 0:
            return
        unit = ctx.resolve_unit(es.EntityRef(trigger_uid))
        if unit:
            ctx.buff_perm(es.EntityRef(trigger_uid), max_atk, 0)

    return _effect


def _make_soc_buff_self_by_highest_board_atk():
    """SoC: set own stats to match the highest-attack friendly minion."""

    def _effect(ctx: EffectContext, _event: Event, trigger_uid: int) -> None:
        es = _event_system()
        pos = ctx.resolve_pos(es.EntityRef(trigger_uid))
        if not pos:
            return
        best = max(
            (
                (unit.cur_atk, unit.cur_hp)
                for _slot, unit in ctx.iter_board_units(pos.side)
                if unit.uid != trigger_uid
            ),
            default=(0, 0),
        )
        if best[0] <= 0:
            return
        unit = ctx.resolve_unit(es.EntityRef(trigger_uid))
        if unit:
            gain_atk = max(0, best[0] - unit.cur_atk)
            gain_hp = max(0, best[1] - unit.cur_hp)
            ctx.buff_perm(es.EntityRef(trigger_uid), gain_atk, gain_hp)

    return _effect


def _make_bc_make_golden_friendly_by_tier(max_tier: int):
    """BC: make a random non-golden friendly minion from tier <= max_tier golden."""

    def _effect(ctx: EffectContext, _event: Event, trigger_uid: int) -> None:
        from .configs import CARD_DB

        es = _event_system()
        pos = ctx.resolve_pos(es.EntityRef(trigger_uid))
        if not pos:
            return
        candidates = [
            unit
            for _slot, unit in ctx.iter_board_units(pos.side)
            if unit.uid != trigger_uid
            and not unit.is_golden
            and CARD_DB.get(unit.card_id, {}).get("tier", 99) <= max_tier
        ]
        if not candidates:
            return
        target = random.choice(candidates)
        target.is_golden = True
        # Double stats for golden
        ctx.buff_perm(es.EntityRef(target.uid), target.cur_atk, target.cur_hp)

    return _effect


def _make_rally_buff_friendly_type_atk(trigger_type: UnitType, atk: int):
    """Rally: give all other friendly units of type +atk permanently."""

    def _effect(ctx: EffectContext, _event: Event, trigger_uid: int) -> None:
        es = _event_system()
        pos = ctx.resolve_pos(es.EntityRef(trigger_uid))
        if not pos:
            return
        for _slot, unit in ctx.iter_board_units(pos.side):
            if unit.uid != trigger_uid and trigger_type in unit.types:
                ctx.buff_perm(es.EntityRef(unit.uid), atk, 0)

    return _effect


def _make_on_friendly_beast_damaged_buff_self(hp: int):
    """When another friendly Beast takes damage, buff self +hp permanently."""

    def _effect(ctx: EffectContext, event: Event, trigger_uid: int) -> None:
        es = _event_system()
        # Determine who took damage
        if not event.target:
            return
        target = ctx.resolve_unit(event.target)
        if not target or UnitType.BEAST not in target.types:
            return
        if target.uid == trigger_uid:
            return
        unit_pos = ctx.resolve_pos(es.EntityRef(trigger_uid))
        if not unit_pos:
            return
        # Must be friendly
        target_pos = ctx.resolve_pos(event.target)
        if target_pos and target_pos.side != unit_pos.side:
            return
        ctx.buff_perm(es.EntityRef(trigger_uid), 0, hp)

    return _effect


def _make_on_friendly_beast_damaged_buff_other(atk: int, hp: int):
    """When a friendly Beast takes damage, give a different friendly Beast +atk/+hp."""

    def _effect(ctx: EffectContext, event: Event, trigger_uid: int) -> None:
        es = _event_system()
        if not event.target:
            return
        damaged = ctx.resolve_unit(event.target)
        if not damaged or UnitType.BEAST not in damaged.types:
            return
        unit_pos = ctx.resolve_pos(es.EntityRef(trigger_uid))
        if not unit_pos:
            return
        target_pos = ctx.resolve_pos(event.target)
        if target_pos and target_pos.side != unit_pos.side:
            return
        candidates = [
            unit
            for _slot, unit in ctx.iter_board_units(unit_pos.side)
            if unit.uid != damaged.uid and UnitType.BEAST in unit.types
        ]
        if not candidates:
            return
        target = random.choice(candidates)
        ctx.buff_perm(es.EntityRef(target.uid), atk, hp)

    return _effect


def _make_avenge_buff_friendly_type_global(
    threshold: int, trigger_type: UnitType, atk: int, hp: int
):
    """Avenge(N): give all friendly units of type +atk globally."""

    # NOTE: This is registered in AVENGE_REGISTRY and handled by avenge system
    # The avenge system calls this effect when threshold is met
    def _effect(ctx: EffectContext, _event: Event, trigger_uid: int) -> None:
        es = _event_system()
        pos = ctx.resolve_pos(es.EntityRef(trigger_uid))
        if not pos:
            return
        player = ctx.players_by_uid.get(pos.side)
        if not player:
            return
        for unit in player.board:
            if trigger_type in unit.types:
                ctx.buff_perm(es.EntityRef(unit.uid), atk, hp)
        for hc in player.hand:
            if hc.unit and trigger_type in hc.unit.types:
                ctx.buff_perm(es.EntityRef(hc.unit.uid), atk, hp)

    return _effect


def _make_dr_destroy_killer():
    """Deathrattle: destroy the minion that killed this."""

    def _effect(ctx: EffectContext, event: Event, _trigger_uid: int) -> None:
        # event.meta stores killer uid
        killer_uid = event.meta
        if not killer_uid:
            return
        es = _event_system()
        killer = ctx.resolve_unit(es.EntityRef(killer_uid))
        if killer and killer.is_alive:
            killer.cur_hp = 0

    return _effect


def _make_sell_for_gold_conditional(amount: int):
    """Sells for extra gold if player lost last combat."""

    def _effect(ctx: EffectContext, event: Event, trigger_uid: int) -> None:
        if event.source_pos:
            side = event.source_pos.side
        else:
            side = next(iter(ctx.players_by_uid), None)
            if side is None:
                return
        player = ctx.players_by_uid.get(side)
        if not player:
            return
        # Check if lost last combat (simplified: always give extra)
        lost_last = getattr(player, "lost_last_combat", False)
        gold = amount if lost_last else 1
        ctx.gain_gold(side, gold - 1)  # -1 because default sell already gives 1

    return _effect


def _make_dr_buff_all_friendlies_global(trigger_type: UnitType, atk: int, hp: int):
    """On death: permanently buff all surviving friendlies of type (Goldrinn)."""

    def _effect(ctx: EffectContext, event: Event, _trigger_uid: int) -> None:
        es = _event_system()
        pos = event.source_pos or (event.snapshot.pos if event.snapshot else None)
        if not pos:
            return
        for _slot, unit in ctx.iter_board_units(pos.side):
            if trigger_type in unit.types or trigger_type == UnitType.ALL:
                ctx.buff_combat(es.EntityRef(unit.uid), atk, hp)

    return _effect


def _make_eot_buff_friendly_type_naga(atk: int, hp: int):
    """EoT: give all other friendly Naga +atk/+hp."""

    def _effect(ctx: EffectContext, _event: Event, trigger_uid: int) -> None:
        es = _event_system()
        pos = ctx.resolve_pos(es.EntityRef(trigger_uid))
        if not pos:
            return
        for _slot, unit in ctx.iter_board_units(pos.side):
            if unit.uid != trigger_uid and UnitType.NAGA in unit.types:
                ctx.buff_perm(es.EntityRef(unit.uid), atk, hp)

    return _effect


def _make_on_friendly_demon_damage_buff(atk: int, hp: int):
    """After a friendly Demon deals damage, buff other friendlies +atk/+hp."""

    def _effect(ctx: EffectContext, event: Event, trigger_uid: int) -> None:
        es = _event_system()
        unit_pos = ctx.resolve_pos(es.EntityRef(trigger_uid))
        if not unit_pos:
            return
        # Check source is a friendly demon
        if not event.source:
            return
        src = ctx.resolve_unit(event.source)
        if not src or UnitType.DEMON not in src.types:
            return
        src_pos = ctx.resolve_pos(event.source)
        if not src_pos or src_pos.side != unit_pos.side:
            return
        for _slot, unit in ctx.iter_board_units(unit_pos.side):
            if unit.uid != src.uid:
                ctx.buff_perm(es.EntityRef(unit.uid), atk, hp)

    return _effect


def _make_eot_consume_tavern_for_demon():
    """EoT: each friendly Demon consumes a tavern minion for its stats."""

    def _effect(ctx: EffectContext, _event: Event, trigger_uid: int) -> None:
        es = _event_system()
        pos = ctx.resolve_pos(es.EntityRef(trigger_uid))
        if not pos:
            return
        demons = [
            unit for _slot, unit in ctx.iter_board_units(pos.side) if UnitType.DEMON in unit.types
        ]
        for demon in demons:
            result = ctx.consume_random_store_unit(pos.side)
            if result:
                gained_atk, gained_hp = result
                ctx.buff_perm(es.EntityRef(demon.uid), gained_atk, gained_hp)

    return _effect


def _make_soc_buff_friendly_type_scaling(trigger_type: UnitType, atk: int, hp: int):
    """SoC: buff all friendly units of type, scaling with a play counter."""

    def _effect(ctx: EffectContext, _event: Event, trigger_uid: int) -> None:
        es = _event_system()
        pos = ctx.resolve_pos(es.EntityRef(trigger_uid))
        if not pos:
            return
        for _slot, unit in ctx.iter_board_units(pos.side):
            if trigger_type in unit.types and unit.uid != trigger_uid:
                ctx.buff_combat(es.EntityRef(unit.uid), atk, hp)

    return _effect


def _make_eot_trigger_adjacent_battlecry():
    """EoT: trigger the battlecry of adjacent minions."""

    def _effect(ctx: EffectContext, _event: Event, trigger_uid: int) -> None:
        es = _event_system()
        pos = ctx.resolve_pos(es.EntityRef(trigger_uid))
        if not pos:
            return
        for _slot, unit in ctx.get_adjacent(pos.side, trigger_uid):
            # Re-fire MINION_PLAYED for adjacent unit to trigger its battlecry
            ctx.emit_event(
                es.Event(
                    event_type=es.EventType.MINION_PLAYED,
                    source=es.EntityRef(unit.uid),
                    source_pos=es.PosRef(side=pos.side, zone=es.Zone.BOARD, slot=_slot),
                )
            )

    return _effect


def _make_rally_deal_damage_equal_to_atk():
    """Rally: deal damage equal to this minion's Attack to a random enemy."""

    def _effect(ctx: EffectContext, _event: Event, trigger_uid: int) -> None:
        es = _event_system()
        pos = ctx.resolve_pos(es.EntityRef(trigger_uid))
        if not pos:
            return
        attacker = ctx.resolve_unit(es.EntityRef(trigger_uid))
        if not attacker:
            return
        enemy_side = 1 - pos.side
        candidates = [unit for _slot, unit in ctx.iter_board_units(enemy_side) if unit.is_alive]
        if not candidates:
            return
        target = random.choice(candidates)
        target.cur_hp -= attacker.cur_atk

    return _effect


def _make_dr_give_friendlies_scaling(buff_atk: int, buff_hp: int, self_damage: int):
    """DR: give all friendly minions +atk/+hp and deal self_damage to them."""

    def _effect(ctx: EffectContext, event: Event, _trigger_uid: int) -> None:
        es = _event_system()
        pos = event.source_pos or (event.snapshot.pos if event.snapshot else None)
        if not pos:
            return
        for _slot, unit in ctx.iter_board_units(pos.side):
            ctx.buff_perm(es.EntityRef(unit.uid), buff_atk, buff_hp)
            if self_damage > 0:
                unit.cur_hp -= self_damage

    return _effect


# ---------------------------------------------------------------------------
# build_trigger_registry  →  produces the same dict as original TRIGGER_REGISTRY
# ---------------------------------------------------------------------------


def build_trigger_registry() -> Dict[str, list]:
    es = _event_system()
    TriggerDef = es.TriggerDef
    EventType = es.EventType

    registry: Dict[str, list] = {}

    for card in ALL_CARDS:
        triggers: list = []

        for eff in card.effects:
            # --- Deathrattle: summon token(s) ---
            if isinstance(eff, DeathrattleSummon):
                triggers.append(
                    TriggerDef(
                        event_type=EventType.MINION_DIED,
                        condition=_is_self_death,
                        effect=_make_dr_summon(eff.token_id, eff.count),
                        name=f"{card.name} Deathrattle",
                    )
                )

            # --- Deathrattle: summon token(s) with extra tag ---
            elif isinstance(eff, DeathrattleSummonWithTag):
                triggers.append(
                    TriggerDef(
                        event_type=EventType.MINION_DIED,
                        condition=_is_self_death,
                        effect=_make_dr_summon_with_tag(eff.token_id, eff.count, eff.tag),
                        name=f"{card.name} Deathrattle",
                    )
                )

            # --- Battlecry: summon unit at right ---
            elif isinstance(eff, BattlecrySummonAtRight):
                triggers.append(
                    TriggerDef(
                        event_type=EventType.MINION_PLAYED,
                        condition=_is_self_play,
                        effect=_make_battlecry_summon_at_right(eff.token_id),
                        name=f"{card.name} Battlecry",
                        priority=10,
                    )
                )

            # --- Battlecry: gain gold ---
            elif isinstance(eff, BattlecryGainGold):
                triggers.append(
                    TriggerDef(
                        event_type=EventType.MINION_PLAYED,
                        condition=_is_self_play,
                        effect=_make_battlecry_gain_gold(eff.amount),
                        name=f"{card.name} Battlecry",
                        priority=10,
                    )
                )

            # --- Battlecry: add spell to hand ---
            elif isinstance(eff, BattlecryAddSpell):
                triggers.append(
                    TriggerDef(
                        event_type=EventType.MINION_PLAYED,
                        condition=_is_self_play,
                        effect=_make_battlecry_add_spell(eff.spell_id, eff.count),
                        name=f"{card.name} Battlecry",
                        priority=10,
                    )
                )

            # --- Battlecry: spell discount ---
            elif isinstance(eff, BattlecrySpellDiscount):
                triggers.append(
                    TriggerDef(
                        event_type=EventType.MINION_PLAYED,
                        condition=_is_self_play,
                        effect=_make_battlecry_spell_discount(eff.amount),
                        name=f"{card.name} Battlecry",
                        priority=10,
                    )
                )

            # --- Battlecry: modify mechanic ---
            elif isinstance(eff, BattlecryModifyMechanic):
                triggers.append(
                    TriggerDef(
                        event_type=EventType.MINION_PLAYED,
                        condition=_is_self_play,
                        effect=_make_battlecry_modify_mechanic(eff.mechanic, eff.atk, eff.hp),
                        name=f"{card.name} Battlecry",
                        priority=10,
                    )
                )

            # --- Battlecry: consume random shop unit ---
            elif isinstance(eff, ConsumeShopUnit):
                triggers.append(
                    TriggerDef(
                        event_type=EventType.MINION_PLAYED,
                        condition=_is_self_play,
                        effect=_make_battlecry_consume_shop_unit(),
                        name=f"{card.name} Battlecry",
                        priority=10,
                    )
                )

            # --- Battlecry: make self golden ---
            elif isinstance(eff, BattlecryMakeGolden):
                triggers.append(
                    TriggerDef(
                        event_type=EventType.MINION_PLAYED,
                        condition=_is_self_play,
                        effect=_make_battlecry_make_golden(),
                        name=f"{card.name} Battlecry",
                        priority=10,
                    )
                )

            # --- Sell: add spell to hand ---
            elif isinstance(eff, SellAddSpell):
                triggers.append(
                    TriggerDef(
                        event_type=EventType.MINION_SOLD,
                        condition=_is_self_play,
                        effect=_make_sell_add_spell(eff.spell_id, eff.count),
                        name=f"{card.name} Sell",
                    )
                )

            # --- Sell: get random unit ---
            elif isinstance(eff, SellGetRandomUnit):
                triggers.append(
                    TriggerDef(
                        event_type=EventType.MINION_SOLD,
                        condition=_is_self_play,
                        effect=_make_sell_get_random_unit(eff.tier),
                        name=f"{card.name} Sell",
                    )
                )

            # --- Sell for gold ---
            elif isinstance(eff, SellForGold):
                triggers.append(
                    TriggerDef(
                        event_type=EventType.MINION_SOLD,
                        condition=_is_self_play,
                        effect=_make_sell_for_gold(eff.amount),
                        name=f"{card.name} Sell",
                    )
                )

            # --- Rally: Blood Gem on self when this attacks ---
            elif isinstance(eff, RallyBuff):
                triggers.append(
                    TriggerDef(
                        event_type=EventType.ATTACK_DECLARED,
                        condition=_is_self_play,  # source.uid == trigger_uid
                        effect=_make_rally_buff(eff.atk, eff.hp, eff.use_blood_gem),
                        name=f"{card.name} Rally",
                    )
                )

            # --- SoC from hand ---
            elif isinstance(eff, StartOfCombatFromHand):
                triggers.append(
                    TriggerDef(
                        event_type=EventType.START_OF_COMBAT,
                        condition=lambda ctx, event, ref: True,
                        effect=_make_soc_from_hand(),
                        name=f"{card.name} SoC",
                    )
                )

            # --- Start of Combat: buff self by tier ---
            elif isinstance(eff, StartOfCombatBuffSelfByTier):
                triggers.append(
                    TriggerDef(
                        event_type=EventType.START_OF_COMBAT,
                        condition=_is_friendly_soc,
                        effect=_make_start_of_combat_buff_self_by_tier(),
                        name=f"{card.name} SoC",
                    )
                )

            elif isinstance(eff, StartOfCombatBuffSelf):
                _soc_a, _soc_h = eff.atk, eff.hp
                def _make_soc_self(a=_soc_a, h=_soc_h):
                    def _fn(ctx, _ev, uid):
                        es = _event_system()
                        ctx.buff_combat(es.EntityRef(uid), a, h)
                    return _fn
                triggers.append(
                    TriggerDef(
                        event_type=EventType.START_OF_COMBAT,
                        condition=_is_friendly_soc,
                        effect=_make_soc_self(),
                        name=f"{card.name} SoC",
                    )
                )

            # --- On friendly death (excl. self): combat buff ---
            elif isinstance(eff, OnFriendlyDeathBuff):
                triggers.append(
                    TriggerDef(
                        event_type=EventType.MINION_DIED,
                        condition=_is_friendly_death_exclude_self,
                        effect=_make_on_friendly_death_buff(eff.atk, eff.hp),
                        name=f"{card.name} Buff",
                    )
                )

            # --- On friendly play of type: damage hero + buff self ---
            elif isinstance(eff, OnFriendlyPlayTypeDamageHero):
                triggers.append(
                    TriggerDef(
                        event_type=EventType.MINION_PLAYED,
                        condition=lambda ctx, event, ref: True,
                        effect=_make_on_play_type_damage_hero(
                            eff.trigger_type, eff.hero_dmg, eff.atk, eff.hp, eff.exclude_self
                        ),
                        name=f"{card.name} Trigger",
                    )
                )

            # --- On friendly play of type: buff self ---
            elif isinstance(eff, OnFriendlyPlayType):
                triggers.append(
                    TriggerDef(
                        event_type=EventType.MINION_PLAYED,
                        condition=lambda ctx, event, ref: True,
                        effect=_make_on_play_type_buff(
                            eff.trigger_type, eff.atk, eff.hp, eff.exclude_self
                        ),
                        name=f"{card.name} Trigger",
                    )
                )

            # --- Deathrattle: buff all friendlies ---
            elif isinstance(eff, DeathrattleBuffAllFriendlies):
                triggers.append(
                    TriggerDef(
                        event_type=EventType.MINION_DIED,
                        condition=lambda ctx, e, uid: bool(e.source and e.source.uid == uid),
                        effect=_make_dr_buff_all_friendlies(eff.atk, eff.hp),
                        name=f"{card.name} DR",
                    )
                )

            # --- Deathrattle: deal damage to random enemy ---
            elif isinstance(eff, DeathrattleRandomEnemyDamage):
                triggers.append(
                    TriggerDef(
                        event_type=EventType.MINION_DIED,
                        condition=lambda ctx, e, uid: bool(e.source and e.source.uid == uid),
                        effect=_make_dr_random_enemy_damage(eff.damage),
                        name=f"{card.name} DR",
                    )
                )

            # --- On friendly summoned of type: buff self + divine shield ---
            elif isinstance(eff, OnFriendlySummonedTypeBuff):
                triggers.append(
                    TriggerDef(
                        event_type=EventType.MINION_SUMMONED,
                        condition=_make_deflect_o_bot_condition(),
                        effect=_make_on_friendly_summoned_type_buff(
                            eff.trigger_type,
                            eff.atk,
                            eff.hp,
                            eff.exclude_self,
                            eff.combat_buff,
                            eff.gain_divine_shield,
                        ),
                        name=f"{card.name} Trigger",
                    )
                )

            # --- End of turn: add spell to hand ---
            elif isinstance(eff, EndOfTurnAddSpell):
                triggers.append(
                    TriggerDef(
                        event_type=EventType.END_OF_TURN,
                        condition=_is_friendly_soc,  # unit must be on board
                        effect=_make_end_of_turn_add_spell(eff.spell_id, eff.count),
                        name=f"{card.name} End of Turn",
                    )
                )

            # --- End of turn: buff adjacent units ---
            elif isinstance(eff, EndOfTurnBuffAdjacent):
                triggers.append(
                    TriggerDef(
                        event_type=EventType.END_OF_TURN,
                        condition=_is_friendly_soc,
                        effect=_make_eot_buff_adjacent(eff.atk, eff.hp),
                        name=f"{card.name} End of Turn",
                    )
                )

            # --- End of turn: buff self ---
            elif isinstance(eff, EndOfTurnBuffSelf):
                triggers.append(
                    TriggerDef(
                        event_type=EventType.END_OF_TURN,
                        condition=_is_friendly_soc,
                        effect=_make_eot_buff_self(eff.atk, eff.hp),
                        name=f"{card.name} End of Turn",
                    )
                )

            # --- End of turn: buff all board units ---
            elif isinstance(eff, EndOfTurnBuffBoard):
                triggers.append(
                    TriggerDef(
                        event_type=EventType.END_OF_TURN,
                        condition=_is_friendly_soc,
                        effect=_make_eot_buff_board(eff.atk, eff.hp),
                        name=f"{card.name} End of Turn",
                    )
                )

            # --- End of turn: buff all board units of type ---
            elif isinstance(eff, EndOfTurnBuffBoardByType):
                triggers.append(
                    TriggerDef(
                        event_type=EventType.END_OF_TURN,
                        condition=_is_friendly_soc,
                        effect=_make_eot_buff_board_by_type(eff.trigger_type, eff.atk, eff.hp),
                        name=f"{card.name} End of Turn",
                    )
                )

            # --- SoC: buff all friendly of type (combat) ---
            elif isinstance(eff, StartOfCombatBuffFriendlyType):
                triggers.append(
                    TriggerDef(
                        event_type=EventType.START_OF_COMBAT,
                        condition=_is_friendly_soc,
                        effect=_make_soc_buff_friendly_type(eff.trigger_type, eff.atk, eff.hp),
                        name=f"{card.name} SoC",
                    )
                )

            # --- BC: buff all friendly of type (perm) ---
            elif isinstance(eff, BattlecryBuffAllByType):
                triggers.append(
                    TriggerDef(
                        event_type=EventType.MINION_PLAYED,
                        condition=_is_self_play,
                        effect=_make_bc_buff_all_by_type(eff.trigger_type, eff.atk, eff.hp),
                        name=f"{card.name} Battlecry",
                        priority=10,
                    )
                )

            # --- On friendly play of type: add spell to hand ---
            elif isinstance(eff, OnFriendlyPlayTypeAddSpell):
                triggers.append(
                    TriggerDef(
                        event_type=EventType.MINION_PLAYED,
                        condition=lambda ctx, event, ref: True,
                        effect=_make_on_play_type_add_spell(
                            eff.trigger_type, eff.spell_id, eff.count, eff.exclude_self
                        ),
                        name=f"{card.name} Trigger",
                    )
                )

            # --- On friendly summoned of type: buff random other ---
            elif isinstance(eff, OnSummonedTypeBuffRandomOther):
                triggers.append(
                    TriggerDef(
                        event_type=EventType.MINION_SUMMONED,
                        condition=lambda ctx, event, ref: True,
                        effect=_make_on_summoned_buff_random_other(
                            eff.trigger_type, eff.atk, eff.hp
                        ),
                        name=f"{card.name} Trigger",
                    )
                )

            # --- Sell: add unit to hand ---
            elif isinstance(eff, SellAddUnit):
                triggers.append(
                    TriggerDef(
                        event_type=EventType.MINION_SOLD,
                        condition=_is_self_play,
                        effect=_make_sell_add_unit_v2(eff.card_id),
                        name=f"{card.name} Sell",
                    )
                )

            # --- Rally: buff random other friendly of type (combat) ---
            elif isinstance(eff, RallyBuffRandomFriendlyType):
                triggers.append(
                    TriggerDef(
                        event_type=EventType.ATTACK_DECLARED,
                        condition=_is_self_play,
                        effect=_make_rally_buff_random_type(eff.trigger_type, eff.atk, eff.hp),
                        name=f"{card.name} Rally",
                    )
                )

            # --- Sell: get random unit of type from pool ---
            elif isinstance(eff, SellGetRandomUnitByType):
                triggers.append(
                    TriggerDef(
                        event_type=EventType.MINION_SOLD,
                        condition=_is_self_play,
                        effect=_make_sell_get_random_by_type(eff.unit_type),
                        name=f"{card.name} Sell",
                    )
                )

            # --- BC: consume shop unit, give stats to random friendly of type ---
            elif isinstance(eff, ConsumeShopUnitForRandomFriendly):
                triggers.append(
                    TriggerDef(
                        event_type=EventType.MINION_PLAYED,
                        condition=_is_self_play,
                        effect=_make_consume_for_random_friendly(eff.trigger_type),
                        name=f"{card.name} Battlecry",
                        priority=10,
                    )
                )

            # --- After tavern refreshed: buff rightmost shop minion ---
            elif isinstance(eff, OnTavernRefreshBuffRightmostShop):
                triggers.append(
                    TriggerDef(
                        event_type=EventType.TAVERN_REFRESHED,
                        condition=_make_on_tavern_refresh_buff_rightmost_shop_condition(),
                        effect=_make_on_tavern_refresh_buff_rightmost_shop(
                            eff.atk, eff.hp, eff.give_reborn, eff.use_blood_gem
                        ),
                        name=f"{card.name} Tavern Refresh",
                    )
                )

            # --- Sell: buff board scaling ---
            elif isinstance(eff, SellBuffBoardScaling):
                triggers.append(
                    TriggerDef(
                        event_type=EventType.MINION_SOLD,
                        condition=_is_self_play,
                        effect=_make_sell_buff_board_scaling(
                            eff.scaling_key, eff.atk_per, eff.hp_per
                        ),
                        name=f"{card.name} Sell",
                    )
                )

            # --- SoC: damage and buff adjacent ---
            elif isinstance(eff, StartOfCombatDamageAndBuffAdjacent):
                triggers.append(
                    TriggerDef(
                        event_type=EventType.START_OF_COMBAT,
                        condition=_is_friendly_soc,
                        effect=_make_soc_damage_and_buff_adjacent(eff.damage, eff.atk, eff.hp),
                        name=f"{card.name} SoC",
                    )
                )

            # --- On hero damaged: heal and buff self ---
            elif isinstance(eff, OnHeroDamagedHealAndBuffSelf):
                triggers.append(
                    TriggerDef(
                        event_type=EventType.HERO_DAMAGED,
                        condition=lambda ctx, event, ref: True,
                        effect=_make_on_hero_damaged_heal_and_buff_self(eff.hp),
                        name=f"{card.name} Trigger",
                    )
                )

            # --- EOT: buff adjacent per golden ---
            elif isinstance(eff, EndOfTurnBuffAdjacentPerGolden):
                triggers.append(
                    TriggerDef(
                        event_type=EventType.END_OF_TURN,
                        condition=_is_friendly_soc,
                        effect=_make_eot_buff_adjacent_per_golden(eff.atk, eff.hp),
                        name=f"{card.name} End of Turn",
                    )
                )

            # --- Sell: discover ---
            elif isinstance(eff, SellDiscover):
                triggers.append(
                    TriggerDef(
                        event_type=EventType.MINION_SOLD,
                        condition=_is_self_play,
                        effect=_make_sell_discover(eff.base_tier, eff.scaling_key),
                        name=f"{card.name} Sell",
                    )
                )

            # --- Deathrattle: add spell to hand ---
            elif isinstance(eff, DeathrattleAddSpell):
                triggers.append(
                    TriggerDef(
                        event_type=EventType.MINION_DIED,
                        condition=_is_self_death,
                        effect=_make_dr_add_spell(eff.spell_id, eff.count),
                        name=f"{card.name} Deathrattle",
                    )
                )

            # --- Rally: add spell to hand ---
            elif isinstance(eff, RallyAddSpell):
                triggers.append(
                    TriggerDef(
                        event_type=EventType.ATTACK_DECLARED,
                        condition=_is_self_play,
                        effect=_make_rally_add_spell(eff.spell_id, eff.count),
                        name=f"{card.name} Rally",
                    )
                )

            # --- EOT: buff self per golden ---
            elif isinstance(eff, EndOfTurnBuffSelfPerGolden):
                triggers.append(
                    TriggerDef(
                        event_type=EventType.END_OF_TURN,
                        condition=_is_friendly_soc,
                        effect=_make_eot_buff_self_per_golden(eff.atk_per, eff.hp_per),
                        name=f"{card.name} End of Turn",
                    )
                )

            # --- On Divine Shield lost: add spell ---
            elif isinstance(eff, OnDivineShieldLostAddSpell):
                triggers.append(
                    TriggerDef(
                        event_type=EventType.DIVINE_SHIELD_LOST,
                        condition=lambda ctx, event, ref: True,
                        effect=_make_on_divine_shield_lost_add_spell(eff.spell_id, eff.count),
                        name=f"{card.name} Trigger",
                    )
                )

            # --- BC: buff all of type in hand and board ---
            elif isinstance(eff, BattlecryBuffAllByTypeIncludeHand):
                triggers.append(
                    TriggerDef(
                        event_type=EventType.MINION_PLAYED,
                        condition=_is_self_play,
                        effect=_make_bc_buff_all_by_type_include_hand(
                            eff.trigger_type, eff.atk, eff.hp
                        ),
                        name=f"{card.name} Battlecry",
                        priority=10,
                    )
                )

            # --- Rally: damage own board ---
            elif isinstance(eff, RallyDamageOwnBoard):
                triggers.append(
                    TriggerDef(
                        event_type=EventType.ATTACK_DECLARED,
                        condition=_is_self_play,
                        effect=_make_rally_damage_own_board(eff.damage),
                        name=f"{card.name} Rally",
                    )
                )

            # --- Deathrattle: modify mechanic ---
            elif isinstance(eff, DeathrattleModifyMechanic):
                triggers.append(
                    TriggerDef(
                        event_type=EventType.MINION_DIED,
                        condition=_is_self_death,
                        effect=_make_dr_modify_mechanic(eff.mechanic, eff.atk, eff.hp),
                        name=f"{card.name} Deathrattle",
                    )
                )

            # --- SoC: buff random friendly of type + divine shield ---
            elif isinstance(eff, StartOfCombatBuffRandomFriendlyTypeAndDS):
                triggers.append(
                    TriggerDef(
                        event_type=EventType.START_OF_COMBAT,
                        condition=_is_friendly_soc,
                        effect=_make_soc_buff_random_friendly_type_and_ds(
                            eff.trigger_type, eff.atk, eff.hp
                        ),
                        name=f"{card.name} SoC",
                    )
                )

            # --- On self damaged: buff board ---
            elif isinstance(eff, OnSelfDamagedBuffBoard):
                triggers.append(
                    TriggerDef(
                        event_type=EventType.MINION_DAMAGED,
                        condition=lambda ctx, event, ref: (
                            event.target is not None and event.target.uid == ref
                        ),
                        effect=_make_on_self_damaged_buff_board(eff.atk, eff.hp),
                        name=f"{card.name} Trigger",
                    )
                )

            # --- On friendly reborn: buff self ---
            elif isinstance(eff, OnFriendlyRebornBuffSelf):
                triggers.append(
                    TriggerDef(
                        event_type=EventType.MINION_SUMMONED,
                        condition=lambda ctx, event, ref: True,
                        effect=_make_on_friendly_reborn_buff_self(eff.atk, eff.hp),
                        name=f"{card.name} Trigger",
                    )
                )

            # --- Deathrattle: buff all friendly of type globally (hand+board) ---
            elif isinstance(eff, DeathrattleBuffFriendlyTypeGlobal):
                triggers.append(
                    TriggerDef(
                        event_type=EventType.MINION_DIED,
                        condition=_is_self_death,
                        effect=_make_dr_buff_friendly_type_global(
                            eff.trigger_type, eff.atk, eff.hp
                        ),
                        name=f"{card.name} Deathrattle",
                    )
                )

            # --- Deathrattle: buff shop minions ---
            elif isinstance(eff, DeathrattleBuffShop):
                triggers.append(
                    TriggerDef(
                        event_type=EventType.MINION_DIED,
                        condition=_is_self_death,
                        effect=_make_dr_buff_shop(eff.atk, eff.hp),
                        name=f"{card.name} Deathrattle",
                    )
                )

            # --- Deathrattle: buff random hand minion ---
            elif isinstance(eff, DeathrattleBuffHandRandom):
                triggers.append(
                    TriggerDef(
                        event_type=EventType.MINION_DIED,
                        condition=_is_self_death,
                        effect=_make_dr_buff_hand_random(eff.atk, eff.hp),
                        name=f"{card.name} Deathrattle",
                    )
                )

            # --- On friendly attack: buff the attacker (Roaring Recruiter) ---
            elif isinstance(eff, OnFriendlyAttackBuffSelf):
                triggers.append(
                    TriggerDef(
                        event_type=EventType.ATTACK_DECLARED,
                        condition=lambda ctx, event, ref: True,
                        effect=_make_on_friendly_attack_buff_attacker(
                            eff.trigger_type, eff.atk, eff.hp
                        ),
                        name=f"{card.name} Trigger",
                    )
                )

            # --- On friendly attack: buff self/trigger unit (Twilight Watcher) ---
            elif isinstance(eff, OnFriendlyAttackBuffTriggerSelf):
                triggers.append(
                    TriggerDef(
                        event_type=EventType.ATTACK_DECLARED,
                        condition=lambda ctx, event, ref: True,
                        effect=_make_on_friendly_attack_buff_trigger(
                            eff.trigger_type, eff.atk, eff.hp
                        ),
                        name=f"{card.name} Trigger",
                    )
                )

            # --- While in hand, on friendly play of type: buff self (Bream Counter) ---
            elif isinstance(eff, OnFriendlyPlayTypeBuffSelfInHand):
                triggers.append(
                    TriggerDef(
                        event_type=EventType.MINION_PLAYED,
                        condition=lambda ctx, event, ref: True,
                        effect=_make_on_friendly_play_type_buff_self_in_hand(
                            eff.trigger_type, eff.atk, eff.hp
                        ),
                        name=f"{card.name} Trigger",
                    )
                )

            # --- On spell cast: buff self (Glad-iator, Timecap'n Hooktail) ---
            elif isinstance(eff, OnSpellCastBuffSelf):
                triggers.append(
                    TriggerDef(
                        event_type=EventType.SPELL_CAST,
                        condition=lambda ctx, event, ref: True,
                        effect=_make_on_spell_cast_buff_self(eff.atk, eff.hp),
                        name=f"{card.name} Trigger",
                    )
                )

            # --- On gain gold (spell cast with GAIN_GOLD): buff self ---
            elif isinstance(eff, OnGainGoldBuffSelf):
                triggers.append(
                    TriggerDef(
                        event_type=EventType.SPELL_CAST,
                        condition=lambda ctx, event, ref: True,
                        effect=_make_on_gain_gold_buff_self(eff.atk, eff.hp),
                        name=f"{card.name} Trigger",
                    )
                )

            # --- Deathrattle: damage all minions (Tunnel Blaster, Silent Enforcer) ---
            elif isinstance(eff, DeathrattleDamageAllMinions):
                triggers.append(
                    TriggerDef(
                        event_type=EventType.MINION_DIED,
                        condition=_is_self_death,
                        effect=_make_dr_damage_all_minions(eff.damage),
                        name=f"{card.name} Deathrattle",
                    )
                )

            # --- SoC: permanently buff all friendly of type (Prized Promo-Drake) ---
            elif isinstance(eff, StartOfCombatBuffAllFriendlyType):
                triggers.append(
                    TriggerDef(
                        event_type=EventType.START_OF_COMBAT,
                        condition=_is_friendly_soc,
                        effect=_make_soc_buff_all_friendly_type(eff.trigger_type, eff.atk, eff.hp),
                        name=f"{card.name} SoC",
                    )
                )

            # --- SoC: give random friendly of type Reborn (Soulsplitter) ---
            elif isinstance(eff, StartOfCombatGiveFriendlyTypeReborn):
                triggers.append(
                    TriggerDef(
                        event_type=EventType.START_OF_COMBAT,
                        condition=_is_friendly_soc,
                        effect=_make_soc_give_friendly_type_reborn(eff.trigger_type),
                        name=f"{card.name} SoC",
                    )
                )

            # --- BC: add random unit of type to hand (Tavern Tempest) ---
            elif isinstance(eff, BattlecryAddRandomUnit):
                triggers.append(
                    TriggerDef(
                        event_type=EventType.MINION_PLAYED,
                        condition=_is_self_play,
                        effect=_make_bc_add_random_unit(eff.unit_type, eff.tier),
                        name=f"{card.name} Battlecry",
                        priority=10,
                    )
                )

            # --- BC: gain free refreshes (Refreshing Anomaly) ---
            elif isinstance(eff, BattlecryGainFreeRefreshes):
                triggers.append(
                    TriggerDef(
                        event_type=EventType.MINION_PLAYED,
                        condition=_is_self_play,
                        effect=_make_bc_gain_free_refreshes(eff.count),
                        name=f"{card.name} Battlecry",
                        priority=10,
                    )
                )

            # --- On DS lost: buff the unit that lost it (Grease Bot) ---
            elif isinstance(eff, OnDivineShieldLostBuffUnit):
                triggers.append(
                    TriggerDef(
                        event_type=EventType.DIVINE_SHIELD_LOST,
                        condition=lambda ctx, event, ref: True,
                        effect=_make_on_ds_lost_buff_unit(eff.atk, eff.hp),
                        name=f"{card.name} Trigger",
                    )
                )

            # --- Rally: play blood gems on all other friendlies (Bonker) ---
            elif isinstance(eff, RallyBuffAllOthersByType):
                triggers.append(
                    TriggerDef(
                        event_type=EventType.ATTACK_DECLARED,
                        condition=_is_self_play,
                        effect=_make_rally_buff_all_others_blood_gems(eff.count),
                        name=f"{card.name} Rally",
                    )
                )

            # --- EOT: add random spell to hand (Marquee Ticker) ---
            elif isinstance(eff, EndOfTurnAddRandomSpell):
                triggers.append(
                    TriggerDef(
                        event_type=EventType.END_OF_TURN,
                        condition=_is_friendly_soc,
                        effect=_make_eot_add_random_spell(),
                        name=f"{card.name} End of Turn",
                    )
                )

            # --- StartOfCombatGainGold: handled as EOT gold (Accord-o-Tron, Industrious Deckhand) ---
            elif isinstance(eff, StartOfCombatGainGold):
                triggers.append(
                    TriggerDef(
                        event_type=EventType.END_OF_TURN,
                        condition=_is_friendly_soc,
                        effect=_make_battlecry_gain_gold(eff.amount),
                        name=f"{card.name} End of Turn",
                    )
                )

            # --- AvengeEffect: extended targets (add_spell, add_unit) ---
            elif isinstance(eff, AvengeEffect) and eff.buff_target in ("add_spell", "add_unit"):
                # These are handled in the avenge system via AVENGE_REGISTRY
                # but we also need to register them so they fire.
                # The avenge system in combat.py reads AVENGE_REGISTRY for threshold.
                pass  # handled by avenge system reading AvengeEffect from registry

            # --- AvengeBuffFriendlyTypeGlobal ---
            elif isinstance(eff, AvengeBuffFriendlyTypeGlobal):
                pass  # handled by avenge system; effect fired via avenge_registry

            # --- OnSpellCastBuffBoard (Plankwalker, Sundered Matriarch) ---
            elif isinstance(eff, OnSpellCastBuffBoard):
                triggers.append(
                    TriggerDef(
                        event_type=EventType.SPELL_CAST,
                        condition=lambda ctx, event, ref: True,
                        effect=_make_on_spell_cast_buff_board(eff.atk, eff.hp, eff.trigger_type),
                        name=f"{card.name} Trigger",
                    )
                )

            # --- DeathrattleSummonTauntToken (Twilight Broodmother) ---
            elif isinstance(eff, DeathrattleSummonTauntToken):
                triggers.append(
                    TriggerDef(
                        event_type=EventType.MINION_DIED,
                        condition=_is_self_death,
                        effect=_make_dr_summon_taunt_token(eff.token_id, eff.count),
                        name=f"{card.name} Deathrattle",
                    )
                )

            # --- StartOfCombatBuffSelfByHighestAllyAtk (Costume Enthusiast) ---
            elif isinstance(eff, StartOfCombatBuffSelfByHighestAllyAtk):
                triggers.append(
                    TriggerDef(
                        event_type=EventType.START_OF_COMBAT,
                        condition=_is_friendly_soc,
                        effect=_make_soc_buff_self_by_highest_ally_atk(),
                        name=f"{card.name} SoC",
                    )
                )

            # --- StartOfCombatBuffSelfByHighestBoardAtk (Psychus) ---
            elif isinstance(eff, StartOfCombatBuffSelfByHighestBoardAtk):
                triggers.append(
                    TriggerDef(
                        event_type=EventType.START_OF_COMBAT,
                        condition=_is_friendly_soc,
                        effect=_make_soc_buff_self_by_highest_board_atk(),
                        name=f"{card.name} SoC",
                    )
                )

            # --- BattlecryMakeGoldenFriendlyByTier (Elite Navigator, Captain Sanders) ---
            elif isinstance(eff, BattlecryMakeGoldenFriendlyByTier):
                triggers.append(
                    TriggerDef(
                        event_type=EventType.MINION_PLAYED,
                        condition=_is_self_play,
                        effect=_make_bc_make_golden_friendly_by_tier(eff.max_tier),
                        name=f"{card.name} Battlecry",
                        priority=10,
                    )
                )

            # --- RallyBuffFriendlyTypeAtk (Sunken Advocate) ---
            elif isinstance(eff, RallyBuffFriendlyTypeAtk):
                triggers.append(
                    TriggerDef(
                        event_type=EventType.ATTACK_DECLARED,
                        condition=_is_self_play,
                        effect=_make_rally_buff_friendly_type_atk(eff.trigger_type, eff.atk),
                        name=f"{card.name} Rally",
                    )
                )

            # --- OnFriendlyBeastDamagedBuffSelf (Trigore the Lasher) ---
            elif isinstance(eff, OnFriendlyBeastDamagedBuffSelf):
                triggers.append(
                    TriggerDef(
                        event_type=EventType.MINION_DAMAGED,
                        condition=lambda ctx, event, ref: True,
                        effect=_make_on_friendly_beast_damaged_buff_self(eff.hp),
                        name=f"{card.name} Trigger",
                    )
                )

            # --- OnFriendlyBeastDamagedBuffOther (Iridescent Skyblazer) ---
            elif isinstance(eff, OnFriendlyBeastDamagedBuffOther):
                triggers.append(
                    TriggerDef(
                        event_type=EventType.MINION_DAMAGED,
                        condition=lambda ctx, event, ref: True,
                        effect=_make_on_friendly_beast_damaged_buff_other(eff.atk, eff.hp),
                        name=f"{card.name} Trigger",
                    )
                )

            # --- DeathrattleDestroyKiller (Leeroy the Reckless) ---
            elif isinstance(eff, DeathrattleDestroyKiller):
                triggers.append(
                    TriggerDef(
                        event_type=EventType.MINION_DIED,
                        condition=_is_self_death,
                        effect=_make_dr_destroy_killer(),
                        name=f"{card.name} Deathrattle",
                    )
                )

            # --- SellForGoldConditional (Tortollan Blue Shell) ---
            elif isinstance(eff, SellForGoldConditional):
                triggers.append(
                    TriggerDef(
                        event_type=EventType.MINION_SOLD,
                        condition=_is_self_play,
                        effect=_make_sell_for_gold_conditional(eff.amount),
                        name=f"{card.name} Sell",
                    )
                )

            # --- DeathrattleBuffAllFriendliesGlobal (Goldrinn) ---
            elif isinstance(eff, DeathrattleBuffAllFriendliesGlobal):
                triggers.append(
                    TriggerDef(
                        event_type=EventType.MINION_DIED,
                        condition=_is_self_death,
                        effect=_make_dr_buff_all_friendlies_global(
                            eff.trigger_type, eff.atk, eff.hp
                        ),
                        name=f"{card.name} Deathrattle",
                    )
                )

            # --- EndOfTurnBuffFriendlyTypeNaga (Slitherspear) ---
            elif isinstance(eff, EndOfTurnBuffFriendlyTypeNaga):
                triggers.append(
                    TriggerDef(
                        event_type=EventType.END_OF_TURN,
                        condition=_is_friendly_soc,
                        effect=_make_eot_buff_friendly_type_naga(eff.atk, eff.hp),
                        name=f"{card.name} End of Turn",
                    )
                )

            # --- OnFriendlyDemonDamageBuff (Lord of the Ruins) ---
            elif isinstance(eff, OnFriendlyDemonDamageBuff):
                triggers.append(
                    TriggerDef(
                        event_type=EventType.DAMAGE_DEALT,
                        condition=lambda ctx, event, ref: True,
                        effect=_make_on_friendly_demon_damage_buff(eff.atk, eff.hp),
                        name=f"{card.name} Trigger",
                    )
                )

            # --- EndOfTurnConsumeTavernForDemon (Famished Felbat) ---
            elif isinstance(eff, EndOfTurnConsumeTavernForDemon):
                triggers.append(
                    TriggerDef(
                        event_type=EventType.END_OF_TURN,
                        condition=_is_friendly_soc,
                        effect=_make_eot_consume_tavern_for_demon(),
                        name=f"{card.name} End of Turn",
                    )
                )

            # --- StartOfCombatBuffFriendlyTypeScaling (Ultraviolet Ascendant) ---
            elif isinstance(eff, StartOfCombatBuffFriendlyTypeScaling):
                triggers.append(
                    TriggerDef(
                        event_type=EventType.START_OF_COMBAT,
                        condition=_is_friendly_soc,
                        effect=_make_soc_buff_friendly_type_scaling(
                            eff.trigger_type, eff.atk, eff.hp
                        ),
                        name=f"{card.name} SoC",
                    )
                )

            # --- EndOfTurnTriggerAdjacentBattlecry (Young Murk-Eye) ---
            elif isinstance(eff, EndOfTurnTriggerAdjacentBattlecry):
                triggers.append(
                    TriggerDef(
                        event_type=EventType.END_OF_TURN,
                        condition=_is_friendly_soc,
                        effect=_make_eot_trigger_adjacent_battlecry(),
                        name=f"{card.name} End of Turn",
                    )
                )

            # --- RallyDealDamageEqualToAtk (Niuzao, Obsidian Ravager) ---
            elif isinstance(eff, RallyDealDamageEqualToAtk):
                triggers.append(
                    TriggerDef(
                        event_type=EventType.ATTACK_DECLARED,
                        condition=_is_self_play,
                        effect=_make_rally_deal_damage_equal_to_atk(),
                        name=f"{card.name} Rally",
                    )
                )

            # --- DeathrattleGiveFriendliesScaling (Spiked Savior) ---
            elif isinstance(eff, DeathrattleGiveFriendliesScaling):
                triggers.append(
                    TriggerDef(
                        event_type=EventType.MINION_DIED,
                        condition=_is_self_death,
                        effect=_make_dr_give_friendlies_scaling(
                            eff.buff_atk, eff.buff_hp, eff.self_damage
                        ),
                        name=f"{card.name} Deathrattle",
                    )
                )

            # --- Pass-through for custom / hand-crafted triggers ---
            elif isinstance(eff, CustomEffect):
                # Felemental: BC gives +2/+1 to all current shop minions, and future ones via mechanic
                if card.card_id == CardIDs.FELEMENTAL:
                    triggers.append(
                        TriggerDef(
                            event_type=EventType.MINION_PLAYED,
                            condition=_is_self_play,
                            effect=_make_felemental_bc(),
                            name="Felemental Battlecry",
                            priority=10,
                        )
                    )
                else:
                    triggers.extend(eff.trigger_defs)

        if triggers:
            registry[card.card_id] = triggers

    # --- Crab Deathrattle (attached dynamically via Surf Spellcraft spell) ---
    def _summon_crab_token(ctx: EffectContext, event: Event, _trigger_uid: int) -> None:
        pos = event.source_pos or (event.snapshot.pos if event.snapshot else None)
        if pos:
            ctx.summon(pos.side, CardIDs.CRAB_TOKEN, pos.slot)

    registry[EffectIDs.CRAB_DEATHRATTLE] = [
        TriggerDef(
            event_type=EventType.MINION_DIED,
            condition=lambda ctx, event, trigger_uid: (
                event.source is not None and event.source.uid == trigger_uid
            ),
            effect=_summon_crab_token,
            name="Attached Crab Deathrattle",
        )
    ]

    return registry


# =====================================================================
# Module-level registries (importable by combat.py, game.py, tavern.py)
# Lazy-initialized to avoid circular import:
#   card_def → event_system → auras → entities → configs → card_def
# =====================================================================

_TRIGGER_REGISTRY = None
GOLDEN_TRIGGER_REGISTRY: dict = {}


def _get_trigger_registry():
    global _TRIGGER_REGISTRY
    if _TRIGGER_REGISTRY is None:
        _TRIGGER_REGISTRY = build_trigger_registry()
    return _TRIGGER_REGISTRY


class _LazyTriggerRegistry:
    """Dict-like proxy that builds TRIGGER_REGISTRY on first access."""

    def __getattr__(self, name):
        return getattr(_get_trigger_registry(), name)

    def __getitem__(self, key):
        return _get_trigger_registry()[key]

    def __contains__(self, key):
        return key in _get_trigger_registry()

    def __iter__(self):
        return iter(_get_trigger_registry())

    def __len__(self):
        return len(_get_trigger_registry())

    def keys(self):
        return _get_trigger_registry().keys()

    def values(self):
        return _get_trigger_registry().values()

    def items(self):
        return _get_trigger_registry().items()

    def get(self, key, default=None):
        return _get_trigger_registry().get(key, default)


TRIGGER_REGISTRY = _LazyTriggerRegistry()


# ---------------------------------------------------------------------------
# AVENGE_REGISTRY — card_id → AvengeEffect for all cards with Avenge mechanic
# ---------------------------------------------------------------------------


def build_avenge_registry() -> Dict[str, AvengeEffect]:
    """Map card_id → AvengeEffect for cards with Avenge mechanic.
    Also maps AvengeBuffFriendlyTypeGlobal to a synthetic AvengeEffect."""
    registry: Dict[str, AvengeEffect] = {}
    for card in ALL_CARDS:
        for eff in card.effects:
            if isinstance(eff, AvengeEffect):
                registry[card.card_id] = eff
                break
            elif isinstance(eff, AvengeBuffFriendlyTypeGlobal):
                # Map to AvengeEffect with friendly_type scope covering board
                registry[card.card_id] = AvengeEffect(
                    threshold=eff.threshold,
                    buff_atk=eff.atk,
                    buff_hp=eff.hp,
                    buff_scope="perm",
                    buff_target="friendly_type",
                    target_type=eff.trigger_type,
                )
                break
    return registry


AVENGE_REGISTRY: Dict[str, AvengeEffect] = build_avenge_registry()
