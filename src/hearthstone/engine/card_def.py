from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List

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
    token_id: str       # CardIDs value (string)
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
    spell_id: str       # SpellIDs value
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
class BattlecryConsumeShopUnit(EffectDef):
    """Consume random shop unit, gain its stats."""
    pass


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
    trigger_defs: list = field(default_factory=list)   # List[TriggerDef]


# ---------------------------------------------------------------------------
# CardDef
# ---------------------------------------------------------------------------

@dataclass
class CardDef:
    card_id: str                # e.g. "101" or "t001"
    name: str
    tier: int
    atk: int
    hp: int
    types: list                 # list[UnitType]
    tags: set = field(default_factory=set)
    is_token: bool = False
    deathrattle: bool = False   # metadata flag used by obs encoding
    effects: list = field(default_factory=list)  # list[EffectDef]


# ---------------------------------------------------------------------------
# ALL_CARDS — single source of truth for every card / token
# ---------------------------------------------------------------------------

ALL_CARDS: List[CardDef] = [
    # -----------------------------------------------------------------------
    # TIER 1 — 21 cards (patch 234747)
    # -----------------------------------------------------------------------
    CardDef(
        CardIDs.ANNOY_O_TRON, "Annoy-o-Tron", 1, 1, 2,
        [UnitType.MECH],
        tags={Tags.DIVINE_SHIELD, Tags.TAUNT},
    ),
    CardDef(
        CardIDs.AUREATE_LAUREATE, "Aureate Laureate", 1, 1, 1,
        [UnitType.PIRATE],
        tags={Tags.DIVINE_SHIELD},
        # Battlecry: Make this minion Golden — shop-phase, TODO
    ),
    CardDef(
        CardIDs.CORD_PULLER, "Cord Puller", 1, 1, 1,
        [UnitType.MECH],
        tags={Tags.DIVINE_SHIELD},
        deathrattle=True,
        effects=[
            DeathrattleSummon(token_id=CardIDs.MICROBOT, count=1)
        ],
    ),
    CardDef(
        CardIDs.CRACKLING_CYCLONE, "Crackling Cyclone", 1, 2, 1,
        [UnitType.ELEMENTAL],
        tags={Tags.DIVINE_SHIELD, Tags.WINDFURY},
    ),
    CardDef(
        CardIDs.DUNE_DWELLER, "Dune Dweller", 1, 3, 2,
        [UnitType.ELEMENTAL],
        effects=[
            BattlecryModifyMechanic(mechanic=MechanicType.ELEMENTAL_BUFF, atk=1, hp=1)
        ],
    ),
    CardDef(
        CardIDs.FLIGHTY_SCOUT, "Flighty Scout", 1, 3, 3,
        [UnitType.MURLOC],
        # SoC: If in hand, summon copy — C++ combat only
    ),
    CardDef(
        CardIDs.HARMLESS_BONEHEAD, "Harmless Bonehead", 1, 1, 1,
        [UnitType.UNDEAD],
        deathrattle=True,
        effects=[
            DeathrattleSummon(token_id=CardIDs.SKELETON, count=2)
        ],
    ),
    CardDef(
        CardIDs.MANASABER, "Manasaber", 1, 4, 1,
        [UnitType.BEAST],
        deathrattle=True,
        effects=[
            DeathrattleSummon(token_id=CardIDs.CUBLING, count=2)
        ],
    ),
    CardDef(
        CardIDs.MINTED_CORSAIR, "Minted Corsair", 1, 1, 3,
        [UnitType.PIRATE],
        effects=[
            SellAddSpell(spell_id=SpellIDs.TAVERN_COIN, count=1)
        ],
    ),
    CardDef(
        CardIDs.MISFIT_DRAGONLING, "Misfit Dragonling", 1, 2, 1,
        [UnitType.DRAGON],
        # SoC: Gain stats equal to your Tier
        effects=[
            StartOfCombatBuffSelfByTier()
        ],
    ),
    CardDef(
        CardIDs.OMINOUS_SEER, "Ominous Seer", 1, 2, 1,
        [UnitType.DEMON, UnitType.NAGA],
        effects=[
            BattlecrySpellDiscount(amount=1)
        ],
    ),
    CardDef(
        CardIDs.PICKY_EATER, "Picky Eater", 1, 1, 1,
        [UnitType.DEMON],
        effects=[
            BattlecryConsumeShopUnit()
        ],
    ),
    CardDef(
        CardIDs.RAZORFEN_GEOMANCER, "Razorfen Geomancer", 1, 2, 1,
        [UnitType.QUILBOAR],
        effects=[
            BattlecryAddSpell(spell_id=SpellIDs.BLOOD_GEM, count=2)
        ],
    ),
    CardDef(
        CardIDs.RISEN_RIDER, "Risen Rider", 1, 2, 1,
        [UnitType.UNDEAD],
        tags={Tags.TAUNT, Tags.REBORN},
    ),
    CardDef(
        CardIDs.RIVER_SKIPPER, "River Skipper", 1, 1, 1,
        [UnitType.MURLOC],
        effects=[
            SellGetRandomUnit(tier=1)
        ],
    ),
    CardDef(
        CardIDs.ROT_HIDE_GNOLL, "Rot Hide Gnoll", 1, 1, 4,
        [UnitType.UNDEAD],
        # +1 Atk per friendly death this combat — combat trigger
        effects=[
            OnFriendlyDeathBuff(atk=1, hp=0)
        ],
    ),
    CardDef(
        CardIDs.SURF_N_SURF, "Surf n' Surf", 1, 1, 1,
        [UnitType.NAGA, UnitType.BEAST],
        # Spellcraft: DR summon 3/2 Crab — shop-phase spellcraft
    ),
    CardDef(
        CardIDs.SWAMPSTRIKER, "Swampstriker", 1, 1, 5,
        [UnitType.MURLOC],
        tags={Tags.WINDFURY},
        effects=[
            OnFriendlyPlayType(
                trigger_type=UnitType.MURLOC,
                atk=1,
                hp=0,
                exclude_self=True,
            )
        ],
    ),
    CardDef(
        CardIDs.TUSKED_CAMPER, "Tusked Camper", 1, 2, 3,
        [UnitType.QUILBOAR],
        # Rally: Blood Gem on self — C++ combat only
    ),
    CardDef(
        CardIDs.TWILIGHT_HATCHLING, "Twilight Hatchling", 1, 1, 1,
        [UnitType.DRAGON],
        deathrattle=True,
        effects=[
            DeathrattleSummonWithTag(
                token_id=CardIDs.TWILIGHT_WHELP,
                count=1,
                tag=Tags.IMMEDIATE_ATTACK,
            )
        ],
    ),
    CardDef(
        CardIDs.WRATH_WEAVER, "Wrath Weaver", 1, 1, 4,
        [UnitType.DEMON],
        effects=[
            OnFriendlyPlayTypeDamageHero(
                trigger_type=UnitType.DEMON,
                hero_dmg=1,
                atk=2,
                hp=1,
                exclude_self=True,
            )
        ],
    ),
    # -----------------------------------------------------------------------
    # TOKENS
    # -----------------------------------------------------------------------
    CardDef(
        CardIDs.MICROBOT, "Microbot", 1, 1, 1,
        [UnitType.MECH],
        is_token=True,
    ),
    CardDef(
        CardIDs.SKELETON, "Skeleton", 1, 1, 1,
        [UnitType.UNDEAD],
        is_token=True,
    ),
    CardDef(
        CardIDs.CUBLING, "Cubling", 1, 0, 1,
        [UnitType.BEAST],
        tags={Tags.TAUNT},
        is_token=True,
    ),
    CardDef(
        CardIDs.TWILIGHT_WHELP, "Twilight Whelp", 1, 3, 3,
        [UnitType.DRAGON],
        is_token=True,
    ),
    CardDef(
        CardIDs.CRAB_TOKEN, "Crab", 1, 3, 2,
        [UnitType.BEAST],
        is_token=True,
    ),
]


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


def _make_sell_get_random_unit(tier: int):
    """On sell: add a random T{tier} pool minion to hand."""
    from .configs import CARD_DB  # local import to avoid circular at module level

    def _effect(ctx: EffectContext, _event: Event, trigger_uid: int) -> None:
        es = _event_system()
        pos = ctx.resolve_pos(es.EntityRef(trigger_uid))
        if not pos:
            return
        t1_ids = [
            cid for cid, data in CARD_DB.items()
            if data["tier"] == tier and not data.get("is_token")
        ]
        if not t1_ids:
            return
        chosen_id = random.choice(t1_ids)
        ctx.add_unit_to_hand(pos.side, chosen_id)
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
            elif isinstance(eff, BattlecryConsumeShopUnit):
                triggers.append(
                    TriggerDef(
                        event_type=EventType.MINION_PLAYED,
                        condition=_is_self_play,
                        effect=_make_battlecry_consume_shop_unit(),
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

            # --- Pass-through for custom / hand-crafted triggers ---
            elif isinstance(eff, CustomEffect):
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
