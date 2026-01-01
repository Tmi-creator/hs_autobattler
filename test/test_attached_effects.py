import itertools

from engine.attached_effects import EFFECT_ID_TO_INDEX
from engine.combat import Combat_Manager
from engine.effects import TRIGGER_REGISTRY
from engine.entities import HandCard, Player, Spell, Unit
from engine.event_system import EntityRef, Event, EventManager, EventType, PosRef, Zone
from engine.pool import CardPool, SpellPool
from engine.tavern import TavernManager


def _uid_provider():
    counter = itertools.count(1000)
    return lambda: next(counter)


def test_spellcraft_attaches_and_resets_next_turn():
    tavern = TavernManager(CardPool(), SpellPool())
    unit = Unit.create_from_db("101", 1, 1)
    player = Player(uid=1, board=[unit], hand=[HandCard(uid=2, spell=Spell.create_from_db("S007"))], tavern_tier=1)

    success, _ = tavern.play_unit(player, hand_index=0, target_index=0)
    assert success

    effect_index = EFFECT_ID_TO_INDEX["E_DR_CRAB32"]
    assert unit.attached_turn[effect_index] == 1

    tavern.start_turn(player, turn_number=2)
    assert unit.attached_turn[effect_index] == 0


def _process_death_with_attached(count: int):
    unit = Unit.create_from_db("101", 1, 1)
    effect_index = EFFECT_ID_TO_INDEX["E_DR_CRAB32"]
    unit.attached_turn[effect_index] = count
    player = Player(uid=1, board=[unit], hand=[], tavern_tier=1)

    combat_manager = Combat_Manager()
    death_event = Event(
        event_type=EventType.MINION_DIED,
        source=EntityRef(unit.uid),
        source_pos=PosRef(side=player.uid, zone=Zone.BOARD, slot=0),
    )
    extra_triggers = combat_manager._collect_death_triggers(unit, 0)
    player.board.pop(0)
    combat_manager.event_manager.process_event(
        death_event,
        {player.uid: player},
        combat_manager.get_uid,
        extra_triggers=extra_triggers,
    )
    return player.board


def test_attached_deathrattle_summons_crab():
    board = _process_death_with_attached(1)
    assert [unit.card_id for unit in board] == ["110t"]


def test_attached_deathrattle_stacks():
    board = _process_death_with_attached(2)
    assert [unit.card_id for unit in board] == ["110t", "110t"]


def test_existing_spells_and_triggers_still_work():
    tavern = TavernManager(CardPool(), SpellPool())
    unit = Unit.create_from_db("101", 1, 1)
    player = Player(uid=1, board=[unit], hand=[HandCard(uid=2, spell=Spell.create_from_db("S002"))], tavern_tier=1)

    success, _ = tavern.play_unit(player, hand_index=0, target_index=0)
    assert success
    assert unit.max_atk == unit.base_atk + 2
    assert unit.max_hp == unit.base_hp + 2

    weaver = Unit.create_from_db("101", 10, 1)
    demon = Unit.create_from_db("108", 11, 1)
    player = Player(uid=1, board=[weaver, demon], hand=[], tavern_tier=1, health=30)
    event_manager = EventManager(TRIGGER_REGISTRY)
    event_manager.process_event(
        Event(
            event_type=EventType.MINION_PLAYED,
            source=EntityRef(demon.uid),
            source_pos=PosRef(side=player.uid, zone=Zone.BOARD, slot=1),
        ),
        {player.uid: player},
        _uid_provider(),
    )
    assert weaver.max_atk == weaver.base_atk + 2
    assert weaver.max_hp == weaver.base_hp + 1
    assert player.health == 29
