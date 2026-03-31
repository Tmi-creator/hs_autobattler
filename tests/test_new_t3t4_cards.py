"""Tests for new T3 and T4 cards added in batch."""
from __future__ import annotations

from collections import deque
from typing import Callable, Dict, List, Tuple

import pytest

from hearthstone.engine.card_def import (
    AVENGE_REGISTRY,
    TRIGGER_REGISTRY,
    GOLDEN_TRIGGER_REGISTRY,
)
from hearthstone.engine.combat import CombatManager
from hearthstone.engine.entities import HandCard, Player, StoreItem, Unit
from hearthstone.engine.enums import CardIDs, MechanicType, SpellIDs, Tags, UnitType
from hearthstone.engine.event_system import (
    EffectContext,
    EntityRef,
    Event,
    EventManager,
    EventType,
    MinionSnapshot,
    PosRef,
    Zone,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _em() -> EventManager:
    return EventManager(TRIGGER_REGISTRY, GOLDEN_TRIGGER_REGISTRY)


def _fire(em, event_type, players, uid_fn, /, **kwargs):
    em.process_event(Event(event_type=event_type, **kwargs), players, uid_fn)


# ===========================================================================
# T3 — Anub'arak, Nerubian King
# ===========================================================================

class TestAnubarakNerubianKing:
    def test_buffs_undead_on_board_on_death(self, combat_players):
        players, boards, cm = combat_players(
            [CardIDs.ANUBARAK_NERUBIAN_KING, CardIDs.SKELETON, CardIDs.RISEN_RIDER],
            [CardIDs.RAMPAGER],
        )
        board0 = boards[0]
        anubarak = board0[0]
        undead1 = board0[1]
        undead2 = board0[2]

        atk1 = undead1.cur_atk
        atk2 = undead2.cur_atk

        anubarak.cur_hp = 0
        cm.cleanup_dead(boards, [0, 0], players)

        assert undead1.cur_atk == atk1 + 1
        assert undead2.cur_atk == atk2 + 1

    def test_does_not_buff_non_undead(self, combat_players):
        players, boards, cm = combat_players(
            [CardIDs.ANUBARAK_NERUBIAN_KING, CardIDs.RAMPAGER],
            [CardIDs.SKELETON],
        )
        board0 = boards[0]
        anubarak = board0[0]
        beast = board0[1]
        atk_before = beast.cur_atk

        anubarak.cur_hp = 0
        cm.cleanup_dead(boards, [0, 0], players)

        assert beast.cur_atk == atk_before


# ===========================================================================
# T3 — Deflect-o-Bot
# ===========================================================================

class TestDeflectOBot:
    def test_gains_atk_and_ds_on_mech_summoned(self, empty_game, player, mock_unit):
        bot = mock_unit(CardIDs.DEFLECT_O_BOT, owner_id=player.uid)
        mech = mock_unit(CardIDs.ANNOY_O_TRON, owner_id=player.uid)
        player.board = [bot, mech]
        bot.tags.discard(Tags.DIVINE_SHIELD)
        atk_before = bot.cur_atk

        _em().process_event(
            Event(
                event_type=EventType.MINION_SUMMONED,
                source=EntityRef(mech.uid),
                source_pos=PosRef(side=player.uid, zone=Zone.BOARD, slot=1),
            ),
            {player.uid: player},
            empty_game.tavern.get_next_uid,
        )
        assert bot.cur_atk == atk_before + 2
        assert Tags.DIVINE_SHIELD in bot.tags

    def test_no_trigger_on_non_mech(self, empty_game, player, mock_unit):
        bot = mock_unit(CardIDs.DEFLECT_O_BOT, owner_id=player.uid)
        beast = mock_unit(CardIDs.RAMPAGER, owner_id=player.uid)
        player.board = [bot, beast]
        atk_before = bot.cur_atk

        _em().process_event(
            Event(
                event_type=EventType.MINION_SUMMONED,
                source=EntityRef(beast.uid),
                source_pos=PosRef(side=player.uid, zone=Zone.BOARD, slot=1),
            ),
            {player.uid: player},
            empty_game.tavern.get_next_uid,
        )
        assert bot.cur_atk == atk_before


# ===========================================================================
# T3 — Roaring Recruiter
# ===========================================================================

class TestRoaringRecruiter:
    def test_buffs_dragon_attacker(self, empty_game, player, mock_unit):
        recruiter = mock_unit(CardIDs.ROARING_RECRUITER, owner_id=player.uid)
        dragon = mock_unit(CardIDs.AMBER_GUARDIAN, owner_id=player.uid)
        player.board = [recruiter, dragon]
        atk_before = dragon.cur_atk
        hp_before = dragon.cur_hp

        _em().process_event(
            Event(
                event_type=EventType.ATTACK_DECLARED,
                source=EntityRef(dragon.uid),
                source_pos=PosRef(side=player.uid, zone=Zone.BOARD, slot=1),
            ),
            {player.uid: player},
            empty_game.tavern.get_next_uid,
        )
        assert dragon.cur_atk == atk_before + 3
        assert dragon.cur_hp == hp_before + 1

    def test_no_self_buff(self, empty_game, player, mock_unit):
        recruiter = mock_unit(CardIDs.ROARING_RECRUITER, owner_id=player.uid)
        player.board = [recruiter]
        atk_before = recruiter.cur_atk

        _em().process_event(
            Event(
                event_type=EventType.ATTACK_DECLARED,
                source=EntityRef(recruiter.uid),
                source_pos=PosRef(side=player.uid, zone=Zone.BOARD, slot=0),
            ),
            {player.uid: player},
            empty_game.tavern.get_next_uid,
        )
        assert recruiter.cur_atk == atk_before


# ===========================================================================
# T3 — Scourfin
# ===========================================================================

class TestScourfin:
    def test_buffs_hand_minion_on_death(self, combat_players, mock_unit, empty_game):
        players, boards, cm = combat_players(
            [CardIDs.SCOURFIN],
            [CardIDs.RAMPAGER],
        )
        board0 = boards[0]
        scourfin = board0[0]
        side = scourfin.owner_id
        player = players[side]

        hand_unit = mock_unit(CardIDs.SKELETON, owner_id=side)
        player.hand = [HandCard(uid=hand_unit.uid, unit=hand_unit)]
        atk_before = hand_unit.cur_atk
        hp_before = hand_unit.cur_hp

        scourfin.cur_hp = 0
        cm.cleanup_dead(boards, [0, 0], players)

        assert hand_unit.cur_atk == atk_before + 5
        assert hand_unit.cur_hp == hp_before + 5

    def test_no_crash_empty_hand(self, combat_players):
        players, boards, cm = combat_players(
            [CardIDs.SCOURFIN],
            [CardIDs.RAMPAGER],
        )
        board0 = boards[0]
        scourfin = board0[0]
        side = scourfin.owner_id
        players[side].hand = []

        scourfin.cur_hp = 0
        cm.cleanup_dead(boards, [0, 0], players)  # should not raise


# ===========================================================================
# T3 — The Glad-iator
# ===========================================================================

class TestTheGladIator:
    def test_has_divine_shield(self, mock_unit):
        unit = mock_unit(CardIDs.THE_GLAD_IATOR)
        assert Tags.DIVINE_SHIELD in unit.tags

    def test_gains_atk_on_spell_cast(self, empty_game, player, mock_unit):
        glad = mock_unit(CardIDs.THE_GLAD_IATOR, owner_id=player.uid)
        player.board = [glad]
        atk_before = glad.cur_atk

        _em().process_event(
            Event(
                event_type=EventType.SPELL_CAST,
                source_pos=PosRef(side=player.uid, zone=Zone.HAND, slot=0),
            ),
            {player.uid: player},
            empty_game.tavern.get_next_uid,
        )
        assert glad.cur_atk == atk_before + 1

    def test_enemy_spell_does_not_buff(self, empty_game, player, enemy, mock_unit):
        glad = mock_unit(CardIDs.THE_GLAD_IATOR, owner_id=player.uid)
        player.board = [glad]
        atk_before = glad.cur_atk

        _em().process_event(
            Event(
                event_type=EventType.SPELL_CAST,
                source_pos=PosRef(side=enemy.uid, zone=Zone.HAND, slot=0),
            ),
            {player.uid: player, enemy.uid: enemy},
            empty_game.tavern.get_next_uid,
        )
        assert glad.cur_atk == atk_before


# ===========================================================================
# T3 — Prehistoric Tinkerer
# ===========================================================================

class TestPrehistoricTinkerer:
    def test_buffs_rightmost_on_refresh(self, empty_game, player, mock_unit):
        tinkerer = mock_unit(CardIDs.PREHISTORIC_TINKERER, owner_id=player.uid)
        player.board = [tinkerer]
        victim = mock_unit(CardIDs.ANNOY_O_TRON, owner_id=player.uid)
        player.store.clear()
        player.store.append(StoreItem(unit=victim))
        atk_before = victim.cur_atk
        hp_before = victim.cur_hp

        _em().process_event(
            Event(
                event_type=EventType.TAVERN_REFRESHED,
                source_pos=PosRef(side=player.uid, zone=Zone.HERO, slot=0),
            ),
            {player.uid: player},
            empty_game.tavern.get_next_uid,
        )
        assert victim.cur_atk == atk_before + 2
        assert victim.cur_hp == hp_before + 2


# ===========================================================================
# T3 — Briarback Drummer
# ===========================================================================

class TestBriarbackDrummer:
    def test_adds_barrage_spell(self, empty_game, player, mock_unit):
        drummer = mock_unit(CardIDs.BRIARBACK_DRUMMER, owner_id=player.uid)
        player.board = [drummer]
        player.hand.clear()

        _em().process_event(
            Event(
                event_type=EventType.MINION_PLAYED,
                source=EntityRef(drummer.uid),
                source_pos=PosRef(side=player.uid, zone=Zone.BOARD, slot=0),
            ),
            {player.uid: player},
            empty_game.tavern.get_next_uid,
        )
        spells = [hc.spell for hc in player.hand if hc.spell]
        assert len(spells) == 1
        assert spells[0].card_id == SpellIDs.BLOOD_GEM_BARRAGE


# ===========================================================================
# T4 — King Bagurgle
# ===========================================================================

class TestKingBagurgle:
    def test_buffs_murlocs_board_and_hand(self, empty_game, player, mock_unit):
        bagurgle = mock_unit(CardIDs.KING_BAGURGLE, owner_id=player.uid)
        murloc_board = mock_unit(CardIDs.CANOPY_SWINGER, owner_id=player.uid)
        murloc_hand = mock_unit(CardIDs.HOT_SPRINGER, owner_id=player.uid)
        player.board = [bagurgle, murloc_board]
        player.hand = [HandCard(uid=murloc_hand.uid, unit=murloc_hand)]

        atk_b, hp_b = murloc_board.cur_atk, murloc_board.cur_hp
        atk_h, hp_h = murloc_hand.cur_atk, murloc_hand.cur_hp

        _em().process_event(
            Event(
                event_type=EventType.MINION_PLAYED,
                source=EntityRef(bagurgle.uid),
                source_pos=PosRef(side=player.uid, zone=Zone.BOARD, slot=0),
            ),
            {player.uid: player},
            empty_game.tavern.get_next_uid,
        )
        assert murloc_board.cur_atk == atk_b + 2
        assert murloc_board.cur_hp == hp_b + 3
        assert murloc_hand.cur_atk == atk_h + 2
        assert murloc_hand.cur_hp == hp_h + 3

    def test_no_buff_non_murloc(self, empty_game, player, mock_unit):
        bagurgle = mock_unit(CardIDs.KING_BAGURGLE, owner_id=player.uid)
        non_murloc = mock_unit(CardIDs.RAMPAGER, owner_id=player.uid)
        player.board = [bagurgle, non_murloc]
        atk_before = non_murloc.cur_atk

        _em().process_event(
            Event(
                event_type=EventType.MINION_PLAYED,
                source=EntityRef(bagurgle.uid),
                source_pos=PosRef(side=player.uid, zone=Zone.BOARD, slot=0),
            ),
            {player.uid: player},
            empty_game.tavern.get_next_uid,
        )
        assert non_murloc.cur_atk == atk_before


# ===========================================================================
# T4 — Prized Promo-Drake
# ===========================================================================

class TestPrizedPromoDrake:
    def test_buffs_all_dragons_at_soc(self, empty_game, player, mock_unit):
        drake = mock_unit(CardIDs.PRIZED_PROMO_DRAKE, owner_id=player.uid)
        dragon = mock_unit(CardIDs.AMBER_GUARDIAN, owner_id=player.uid)
        non_dragon = mock_unit(CardIDs.SKELETON, owner_id=player.uid)
        player.board = [drake, dragon, non_dragon]

        atk_d, hp_d = dragon.cur_atk, dragon.cur_hp
        atk_nd = non_dragon.cur_atk

        _em().process_event(
            Event(event_type=EventType.START_OF_COMBAT),
            {player.uid: player},
            empty_game.tavern.get_next_uid,
        )
        assert dragon.cur_atk == atk_d + 4
        assert dragon.cur_hp == hp_d + 4
        assert non_dragon.cur_atk == atk_nd


# ===========================================================================
# T4 — Soulsplitter
# ===========================================================================

class TestSoulsplitter:
    def test_has_reborn(self, mock_unit):
        splitter = mock_unit(CardIDs.SOULSPLITTER)
        assert Tags.REBORN in splitter.tags

    def test_gives_undead_reborn_at_soc(self, empty_game, player, mock_unit):
        splitter = mock_unit(CardIDs.SOULSPLITTER, owner_id=player.uid)
        undead = mock_unit(CardIDs.SKELETON, owner_id=player.uid)
        undead.tags.discard(Tags.REBORN)
        player.board = [splitter, undead]

        _em().process_event(
            Event(event_type=EventType.START_OF_COMBAT),
            {player.uid: player},
            empty_game.tavern.get_next_uid,
        )
        assert Tags.REBORN in undead.tags


# ===========================================================================
# T4 — Tunnel Blaster
# ===========================================================================

class TestTunnelBlaster:
    def test_has_taunt(self, mock_unit):
        assert Tags.TAUNT in mock_unit(CardIDs.TUNNEL_BLASTER).tags

    def test_deals_3_damage_to_all(self, combat_players):
        players, boards, cm = combat_players(
            [CardIDs.TUNNEL_BLASTER, CardIDs.RAMPAGER],
            [CardIDs.RAMPAGER],
        )
        blaster = boards[0][0]
        ally = boards[0][1]
        foe = boards[1][0]

        hp_ally = ally.cur_hp
        hp_foe = foe.cur_hp

        blaster.cur_hp = 0
        cm.cleanup_dead(boards, [0, 0], players)

        assert ally.cur_hp == hp_ally - 3
        assert foe.cur_hp == hp_foe - 3


# ===========================================================================
# T4 — Silent Enforcer
# ===========================================================================

class TestSilentEnforcer:
    def test_has_taunt(self, mock_unit):
        assert Tags.TAUNT in mock_unit(CardIDs.SILENT_ENFORCER).tags

    def test_deals_2_damage_to_all(self, combat_players):
        players, boards, cm = combat_players(
            [CardIDs.SILENT_ENFORCER, CardIDs.RAMPAGER],
            [CardIDs.RAMPAGER],
        )
        enforcer = boards[0][0]
        ally = boards[0][1]
        foe = boards[1][0]

        hp_ally = ally.cur_hp
        hp_foe = foe.cur_hp

        enforcer.cur_hp = 0
        cm.cleanup_dead(boards, [0, 0], players)

        assert ally.cur_hp == hp_ally - 2
        assert foe.cur_hp == hp_foe - 2


# ===========================================================================
# T4 — Bonker
# ===========================================================================

class TestBonker:
    def test_plays_blood_gems_on_others(self, empty_game, player, mock_unit):
        bonker = mock_unit(CardIDs.BONKER, owner_id=player.uid)
        ally1 = mock_unit(CardIDs.SKELETON, owner_id=player.uid)
        ally2 = mock_unit(CardIDs.RISEN_RIDER, owner_id=player.uid)
        player.board = [bonker, ally1, ally2]

        gem_atk, gem_hp = player.mechanics.get_stat(MechanicType.BLOOD_GEM)
        atk1, hp1 = ally1.cur_atk, ally1.cur_hp
        atk2, hp2 = ally2.cur_atk, ally2.cur_hp

        _em().process_event(
            Event(
                event_type=EventType.ATTACK_DECLARED,
                source=EntityRef(bonker.uid),
                source_pos=PosRef(side=player.uid, zone=Zone.BOARD, slot=0),
            ),
            {player.uid: player},
            empty_game.tavern.get_next_uid,
        )
        assert ally1.cur_atk == atk1 + gem_atk * 2
        assert ally1.cur_hp == hp1 + gem_hp * 2
        assert ally2.cur_atk == atk2 + gem_atk * 2
        assert ally2.cur_hp == hp2 + gem_hp * 2


# ===========================================================================
# T4 — Grease Bot
# ===========================================================================

class TestGreaseBot:
    def test_has_divine_shield(self, mock_unit):
        assert Tags.DIVINE_SHIELD in mock_unit(CardIDs.GREASE_BOT).tags

    def test_buffs_unit_that_lost_ds(self, empty_game, player, mock_unit):
        grease = mock_unit(CardIDs.GREASE_BOT, owner_id=player.uid)
        shielded = mock_unit(CardIDs.ANNOY_O_TRON, owner_id=player.uid)
        player.board = [grease, shielded]
        atk_before = shielded.cur_atk
        hp_before = shielded.cur_hp

        _em().process_event(
            Event(
                event_type=EventType.DIVINE_SHIELD_LOST,
                source=EntityRef(shielded.uid),
                source_pos=PosRef(side=player.uid, zone=Zone.BOARD, slot=1),
            ),
            {player.uid: player},
            empty_game.tavern.get_next_uid,
        )
        assert shielded.cur_atk == atk_before + 2
        assert shielded.cur_hp == hp_before + 2


# ===========================================================================
# T4 — Marquee Ticker
# ===========================================================================

class TestMarqueeTicker:
    def test_adds_spell_at_eot(self, empty_game, player, mock_unit):
        ticker = mock_unit(CardIDs.MARQUEE_TICKER, owner_id=player.uid)
        player.board = [ticker]
        player.hand.clear()

        _em().process_event(
            Event(
                event_type=EventType.END_OF_TURN,
                source_pos=PosRef(side=player.uid, zone=Zone.HERO, slot=0),
            ),
            {player.uid: player},
            empty_game.tavern.get_next_uid,
        )
        spells = [hc.spell for hc in player.hand if hc.spell]
        assert len(spells) == 1


# ===========================================================================
# T4 — Refreshing Anomaly
# ===========================================================================

class TestRefreshingAnomaly:
    def test_grants_two_free_refreshes(self, empty_game, player, mock_unit):
        anomaly = mock_unit(CardIDs.REFRESHING_ANOMALY, owner_id=player.uid)
        player.board = [anomaly]
        player.free_refreshes = 0

        _em().process_event(
            Event(
                event_type=EventType.MINION_PLAYED,
                source=EntityRef(anomaly.uid),
                source_pos=PosRef(side=player.uid, zone=Zone.BOARD, slot=0),
            ),
            {player.uid: player},
            empty_game.tavern.get_next_uid,
        )
        assert player.free_refreshes == 2


# ===========================================================================
# T4 — Wannabe Gargoyle
# ===========================================================================

class TestWannabeGargoyle:
    def test_has_reborn(self, mock_unit):
        assert Tags.REBORN in mock_unit(CardIDs.WANNABE_GARGOYLE).tags

    def test_stats(self, mock_unit):
        g = mock_unit(CardIDs.WANNABE_GARGOYLE)
        assert g.cur_atk == 9
        assert g.cur_hp == 1


# ===========================================================================
# T4 — Spirit Drake
# ===========================================================================

class TestSpiritDrake:
    def test_avenge_threshold_3_and_add_spell_target(self):
        avenge_def = AVENGE_REGISTRY.get(CardIDs.SPIRIT_DRAKE)
        assert avenge_def is not None
        assert avenge_def.threshold == 3
        assert avenge_def.buff_target == "add_spell"


# ===========================================================================
# T4 — Witchwing Nestmatron
# ===========================================================================

class TestWitchwingNestmatron:
    def test_avenge_threshold_3_and_add_unit_target(self):
        avenge_def = AVENGE_REGISTRY.get(CardIDs.WITCHWING_NESTMATRON)
        assert avenge_def is not None
        assert avenge_def.threshold == 3
        assert avenge_def.buff_target == "add_unit"


# ===========================================================================
# T4 — Trench Fighter
# ===========================================================================

class TestTrenchFighter:
    def test_adds_gem_confiscation_at_eot(self, empty_game, player, mock_unit):
        fighter = mock_unit(CardIDs.TRENCH_FIGHTER, owner_id=player.uid)
        player.board = [fighter]
        player.hand.clear()

        _em().process_event(
            Event(
                event_type=EventType.END_OF_TURN,
                source_pos=PosRef(side=player.uid, zone=Zone.HERO, slot=0),
            ),
            {player.uid: player},
            empty_game.tavern.get_next_uid,
        )
        spells = [hc.spell for hc in player.hand if hc.spell]
        assert len(spells) == 1
        assert spells[0].card_id == SpellIDs.GEM_CONFISCATION


# ===========================================================================
# T4 — Razorfen Flapper deathrattle
# ===========================================================================

class TestRazorfenFlapper:
    def test_dr_adds_blood_gem_barrage(self, combat_players):
        players, boards, cm = combat_players(
            [CardIDs.RAZORFEN_FLAPPER],
            [CardIDs.RAMPAGER],
        )
        flapper = boards[0][0]
        side = flapper.owner_id
        player = players[side]
        player.hand.clear()

        flapper.cur_hp = 0
        cm.cleanup_dead(boards, [0, 0], players)

        spells = [hc.spell for hc in player.hand if hc.spell]
        assert len(spells) == 1
        assert spells[0].card_id == SpellIDs.BLOOD_GEM_BARRAGE


# ===========================================================================
# Stat + keyword smoke tests
# ===========================================================================

class TestCardStatsSmokeT3T4:
    @pytest.mark.parametrize("card_id, exp_atk, exp_hp, exp_tier", [
        (CardIDs.ANUBARAK_NERUBIAN_KING, 3, 2, 3),
        (CardIDs.ARANASI_ALCHEMIST, 1, 2, 3),
        (CardIDs.BASSGILL, 5, 2, 3),
        (CardIDs.BRIARBACK_DRUMMER, 5, 2, 3),
        (CardIDs.DEFLECT_O_BOT, 3, 2, 3),
        (CardIDs.PEGGY_STURDYBONE, 2, 1, 3),
        (CardIDs.PREHISTORIC_TINKERER, 4, 2, 3),
        (CardIDs.ROARING_RECRUITER, 2, 8, 3),
        (CardIDs.SCOURFIN, 3, 3, 3),
        (CardIDs.TARDY_TRAVELER, 3, 4, 3),
        (CardIDs.TECHNICAL_ELEMENT, 5, 6, 3),
        (CardIDs.THE_GLAD_IATOR, 3, 3, 3),
        (CardIDs.TIMECAPN_HOOKTAIL, 1, 4, 3),
        (CardIDs.UNDERHANDED_DEALER, 3, 3, 3),
        (CardIDs.WAVELING, 6, 1, 3),
        (CardIDs.WHEELED_CREWMATE, 6, 3, 3),
        (CardIDs.WILDFIRE_ELEMENTAL, 6, 3, 3),
        (CardIDs.ACCORD_O_TRON, 5, 5, 4),
        (CardIDs.BLADE_COLLECTOR, 3, 2, 4),
        (CardIDs.BONKER, 2, 7, 4),
        (CardIDs.DEVOUT_HELLCALLER, 2, 2, 4),
        (CardIDs.EN_DJINN_BLAZER, 4, 4, 4),
        (CardIDs.FRIENDLY_GEIST, 6, 3, 4),
        (CardIDs.GEOMAGUS_ROOGUG, 4, 6, 4),
        (CardIDs.GREASE_BOT, 2, 4, 4),
        (CardIDs.GUNPOWDER_COURIER, 2, 6, 4),
        (CardIDs.HEROIC_UNDERDOG, 1, 10, 4),
        (CardIDs.HUMON_GOZZ, 5, 5, 4),
        (CardIDs.INDUSTRIOUS_DECKHAND, 3, 5, 4),
        (CardIDs.KING_BAGURGLE, 3, 4, 4),
        (CardIDs.MARQUEE_TICKER, 1, 5, 4),
        (CardIDs.PRIZED_PROMO_DRAKE, 1, 1, 4),
        (CardIDs.PROSTHETIC_HAND, 3, 1, 4),
        (CardIDs.RAZORFEN_FLAPPER, 5, 3, 4),
        (CardIDs.REFRESHING_ANOMALY, 4, 5, 4),
        (CardIDs.SILENT_ENFORCER, 6, 2, 4),
        (CardIDs.SIN_DOREI_STRAIGHT_SHOT, 3, 4, 4),
        (CardIDs.SLY_RAPTOR, 1, 4, 4),
        (CardIDs.SOULSPLITTER, 4, 2, 4),
        (CardIDs.SPIRIT_DRAKE, 1, 8, 4),
        (CardIDs.TAVERN_TEMPEST, 2, 2, 4),
        (CardIDs.TRENCH_FIGHTER, 6, 6, 4),
        (CardIDs.TUNNEL_BLASTER, 3, 7, 4),
        (CardIDs.WANNABE_GARGOYLE, 9, 1, 4),
        (CardIDs.WITCHWING_NESTMATRON, 3, 5, 4),
    ])
    def test_stats(self, mock_unit, card_id, exp_atk, exp_hp, exp_tier):
        unit = mock_unit(card_id)
        assert unit.cur_atk == exp_atk, f"{card_id} atk"
        assert unit.cur_hp == exp_hp, f"{card_id} hp"
        assert unit.tier == exp_tier, f"{card_id} tier"


class TestKeywordTagsSmokeT3T4:
    @pytest.mark.parametrize("card_id, tags", [
        (CardIDs.TECHNICAL_ELEMENT, {Tags.MAGNETIC}),
        (CardIDs.WILDFIRE_ELEMENTAL, {Tags.CLEAVE}),
        (CardIDs.ARANASI_ALCHEMIST, {Tags.TAUNT, Tags.REBORN}),
        (CardIDs.DEFLECT_O_BOT, {Tags.DIVINE_SHIELD}),
        (CardIDs.PREHISTORIC_TINKERER, {Tags.DIVINE_SHIELD}),
        (CardIDs.THE_GLAD_IATOR, {Tags.DIVINE_SHIELD}),
        (CardIDs.ACCORD_O_TRON, {Tags.MAGNETIC}),
        (CardIDs.BLADE_COLLECTOR, {Tags.CLEAVE}),
        (CardIDs.GEOMAGUS_ROOGUG, {Tags.DIVINE_SHIELD}),
        (CardIDs.GREASE_BOT, {Tags.DIVINE_SHIELD}),
        (CardIDs.HEROIC_UNDERDOG, {Tags.STEALTH}),
        (CardIDs.HUMON_GOZZ, {Tags.DIVINE_SHIELD}),
        (CardIDs.PROSTHETIC_HAND, {Tags.MAGNETIC, Tags.REBORN}),
        (CardIDs.SIN_DOREI_STRAIGHT_SHOT, {Tags.DIVINE_SHIELD, Tags.WINDFURY}),
        (CardIDs.SILENT_ENFORCER, {Tags.TAUNT}),
        (CardIDs.SOULSPLITTER, {Tags.REBORN}),
        (CardIDs.TUNNEL_BLASTER, {Tags.TAUNT}),
        (CardIDs.WANNABE_GARGOYLE, {Tags.REBORN}),
    ])
    def test_tags(self, mock_unit, card_id, tags):
        unit = mock_unit(card_id)
        for tag in tags:
            assert tag in unit.tags, f"{card_id} missing {tag}"
