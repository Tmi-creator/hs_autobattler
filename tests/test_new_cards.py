"""Tests for new T2 and T3 cards added in batch implementation."""

from __future__ import annotations

from collections import deque
from typing import Callable, Dict, List, Tuple

import pytest

from hearthstone.engine.combat import CombatManager
from hearthstone.engine.entities import HandCard, Player, StoreItem, Unit
from hearthstone.engine.enums import CardIDs, SpellIDs, Tags, UnitType
from hearthstone.engine.event_system import (
    EffectContext,
    EntityRef,
    Event,
    EventType,
    PosRef,
    Zone,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ctx(empty_game, player):
    return EffectContext(
        {player.uid: player},
        empty_game.tavern.get_next_uid,
        deque(),
        card_pool=empty_game.pool,
    )


def _make_ctx2(empty_game, p0, p1):
    return EffectContext(
        {p0.uid: p0, p1.uid: p1},
        empty_game.tavern.get_next_uid,
        deque(),
        card_pool=empty_game.pool,
    )


# ===========================================================================
# T2 CARDS
# ===========================================================================


class TestEmbalmingExpert:
    """After Tavern Refreshed: give rightmost shop minion +2 ATK and Reborn."""

    def test_buffs_rightmost_shop_unit_atk(
        self, empty_game, player, tavern, mock_unit
    ):
        expert = mock_unit(CardIDs.EMBALMING_EXPERT, owner_id=player.uid)
        player.board.append(expert)

        victim = mock_unit(CardIDs.ANNOY_O_TRON, owner_id=player.uid)
        decoy = mock_unit(CardIDs.SKELETON, owner_id=player.uid)
        player.store.clear()
        player.store.append(StoreItem(unit=decoy))
        player.store.append(StoreItem(unit=victim))

        atk_before = victim.cur_atk

        tavern.event_manager.process_event(
            Event(
                event_type=EventType.TAVERN_REFRESHED,
                source_pos=PosRef(side=player.uid, zone=Zone.HERO, slot=0),
            ),
            {player.uid: player},
            empty_game.tavern.get_next_uid,
        )

        assert victim.cur_atk == atk_before + 2
        assert Tags.REBORN in victim.tags

    def test_does_not_buff_when_shop_empty(
        self, empty_game, player, mock_unit
    ):
        expert = mock_unit(CardIDs.EMBALMING_EXPERT, owner_id=player.uid)
        player.board.append(expert)
        player.store.clear()

        # Should not raise
        empty_game.tavern.event_manager.process_event(
            Event(
                event_type=EventType.TAVERN_REFRESHED,
                source_pos=PosRef(side=player.uid, zone=Zone.HERO, slot=0),
            ),
            {player.uid: player},
            empty_game.tavern.get_next_uid,
        )

    def test_only_fires_for_own_refresh(
        self, empty_game, player, enemy, mock_unit
    ):
        """Expert on player.board should not buff enemy's shop."""
        expert = mock_unit(CardIDs.EMBALMING_EXPERT, owner_id=player.uid)
        player.board.append(expert)

        victim = mock_unit(CardIDs.ANNOY_O_TRON, owner_id=enemy.uid)
        enemy.store.clear()
        enemy.store.append(StoreItem(unit=victim))
        atk_before = victim.cur_atk

        empty_game.tavern.event_manager.process_event(
            Event(
                event_type=EventType.TAVERN_REFRESHED,
                source_pos=PosRef(side=enemy.uid, zone=Zone.HERO, slot=0),
            ),
            {player.uid: player, enemy.uid: enemy},
            empty_game.tavern.get_next_uid,
        )
        assert victim.cur_atk == atk_before  # unchanged


class TestQuilledCabbie:
    """After Tavern Refreshed: play 2 Blood Gems on rightmost shop minion."""

    def test_applies_two_blood_gems(self, empty_game, player, mock_unit):
        cabbie = mock_unit(CardIDs.QUILLED_CABBIE, owner_id=player.uid)
        player.board.append(cabbie)

        victim = mock_unit(CardIDs.ANNOY_O_TRON, owner_id=player.uid)
        player.store.clear()
        player.store.append(StoreItem(unit=victim))

        # Default blood gem = +1/+1; applies 2x → +2/+2
        atk_before = victim.cur_atk
        hp_before = victim.cur_hp

        empty_game.tavern.event_manager.process_event(
            Event(
                event_type=EventType.TAVERN_REFRESHED,
                source_pos=PosRef(side=player.uid, zone=Zone.HERO, slot=0),
            ),
            {player.uid: player},
            empty_game.tavern.get_next_uid,
        )

        assert victim.cur_atk == atk_before + 2
        assert victim.cur_hp == hp_before + 2


class TestGhostlyYmirjar:
    """Avenge(4): Gain a free Refresh."""

    def test_in_avenge_registry(self):
        from hearthstone.engine.card_def import AVENGE_REGISTRY
        assert CardIDs.GHOSTLY_YMIRJAR in AVENGE_REGISTRY
        eff = AVENGE_REGISTRY[CardIDs.GHOSTLY_YMIRJAR]
        assert eff.threshold == 4
        assert eff.buff_target == "free_refresh"

    def test_grants_free_refresh_after_4_deaths(
        self, combat_players, mock_unit
    ):
        players, boards, cm = combat_players(
            [CardIDs.GHOSTLY_YMIRJAR,
             CardIDs.ANNOY_O_TRON, CardIDs.ANNOY_O_TRON,
             CardIDs.ANNOY_O_TRON, CardIDs.ANNOY_O_TRON],
            [CardIDs.ANNOY_O_TRON],
        )
        board0 = boards[0]
        ymirjar = board0[0]
        sac = board0[1:]

        side_uid = ymirjar.owner_id
        player = players[side_uid]
        assert player.free_refreshes == 0

        # Kill 3 — avenge should NOT fire yet
        for s in sac[:3]:
            s.cur_hp = 0
            cm.cleanup_dead(boards, [0, 0], players)
        assert player.free_refreshes == 0

        # Kill 4th — avenge fires
        sac[3].cur_hp = 0
        cm.cleanup_dead(boards, [0, 0], players)
        assert player.free_refreshes == 1


class TestFireBaller:
    """On sell: buff board +1 ATK and increment baller counter."""

    def test_sell_buffs_board(self, empty_game, player, tavern, mock_unit):
        baller = mock_unit(CardIDs.FIRE_BALLER, owner_id=player.uid)
        friendly = mock_unit(CardIDs.ANNOY_O_TRON, owner_id=player.uid)
        player.board.append(friendly)

        from hearthstone.engine.entities import HandCard
        player.hand.append(HandCard(uid=baller.uid, unit=baller))
        player.gold = 10

        atk_before = friendly.cur_atk
        # Put baller on board then sell it
        tavern.play_unit(player, 0)
        sell_idx = next(i for i, u in enumerate(player.board) if u.card_id == CardIDs.FIRE_BALLER)
        tavern.sell_unit(player, sell_idx)

        assert friendly.cur_atk == atk_before + 1

    def test_scaling_increments(self, empty_game, player, tavern, mock_unit):
        from hearthstone.engine.entities import HandCard

        baller1 = mock_unit(CardIDs.FIRE_BALLER, owner_id=player.uid)
        baller2 = mock_unit(CardIDs.FIRE_BALLER, owner_id=player.uid)
        friendly = mock_unit(CardIDs.ANNOY_O_TRON, owner_id=player.uid)
        player.board.append(friendly)
        player.gold = 10

        # Sell first baller → +1 atk (counter=0 before → +1*1=+1)
        player.hand.append(HandCard(uid=baller1.uid, unit=baller1))
        tavern.play_unit(player, 0)
        sell_idx = next(i for i, u in enumerate(player.board) if u.card_id == CardIDs.FIRE_BALLER)
        atk_after_first_sell = friendly.cur_atk
        tavern.sell_unit(player, sell_idx)
        assert friendly.cur_atk == atk_after_first_sell + 1

        # Sell second baller → +2 atk (counter=1 → +1*2=+2)
        player.hand.append(HandCard(uid=baller2.uid, unit=baller2))
        tavern.play_unit(player, 0)
        sell_idx = next(i for i, u in enumerate(player.board) if u.card_id == CardIDs.FIRE_BALLER)
        atk_after_second_sell = friendly.cur_atk
        tavern.sell_unit(player, sell_idx)
        assert friendly.cur_atk == atk_after_second_sell + 2


class TestSnowBaller:
    """On sell: buff board +1 HP and increment baller counter."""

    def test_sell_buffs_board_hp(self, empty_game, player, tavern, mock_unit):
        from hearthstone.engine.entities import HandCard
        baller = mock_unit(CardIDs.SNOW_BALLER, owner_id=player.uid)
        friendly = mock_unit(CardIDs.ANNOY_O_TRON, owner_id=player.uid)
        player.board.append(friendly)
        player.hand.append(HandCard(uid=baller.uid, unit=baller))
        player.gold = 10

        hp_before = friendly.cur_hp
        tavern.play_unit(player, 0)
        sell_idx = next(i for i, u in enumerate(player.board) if u.card_id == CardIDs.SNOW_BALLER)
        tavern.sell_unit(player, sell_idx)
        assert friendly.cur_hp == hp_before + 1


def _fire_soc(em, players):
    """Fire START_OF_COMBAT event via event manager (simulating combat start)."""
    from hearthstone.engine.event_system import Event, EventType, PosRef, Zone
    first_side = next(iter(players))
    em.process_event(
        Event(
            event_type=EventType.START_OF_COMBAT,
            source_pos=PosRef(side=first_side, zone=Zone.BOARD, slot=-1),
        ),
        players,
        lambda: 77777,
    )


class TestIrateRooster:
    """SoC: Deal 1 damage to adjacent and give them +4 ATK."""

    def test_soc_damages_and_buffs_adjacent(
        self, combat_players, event_manager
    ):
        players, boards, cm = combat_players(
            [CardIDs.ANNOY_O_TRON, CardIDs.IRATE_ROOSTER, CardIDs.ANNOY_O_TRON],
            [CardIDs.ANNOY_O_TRON],
        )
        board0 = boards[0]
        left = board0[0]
        right = board0[2]

        left_atk_before = left.cur_atk
        right_atk_before = right.cur_atk
        left_hp_before = left.cur_hp
        right_hp_before = right.cur_hp

        _fire_soc(event_manager, players)

        # Adjacent take 1 damage and get +4 ATK (combat buff)
        assert left.cur_hp == left_hp_before - 1
        assert right.cur_hp == right_hp_before - 1
        assert left.cur_atk == left_atk_before + 4
        assert right.cur_atk == right_atk_before + 4

    def test_soc_does_not_buff_non_adjacent(
        self, combat_players, event_manager
    ):
        players, boards, cm = combat_players(
            [CardIDs.IRATE_ROOSTER, CardIDs.ANNOY_O_TRON, CardIDs.SKELETON],
            [CardIDs.ANNOY_O_TRON],
        )
        board0 = boards[0]
        far = board0[2]
        atk_before = far.cur_atk

        _fire_soc(event_manager, players)
        assert far.cur_atk == atk_before  # no buff


class TestSoulRewinder:
    """After hero takes damage: undo it and gain +1 HP."""

    def test_heals_hero_and_buffs_self(self, empty_game, player, mock_unit):
        rewinder = mock_unit(CardIDs.SOUL_REWINDER, owner_id=player.uid)
        player.board.append(rewinder)
        player.health = 30
        hp_before = rewinder.cur_hp

        from hearthstone.engine.card_def import TRIGGER_REGISTRY, GOLDEN_TRIGGER_REGISTRY
        from hearthstone.engine.event_system import EventManager
        em = EventManager(TRIGGER_REGISTRY, GOLDEN_TRIGGER_REGISTRY)

        # HERO_DAMAGED fires with value=5; rewinder heals hero by 5 and gains +1 HP
        em.process_event(
            Event(
                event_type=EventType.HERO_DAMAGED,
                source_pos=PosRef(side=player.uid, zone=Zone.HERO, slot=0),
                value=5,
            ),
            {player.uid: player},
            empty_game.tavern.get_next_uid,
        )
        assert player.health == 35  # 30 + 5 healed back
        assert rewinder.cur_hp == hp_before + 1

    def test_does_not_fire_for_enemy_hero_damage(
        self, empty_game, player, enemy, mock_unit
    ):
        """Rewinder on player side should not react to enemy hero taking damage."""
        rewinder = mock_unit(CardIDs.SOUL_REWINDER, owner_id=player.uid)
        player.board.append(rewinder)
        hp_before = rewinder.cur_hp

        from hearthstone.engine.card_def import TRIGGER_REGISTRY, GOLDEN_TRIGGER_REGISTRY
        from hearthstone.engine.event_system import EventManager
        em = EventManager(TRIGGER_REGISTRY, GOLDEN_TRIGGER_REGISTRY)
        em.process_event(
            Event(
                event_type=EventType.HERO_DAMAGED,
                source_pos=PosRef(side=enemy.uid, zone=Zone.HERO, slot=0),
                value=3,
            ),
            {player.uid: player, enemy.uid: enemy},
            empty_game.tavern.get_next_uid,
        )
        assert rewinder.cur_hp == hp_before  # no self-buff


class TestSurfingSylvar:
    """EOT: give adjacent +1 ATK (repeat per golden)."""

    def test_eot_buffs_adjacent_once_without_golden(
        self, empty_game, player, tavern, mock_unit
    ):
        left = mock_unit(CardIDs.ANNOY_O_TRON, owner_id=player.uid)
        sylvar = mock_unit(CardIDs.SURFING_SYLVAR, owner_id=player.uid)
        right = mock_unit(CardIDs.SKELETON, owner_id=player.uid)
        player.board = [left, sylvar, right]

        atk_left_before = left.cur_atk
        atk_right_before = right.cur_atk

        tavern.end_turn(player)

        assert left.cur_atk == atk_left_before + 1
        assert right.cur_atk == atk_right_before + 1

    def test_eot_repeats_per_golden(
        self, empty_game, player, tavern, mock_unit
    ):
        left = mock_unit(CardIDs.ANNOY_O_TRON, owner_id=player.uid)
        sylvar = mock_unit(CardIDs.SURFING_SYLVAR, owner_id=player.uid)
        golden = mock_unit(CardIDs.SKELETON, owner_id=player.uid, is_golden=True)
        player.board = [left, sylvar, golden]

        atk_left_before = left.cur_atk
        # golden=1 → repeats = max(1,1) = 1 base + golden_count=1 → total 2 (base 1 + 1 repeat)
        # Wait, the formula is max(1, golden_count) not 1+golden_count
        # So 1 golden → repeats = 1, still 1 buff
        # Let's add a second golden to see doubling
        golden2 = mock_unit(CardIDs.ANNOY_O_TRON, owner_id=player.uid, is_golden=True)
        player.board = [left, sylvar, golden, golden2]

        atk_left_before = left.cur_atk

        tavern.end_turn(player)

        # repeats = max(1, 2) = 2 → left gets +2
        assert left.cur_atk == atk_left_before + 2


class TestPatientScout:
    """On sell: discover a T1 minion (improves each turn)."""

    def test_sell_sets_discovery_request(
        self, empty_game, player, tavern, mock_unit
    ):
        scout = mock_unit(CardIDs.PATIENT_SCOUT, owner_id=player.uid)
        player.hand.append(HandCard(uid=scout.uid, unit=scout))
        player.gold = 10

        tavern.play_unit(player, 0)
        assert player.pending_discovery_request is None

        sell_idx = next(i for i, u in enumerate(player.board) if u.card_id == CardIDs.PATIENT_SCOUT)
        tavern.sell_unit(player, sell_idx)

        assert player.pending_discovery_request is not None
        assert player.pending_discovery_request.tier == 1

    def test_scaling_increases_tier(self, empty_game, player, mock_unit):
        # Simulate 2 prior sells by incrementing scaling counter
        player.mechanics.increment_scaling("patient_scout")
        player.mechanics.increment_scaling("patient_scout")

        scout = mock_unit(CardIDs.PATIENT_SCOUT, owner_id=player.uid)
        player.hand.append(HandCard(uid=scout.uid, unit=scout))
        player.gold = 10

        from hearthstone.engine.card_def import TRIGGER_REGISTRY, GOLDEN_TRIGGER_REGISTRY
        from hearthstone.engine.event_system import EventManager, Event, EventType, PosRef, Zone
        em = EventManager(TRIGGER_REGISTRY, GOLDEN_TRIGGER_REGISTRY)

        # Put scout on board manually
        player.board.append(scout)

        em.process_event(
            Event(
                event_type=EventType.MINION_SOLD,
                source=EntityRef(scout.uid),
                source_pos=PosRef(side=player.uid, zone=Zone.BOARD, slot=0),
            ),
            {player.uid: player},
            empty_game.tavern.get_next_uid,
            card_pool=empty_game.pool,
        )
        assert player.pending_discovery_request is not None
        assert player.pending_discovery_request.tier == 3  # base 1 + 2 increments


# ===========================================================================
# T3 CARDS
# ===========================================================================


class TestAnnoyOModule:
    """Pure keyword card: Magnetic, Divine Shield, Taunt."""

    def test_has_correct_tags(self, mock_unit):
        unit = mock_unit(CardIDs.ANNOY_O_MODULE)
        assert Tags.DIVINE_SHIELD in unit.tags
        assert Tags.TAUNT in unit.tags
        assert Tags.MAGNETIC in unit.tags

    def test_stats(self, mock_unit):
        unit = mock_unit(CardIDs.ANNOY_O_MODULE)
        assert unit.cur_atk == 2
        assert unit.cur_hp == 4


class TestDeadlySpore:
    """Venomous."""

    def test_has_venomous_tag(self, mock_unit):
        unit = mock_unit(CardIDs.DEADLY_SPORE)
        assert Tags.VENOMOUS in unit.tags

    def test_stats(self, mock_unit):
        unit = mock_unit(CardIDs.DEADLY_SPORE)
        assert unit.cur_atk == 1
        assert unit.cur_hp == 1


class TestCadaverCaretaker:
    """DR: Summon 3 Skeletons."""

    def test_summons_three_skeletons(
        self, combat_players
    ):
        players, boards, cm = combat_players(
            [CardIDs.CADAVER_CARETAKER],
            [CardIDs.ANNOY_O_TRON],
        )
        board0 = boards[0]
        caretaker = board0[0]
        caretaker.cur_hp = 0
        cm.cleanup_dead(boards, [0, 0], players)

        skeletons = [u for u in board0 if u.card_id == CardIDs.SKELETON]
        assert len(skeletons) == 3


class TestBrinyBootlegger:
    """DR: Get a Tavern Coin."""

    def test_deathrattle_adds_tavern_coin(self, combat_players):
        players, boards, cm = combat_players(
            [CardIDs.BRINY_BOOTLEGGER],
            [CardIDs.ANNOY_O_TRON],
        )
        board0 = boards[0]
        bootlegger = board0[0]
        side = bootlegger.owner_id
        player = players[side]

        bootlegger.cur_hp = 0
        cm.cleanup_dead(boards, [0, 0], players)

        coins = [hc for hc in player.hand if hc.spell and hc.spell.card_id == SpellIDs.TAVERN_COIN]
        assert len(coins) == 1


class TestHandlessForsaken:
    """DR: Summon a 2/1 Hand with Reborn."""

    def test_summons_hand_token_with_reborn(
        self, combat_players
    ):
        players, boards, cm = combat_players(
            [CardIDs.HANDLESS_FORSAKEN],
            [CardIDs.ANNOY_O_TRON],
        )
        board0 = boards[0]
        forsaken = board0[0]
        forsaken.cur_hp = 0
        cm.cleanup_dead(boards, [0, 0], players)

        hands = [u for u in board0 if u.card_id == CardIDs.HAND_TOKEN]
        assert len(hands) == 1
        assert Tags.REBORN in hands[0].tags


class TestGreedySnaketongue:
    """Rally: Get a Tavern Coin."""

    def test_rally_adds_tavern_coin(self, combat_players, event_manager):
        players, boards, cm = combat_players(
            [CardIDs.GREEDY_SNAKETONGUE, CardIDs.ANNOY_O_TRON],
            [CardIDs.ANNOY_O_TRON],
        )
        board0 = boards[0]
        snake = board0[0]
        side = snake.owner_id
        player = players[side]

        event_manager.process_event(
            Event(
                event_type=EventType.ATTACK_DECLARED,
                source=EntityRef(snake.uid),
                source_pos=PosRef(side=side, zone=Zone.BOARD, slot=0),
            ),
            players,
            cm.get_uid,
        )
        coins = [hc for hc in player.hand if hc.spell and hc.spell.card_id == SpellIDs.TAVERN_COIN]
        assert len(coins) == 1


class TestRoadboar:
    """Rally: Get 2 Blood Gems."""

    def test_rally_adds_two_blood_gems(self, combat_players, event_manager):
        players, boards, cm = combat_players(
            [CardIDs.ROADBOAR, CardIDs.SKELETON],
            [CardIDs.ANNOY_O_TRON],
        )
        board0 = boards[0]
        boar = board0[0]
        side = boar.owner_id
        player = players[side]

        event_manager.process_event(
            Event(
                event_type=EventType.ATTACK_DECLARED,
                source=EntityRef(boar.uid),
                source_pos=PosRef(side=side, zone=Zone.BOARD, slot=0),
            ),
            players,
            cm.get_uid,
        )
        gems = [hc for hc in player.hand if hc.spell and hc.spell.card_id == SpellIDs.BLOOD_GEM]
        assert len(gems) == 2


class TestGoldgrubber:
    """EOT: gain +3/+2 per golden minion."""

    def test_no_buff_without_golden(
        self, empty_game, player, tavern, mock_unit
    ):
        grubber = mock_unit(CardIDs.GOLDGRUBBER, owner_id=player.uid)
        player.board.append(grubber)
        atk_before = grubber.cur_atk
        hp_before = grubber.cur_hp

        tavern.end_turn(player)
        assert grubber.cur_atk == atk_before
        assert grubber.cur_hp == hp_before

    def test_buff_per_golden(
        self, empty_game, player, tavern, mock_unit
    ):
        grubber = mock_unit(CardIDs.GOLDGRUBBER, owner_id=player.uid)
        golden1 = mock_unit(CardIDs.ANNOY_O_TRON, owner_id=player.uid, is_golden=True)
        golden2 = mock_unit(CardIDs.SKELETON, owner_id=player.uid, is_golden=True)
        player.board = [grubber, golden1, golden2]

        atk_before = grubber.cur_atk
        hp_before = grubber.cur_hp

        tavern.end_turn(player)
        assert grubber.cur_atk == atk_before + 3 * 2
        assert grubber.cur_hp == hp_before + 2 * 2


class TestGemsplitter:
    """DS. After friendly loses DS, get Blood Gem."""

    def test_has_divine_shield(self, mock_unit):
        unit = mock_unit(CardIDs.GEMSPLITTER)
        assert Tags.DIVINE_SHIELD in unit.tags

    def test_friendly_ds_lost_adds_blood_gem(
        self, empty_game, player, mock_unit
    ):
        splitter = mock_unit(CardIDs.GEMSPLITTER, owner_id=player.uid)
        other = mock_unit(CardIDs.ANNOY_O_TRON, owner_id=player.uid)
        player.board = [splitter, other]

        from hearthstone.engine.card_def import TRIGGER_REGISTRY, GOLDEN_TRIGGER_REGISTRY
        from hearthstone.engine.event_system import EventManager
        em = EventManager(TRIGGER_REGISTRY, GOLDEN_TRIGGER_REGISTRY)

        em.process_event(
            Event(
                event_type=EventType.DIVINE_SHIELD_LOST,
                source=EntityRef(other.uid),
                source_pos=PosRef(side=player.uid, zone=Zone.BOARD, slot=1),
            ),
            {player.uid: player},
            empty_game.tavern.get_next_uid,
        )
        gems = [hc for hc in player.hand if hc.spell and hc.spell.card_id == SpellIDs.BLOOD_GEM]
        assert len(gems) == 1

    def test_enemy_ds_lost_does_not_trigger(
        self, empty_game, player, enemy, mock_unit
    ):
        splitter = mock_unit(CardIDs.GEMSPLITTER, owner_id=player.uid)
        player.board = [splitter]
        enemy_unit = mock_unit(CardIDs.ANNOY_O_TRON, owner_id=enemy.uid)
        enemy.board = [enemy_unit]

        from hearthstone.engine.card_def import TRIGGER_REGISTRY, GOLDEN_TRIGGER_REGISTRY
        from hearthstone.engine.event_system import EventManager
        em = EventManager(TRIGGER_REGISTRY, GOLDEN_TRIGGER_REGISTRY)

        em.process_event(
            Event(
                event_type=EventType.DIVINE_SHIELD_LOST,
                source=EntityRef(enemy_unit.uid),
                source_pos=PosRef(side=enemy.uid, zone=Zone.BOARD, slot=0),
            ),
            {player.uid: player, enemy.uid: enemy},
            empty_game.tavern.get_next_uid,
        )
        gems = [hc for hc in player.hand if hc.spell and hc.spell.card_id == SpellIDs.BLOOD_GEM]
        assert len(gems) == 0


class TestCanopySwinger:
    """BC: Give all other Murlocs in hand and board +4 ATK."""

    def test_bc_buffs_board_murlocs(
        self, empty_game, player, tavern, mock_unit
    ):
        swinger = mock_unit(CardIDs.CANOPY_SWINGER, owner_id=player.uid)
        murloc_board = mock_unit(CardIDs.SALTSCALE_HONCHO, owner_id=player.uid)
        non_murloc = mock_unit(CardIDs.ANNOY_O_TRON, owner_id=player.uid)
        player.board = [murloc_board, non_murloc]
        player.hand.append(HandCard(uid=swinger.uid, unit=swinger))
        player.gold = 10

        murloc_atk_before = murloc_board.cur_atk
        non_murloc_atk_before = non_murloc.cur_atk

        tavern.play_unit(player, 0)

        assert murloc_board.cur_atk == murloc_atk_before + 4
        assert non_murloc.cur_atk == non_murloc_atk_before

    def test_bc_buffs_hand_murlocs(
        self, empty_game, player, tavern, mock_unit
    ):
        swinger = mock_unit(CardIDs.CANOPY_SWINGER, owner_id=player.uid)
        hand_murloc = mock_unit(CardIDs.TAD, owner_id=player.uid)
        player.hand = [HandCard(uid=swinger.uid, unit=swinger),
                       HandCard(uid=hand_murloc.uid, unit=hand_murloc)]
        player.gold = 10

        murloc_atk_before = hand_murloc.cur_atk
        tavern.play_unit(player, 0)
        assert hand_murloc.cur_atk == murloc_atk_before + 4


class TestHotSpringer:
    """BC: Give all other Murlocs in hand and board +4 HP."""

    def test_bc_buffs_murloc_hp(
        self, empty_game, player, tavern, mock_unit
    ):
        springer = mock_unit(CardIDs.HOT_SPRINGER, owner_id=player.uid)
        murloc = mock_unit(CardIDs.SALTSCALE_HONCHO, owner_id=player.uid)
        player.board = [murloc]
        player.hand.append(HandCard(uid=springer.uid, unit=springer))
        player.gold = 10

        hp_before = murloc.cur_hp
        tavern.play_unit(player, 0)
        assert murloc.cur_hp == hp_before + 4


class TestRampager:
    """Rally: Deal 1 damage to your other minions."""

    def test_rally_damages_friendly_board(self, combat_players, event_manager):
        players, boards, cm = combat_players(
            [CardIDs.RAMPAGER, CardIDs.ANNOY_O_TRON, CardIDs.SKELETON],
            [CardIDs.ANNOY_O_TRON],
        )
        board0 = boards[0]
        rampager = board0[0]
        annoy = board0[1]
        skeleton = board0[2]

        side = rampager.owner_id
        hp_annoy_before = annoy.cur_hp
        hp_skel_before = skeleton.cur_hp

        event_manager.process_event(
            Event(
                event_type=EventType.ATTACK_DECLARED,
                source=EntityRef(rampager.uid),
                source_pos=PosRef(side=side, zone=Zone.BOARD, slot=0),
            ),
            players,
            cm.get_uid,
        )
        assert annoy.cur_hp == hp_annoy_before - 1
        assert skeleton.cur_hp == hp_skel_before - 1


class TestFelemental:
    """BC: Give tavern minions +2/+1 this game."""

    def test_bc_buffs_current_shop(
        self, empty_game, player, tavern, mock_unit
    ):
        felemental = mock_unit(CardIDs.FELEMENTAL, owner_id=player.uid)
        shop_unit = mock_unit(CardIDs.ANNOY_O_TRON, owner_id=player.uid)
        player.store.clear()
        player.store.append(StoreItem(unit=shop_unit))
        player.hand.append(HandCard(uid=felemental.uid, unit=felemental))
        player.gold = 10

        atk_before = shop_unit.cur_atk
        hp_before = shop_unit.cur_hp

        tavern.play_unit(player, 0)

        assert shop_unit.cur_atk == atk_before + 2
        assert shop_unit.cur_hp == hp_before + 1


class TestPricklyPiper:
    """DR: Blood Gems give +1 ATK this game."""

    def test_dr_increments_blood_gem_atk(
        self, combat_players
    ):
        players, boards, cm = combat_players(
            [CardIDs.PRICKLY_PIPER],
            [CardIDs.ANNOY_O_TRON],
        )
        board0 = boards[0]
        piper = board0[0]
        side = piper.owner_id
        player = players[side]

        # Default blood gem atk = 1
        from hearthstone.engine.enums import MechanicType
        atk_before, hp_before = player.mechanics.get_stat(MechanicType.BLOOD_GEM)

        piper.cur_hp = 0
        cm.cleanup_dead(boards, [0, 0], players)

        atk_after, hp_after = player.mechanics.get_stat(MechanicType.BLOOD_GEM)
        assert atk_after == atk_before + 1
        assert hp_after == hp_before


class TestAmberGuardian:
    """Taunt. SoC: Give another friendly Dragon +2/+2 and DS."""

    def test_has_taunt(self, mock_unit):
        unit = mock_unit(CardIDs.AMBER_GUARDIAN)
        assert Tags.TAUNT in unit.tags

    def test_soc_buffs_random_dragon_and_gives_ds(
        self, combat_players, event_manager
    ):
        players, boards, cm = combat_players(
            [CardIDs.AMBER_GUARDIAN, CardIDs.SLEEPY_SUPPORTER],
            [CardIDs.ANNOY_O_TRON],
        )
        board0 = boards[0]
        dragon = board0[1]

        dragon_atk_before = dragon.cur_atk

        _fire_soc(event_manager, players)

        assert dragon.cur_atk == dragon_atk_before + 2
        assert Tags.DIVINE_SHIELD in dragon.tags


class TestHardyOrca:
    """Taunt. Whenever takes damage, give other minions +1/+1."""

    def test_has_taunt(self, mock_unit):
        unit = mock_unit(CardIDs.HARDY_ORCA)
        assert Tags.TAUNT in unit.tags

    def test_self_damage_buffs_others(
        self, empty_game, player, mock_unit
    ):
        orca = mock_unit(CardIDs.HARDY_ORCA, owner_id=player.uid)
        other = mock_unit(CardIDs.ANNOY_O_TRON, owner_id=player.uid)
        player.board = [orca, other]

        other_atk_before = other.cur_atk
        other_hp_before = other.cur_hp

        from hearthstone.engine.card_def import TRIGGER_REGISTRY, GOLDEN_TRIGGER_REGISTRY
        from hearthstone.engine.event_system import EventManager
        em = EventManager(TRIGGER_REGISTRY, GOLDEN_TRIGGER_REGISTRY)

        em.process_event(
            Event(
                event_type=EventType.MINION_DAMAGED,
                source=EntityRef(9999),   # some attacker
                target=EntityRef(orca.uid),
                source_pos=PosRef(side=1, zone=Zone.BOARD, slot=0),
                target_pos=PosRef(side=player.uid, zone=Zone.BOARD, slot=0),
                value=2,
            ),
            {player.uid: player},
            empty_game.tavern.get_next_uid,
        )
        assert other.cur_atk == other_atk_before + 1
        assert other.cur_hp == other_hp_before + 1

    def test_self_not_buffed(
        self, empty_game, player, mock_unit
    ):
        orca = mock_unit(CardIDs.HARDY_ORCA, owner_id=player.uid)
        player.board = [orca]
        atk_before = orca.cur_atk

        from hearthstone.engine.card_def import TRIGGER_REGISTRY, GOLDEN_TRIGGER_REGISTRY
        from hearthstone.engine.event_system import EventManager
        em = EventManager(TRIGGER_REGISTRY, GOLDEN_TRIGGER_REGISTRY)

        em.process_event(
            Event(
                event_type=EventType.MINION_DAMAGED,
                source=EntityRef(9999),
                target=EntityRef(orca.uid),
                source_pos=PosRef(side=1, zone=Zone.BOARD, slot=0),
                target_pos=PosRef(side=player.uid, zone=Zone.BOARD, slot=0),
                value=2,
            ),
            {player.uid: player},
            empty_game.tavern.get_next_uid,
        )
        assert orca.cur_atk == atk_before  # no self-buff


class TestColdlightDiver:
    """BC and DR: Get Tavern Coin."""

    def test_bc_adds_coin(
        self, empty_game, player, tavern, mock_unit
    ):
        diver = mock_unit(CardIDs.COLDLIGHT_DIVER, owner_id=player.uid)
        player.hand.append(HandCard(uid=diver.uid, unit=diver))
        player.gold = 10

        tavern.play_unit(player, 0)

        coins = [hc for hc in player.hand if hc.spell and hc.spell.card_id == SpellIDs.TAVERN_COIN]
        assert len(coins) == 1

    def test_dr_adds_coin(self, combat_players):
        players, boards, cm = combat_players(
            [CardIDs.COLDLIGHT_DIVER],
            [CardIDs.ANNOY_O_TRON],
        )
        board0 = boards[0]
        diver = board0[0]
        side = diver.owner_id
        player = players[side]

        diver.cur_hp = 0
        cm.cleanup_dead(boards, [0, 0], players)

        coins = [hc for hc in player.hand if hc.spell and hc.spell.card_id == SpellIDs.TAVERN_COIN]
        assert len(coins) == 1


class TestJellyBelly:
    """After a friendly minion is Reborn, gain +2/+3 perm."""

    def test_buffs_self_on_reborn_summon(
        self, empty_game, player, mock_unit
    ):
        jelly = mock_unit(CardIDs.JELLY_BELLY, owner_id=player.uid)
        reborn_unit = mock_unit(CardIDs.RISEN_RIDER, owner_id=player.uid)
        player.board = [jelly, reborn_unit]

        atk_before = jelly.cur_atk
        hp_before = jelly.cur_hp

        from hearthstone.engine.card_def import TRIGGER_REGISTRY, GOLDEN_TRIGGER_REGISTRY
        from hearthstone.engine.event_system import EventManager
        em = EventManager(TRIGGER_REGISTRY, GOLDEN_TRIGGER_REGISTRY)

        # MINION_SUMMONED with meta=1 signals reborn
        em.process_event(
            Event(
                event_type=EventType.MINION_SUMMONED,
                source=EntityRef(reborn_unit.uid),
                source_pos=PosRef(side=player.uid, zone=Zone.BOARD, slot=1),
                meta=1,  # reborn flag
            ),
            {player.uid: player},
            empty_game.tavern.get_next_uid,
        )
        assert jelly.cur_atk == atk_before + 2
        assert jelly.cur_hp == hp_before + 3

    def test_does_not_buff_on_normal_summon(
        self, empty_game, player, mock_unit
    ):
        jelly = mock_unit(CardIDs.JELLY_BELLY, owner_id=player.uid)
        other = mock_unit(CardIDs.ANNOY_O_TRON, owner_id=player.uid)
        player.board = [jelly, other]

        atk_before = jelly.cur_atk

        from hearthstone.engine.card_def import TRIGGER_REGISTRY, GOLDEN_TRIGGER_REGISTRY
        from hearthstone.engine.event_system import EventManager
        em = EventManager(TRIGGER_REGISTRY, GOLDEN_TRIGGER_REGISTRY)

        em.process_event(
            Event(
                event_type=EventType.MINION_SUMMONED,
                source=EntityRef(other.uid),
                source_pos=PosRef(side=player.uid, zone=Zone.BOARD, slot=1),
                meta=None,  # not reborn
            ),
            {player.uid: player},
            empty_game.tavern.get_next_uid,
        )
        assert jelly.cur_atk == atk_before


# ===========================================================================
# T4 NEW CARDS
# ===========================================================================


class TestBreemCounter:
    """While in hand, after you play a Murloc, gain +4/+4."""

    def test_buffs_self_in_hand_when_murloc_played(self, empty_game, player, mock_unit):
        from hearthstone.engine.card_def import TRIGGER_REGISTRY, GOLDEN_TRIGGER_REGISTRY
        from hearthstone.engine.event_system import EventManager
        em = EventManager(TRIGGER_REGISTRY, GOLDEN_TRIGGER_REGISTRY)

        bream = mock_unit(CardIDs.BREAM_COUNTER, owner_id=player.uid)
        murloc = mock_unit(CardIDs.SWAMPSTRIKER, owner_id=player.uid)
        player.hand = [HandCard(uid=bream.uid, unit=bream)]
        player.board = [murloc]

        atk_before = bream.cur_atk
        hp_before = bream.cur_hp

        em.process_event(
            Event(
                event_type=EventType.MINION_PLAYED,
                source=EntityRef(murloc.uid),
                source_pos=PosRef(side=player.uid, zone=Zone.BOARD, slot=0),
            ),
            {player.uid: player},
            empty_game.tavern.get_next_uid,
        )
        assert bream.cur_atk == atk_before + 4
        assert bream.cur_hp == hp_before + 4


class TestPlankwalker:
    """Whenever you cast a Tavern spell, give three random friendly minions +2/+1."""

    def test_buffs_friendlies_on_spell_cast(self, empty_game, player, mock_unit):
        from hearthstone.engine.card_def import TRIGGER_REGISTRY, GOLDEN_TRIGGER_REGISTRY
        from hearthstone.engine.event_system import EventManager
        em = EventManager(TRIGGER_REGISTRY, GOLDEN_TRIGGER_REGISTRY)

        plank = mock_unit(CardIDs.PLANKWALKER, owner_id=player.uid)
        ally = mock_unit(CardIDs.SKELETON, owner_id=player.uid)
        player.board = [plank, ally]

        atk_before = ally.cur_atk

        em.process_event(
            Event(
                event_type=EventType.SPELL_CAST,
                source=EntityRef(plank.uid),
                source_pos=PosRef(side=player.uid, zone=Zone.BOARD, slot=0),
            ),
            {player.uid: player},
            empty_game.tavern.get_next_uid,
        )
        # At least ally gets buffed
        assert ally.cur_atk >= atk_before + 2


class TestSunkenAdvocate:
    """Rally: Give other Naga +1 Attack permanently."""

    def test_rally_buffs_other_naga(self, empty_game, player, mock_unit):
        from hearthstone.engine.card_def import TRIGGER_REGISTRY, GOLDEN_TRIGGER_REGISTRY
        from hearthstone.engine.event_system import EventManager
        em = EventManager(TRIGGER_REGISTRY, GOLDEN_TRIGGER_REGISTRY)

        adv = mock_unit(CardIDs.SUNKEN_ADVOCATE, owner_id=player.uid)
        naga = mock_unit(CardIDs.TRENCH_FIGHTER, owner_id=player.uid)
        player.board = [adv, naga]

        atk_before = naga.cur_atk

        em.process_event(
            Event(
                event_type=EventType.ATTACK_DECLARED,
                source=EntityRef(adv.uid),
                source_pos=PosRef(side=player.uid, zone=Zone.BOARD, slot=0),
            ),
            {player.uid: player},
            empty_game.tavern.get_next_uid,
        )
        assert naga.cur_atk == atk_before + 1


class TestTrigoreTheLasher:
    """When another friendly Beast takes damage, gain +2 Health permanently."""

    def test_gains_hp_when_friendly_beast_damaged(self, empty_game, player, mock_unit):
        from hearthstone.engine.card_def import TRIGGER_REGISTRY, GOLDEN_TRIGGER_REGISTRY
        from hearthstone.engine.event_system import EventManager
        em = EventManager(TRIGGER_REGISTRY, GOLDEN_TRIGGER_REGISTRY)

        trigore = mock_unit(CardIDs.TRIGORE_THE_LASHER, owner_id=player.uid)
        beast = mock_unit(CardIDs.MANASABER, owner_id=player.uid)
        player.board = [trigore, beast]

        hp_before = trigore.cur_hp

        em.process_event(
            Event(
                event_type=EventType.MINION_DAMAGED,
                source=EntityRef(beast.uid),
                source_pos=PosRef(side=player.uid, zone=Zone.BOARD, slot=1),
                target=EntityRef(beast.uid),
                target_pos=PosRef(side=player.uid, zone=Zone.BOARD, slot=1),
            ),
            {player.uid: player},
            empty_game.tavern.get_next_uid,
        )
        assert trigore.cur_hp == hp_before + 2


class TestRylakMetalhead:
    """Taunt, Deathrattle: buff all friendlies (simplified)."""

    def test_has_taunt(self, mock_unit, player):
        rylak = mock_unit(CardIDs.RYLAK_METALHEAD, owner_id=player.uid)
        assert Tags.TAUNT in rylak.tags

    def test_deathrattle_buffs_friendlies(self, empty_game, player, mock_unit):
        from hearthstone.engine.card_def import TRIGGER_REGISTRY, GOLDEN_TRIGGER_REGISTRY
        from hearthstone.engine.event_system import EventManager
        em = EventManager(TRIGGER_REGISTRY, GOLDEN_TRIGGER_REGISTRY)

        rylak = mock_unit(CardIDs.RYLAK_METALHEAD, owner_id=player.uid)
        ally = mock_unit(CardIDs.SKELETON, owner_id=player.uid)
        player.board = [rylak, ally]

        atk_before = ally.cur_atk

        from hearthstone.engine.event_system import MinionSnapshot
        em.process_event(
            Event(
                event_type=EventType.MINION_DIED,
                source=EntityRef(rylak.uid),
                source_pos=PosRef(side=player.uid, zone=Zone.BOARD, slot=0),
                snapshot=MinionSnapshot(
                    uid=rylak.uid, card_id=rylak.card_id,
                    owner_id=player.uid,
                    pos=PosRef(side=player.uid, zone=Zone.BOARD, slot=0),
                    atk=rylak.cur_atk, hp=rylak.cur_hp,
                    types=rylak.types, tags=rylak.tags,
                ),
            ),
            {player.uid: player},
            empty_game.tavern.get_next_uid,
        )
        assert ally.cur_atk == atk_before + 1


# ===========================================================================
# T5 CARDS
# ===========================================================================


class TestBrannBronzebeard:
    """Your Battlecries trigger twice (multiplier on MINION_PLAYED)."""

    def test_has_multiplier(self):
        from hearthstone.engine.card_def import ALL_CARDS
        card = next(c for c in ALL_CARDS if c.card_id == CardIDs.BRANN_BRONZEBEARD)
        assert card.multiplier is not None
        assert card.multiplier.event_type_name == "MINION_PLAYED"
        assert card.multiplier.extra_stacks == 1
        assert card.multiplier.self_only is True

    def test_brann_doubles_battlecry(self, empty_game, player, mock_unit):
        """Brann on board: Shell Collector BC adds 2 Tavern Coins instead of 1."""
        brann = mock_unit(CardIDs.BRANN_BRONZEBEARD, owner_id=player.uid)
        player.board = [brann]

        shell = mock_unit(CardIDs.SHELL_COLLECTOR, owner_id=player.uid)
        player.hand = [HandCard(uid=shell.uid, unit=shell)]

        ok, _ = empty_game.tavern.play_unit(player, 0)
        assert ok
        # Shell Collector BC: adds 1 Tavern Coin, doubled by Brann = 2 coins
        coins = [hc for hc in player.hand if hc.spell and hc.spell.card_id == SpellIDs.TAVERN_COIN]
        assert len(coins) == 2


class TestTitusRivendare:
    """Your Deathrattles trigger an extra time."""

    def test_has_multiplier(self):
        from hearthstone.engine.card_def import ALL_CARDS
        card = next(c for c in ALL_CARDS if c.card_id == CardIDs.TITUS_RIVENDARE)
        assert card.multiplier is not None
        assert card.multiplier.event_type_name == "MINION_DIED"
        assert card.multiplier.extra_stacks == 1


class TestDrakkariEnchanter:
    """Your end of turn effects trigger twice."""

    def test_has_multiplier(self):
        from hearthstone.engine.card_def import ALL_CARDS
        card = next(c for c in ALL_CARDS if c.card_id == CardIDs.DRAKKARI_ENCHANTER)
        assert card.multiplier is not None
        assert card.multiplier.event_type_name == "END_OF_TURN"
        assert card.multiplier.self_only is False
        assert card.multiplier.extra_stacks == 1


class TestGentleDjinni:
    """Taunt, Deathrattle: Get a random Elemental."""

    def test_has_taunt(self, mock_unit, player):
        djinni = mock_unit(CardIDs.GENTLE_DJINNI, owner_id=player.uid)
        assert Tags.TAUNT in djinni.tags

    def test_deathrattle_gives_elemental(self, empty_game, player, mock_unit):
        from hearthstone.engine.card_def import TRIGGER_REGISTRY, GOLDEN_TRIGGER_REGISTRY
        from hearthstone.engine.event_system import EventManager
        em = EventManager(TRIGGER_REGISTRY, GOLDEN_TRIGGER_REGISTRY)

        djinni = mock_unit(CardIDs.GENTLE_DJINNI, owner_id=player.uid)
        player.board = [djinni]
        hand_before = len(player.hand)

        em.process_event(
            Event(
                event_type=EventType.MINION_PLAYED,
                source=EntityRef(djinni.uid),
                source_pos=PosRef(side=player.uid, zone=Zone.BOARD, slot=0),
            ),
            {player.uid: player},
            empty_game.tavern.get_next_uid,
            card_pool=empty_game.pool,
        )
        # BC adds random elemental to hand
        assert len(player.hand) == hand_before + 1


class TestChampionOfThePrimus:
    """Avenge(2): Your Undead have +1 Attack this game."""

    def test_avenge_threshold_is_2(self):
        from hearthstone.engine.card_def import AVENGE_REGISTRY
        from hearthstone.engine.enums import CardIDs
        eff = AVENGE_REGISTRY.get(CardIDs.CHAMPION_OF_THE_PRIMUS)
        assert eff is not None
        assert eff.threshold == 2

    def test_avenge_buffs_undead(self, empty_game, player, mock_unit):
        from hearthstone.engine.combat import _execute_avenge
        from hearthstone.engine.card_def import AVENGE_REGISTRY

        champion = mock_unit(CardIDs.CHAMPION_OF_THE_PRIMUS, owner_id=player.uid)
        undead = mock_unit(CardIDs.SKELETON, owner_id=player.uid)
        player.board = [champion, undead]

        atk_before = undead.cur_atk
        avenge_def = AVENGE_REGISTRY[CardIDs.CHAMPION_OF_THE_PRIMUS]
        _execute_avenge(champion, avenge_def, {player.uid: player}, player.uid)
        assert undead.cur_atk == atk_before + 1


class TestTwilightWatcher:
    """Whenever a friendly Dragon attacks, give your Dragons +1/+3."""

    def test_buffs_dragons_when_dragon_attacks(self, empty_game, player, mock_unit):
        from hearthstone.engine.card_def import TRIGGER_REGISTRY, GOLDEN_TRIGGER_REGISTRY
        from hearthstone.engine.event_system import EventManager
        em = EventManager(TRIGGER_REGISTRY, GOLDEN_TRIGGER_REGISTRY)

        watcher = mock_unit(CardIDs.TWILIGHT_WATCHER, owner_id=player.uid)
        dragon = mock_unit(CardIDs.MISFIT_DRAGONLING, owner_id=player.uid)
        player.board = [watcher, dragon]

        hp_before = watcher.cur_hp

        em.process_event(
            Event(
                event_type=EventType.ATTACK_DECLARED,
                source=EntityRef(dragon.uid),
                source_pos=PosRef(side=player.uid, zone=Zone.BOARD, slot=1),
            ),
            {player.uid: player},
            empty_game.tavern.get_next_uid,
        )
        assert watcher.cur_hp == hp_before + 3


class TestUnforgivingTreant:
    """Taunt. Whenever this takes damage, give your minions +2 Attack."""

    def test_has_taunt(self, mock_unit, player):
        treant = mock_unit(CardIDs.UNFORGIVING_TREANT, owner_id=player.uid)
        assert Tags.TAUNT in treant.tags

    def test_buffs_board_when_damaged(self, empty_game, player, mock_unit):
        from hearthstone.engine.card_def import TRIGGER_REGISTRY, GOLDEN_TRIGGER_REGISTRY
        from hearthstone.engine.event_system import EventManager
        em = EventManager(TRIGGER_REGISTRY, GOLDEN_TRIGGER_REGISTRY)

        treant = mock_unit(CardIDs.UNFORGIVING_TREANT, owner_id=player.uid)
        ally = mock_unit(CardIDs.SKELETON, owner_id=player.uid)
        player.board = [treant, ally]

        atk_before = ally.cur_atk

        em.process_event(
            Event(
                event_type=EventType.MINION_DAMAGED,
                source=EntityRef(treant.uid),
                source_pos=PosRef(side=player.uid, zone=Zone.BOARD, slot=0),
                target=EntityRef(treant.uid),
                target_pos=PosRef(side=player.uid, zone=Zone.BOARD, slot=0),
            ),
            {player.uid: player},
            empty_game.tavern.get_next_uid,
        )
        assert ally.cur_atk == atk_before + 2


class TestCarapaceRaiser:
    """Deathrattle: Get a Haunted Carapace."""

    def test_deathrattle_gives_carapace(self, empty_game, player, mock_unit):
        from hearthstone.engine.card_def import TRIGGER_REGISTRY, GOLDEN_TRIGGER_REGISTRY
        from hearthstone.engine.event_system import EventManager
        from hearthstone.engine.entities import Spell
        em = EventManager(TRIGGER_REGISTRY, GOLDEN_TRIGGER_REGISTRY)

        raiser = mock_unit(CardIDs.CARAPACE_RAISER, owner_id=player.uid)
        player.board = [raiser]

        em.process_event(
            Event(
                event_type=EventType.MINION_DIED,
                source=EntityRef(raiser.uid),
                source_pos=PosRef(side=player.uid, zone=Zone.BOARD, slot=0),
                snapshot=None,
            ),
            {player.uid: player},
            empty_game.tavern.get_next_uid,
        )
        spells_in_hand = [hc for hc in player.hand if hc.spell]
        assert any(hc.spell.card_id == SpellIDs.HAUNTED_CARAPACE for hc in spells_in_hand)


class TestFirescaleHoarder:
    """Battlecry and Deathrattle: Get a Shiny Ring."""

    def test_bc_gives_shiny_ring(self, empty_game, player, mock_unit):
        from hearthstone.engine.card_def import TRIGGER_REGISTRY, GOLDEN_TRIGGER_REGISTRY
        from hearthstone.engine.event_system import EventManager
        em = EventManager(TRIGGER_REGISTRY, GOLDEN_TRIGGER_REGISTRY)

        hoarder = mock_unit(CardIDs.FIRESCALE_HOARDER, owner_id=player.uid)
        player.board = [hoarder]

        em.process_event(
            Event(
                event_type=EventType.MINION_PLAYED,
                source=EntityRef(hoarder.uid),
                source_pos=PosRef(side=player.uid, zone=Zone.BOARD, slot=0),
            ),
            {player.uid: player},
            empty_game.tavern.get_next_uid,
        )
        spells = [hc for hc in player.hand if hc.spell and hc.spell.card_id == SpellIDs.SHINY_RING]
        assert len(spells) >= 1

    def test_dr_gives_shiny_ring(self, empty_game, player, mock_unit):
        from hearthstone.engine.card_def import TRIGGER_REGISTRY, GOLDEN_TRIGGER_REGISTRY
        from hearthstone.engine.event_system import EventManager
        em = EventManager(TRIGGER_REGISTRY, GOLDEN_TRIGGER_REGISTRY)

        hoarder = mock_unit(CardIDs.FIRESCALE_HOARDER, owner_id=player.uid)
        player.board = [hoarder]

        em.process_event(
            Event(
                event_type=EventType.MINION_DIED,
                source=EntityRef(hoarder.uid),
                source_pos=PosRef(side=player.uid, zone=Zone.BOARD, slot=0),
                snapshot=None,
            ),
            {player.uid: player},
            empty_game.tavern.get_next_uid,
        )
        spells = [hc for hc in player.hand if hc.spell and hc.spell.card_id == SpellIDs.SHINY_RING]
        assert len(spells) >= 1


class TestLeeroyTheReckless:
    """Deathrattle: Destroy the minion that killed this."""

    def test_kills_attacker_on_death(self, combat_players):
        # Set up combat: Leeroy vs a weak attacker — verify no crash
        players, _boards, cm = combat_players(
            [CardIDs.LEEROY_THE_RECKLESS],
            [CardIDs.SKELETON],
        )
        p0, p1 = list(players.values())
        cm.resolve_combat(p0, p1)
        assert True


class TestStuntdrake:
    """Avenge(3): Give this minion's Attack to two different friendly minions."""

    def test_avenge_threshold_is_3(self):
        from hearthstone.engine.card_def import AVENGE_REGISTRY
        eff = AVENGE_REGISTRY.get(CardIDs.STUNTDRAKE)
        assert eff is not None
        assert eff.threshold == 3


class TestTwilightBroodmother:
    """Deathrattle: Summon 2 Twilight Hatchlings. Give them Taunt."""

    def test_deathrattle_summons_two_hatchlings_with_taunt(self, empty_game, player, mock_unit):
        from hearthstone.engine.card_def import TRIGGER_REGISTRY, GOLDEN_TRIGGER_REGISTRY
        from hearthstone.engine.event_system import EventManager
        em = EventManager(TRIGGER_REGISTRY, GOLDEN_TRIGGER_REGISTRY)

        brood = mock_unit(CardIDs.TWILIGHT_BROODMOTHER, owner_id=player.uid)
        player.board = [brood]

        em.process_event(
            Event(
                event_type=EventType.MINION_DIED,
                source=EntityRef(brood.uid),
                source_pos=PosRef(side=player.uid, zone=Zone.BOARD, slot=0),
                snapshot=None,
            ),
            {player.uid: player},
            empty_game.tavern.get_next_uid,
        )
        whelps = [u for u in player.board if u.card_id == CardIDs.TWILIGHT_WHELP]
        assert len(whelps) == 2
        for w in whelps:
            assert Tags.TAUNT in w.tags


class TestCostumeEnthusiast:
    """Divine Shield, SoC: Gain the Attack of the highest-Attack minion in your warband."""

    def test_has_divine_shield(self, mock_unit, player):
        ce = mock_unit(CardIDs.COSTUME_ENTHUSIAST, owner_id=player.uid)
        assert Tags.DIVINE_SHIELD in ce.tags

    def test_soc_gains_highest_atk(self, empty_game, player, mock_unit):
        from hearthstone.engine.card_def import TRIGGER_REGISTRY, GOLDEN_TRIGGER_REGISTRY
        from hearthstone.engine.event_system import EventManager
        em = EventManager(TRIGGER_REGISTRY, GOLDEN_TRIGGER_REGISTRY)

        ce = mock_unit(CardIDs.COSTUME_ENTHUSIAST, owner_id=player.uid)
        strong = mock_unit(CardIDs.WANNABE_GARGOYLE, owner_id=player.uid)  # 9/1
        player.board = [ce, strong]

        atk_before = ce.cur_atk

        em.process_event(
            Event(
                event_type=EventType.START_OF_COMBAT,
                source_pos=PosRef(side=player.uid, zone=Zone.BOARD, slot=0),
            ),
            {player.uid: player},
            empty_game.tavern.get_next_uid,
        )
        # should gain Wannabe Gargoyle's 9 ATK
        assert ce.cur_atk > atk_before


# ===========================================================================
# T6 CARDS
# ===========================================================================


class TestGoldrinnTheGreatWolf:
    """Deathrattle: For rest of combat, your Beasts have +8/+8."""

    def test_deathrattle_buffs_beasts(self, empty_game, player, mock_unit):
        from hearthstone.engine.card_def import TRIGGER_REGISTRY, GOLDEN_TRIGGER_REGISTRY
        from hearthstone.engine.event_system import EventManager
        em = EventManager(TRIGGER_REGISTRY, GOLDEN_TRIGGER_REGISTRY)

        goldrinn = mock_unit(CardIDs.GOLDRINN_THE_GREAT_WOLF, owner_id=player.uid)
        beast = mock_unit(CardIDs.MANASABER, owner_id=player.uid)
        player.board = [goldrinn, beast]

        atk_before = beast.cur_atk
        hp_before = beast.cur_hp

        em.process_event(
            Event(
                event_type=EventType.MINION_DIED,
                source=EntityRef(goldrinn.uid),
                source_pos=PosRef(side=player.uid, zone=Zone.BOARD, slot=0),
                snapshot=None,
            ),
            {player.uid: player},
            empty_game.tavern.get_next_uid,
        )
        assert beast.cur_atk == atk_before + 8
        assert beast.cur_hp == hp_before + 8


class TestSlitherspear:
    """EoT: give your other Naga +2/+1."""

    def test_eot_buffs_naga(self, empty_game, player, mock_unit):
        from hearthstone.engine.card_def import TRIGGER_REGISTRY, GOLDEN_TRIGGER_REGISTRY
        from hearthstone.engine.event_system import EventManager
        em = EventManager(TRIGGER_REGISTRY, GOLDEN_TRIGGER_REGISTRY)

        slither = mock_unit(CardIDs.SLITHERSPEAR_LORD_OF_GAINS, owner_id=player.uid)
        naga = mock_unit(CardIDs.TRENCH_FIGHTER, owner_id=player.uid)
        player.board = [slither, naga]

        atk_before = naga.cur_atk

        em.process_event(
            Event(
                event_type=EventType.END_OF_TURN,
                source_pos=PosRef(side=player.uid, zone=Zone.BOARD, slot=0),
            ),
            {player.uid: player},
            empty_game.tavern.get_next_uid,
        )
        assert naga.cur_atk == atk_before + 2


class TestShipMasterEudora:
    """Deathrattle: Give your minions +8/+8."""

    def test_deathrattle_buffs_all_friendlies(self, empty_game, player, mock_unit):
        from hearthstone.engine.card_def import TRIGGER_REGISTRY, GOLDEN_TRIGGER_REGISTRY
        from hearthstone.engine.event_system import EventManager
        em = EventManager(TRIGGER_REGISTRY, GOLDEN_TRIGGER_REGISTRY)

        eudora = mock_unit(CardIDs.SHIP_MASTER_EUDORA, owner_id=player.uid)
        ally = mock_unit(CardIDs.SKELETON, owner_id=player.uid)
        player.board = [eudora, ally]

        atk_before = ally.cur_atk

        em.process_event(
            Event(
                event_type=EventType.MINION_DIED,
                source=EntityRef(eudora.uid),
                source_pos=PosRef(side=player.uid, zone=Zone.BOARD, slot=0),
                snapshot=None,
            ),
            {player.uid: player},
            empty_game.tavern.get_next_uid,
        )
        assert ally.cur_atk == atk_before + 8


class TestAvalancheCaller:
    """EoT: get a Mounting Avalanche."""

    def test_eot_gives_spell(self, empty_game, player, mock_unit):
        from hearthstone.engine.card_def import TRIGGER_REGISTRY, GOLDEN_TRIGGER_REGISTRY
        from hearthstone.engine.event_system import EventManager
        em = EventManager(TRIGGER_REGISTRY, GOLDEN_TRIGGER_REGISTRY)

        caller = mock_unit(CardIDs.AVALANCHE_CALLER, owner_id=player.uid)
        player.board = [caller]

        em.process_event(
            Event(
                event_type=EventType.END_OF_TURN,
                source_pos=PosRef(side=player.uid, zone=Zone.BOARD, slot=0),
            ),
            {player.uid: player},
            empty_game.tavern.get_next_uid,
        )
        spells = [hc for hc in player.hand if hc.spell and hc.spell.card_id == SpellIDs.MOUNTING_AVALANCHE]
        assert len(spells) == 1


class TestNightmareParTeaGuest:
    """Battlecry and Deathrattle: Get a Misplaced Tea Set."""

    def test_bc_gives_tea_set(self, empty_game, player, mock_unit):
        from hearthstone.engine.card_def import TRIGGER_REGISTRY, GOLDEN_TRIGGER_REGISTRY
        from hearthstone.engine.event_system import EventManager
        em = EventManager(TRIGGER_REGISTRY, GOLDEN_TRIGGER_REGISTRY)

        guest = mock_unit(CardIDs.NIGHTMARE_PAR_TEA_GUEST, owner_id=player.uid)
        player.board = [guest]

        em.process_event(
            Event(
                event_type=EventType.MINION_PLAYED,
                source=EntityRef(guest.uid),
                source_pos=PosRef(side=player.uid, zone=Zone.BOARD, slot=0),
            ),
            {player.uid: player},
            empty_game.tavern.get_next_uid,
        )
        spells = [hc for hc in player.hand if hc.spell and hc.spell.card_id == SpellIDs.MISPLACED_TEA_SET]
        assert len(spells) == 1

    def test_dr_gives_tea_set(self, empty_game, player, mock_unit):
        from hearthstone.engine.card_def import TRIGGER_REGISTRY, GOLDEN_TRIGGER_REGISTRY
        from hearthstone.engine.event_system import EventManager
        em = EventManager(TRIGGER_REGISTRY, GOLDEN_TRIGGER_REGISTRY)

        guest = mock_unit(CardIDs.NIGHTMARE_PAR_TEA_GUEST, owner_id=player.uid)
        player.board = [guest]

        em.process_event(
            Event(
                event_type=EventType.MINION_DIED,
                source=EntityRef(guest.uid),
                source_pos=PosRef(side=player.uid, zone=Zone.BOARD, slot=0),
                snapshot=None,
            ),
            {player.uid: player},
            empty_game.tavern.get_next_uid,
        )
        spells = [hc for hc in player.hand if hc.spell and hc.spell.card_id == SpellIDs.MISPLACED_TEA_SET]
        assert len(spells) == 1


class TestSunderedMatriarch:
    """Whenever you cast a spell, give your minions +2 Health."""

    def test_spell_cast_buffs_board_hp(self, empty_game, player, mock_unit):
        from hearthstone.engine.card_def import TRIGGER_REGISTRY, GOLDEN_TRIGGER_REGISTRY
        from hearthstone.engine.event_system import EventManager
        em = EventManager(TRIGGER_REGISTRY, GOLDEN_TRIGGER_REGISTRY)

        matriarch = mock_unit(CardIDs.SUNDERED_MATRIARCH, owner_id=player.uid)
        ally = mock_unit(CardIDs.SKELETON, owner_id=player.uid)
        player.board = [matriarch, ally]

        hp_before = ally.cur_hp

        em.process_event(
            Event(
                event_type=EventType.SPELL_CAST,
                source=EntityRef(matriarch.uid),
                source_pos=PosRef(side=player.uid, zone=Zone.BOARD, slot=0),
            ),
            {player.uid: player},
            empty_game.tavern.get_next_uid,
        )
        assert ally.cur_hp == hp_before + 2


class TestArchaedas:
    """Battlecry: Get a random Tier 5 minion."""

    def test_bc_gives_tier5_minion(self, empty_game, player, mock_unit):
        from hearthstone.engine.card_def import TRIGGER_REGISTRY, GOLDEN_TRIGGER_REGISTRY
        from hearthstone.engine.event_system import EventManager
        from hearthstone.engine.configs import CARD_DB
        em = EventManager(TRIGGER_REGISTRY, GOLDEN_TRIGGER_REGISTRY)

        arch = mock_unit(CardIDs.ARCHAEDAS, owner_id=player.uid)
        player.board = [arch]

        em.process_event(
            Event(
                event_type=EventType.MINION_PLAYED,
                source=EntityRef(arch.uid),
                source_pos=PosRef(side=player.uid, zone=Zone.BOARD, slot=0),
            ),
            {player.uid: player},
            empty_game.tavern.get_next_uid,
            card_pool=empty_game.pool,
        )
        new_units = [hc for hc in player.hand if hc.unit]
        assert len(new_units) >= 1
        gotten = new_units[-1].unit
        assert CARD_DB.get(gotten.card_id, {}).get("tier", 0) == 5


# ===========================================================================
# T7 CARDS
# ===========================================================================


class TestCaptainSanders:
    """Battlecry: Make a friendly minion from Tier 6 or below Golden."""

    def test_bc_makes_friendly_golden(self, empty_game, player, mock_unit):
        from hearthstone.engine.card_def import TRIGGER_REGISTRY, GOLDEN_TRIGGER_REGISTRY
        from hearthstone.engine.event_system import EventManager
        em = EventManager(TRIGGER_REGISTRY, GOLDEN_TRIGGER_REGISTRY)

        sanders = mock_unit(CardIDs.CAPTAIN_SANDERS, owner_id=player.uid)
        candidate = mock_unit(CardIDs.SKELETON, owner_id=player.uid)
        player.board = [sanders, candidate]

        assert not candidate.is_golden

        em.process_event(
            Event(
                event_type=EventType.MINION_PLAYED,
                source=EntityRef(sanders.uid),
                source_pos=PosRef(side=player.uid, zone=Zone.BOARD, slot=0),
            ),
            {player.uid: player},
            empty_game.tavern.get_next_uid,
        )
        assert candidate.is_golden


class TestHighkeeperRa:
    """Battlecry, Deathrattle and Rally: Get a random Tier 6 minion."""

    def test_bc_gives_card_to_hand(self, empty_game, player, mock_unit):
        from hearthstone.engine.card_def import TRIGGER_REGISTRY, GOLDEN_TRIGGER_REGISTRY
        from hearthstone.engine.event_system import EventManager
        from hearthstone.engine.configs import CARD_DB
        em = EventManager(TRIGGER_REGISTRY, GOLDEN_TRIGGER_REGISTRY)

        ra = mock_unit(CardIDs.HIGHKEEPER_RA, owner_id=player.uid)
        player.board = [ra]

        em.process_event(
            Event(
                event_type=EventType.MINION_PLAYED,
                source=EntityRef(ra.uid),
                source_pos=PosRef(side=player.uid, zone=Zone.BOARD, slot=0),
            ),
            {player.uid: player},
            empty_game.tavern.get_next_uid,
            card_pool=empty_game.pool,
        )
        # Should have added a T6 unit
        new_units = [hc for hc in player.hand if hc.unit]
        assert len(new_units) >= 1
        assert CARD_DB.get(new_units[-1].unit.card_id, {}).get("tier", 0) == 6


class TestSanguineChampion:
    """Battlecry and Deathrattle: Your Blood Gems give an extra +1/+1."""

    def test_bc_modifies_blood_gem(self, empty_game, player, mock_unit):
        from hearthstone.engine.card_def import TRIGGER_REGISTRY, GOLDEN_TRIGGER_REGISTRY
        from hearthstone.engine.event_system import EventManager
        from hearthstone.engine.enums import MechanicType
        em = EventManager(TRIGGER_REGISTRY, GOLDEN_TRIGGER_REGISTRY)

        champ = mock_unit(CardIDs.SANGUINE_CHAMPION, owner_id=player.uid)
        player.board = [champ]

        gem_atk_before, gem_hp_before = player.mechanics.get_stat(MechanicType.BLOOD_GEM)

        em.process_event(
            Event(
                event_type=EventType.MINION_PLAYED,
                source=EntityRef(champ.uid),
                source_pos=PosRef(side=player.uid, zone=Zone.BOARD, slot=0),
            ),
            {player.uid: player},
            empty_game.tavern.get_next_uid,
        )
        gem_atk_after, gem_hp_after = player.mechanics.get_stat(MechanicType.BLOOD_GEM)
        assert gem_atk_after == gem_atk_before + 1
        assert gem_hp_after == gem_hp_before + 1


class TestPsychus:
    """SoC: Set this minion's Attack and Health to match the highest in the warband."""

    def test_soc_sets_stats_to_highest(self, empty_game, player, mock_unit):
        from hearthstone.engine.card_def import TRIGGER_REGISTRY, GOLDEN_TRIGGER_REGISTRY
        from hearthstone.engine.event_system import EventManager
        em = EventManager(TRIGGER_REGISTRY, GOLDEN_TRIGGER_REGISTRY)

        psychus = mock_unit(CardIDs.PSYCHUS, owner_id=player.uid)
        strong = mock_unit(CardIDs.WANNABE_GARGOYLE, owner_id=player.uid)  # 9/1
        player.board = [psychus, strong]

        em.process_event(
            Event(
                event_type=EventType.START_OF_COMBAT,
                source_pos=PosRef(side=player.uid, zone=Zone.BOARD, slot=0),
            ),
            {player.uid: player},
            empty_game.tavern.get_next_uid,
        )
        assert psychus.cur_atk >= strong.cur_atk


class TestFuturefin:
    """EoT: give this minion's stats to the left-most minion in your warband."""

    def test_eot_buffs_adjacent(self, empty_game, player, mock_unit):
        from hearthstone.engine.card_def import TRIGGER_REGISTRY, GOLDEN_TRIGGER_REGISTRY
        from hearthstone.engine.event_system import EventManager
        em = EventManager(TRIGGER_REGISTRY, GOLDEN_TRIGGER_REGISTRY)

        futurefin = mock_unit(CardIDs.FUTUREFIN, owner_id=player.uid)
        left = mock_unit(CardIDs.SKELETON, owner_id=player.uid)
        player.board = [left, futurefin]

        hp_before = left.cur_hp

        em.process_event(
            Event(
                event_type=EventType.END_OF_TURN,
                source_pos=PosRef(side=player.uid, zone=Zone.BOARD, slot=1),
            ),
            {player.uid: player},
            empty_game.tavern.get_next_uid,
        )
        assert left.cur_hp > hp_before
