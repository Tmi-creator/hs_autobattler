"""Tests for card_def.py declarative system and new EffectContext methods.

Covers:
- build_card_db() parity with original CARD_DB
- build_trigger_registry() parity with original TRIGGER_REGISTRY
- BattlecryMakeGolden (Aureate Laureate)
- consume_random_store_unit returns to pool
- draw_from_pool (River Skipper)
- make_golden ctx method
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Dict, List, Tuple

from hearthstone.engine.card_def import ALL_CARDS, AVENGE_REGISTRY, build_card_db, build_trigger_registry
from hearthstone.engine.configs import CARD_DB
from hearthstone.engine.card_def import TRIGGER_REGISTRY
from hearthstone.engine.combat import CombatManager
from hearthstone.engine.entities import Player, StoreItem, Unit
from hearthstone.engine.enums import CardIDs, UnitType

if TYPE_CHECKING:
    from hearthstone.engine.game import Game
    from hearthstone.engine.tavern import TavernManager


# -----------------------------------------------------------------------
# Registry parity tests
# -----------------------------------------------------------------------


class TestBuildCardDB:
    def test_same_keys(self) -> None:
        new_db = build_card_db()
        orig_keys = set(CARD_DB.keys())
        new_keys = set(new_db.keys())
        assert orig_keys == new_keys

    def test_same_stats(self) -> None:
        new_db = build_card_db()
        for card_id, orig in CARD_DB.items():
            new = new_db[card_id]
            for field in ("name", "tier", "atk", "hp"):
                assert orig[field] == new[field], f"{card_id}.{field} mismatch"
            assert orig.get("type") == new.get("type"), f"{card_id}.type mismatch"

    def test_card_count_matches_all_cards(self) -> None:
        assert len(build_card_db()) == len(ALL_CARDS)


class TestBuildTriggerRegistry:
    def test_new_registry_is_superset_of_original(self) -> None:
        """New registry may have extra keys (e.g. Aureate Laureate)
        but must contain all original keys."""
        new_reg = build_trigger_registry()
        orig_keys = set(TRIGGER_REGISTRY.keys())
        new_keys = set(new_reg.keys())
        assert orig_keys.issubset(new_keys)

    def test_same_event_types(self) -> None:
        new_reg = build_trigger_registry()
        for card_id, orig_triggers in TRIGGER_REGISTRY.items():
            new_triggers = new_reg[card_id]
            assert len(orig_triggers) == len(new_triggers), f"{card_id} trigger count"
            for i, (ot, nt) in enumerate(zip(orig_triggers, new_triggers)):
                assert ot.event_type == nt.event_type, (
                    f"{card_id}[{i}] event type: {ot.event_type} vs {nt.event_type}"
                )


# -----------------------------------------------------------------------
# EffectContext.make_golden
# -----------------------------------------------------------------------


class TestMakeGolden:
    def test_doubles_base_stats(
        self, mock_unit: Callable[..., Unit],
    ) -> None:
        unit = mock_unit(CardIDs.ANNOY_O_TRON)
        orig_atk = unit.base_atk
        orig_hp = unit.base_hp
        assert not unit.is_golden

        unit.is_golden = True
        unit.base_atk *= 2
        unit.base_hp *= 2
        unit.recalc_stats()

        assert unit.is_golden
        assert unit.base_atk == orig_atk * 2
        assert unit.base_hp == orig_hp * 2
        assert unit.cur_atk == orig_atk * 2
        assert unit.cur_hp == orig_hp * 2

    def test_make_golden_via_ctx(
        self,
        empty_game: Game,
        player: Player,
        mock_unit: Callable[..., Unit],
    ) -> None:
        unit = mock_unit(CardIDs.ANNOY_O_TRON, owner_id=player.uid)
        orig_atk = unit.base_atk
        orig_hp = unit.base_hp
        player.board.append(unit)

        from hearthstone.engine.event_system import EffectContext, EntityRef
        from collections import deque
        ctx = EffectContext(
            {player.uid: player},
            empty_game.tavern.get_next_uid,
            deque(),
        )
        result = ctx.make_golden(EntityRef(unit.uid))

        assert result is True
        assert unit.is_golden
        assert unit.base_atk == orig_atk * 2
        assert unit.base_hp == orig_hp * 2

    def test_make_golden_already_golden_is_noop(
        self,
        empty_game: Game,
        player: Player,
        mock_unit: Callable[..., Unit],
    ) -> None:
        unit = mock_unit(CardIDs.ANNOY_O_TRON, owner_id=player.uid, is_golden=True)
        player.board.append(unit)

        from hearthstone.engine.event_system import EffectContext, EntityRef
        from collections import deque
        ctx = EffectContext(
            {player.uid: player},
            empty_game.tavern.get_next_uid,
            deque(),
        )
        result = ctx.make_golden(EntityRef(unit.uid))
        assert result is False


# -----------------------------------------------------------------------
# Aureate Laureate Battlecry
# -----------------------------------------------------------------------


class TestAureateLaureate:
    def test_becomes_golden_on_play(
        self,
        empty_game: Game,
        player: Player,
        tavern: TavernManager,
        mock_unit: Callable[..., Unit],
    ) -> None:
        laureate = mock_unit(CardIDs.AUREATE_LAUREATE, owner_id=player.uid)
        assert not laureate.is_golden
        orig_atk = laureate.base_atk
        orig_hp = laureate.base_hp

        from hearthstone.engine.entities import HandCard
        player.hand.append(HandCard(uid=laureate.uid, unit=laureate))
        player.gold = 10

        tavern.play_unit(player, 0)

        board_laureates = [u for u in player.board if u.card_id == CardIDs.AUREATE_LAUREATE]
        assert len(board_laureates) == 1
        played = board_laureates[0]
        assert played.is_golden
        assert played.base_atk == orig_atk * 2
        assert played.base_hp == orig_hp * 2


# -----------------------------------------------------------------------
# consume_random_store_unit returns to pool
# -----------------------------------------------------------------------


class TestConsumeReturnsToPool:
    def test_consumed_unit_goes_back_to_pool(
        self,
        empty_game: Game,
        player: Player,
        mock_unit: Callable[..., Unit],
    ) -> None:
        pool = empty_game.pool
        pool_before = sum(len(t) for t in pool.tiers.values())

        # Put a unit in store manually
        unit = mock_unit(CardIDs.ANNOY_O_TRON, owner_id=player.uid)
        player.store.insert(0, StoreItem(unit=unit))

        from hearthstone.engine.event_system import EffectContext, EntityRef
        from collections import deque
        ctx = EffectContext(
            {player.uid: player},
            empty_game.tavern.get_next_uid,
            deque(),
            card_pool=pool,
        )

        store_units_before = len([s for s in player.store if s.unit])

        result = ctx.consume_random_store_unit(player.uid)
        assert result is not None

        store_units_after = len([s for s in player.store if s.unit])
        assert store_units_after == store_units_before - 1

        pool_after = sum(len(t) for t in pool.tiers.values())
        assert pool_after == pool_before + 1  # returned to pool


# -----------------------------------------------------------------------
# draw_from_pool
# -----------------------------------------------------------------------


class TestDrawFromPool:
    def test_draws_unit_to_hand(
        self,
        empty_game: Game,
        player: Player,
    ) -> None:
        pool = empty_game.pool
        player.hand.clear()

        from hearthstone.engine.event_system import EffectContext
        from collections import deque
        ctx = EffectContext(
            {player.uid: player},
            empty_game.tavern.get_next_uid,
            deque(),
            card_pool=pool,
        )

        hand_before = len(player.hand)
        pool_before = sum(len(t) for t in pool.tiers.values())

        drawn = ctx.draw_from_pool(player.uid, tier=1, count=1)

        assert len(drawn) == 1
        assert len(player.hand) == hand_before + 1
        pool_after = sum(len(t) for t in pool.tiers.values())
        assert pool_after == pool_before - 1  # one less in pool

    def test_returns_to_pool_when_hand_full(
        self,
        empty_game: Game,
        player: Player,
        mock_unit: Callable[..., Unit],
    ) -> None:
        pool = empty_game.pool

        # Fill hand to max
        from hearthstone.engine.entities import HandCard
        player.hand.clear()
        for _ in range(10):
            u = mock_unit(CardIDs.ANNOY_O_TRON, owner_id=player.uid)
            player.hand.append(HandCard(uid=u.uid, unit=u))

        from hearthstone.engine.event_system import EffectContext
        from collections import deque
        ctx = EffectContext(
            {player.uid: player},
            empty_game.tavern.get_next_uid,
            deque(),
            card_pool=pool,
        )

        pool_before = sum(len(t) for t in pool.tiers.values())
        drawn = ctx.draw_from_pool(player.uid, tier=1, count=1)

        assert len(drawn) == 0
        assert len(player.hand) == 10  # unchanged
        pool_after = sum(len(t) for t in pool.tiers.values())
        assert pool_after == pool_before  # returned back

    def test_no_pool_returns_empty(
        self,
        empty_game: Game,
        player: Player,
    ) -> None:
        from hearthstone.engine.event_system import EffectContext
        from collections import deque
        ctx = EffectContext(
            {player.uid: player},
            empty_game.tavern.get_next_uid,
            deque(),
            card_pool=None,
        )
        drawn = ctx.draw_from_pool(player.uid, tier=1, count=1)
        assert drawn == []


# -----------------------------------------------------------------------
# Avenge system tests
# -----------------------------------------------------------------------


class TestAvengeRegistry:
    def test_bird_buddy_in_registry(self) -> None:
        assert CardIDs.BIRD_BUDDY in AVENGE_REGISTRY
        eff = AVENGE_REGISTRY[CardIDs.BIRD_BUDDY]
        assert eff.threshold == 1
        assert eff.buff_atk == 1
        assert eff.buff_hp == 1
        assert eff.buff_target == "friendly_type"
        assert eff.target_type == UnitType.BEAST

    def test_budding_greenthumb_in_registry(self) -> None:
        assert CardIDs.BUDDING_GREENTHUMB in AVENGE_REGISTRY
        eff = AVENGE_REGISTRY[CardIDs.BUDDING_GREENTHUMB]
        assert eff.threshold == 3
        assert eff.buff_atk == 2
        assert eff.buff_hp == 2
        assert eff.buff_scope == "perm"
        assert eff.buff_target == "adjacent"

    def test_non_avenge_card_not_in_registry(self) -> None:
        assert CardIDs.ANNOY_O_TRON not in AVENGE_REGISTRY


class TestAvengeSystem:
    def test_avenge_counter_initialized_on_combat_copy(
        self,
        mock_unit: Callable[..., Unit],
        player: Player,
    ) -> None:
        bird = mock_unit(CardIDs.BIRD_BUDDY, owner_id=player.uid)
        player.board.append(bird)
        cp = player.combat_copy()
        bird_copy = cp.board[0]
        assert bird_copy.avenge_counter == 1  # threshold = 1

    def test_avenge_threshold_3_initialized(
        self,
        mock_unit: Callable[..., Unit],
        player: Player,
    ) -> None:
        gt = mock_unit(CardIDs.BUDDING_GREENTHUMB, owner_id=player.uid)
        player.board.append(gt)
        cp = player.combat_copy()
        assert cp.board[0].avenge_counter == 3

    def test_bird_buddy_avenge_fires_on_first_death(
        self,
        combat_players: Callable[..., Tuple[Dict[int, Player], List[List[Unit]], CombatManager]],
    ) -> None:
        """Bird Buddy (avenge 1): any friendly death triggers buff to all Beasts."""
        players, boards, cm = combat_players(
            [CardIDs.BIRD_BUDDY, CardIDs.MANASABER],   # side 0: avenger + a Beast
            [CardIDs.ANNOY_O_TRON],                     # side 1: enemy
        )
        board0 = boards[0]
        bird = board0[0]
        manasaber = board0[1]

        # Kill the manasaber directly
        manasaber.cur_hp = 0
        cm.cleanup_dead(boards, [0, 0], players)

        # Manasaber is dead and removed; Bird Buddy's avenge (1) should have fired
        assert bird in board0  # bird survived
        # Bird Buddy is also a Beast, so it should get +1/+1 too
        assert bird.cur_atk == 3 + 1   # base 3 + avenge buff
        assert bird.cur_hp >= 1        # still alive

    def test_bird_buddy_avenge_resets_and_fires_again(
        self,
        combat_players: Callable[..., Tuple[Dict[int, Player], List[List[Unit]], CombatManager]],
    ) -> None:
        """After avenge fires, counter resets and fires again on next death."""
        players, boards, cm = combat_players(
            [CardIDs.BIRD_BUDDY, CardIDs.MANASABER, CardIDs.ANNOY_O_TRON],
            [CardIDs.ANNOY_O_TRON],
        )
        board0 = boards[0]
        bird = board0[0]
        manasaber = board0[1]
        annoy = board0[2]

        # Kill manasaber (first death → avenge fires, counter resets to 1)
        manasaber.cur_hp = 0
        cm.cleanup_dead(boards, [0, 0], players)
        assert bird.avenge_counter == 1  # reset after firing

        atk_after_first = bird.cur_atk

        # Kill annoy-o-tron (second death → avenge fires again)
        annoy.cur_hp = 0
        cm.cleanup_dead(boards, [0, 0], players)

        assert bird.cur_atk == atk_after_first + 1  # buffed again

    def test_budding_greenthumb_avenge_buffs_adjacent_perm(
        self,
        combat_players: Callable[..., Tuple[Dict[int, Player], List[List[Unit]], CombatManager]],
    ) -> None:
        """Budding Greenthumb (avenge 3, perm): fires after 3 deaths, buffs adjacent."""
        players, boards, cm = combat_players(
            [CardIDs.ANNOY_O_TRON, CardIDs.BUDDING_GREENTHUMB, CardIDs.ANNOY_O_TRON,
             CardIDs.MANASABER, CardIDs.MANASABER, CardIDs.MANASABER],
            [CardIDs.ANNOY_O_TRON],
        )
        board0 = boards[0]
        left_neighbor = board0[0]   # Annoy-o-Tron at index 0
        greenthumb = board0[1]      # Budding Greenthumb at index 1
        right_neighbor = board0[2]  # Annoy-o-Tron at index 2
        # Sacrificial units at indices 3, 4, 5
        sac1, sac2, sac3 = board0[3], board0[4], board0[5]

        left_atk_before = left_neighbor.perm_atk_add
        right_atk_before = right_neighbor.perm_atk_add

        # Kill 3 units — avenge should fire after the 3rd
        sac1.cur_hp = 0
        cm.cleanup_dead(boards, [0, 0], players)
        assert greenthumb.avenge_counter == 2

        sac2.cur_hp = 0
        cm.cleanup_dead(boards, [0, 0], players)
        assert greenthumb.avenge_counter == 1

        sac3.cur_hp = 0
        cm.cleanup_dead(boards, [0, 0], players)
        assert greenthumb.avenge_counter == 3  # reset to threshold

        # Adjacent units get +2/+2 perm
        assert left_neighbor.perm_atk_add == left_atk_before + 2
        assert right_neighbor.perm_atk_add == right_atk_before + 2

    def test_avenge_only_counts_friendly_deaths(
        self,
        combat_players: Callable[..., Tuple[Dict[int, Player], List[List[Unit]], CombatManager]],
    ) -> None:
        """Enemy deaths must NOT decrement the avenge counter."""
        players, boards, cm = combat_players(
            [CardIDs.BIRD_BUDDY],
            [CardIDs.ANNOY_O_TRON, CardIDs.MANASABER],
        )
        board0 = boards[0]
        board1 = boards[1]
        bird = board0[0]
        enemy = board1[1]

        counter_before = bird.avenge_counter

        # Kill an enemy unit
        enemy.cur_hp = 0
        cm.cleanup_dead(boards, [0, 0], players)

        # Bird Buddy's counter must not have changed
        assert bird.avenge_counter == counter_before
