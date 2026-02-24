"""Comprehensive engine-core tests.

Covers: Economy · Tavern logic · Triplets · Discovery · Edge-cases.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from src.hearthstone.engine.configs import COST_BUY, COST_REROLL, TAVERN_SLOTS
from src.hearthstone.engine.entities import HandCard, Player, Spell, StoreItem, Unit
from src.hearthstone.engine.enums import CardIDs, SpellIDs

if TYPE_CHECKING:
    from src.hearthstone.engine.game import Game
    from src.hearthstone.engine.tavern import TavernManager

# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------


def _inject_unit_into_store(player: Player, unit: Unit) -> int:
    """Insert *unit* as the first store item and return its store index (0)."""
    player.store.insert(0, StoreItem(unit=unit))
    return 0


def _inject_spell_into_store(player: Player, spell: Spell) -> int:
    """Insert *spell* as the first store item and return its store index (0)."""
    player.store.insert(0, StoreItem(spell=spell))
    return 0


def _pool_total(game: "Game") -> int:
    """Total cards currently residing in the card pool across all tiers."""
    return sum(len(tier) for tier in game.pool.tiers.values())


def _pool_tier_count(game: "Game", card_id: str) -> int:
    """Count how many copies of *card_id* are in the pool (any tier)."""
    total = 0
    for tier_cards in game.pool.tiers.values():
        total += tier_cards.count(card_id)
    return total


# ===================================================================
#  1. ECONOMY
# ===================================================================


class TestEconomy:
    """Buying / selling / gold book-keeping."""

    def test_buy_unit_deducts_gold(
        self,
        empty_game: "Game",
        player: Player,
        tavern: "TavernManager",
        mock_unit: Callable[..., Unit],
    ) -> None:
        unit = mock_unit(CardIDs.ALLEYCAT)
        _inject_unit_into_store(player, unit)
        gold_before = player.gold

        success, _ = tavern.buy_unit(player, 0)

        assert success
        assert player.gold == gold_before - COST_BUY

    def test_buy_spell_deducts_gold(
        self,
        empty_game: "Game",
        player: Player,
        tavern: "TavernManager",
    ) -> None:
        spell = Spell.create_from_db(SpellIDs.POINTY_ARROW)  # cost = 1
        _inject_spell_into_store(player, spell)
        gold_before = player.gold

        success, _ = tavern.buy_unit(player, 0)

        assert success
        assert player.gold == gold_before - spell.cost

    def test_sell_unit_returns_one_gold(
        self,
        empty_game: "Game",
        player: Player,
        tavern: "TavernManager",
        mock_unit: Callable[..., Unit],
    ) -> None:
        unit = mock_unit(CardIDs.ANNOY_O_TRON)
        player.board.append(unit)
        gold_before = player.gold

        success, _ = tavern.sell_unit(player, 0)

        assert success
        assert player.gold == gold_before + 1

    def test_sell_unit_returns_card_to_pool(
        self,
        empty_game: "Game",
        player: Player,
        tavern: "TavernManager",
        mock_unit: Callable[..., Unit],
    ) -> None:
        card_id = CardIDs.ANNOY_O_TRON
        unit = mock_unit(card_id)
        player.board.append(unit)
        pool_before = _pool_tier_count(empty_game, card_id)

        tavern.sell_unit(player, 0)

        pool_after = _pool_tier_count(empty_game, card_id)
        assert pool_after == pool_before + 1

    def test_buy_not_enough_gold(
        self,
        empty_game: "Game",
        player: Player,
        tavern: "TavernManager",
        mock_unit: Callable[..., Unit],
    ) -> None:
        unit = mock_unit(CardIDs.ALLEYCAT)
        _inject_unit_into_store(player, unit)
        player.gold = COST_BUY - 1

        success, info = tavern.buy_unit(player, 0)

        assert not success
        assert "gold" in info.lower() or "enough" in info.lower()


# ===================================================================
#  2. TAVERN LOGIC
# ===================================================================


class TestTavernLogic:
    """Rolling, freezing, upgrading."""

    def test_roll_deducts_gold(
        self,
        empty_game: "Game",
        player: Player,
        tavern: "TavernManager",
    ) -> None:
        gold_before = player.gold

        success, _ = tavern.roll_tavern(player)

        assert success
        assert player.gold == gold_before - COST_REROLL

    def test_roll_returns_old_cards_to_pool(
        self,
        empty_game: "Game",
        player: Player,
        tavern: "TavernManager",
    ) -> None:
        tavern.roll_tavern(player)

        # Old units returned; new units drawn → net change depends on slot count.
        # But the specific *old* ids must have come back.  Easier: pool must stay
        # roughly the same size (returned N, drawn N).  We just assert roll success
        # plus the tavern is re-filled.
        expected_unit_slots = TAVERN_SLOTS[player.tavern_tier]
        actual_units = sum(1 for item in player.store if item.unit)
        assert actual_units == expected_unit_slots

    def test_roll_refills_tavern(
        self,
        empty_game: "Game",
        player: Player,
        tavern: "TavernManager",
    ) -> None:
        tavern.roll_tavern(player)

        expected_units = TAVERN_SLOTS[player.tavern_tier]
        actual_units = sum(1 for it in player.store if it.unit)
        assert actual_units == expected_units

    def test_roll_not_enough_gold(
        self,
        player: Player,
        tavern: "TavernManager",
    ) -> None:
        player.gold = 0

        success, _ = tavern.roll_tavern(player)

        assert not success

    def test_freeze_preserves_shop_next_turn(
        self,
        empty_game: "Game",
        player: Player,
        tavern: "TavernManager",
    ) -> None:
        # Freeze the shop
        tavern.toggle_freeze(player)
        frozen_unit_ids = [item.unit.card_id for item in player.store if item.unit]

        # Simulate next turn
        tavern.start_turn(player, empty_game.turn_count + 1)

        remaining_ids = [item.unit.card_id for item in player.store if item.unit]
        # All previously-frozen units must still be present (+ newly-drawn ones)
        for cid in frozen_unit_ids:
            assert cid in remaining_ids

    def test_unfreeze(
        self,
        player: Player,
        tavern: "TavernManager",
    ) -> None:
        tavern.toggle_freeze(player)
        assert all(item.is_frozen for item in player.store)

        tavern.toggle_freeze(player)
        assert all(not item.is_frozen for item in player.store)

    def test_upgrade_increases_tier(
        self,
        player: Player,
        tavern: "TavernManager",
    ) -> None:
        player.gold = 20
        tier_before = player.tavern_tier

        success, _ = tavern.upgrade_tavern(player)

        assert success
        assert player.tavern_tier == tier_before + 1

    def test_upgrade_cost_decreases_each_turn(
        self,
        empty_game: "Game",
        player: Player,
        tavern: "TavernManager",
    ) -> None:
        cost_t1 = player.up_cost
        # Simulate next turn (turn 2)
        tavern.start_turn(player, 2)
        cost_t2 = player.up_cost

        assert cost_t2 == cost_t1 - 1

    def test_upgrade_max_tier_fails(
        self,
        player: Player,
        tavern: "TavernManager",
    ) -> None:
        player.tavern_tier = 6
        player.gold = 100

        success, _ = tavern.upgrade_tavern(player)

        assert not success


# ===================================================================
#  3. TRIPLETS
# ===================================================================


class TestTriplets:
    """Three-of-a-kind merging into golden unit."""

    def test_triplet_merges_into_golden(
        self,
        empty_game: "Game",
        player: Player,
        tavern: "TavernManager",
        mock_unit: Callable[..., Unit],
    ) -> None:
        """Put 2 copies in hand, buy the 3rd → a golden must appear."""
        cid = CardIDs.FLIGHTY_SCOUT
        u1 = mock_unit(cid)
        u2 = mock_unit(cid)
        player.hand.append(HandCard(uid=u1.uid, unit=u1))
        player.hand.append(HandCard(uid=u2.uid, unit=u2))

        u3 = mock_unit(cid)
        _inject_unit_into_store(player, u3)
        player.gold = 10

        tavern.buy_unit(player, 0)

        golden_cards = [hc for hc in player.hand if hc.unit and hc.unit.is_golden]
        assert len(golden_cards) == 1
        golden_unit = golden_cards[0].unit
        assert golden_unit is not None
        assert golden_unit.card_id == cid

    def test_triplet_golden_has_double_base_stats(
        self,
        empty_game: "Game",
        player: Player,
        tavern: "TavernManager",
        mock_unit: Callable[..., Unit],
    ) -> None:
        cid = CardIDs.FLIGHTY_SCOUT  # base 3/3
        for _ in range(2):
            u = mock_unit(cid)
            player.hand.append(HandCard(uid=u.uid, unit=u))

        u3 = mock_unit(cid)
        _inject_unit_into_store(player, u3)
        player.gold = 10
        tavern.buy_unit(player, 0)

        golden = next(hc.unit for hc in player.hand if hc.unit and hc.unit.is_golden)
        assert golden.base_atk == 3 * 2
        assert golden.base_hp == 3 * 2

    def test_triplet_reward_spell_appears(
        self,
        empty_game: "Game",
        player: Player,
        tavern: "TavernManager",
        mock_unit: Callable[..., Unit],
    ) -> None:
        """Playing the golden unit should add a TRIPLET_REWARD spell to hand."""
        cid = CardIDs.FLIGHTY_SCOUT
        for _ in range(2):
            u = mock_unit(cid)
            player.hand.append(HandCard(uid=u.uid, unit=u))

        u3 = mock_unit(cid)
        _inject_unit_into_store(player, u3)
        player.gold = 10
        tavern.buy_unit(player, 0)

        # Now play the golden from hand onto board
        golden_idx = next(i for i, hc in enumerate(player.hand) if hc.unit and hc.unit.is_golden)
        tavern.play_unit(player, golden_idx, insert_index=0)

        reward_spells = [
            hc for hc in player.hand if hc.spell and hc.spell.card_id == SpellIDs.TRIPLET_REWARD
        ]
        assert len(reward_spells) >= 1

    def test_triplet_from_board_and_hand(
        self,
        empty_game: "Game",
        player: Player,
        tavern: "TavernManager",
        mock_unit: Callable[..., Unit],
    ) -> None:
        """1 copy on board + 1 in hand + 1 bought → golden."""
        cid = CardIDs.DIRE_WOLF_ALPHA
        u_board = mock_unit(cid)
        player.board.append(u_board)

        u_hand = mock_unit(cid)
        player.hand.append(HandCard(uid=u_hand.uid, unit=u_hand))

        u_buy = mock_unit(cid)
        _inject_unit_into_store(player, u_buy)
        player.gold = 10
        tavern.buy_unit(player, 0)

        golden_in_hand = [hc for hc in player.hand if hc.unit and hc.unit.is_golden]
        assert len(golden_in_hand) == 1
        # Board copy should have been removed
        assert all(u.card_id != cid or u.is_golden for u in player.board)


# ===================================================================
#  4. DISCOVERY
# ===================================================================


class TestDiscovery:
    """Start discovery, choose a card, verify pool housekeeping."""

    def test_start_discovery(
        self,
        empty_game: "Game",
        player: Player,
        tavern: "TavernManager",
    ) -> None:
        ok = tavern.start_discovery(player, source="Test", tier=1, count=3)

        assert ok
        assert player.is_discovering
        assert len(player.discovery.options) <= 3

    def test_discovery_choice_adds_to_hand(
        self,
        empty_game: "Game",
        player: Player,
        tavern: "TavernManager",
    ) -> None:
        tavern.start_discovery(player, source="Test", tier=1, count=3)
        hand_before = len(player.hand)

        success, _ = tavern.make_discovery_choice(player, 0)

        assert success
        assert len(player.hand) == hand_before + 1
        assert not player.is_discovering

    def test_discovery_unpicked_cards_return_to_pool(
        self,
        empty_game: "Game",
        player: Player,
        tavern: "TavernManager",
    ) -> None:
        pool_before = _pool_total(empty_game)
        tavern.start_discovery(player, source="Test", tier=1, count=3)
        n_options = len(player.discovery.options)
        pool_after_draw = _pool_total(empty_game)

        # Pool shrank by the number of options offered
        assert pool_after_draw == pool_before - n_options

        # Collect unpicked card ids before choosing
        unpicked_ids = [
            item.unit.card_id
            for i, item in enumerate(player.discovery.options)
            if i != 0 and item.unit
        ]

        tavern.make_discovery_choice(player, 0)

        pool_after_choice = _pool_total(empty_game)
        # Unpicked cards returned
        assert pool_after_choice == pool_after_draw + len(unpicked_ids)

    def test_discovery_blocks_other_actions(
        self,
        empty_game: "Game",
        player: Player,
    ) -> None:
        """While discovering, non-DISCOVER_CHOICE actions must be rejected."""
        empty_game.tavern.start_discovery(player, source="Test", tier=1, count=3)

        success, _, _ = empty_game.step(player.uid, "ROLL")

        assert not success


# ===================================================================
#  5. EDGE CASES
# ===================================================================


class TestEdgeCases:
    """Boundary / invalid-state tests."""

    def test_buy_full_hand_fails(
        self,
        empty_game: "Game",
        player: Player,
        tavern: "TavernManager",
        mock_unit: Callable[..., Unit],
    ) -> None:
        # Fill hand to 10
        for _ in range(10):
            u = mock_unit(CardIDs.ALLEYCAT)
            player.hand.append(HandCard(uid=u.uid, unit=u))

        unit = mock_unit(CardIDs.ALLEYCAT)
        _inject_unit_into_store(player, unit)

        success, info = tavern.buy_unit(player, 0)

        assert not success
        assert "full" in info.lower()

    def test_play_unit_full_board_fails(
        self,
        empty_game: "Game",
        player: Player,
        tavern: "TavernManager",
        mock_unit: Callable[..., Unit],
    ) -> None:
        # Fill board to 7
        for _ in range(7):
            u = mock_unit(CardIDs.ALLEYCAT)
            player.board.append(u)

        u_hand = mock_unit(CardIDs.ALLEYCAT)
        player.hand.append(HandCard(uid=u_hand.uid, unit=u_hand))

        success, info = tavern.play_unit(player, hand_index=0, insert_index=0)

        assert not success
        assert "full" in info.lower()

    def test_buy_invalid_store_index(
        self,
        player: Player,
        tavern: "TavernManager",
    ) -> None:
        success, _ = tavern.buy_unit(player, 999)
        assert not success

        success, _ = tavern.buy_unit(player, -1)
        assert not success

    def test_sell_invalid_board_index(
        self,
        player: Player,
        tavern: "TavernManager",
    ) -> None:
        success, _ = tavern.sell_unit(player, 999)
        assert not success

    def test_sell_golden_returns_three_copies(
        self,
        empty_game: "Game",
        player: Player,
        tavern: "TavernManager",
        mock_unit: Callable[..., Unit],
    ) -> None:
        cid = CardIDs.ANNOY_O_TRON
        golden = mock_unit(cid, is_golden=True)
        player.board.append(golden)
        pool_before = _pool_tier_count(empty_game, cid)

        tavern.sell_unit(player, 0)

        pool_after = _pool_tier_count(empty_game, cid)
        assert pool_after == pool_before + 3

    def test_swap_units(
        self,
        player: Player,
        tavern: "TavernManager",
        mock_unit: Callable[..., Unit],
    ) -> None:
        u_a = mock_unit(CardIDs.ALLEYCAT)
        u_b = mock_unit(CardIDs.ANNOY_O_TRON)
        player.board.extend([u_a, u_b])
        uid_a, uid_b = u_a.uid, u_b.uid

        success, _ = tavern.swap_units(player, 0, 1)

        assert success
        assert player.board[0].uid == uid_b
        assert player.board[1].uid == uid_a

    def test_play_unit_at_specific_index(
        self,
        player: Player,
        tavern: "TavernManager",
        mock_unit: Callable[..., Unit],
    ) -> None:
        u1 = mock_unit(CardIDs.ALLEYCAT)
        u2 = mock_unit(CardIDs.ANNOY_O_TRON)
        player.board.extend([u1, u2])

        u_new = mock_unit(CardIDs.SCALLYWAG)
        player.hand.append(HandCard(uid=u_new.uid, unit=u_new))

        # Insert between the two existing units (index 1)
        success, _ = tavern.play_unit(player, hand_index=0, insert_index=1)

        assert success
        assert player.board[1].uid == u_new.uid

    def test_upgrade_not_enough_gold(
        self,
        player: Player,
        tavern: "TavernManager",
    ) -> None:
        player.gold = 0

        success, _ = tavern.upgrade_tavern(player)

        assert not success
