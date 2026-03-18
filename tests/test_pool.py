"""Tests for CardPool and SpellPool.

Covers: initialization, draw mechanics, return mechanics,
discovery (unique, exact tier, predicate), token exclusion, spell pool.
"""

from __future__ import annotations

from hearthstone.engine.configs import CARD_DB, SPELL_DB, TIER_COPIES
from hearthstone.engine.enums import CardIDs, UnitType
from hearthstone.engine.pool import CardPool, SpellPool

# ===================================================================
#  1. CARD POOL INITIALIZATION
# ===================================================================


class TestCardPoolInit:
    """Pool initialized correctly from CARD_DB + TIER_COPIES."""

    def test_pool_has_all_tiers(self) -> None:
        pool = CardPool()
        for tier in TIER_COPIES:
            assert tier in pool.tiers

    def test_pool_excludes_tokens(self) -> None:
        pool = CardPool()
        token_ids = {cid for cid, data in CARD_DB.items() if data.get("is_token", False)}
        for tier_cards in pool.tiers.values():
            for card_id in tier_cards:
                assert card_id not in token_ids, f"Token {card_id} found in pool"

    def test_each_non_token_card_has_correct_copies(self) -> None:
        pool = CardPool()
        for card_id, data in CARD_DB.items():
            if data.get("is_token", False):
                continue
            tier = data["tier"]
            expected = TIER_COPIES[tier]
            actual = pool.tiers[tier].count(card_id)
            assert actual == expected, f"{card_id}: {actual} copies, expected {expected}"

    def test_total_pool_size_matches_config(self) -> None:
        pool = CardPool()
        non_tokens = [cid for cid, d in CARD_DB.items() if not d.get("is_token", False)]
        expected_total = sum(TIER_COPIES[CARD_DB[cid]["tier"]] for cid in non_tokens)
        actual_total = sum(len(t) for t in pool.tiers.values())
        assert actual_total == expected_total


# ===================================================================
#  2. DRAWING CARDS
# ===================================================================


class TestCardPoolDraw:
    """Drawing cards from the pool."""

    def test_draw_removes_from_pool(self) -> None:
        pool = CardPool()
        total_before = sum(len(t) for t in pool.tiers.values())
        drawn = pool.draw_cards(3, max_tier=1)
        total_after = sum(len(t) for t in pool.tiers.values())

        assert len(drawn) == 3
        assert total_after == total_before - 3

    def test_draw_respects_max_tier(self) -> None:
        pool = CardPool()
        drawn = pool.draw_cards(20, max_tier=1)
        for card_id in drawn:
            assert CARD_DB[card_id]["tier"] <= 1

    def test_draw_higher_tier_includes_lower(self) -> None:
        pool = CardPool()
        drawn = pool.draw_cards(30, max_tier=3)
        for card_id in drawn:
            assert CARD_DB[card_id]["tier"] <= 3

    def test_draw_returns_correct_count(self) -> None:
        pool = CardPool()
        for n in [1, 3, 5, 7]:
            drawn = pool.draw_cards(n, max_tier=6)
            assert len(drawn) == n


# ===================================================================
#  3. RETURNING CARDS
# ===================================================================


class TestCardPoolReturn:
    """Returning cards to the pool."""

    def test_return_adds_cards_back(self) -> None:
        pool = CardPool()
        cid = CardIDs.ALLEYCAT
        tier = CARD_DB[cid]["tier"]
        count_before = pool.tiers[tier].count(cid)
        pool.return_cards([cid])
        count_after = pool.tiers[tier].count(cid)
        assert count_after == count_before + 1

    def test_return_tokens_ignored(self) -> None:
        pool = CardPool()
        total_before = sum(len(t) for t in pool.tiers.values())
        pool.return_cards([CardIDs.TABBYCAT, CardIDs.IMP_TOKEN, CardIDs.PIRATE_TOKEN])
        total_after = sum(len(t) for t in pool.tiers.values())
        assert total_after == total_before

    def test_return_unknown_card_ignored(self) -> None:
        pool = CardPool()
        total_before = sum(len(t) for t in pool.tiers.values())
        pool.return_cards(["NONEXISTENT_CARD_999"])
        total_after = sum(len(t) for t in pool.tiers.values())
        assert total_after == total_before

    def test_return_multiple_copies(self) -> None:
        pool = CardPool()
        cid = CardIDs.ANNOY_O_TRON
        tier = CARD_DB[cid]["tier"]
        count_before = pool.tiers[tier].count(cid)
        pool.return_cards([cid, cid, cid])
        count_after = pool.tiers[tier].count(cid)
        assert count_after == count_before + 3


# ===================================================================
#  4. DISCOVERY CARDS
# ===================================================================


class TestCardPoolDiscovery:
    """Discovery drawing: unique, exact tier, predicate filtering."""

    def test_discovery_draws_unique_cards(self) -> None:
        pool = CardPool()
        drawn = pool.draw_discovery_cards(3, tier=1, exact_tier=False)
        assert len(drawn) == len(set(drawn))

    def test_discovery_exact_tier(self) -> None:
        pool = CardPool()
        drawn = pool.draw_discovery_cards(3, tier=2, exact_tier=True)
        for card_id in drawn:
            assert CARD_DB[card_id]["tier"] == 2

    def test_discovery_non_exact_includes_lower(self) -> None:
        pool = CardPool()
        drawn = pool.draw_discovery_cards(10, tier=3, exact_tier=False)
        tiers_seen = {CARD_DB[cid]["tier"] for cid in drawn}
        assert all(t <= 3 for t in tiers_seen)

    def test_discovery_with_predicate(self) -> None:
        pool = CardPool()
        predicate = lambda data: UnitType.MURLOC in data.get("type", [])
        drawn = pool.draw_discovery_cards(3, tier=2, exact_tier=False, predicate=predicate)
        for cid in drawn:
            assert UnitType.MURLOC in CARD_DB[cid].get("type", [])

    def test_discovery_removes_from_pool(self) -> None:
        pool = CardPool()
        total_before = sum(len(t) for t in pool.tiers.values())
        drawn = pool.draw_discovery_cards(3, tier=1, exact_tier=False)
        total_after = sum(len(t) for t in pool.tiers.values())
        assert total_after == total_before - len(drawn)

    def test_discovery_empty_pool_returns_empty(self) -> None:
        """If no cards match, return empty list."""
        pool = CardPool()
        # Dragon type has no cards in our DB
        predicate = lambda data: UnitType.DRAGON in data.get("type", [])
        drawn = pool.draw_discovery_cards(3, tier=1, exact_tier=False, predicate=predicate)
        assert drawn == []


# ===================================================================
#  5. SPELL POOL
# ===================================================================


class TestSpellPool:
    """SpellPool initialization and drawing."""

    def test_spell_pool_excludes_non_pool_spells(self) -> None:
        pool = SpellPool()
        non_pool_ids = {sid for sid, data in SPELL_DB.items() if not data.get("pool", True)}
        for tier_spells in pool.tiers.values():
            for spell_id in tier_spells:
                assert spell_id not in non_pool_ids

    def test_draw_spells_respects_max_tier(self) -> None:
        pool = SpellPool()
        drawn = pool.draw_spells(5, max_tier=1)
        for spell_id in drawn:
            assert SPELL_DB[spell_id]["tier"] <= 1

    def test_draw_spells_returns_correct_count(self) -> None:
        pool = SpellPool()
        drawn = pool.draw_spells(3, max_tier=6)
        assert len(drawn) == 3

    def test_draw_spells_empty_tier_returns_empty(self) -> None:
        pool = SpellPool()
        # Tier 0 has only Blood Gem but it's pool=False
        drawn = pool.draw_spells(3, max_tier=0)
        # If no tiers available at max_tier=0, should return empty
        # (SpellPool only has tier 1 spells in the current config)
        if 0 not in pool.tiers:
            assert drawn == []
