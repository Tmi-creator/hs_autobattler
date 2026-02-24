"""Golden-unit (triplet) logic tests — pure pytest, no unittest.

Covers: stat segregation (perm vs turn), reward tier baking, discovery flow.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import pytest

from hearthstone.engine.entities import HandCard, Player, Spell, Unit
from hearthstone.engine.enums import CardIDs, SpellIDs

if TYPE_CHECKING:
    from hearthstone.engine.game import Game
    from hearthstone.engine.tavern import TavernManager


# ===================================================================
#  TRIPLET STAT SEGREGATION
# ===================================================================


class TestTripletStatSegregation:
    """When three copies merge, perm buffs stay perm, turn buffs stay turn."""

    def test_perm_and_turn_buffs_segregate_after_merge(
        self,
        empty_game: "Game",
        player: Player,
        tavern: "TavernManager",
        mock_unit: Callable[..., Unit],
    ) -> None:
        cid = CardIDs.WRATH_WEAVER  # base 1/3

        # Unit 1: in hand with +2/+2 perm buff
        u1 = mock_unit(cid)
        u1.perm_atk_add = 2
        u1.perm_hp_add = 2
        u1.recalc_stats()
        player.hand.append(HandCard(uid=u1.uid, unit=u1))

        # Unit 2: on board with +3/+0 turn buff
        u2 = mock_unit(cid)
        u2.turn_atk_add = 3
        u2.recalc_stats()
        player.board.append(u2)

        # Unit 3: on board, clean
        u3 = mock_unit(cid)
        player.board.append(u3)

        tavern._check_triplet(player, cid)

        # Board must be empty (both copies consumed)
        assert len(player.board) == 0
        assert len(player.hand) == 1

        golden = player.hand[0].unit
        assert golden is not None
        assert golden.is_golden

        # Perm layer preserved
        assert golden.perm_atk_add == 2
        # Turn layer preserved
        assert golden.turn_atk_add == 3

        # Golden base: 2×(1/3) = 2/6
        # Cur atk: base(2) + perm(2) + turn(3) = 7
        assert golden.cur_atk == 7

    def test_turn_buffs_disappear_after_reset(
        self,
        empty_game: "Game",
        player: Player,
        tavern: "TavernManager",
        mock_unit: Callable[..., Unit],
    ) -> None:
        cid = CardIDs.WRATH_WEAVER  # base 1/3

        u1 = mock_unit(cid)
        u1.perm_atk_add = 2
        u1.perm_hp_add = 2
        u1.recalc_stats()
        player.hand.append(HandCard(uid=u1.uid, unit=u1))

        u2 = mock_unit(cid)
        u2.turn_atk_add = 3
        u2.recalc_stats()
        player.board.append(u2)

        u3 = mock_unit(cid)
        player.board.append(u3)

        tavern._check_triplet(player, cid)

        golden = player.hand[0].unit
        assert golden is not None

        golden.reset_turn_layer()
        golden.recalc_stats()

        # After end-of-turn: base(2) + perm(2) = 4, turn gone
        assert golden.cur_atk == 4


# ===================================================================
#  TRIPLET REWARD TIER
# ===================================================================


class TestTripletRewardTier:
    """Reward spell tier is baked when the golden unit is PLAYED, not merged."""

    def test_reward_tier_equals_tavern_plus_one(
        self,
        empty_game: "Game",
        player: Player,
        tavern: "TavernManager",
        mock_unit: Callable[..., Unit],
    ) -> None:
        cid = CardIDs.WRATH_WEAVER
        for _ in range(3):
            u = mock_unit(cid)
            player.hand.append(HandCard(uid=u.uid, unit=u))

        tavern._check_triplet(player, cid)
        assert len(player.hand) == 1  # golden only

        # Level up to tier 4 before playing
        player.tavern_tier = 4

        golden_idx = 0
        tavern.play_unit(player, golden_idx)

        # Reward must be in hand
        reward_spells = [
            hc for hc in player.hand if hc.spell and hc.spell.card_id == SpellIDs.TRIPLET_REWARD
        ]
        assert len(reward_spells) >= 1

        # Reward tier = min(6, tavern_tier + 1) = min(6, 5) = 5
        recorded_tier = reward_spells[0].spell.params.get("tier")
        assert recorded_tier == 5

    def test_reward_tier_capped_at_6(
        self,
        empty_game: "Game",
        player: Player,
        tavern: "TavernManager",
        mock_unit: Callable[..., Unit],
    ) -> None:
        cid = CardIDs.WRATH_WEAVER
        for _ in range(3):
            u = mock_unit(cid)
            player.hand.append(HandCard(uid=u.uid, unit=u))

        tavern._check_triplet(player, cid)
        player.tavern_tier = 6

        tavern.play_unit(player, 0)

        reward_spells = [
            hc for hc in player.hand if hc.spell and hc.spell.card_id == SpellIDs.TRIPLET_REWARD
        ]
        assert len(reward_spells) >= 1
        # min(6, 6+1) = 6
        assert reward_spells[0].spell.params.get("tier") == 6


# ===================================================================
#  DISCOVERY FLOW FROM TRIPLET
# ===================================================================


class TestDiscoveryFromTriplet:
    """Playing the TRIPLET_REWARD spell must start discovery."""

    def test_play_reward_starts_exact_tier_discovery(
        self,
        empty_game: "Game",
        player: Player,
        tavern: "TavernManager",
    ) -> None:
        spell = Spell.create_from_db(SpellIDs.TRIPLET_REWARD)
        spell.params["tier"] = 1
        player.hand.append(HandCard(uid=999, spell=spell))

        success, msg = tavern.play_unit(player, 0)

        assert success
        assert player.is_discovering
        # Triplet discovery is exact-tier
        assert player.discovery.is_exact_tier
        assert player.discovery.discover_tier == 1
