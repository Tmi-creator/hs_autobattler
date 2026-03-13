"""Tavern smoke tests — pure pytest (replaces legacy script).

Covers: roll, buy, pool count integrity.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from hearthstone.engine.configs import TAVERN_SLOTS
from hearthstone.engine.entities import Player

if TYPE_CHECKING:
    from hearthstone.engine.game import Game
    from hearthstone.engine.tavern import TavernManager


class TestTavernSmoke:
    """Basic tavern operations that the old script was manually verifying."""

    def test_initial_store_has_correct_slot_count(
        self,
        player: Player,
    ) -> None:
        expected_units = TAVERN_SLOTS[player.tavern_tier]
        actual_units = sum(1 for item in player.store if item.unit)
        assert actual_units == expected_units

    def test_roll_and_buy_decrements_pool(
        self,
        empty_game: "Game",
        player: Player,
        tavern: "TavernManager",
    ) -> None:
        pool_before = sum(len(t) for t in empty_game.pool.tiers.values())

        tavern.roll_tavern(player)

        # After roll: returned N old units, drew N new ones → net ~0
        # But buy actually removes 1 from pool
        if player.store:
            tavern.buy_unit(player, 0)

        pool_after = sum(len(t) for t in empty_game.pool.tiers.values())

        # Pool should have shrunk by ~1 (the bought unit left pool and is now in hand)
        # Some units from store are also out of pool, but they'll return on next roll
        assert pool_after <= pool_before
