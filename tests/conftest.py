from __future__ import annotations

from typing import TYPE_CHECKING, Callable, cast

import pytest

from src.hearthstone.engine.entities import Player, Unit
from src.hearthstone.engine.enums import CardIDs
from src.hearthstone.engine.game import Game

if TYPE_CHECKING:
    from src.hearthstone.engine.tavern import TavernManager


@pytest.fixture()
def empty_game() -> Game:
    """Fresh ``Game`` instance (turn 1 already started for both players)."""
    return Game()


@pytest.fixture()
def player(empty_game: Game) -> Player:
    """Player 0 of *empty_game* with gold set to **10** for comfortable testing."""
    p = empty_game.players[0]
    p.gold = 10
    return p


@pytest.fixture()
def tavern(empty_game: Game) -> TavernManager:
    """``TavernManager`` bound to *empty_game*."""
    from src.hearthstone.engine.tavern import TavernManager

    return cast(TavernManager, empty_game.tavern)


@pytest.fixture()
def mock_unit(empty_game: Game) -> Callable[..., Unit]:
    """Factory fixture: ``mock_unit(CardIDs.ALLEYCAT)`` → ready-to-use ``Unit``.

    Accepts optional *owner_id* (default ``0``) and *is_golden* (default ``False``).
    Each call produces a unit with a unique UID.
    """

    def _factory(
        card_id: CardIDs | str,
        owner_id: int = 0,
        is_golden: bool = False,
    ) -> Unit:
        uid = empty_game.tavern._get_next_uid()
        raw_id = card_id.value if isinstance(card_id, CardIDs) else card_id
        return cast(Unit, Unit.create_from_db(raw_id, uid, owner_id, is_golden))

    return _factory
