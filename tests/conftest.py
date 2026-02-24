"""Shared fixtures for the HS Autobattler test suite.

Every test must use these fixtures — manual instantiation of ``Game``,
``Player``, or ``EventManager`` inside test functions is forbidden.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Tuple

import pytest

from hearthstone.engine.combat import CombatManager
from hearthstone.engine.entities import HandCard, Player, Spell, StoreItem, Unit
from hearthstone.engine.enums import CardIDs, SpellIDs
from hearthstone.engine.event_system import EventManager
from hearthstone.engine.game import Game
from hearthstone.engine.tavern import TavernManager

# ---------------------------------------------------------------------------
#  Core fixtures
# ---------------------------------------------------------------------------


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
def enemy(empty_game: Game) -> Player:
    """Player 1 of *empty_game* with gold set to **10**."""
    p = empty_game.players[1]
    p.gold = 10
    return p


@pytest.fixture()
def tavern(empty_game: Game) -> TavernManager:
    """``TavernManager`` bound to *empty_game*."""
    return empty_game.tavern


@pytest.fixture()
def event_manager(empty_game: Game) -> EventManager:
    """``EventManager`` already wired into *empty_game*."""
    return empty_game.event_manager


@pytest.fixture()
def combat_manager(empty_game: Game) -> CombatManager:
    """``CombatManager`` already wired into *empty_game*."""
    return empty_game.combat


# ---------------------------------------------------------------------------
#  Factory fixtures
# ---------------------------------------------------------------------------


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
        uid = empty_game.tavern.get_next_uid()
        raw_id = card_id.value if isinstance(card_id, CardIDs) else card_id
        return Unit.create_from_db(raw_id, uid, owner_id, is_golden)

    return _factory


@pytest.fixture()
def combat_players(
    empty_game: Game,
    mock_unit: Callable[..., Unit],
) -> Callable[..., Tuple[Dict[int, Player], List[List[Unit]], CombatManager]]:
    """Factory: build a combat scenario from two lists of card IDs.

    Returns ``(players_dict, [board_0, board_1], combat_manager)``.

    Usage::

        players, boards, cm = combat_players(
            [CardIDs.ALLEYCAT, CardIDs.SCALLYWAG],
            [CardIDs.ANNOY_O_TRON],
        )
    """

    def _factory(
        side_0_ids: list[CardIDs | str],
        side_1_ids: list[CardIDs | str],
    ) -> Tuple[Dict[int, Player], List[List[Unit]], CombatManager]:
        p0 = empty_game.players[0]
        p1 = empty_game.players[1]
        p0.board = [mock_unit(cid, owner_id=p0.uid) for cid in side_0_ids]
        p1.board = [mock_unit(cid, owner_id=p1.uid) for cid in side_1_ids]

        # Combat copies (so original boards are untouched)
        cp0 = p0.combat_copy()
        cp1 = p1.combat_copy()
        players_dict: Dict[int, Player] = {cp0.uid: cp0, cp1.uid: cp1}
        boards = [cp0.board, cp1.board]
        cm = empty_game.combat
        return players_dict, boards, cm

    return _factory


# ---------------------------------------------------------------------------
#  Store / hand injection helpers (exposed as fixtures for DRY)
# ---------------------------------------------------------------------------


@pytest.fixture()
def inject_unit_into_store() -> Callable[[Player, Unit], int]:
    """Insert *unit* as the first store item; returns store index ``0``."""

    def _inject(player: Player, unit: Unit) -> int:
        player.store.insert(0, StoreItem(unit=unit))
        return 0

    return _inject


@pytest.fixture()
def inject_spell_into_store() -> Callable[[Player, Spell], int]:
    """Insert *spell* as the first store item; returns store index ``0``."""

    def _inject(player: Player, spell: Spell) -> int:
        player.store.insert(0, StoreItem(spell=spell))
        return 0

    return _inject
