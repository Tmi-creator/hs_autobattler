"""Ghost Pool: Zero-inference self-play via historical board replay.

Instead of running model.predict() 30 times per enemy turn, we record
the agent's boards at end-of-turn and replay them as opponents in future games.
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from hearthstone.engine.entities import Player, Unit
from hearthstone.engine.enums import Tags, UnitType


@dataclass(slots=True)
class UnitSnapshot:
    """Lightweight snapshot of a Unit for combat replay."""

    card_id: str
    base_atk: int
    base_hp: int
    cur_atk: int
    cur_hp: int
    perm_atk_add: int
    perm_hp_add: int
    tier: int
    is_golden: bool
    tags: Set[str]  # store tag names, not enums (picklable)
    types: List[str]  # store type values
    attached_perm: Dict[str, int]

    @staticmethod
    def from_unit(unit: Unit) -> UnitSnapshot:
        return UnitSnapshot(
            card_id=unit.card_id,
            base_atk=unit.base_atk,
            base_hp=unit.base_hp,
            cur_atk=unit.cur_atk,
            cur_hp=unit.cur_hp,
            perm_atk_add=unit.perm_atk_add,
            perm_hp_add=unit.perm_hp_add,
            tier=unit.tier,
            is_golden=unit.is_golden,
            tags={t.name for t in unit.tags},
            types=[t.value for t in unit.types],
            attached_perm=dict(unit.attached_perm),
        )

    def to_unit(self, uid: int, owner_id: int) -> Unit:
        """Recreate a combat-ready Unit from snapshot."""
        tags_set: set[Tags] = set()
        for name in self.tags:
            try:
                tags_set.add(Tags[name])
            except KeyError:
                pass

        type_list: list[UnitType] = []
        for val in self.types:
            try:
                type_list.append(UnitType(val))
            except ValueError:
                pass

        unit = Unit(
            uid=uid,
            card_id=self.card_id,
            owner_id=owner_id,
            tier=self.tier,
            base_hp=self.base_hp,
            base_atk=self.base_atk,
            max_hp=self.base_hp,
            max_atk=self.base_atk,
            perm_atk_add=self.perm_atk_add,
            perm_hp_add=self.perm_hp_add,
            types=type_list,
            tags=tags_set,
            is_golden=self.is_golden,
            attached_perm=dict(self.attached_perm),
        )
        unit.recalc_stats()
        unit.restore_stats()
        return unit


@dataclass
class BoardSnapshot:
    """A single turn's board state."""

    units: List[UnitSnapshot]
    tavern_tier: int


class GhostPool:
    """Pool of historical game trajectories for ghost self-play.

    Usage:
        pool = GhostPool(max_games=2000)

        # During training, after each END_TURN:
        pool.record_turn(env_id=0, turn=3, player=agent_player)

        # At episode end:
        pool.finish_game(env_id=0)

        # At env.reset():
        ghost = pool.sample_trajectory()

        # At enemy's turn:
        pool.materialize_board(ghost[turn], enemy_player, uid_fn)
    """

    def __init__(self, max_games: int = 2000) -> None:
        self.max_games = max_games
        self.trajectories: deque[Dict[int, BoardSnapshot]] = deque(
            maxlen=max_games,
        )
        # Per-env recording buffers
        self._current: Dict[int, Dict[int, BoardSnapshot]] = {}

    @property
    def size(self) -> int:
        return len(self.trajectories)

    def record_turn(
        self, env_id: int, turn: int, player: Player
    ) -> None:
        """Snapshot the player's board at end of this turn."""
        if env_id not in self._current:
            self._current[env_id] = {}

        unit_snaps = [UnitSnapshot.from_unit(u) for u in player.board]
        self._current[env_id][turn] = BoardSnapshot(
            units=unit_snaps,
            tavern_tier=player.tavern_tier,
        )

    def finish_game(self, env_id: int) -> None:
        """Push completed trajectory into pool and reset buffer."""
        traj = self._current.pop(env_id, None)
        if traj and len(traj) >= 2:  # need at least 2 turns
            self.trajectories.append(traj)

    def sample_trajectory(
        self,
    ) -> Optional[Dict[int, BoardSnapshot]]:
        """Random trajectory from pool. None if pool is empty."""
        if not self.trajectories:
            return None
        return random.choice(self.trajectories)

    @staticmethod
    def materialize_board(
        snapshot: BoardSnapshot,
        player: Player,
        uid_fn: Any,
    ) -> None:
        """Overwrite player's board with units from snapshot."""
        player.board.clear()
        player.tavern_tier = snapshot.tavern_tier
        for unit_snap in snapshot.units:
            uid = uid_fn()
            unit = unit_snap.to_unit(uid, player.uid)
            player.board.append(unit)
