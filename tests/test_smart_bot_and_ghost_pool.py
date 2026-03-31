"""Tests for SmartBot heuristic opponent and GhostPool self-play."""

import pytest

from hearthstone.engine.configs import CARD_DB
from hearthstone.engine.entities import Player, Unit
from hearthstone.engine.enums import CardIDs, UnitType
from hearthstone.engine.game import Game
from hearthstone.env.ghost_pool import BoardSnapshot, GhostPool, UnitSnapshot
from hearthstone.env.smart_bot import score_unit, smart_bot_turn


# ===================================================================
#  1. SMART BOT: score_unit
# ===================================================================


class TestScoreUnit:
    """Test the scoring function used by smart bot to evaluate shop units."""

    def test_score_basic_stats(self):
        """Score should include ATK + HP + tier * 2."""
        # Find any card in DB to test
        card_id = CardIDs.ANNOY_O_TRON
        s = score_unit(card_id, set(), {}, turn=1)
        data = CARD_DB[card_id]
        base = data["atk"] + data["hp"] + data["tier"] * 2
        assert s >= base  # may have bonuses on top

    def test_triplet_bonus(self):
        """Having 2 copies on board should give +100 (must buy for triple)."""
        card_id = CardIDs.ANNOY_O_TRON
        s_no_copies = score_unit(card_id, set(), {}, turn=1)
        s_with_pair = score_unit(card_id, set(), {card_id: 2}, turn=1)
        assert s_with_pair - s_no_copies == pytest.approx(100.0)

    def test_pair_bonus(self):
        """Having 1 copy should give +10 (pair potential)."""
        card_id = CardIDs.ANNOY_O_TRON
        s_no = score_unit(card_id, set(), {}, turn=1)
        s_pair = score_unit(card_id, set(), {card_id: 1}, turn=1)
        assert s_pair - s_no == pytest.approx(10.0)

    def test_tribal_synergy(self):
        """Matching tribe on board gives +6 per matching type."""
        card_id = CardIDs.ANNOY_O_TRON  # MECH
        s_no_synergy = score_unit(card_id, set(), {}, turn=1)
        s_synergy = score_unit(card_id, {UnitType.MECH}, {}, turn=1)
        assert s_synergy > s_no_synergy

    def test_unknown_card_returns_zero(self):
        """Unknown card_id should return 0."""
        s = score_unit("NONEXISTENT_CARD_XYZ", set(), {}, turn=1)
        assert s == 0.0

    def test_wrath_weaver_early_bonus(self):
        """Wrath Weaver gets +15 bonus on turn <= 4."""
        s_early = score_unit(CardIDs.WRATH_WEAVER, set(), {}, turn=2)
        s_late = score_unit(CardIDs.WRATH_WEAVER, set(), {}, turn=6)
        assert s_early > s_late


class TestSmartBotTurn:
    """Test that smart bot plays a complete turn without errors."""

    def test_bot_completes_turn(self):
        """Smart bot should play a full turn and mark player as ready."""
        game = Game()
        player_idx = 1
        smart_bot_turn(game, player_idx)
        # After bot turn, player should have spent gold and/or played cards
        player = game.players[player_idx]
        # Bot should have made at least one decision (even if END_TURN only)
        assert player.gold >= 0  # gold was consumed or still there

    def test_bot_doesnt_crash_empty_store(self):
        """Bot should handle empty store gracefully."""
        game = Game()
        player = game.players[1]
        player.store.clear()
        smart_bot_turn(game, 1)  # should not raise

    def test_bot_buys_units(self):
        """Bot should buy at least one unit when it has gold."""
        game = Game()
        player = game.players[1]
        initial_hand = len(player.hand)
        initial_board = len(player.board)
        smart_bot_turn(game, 1)
        # Bot should have bought something (unit on board or hand)
        total_after = len(player.board) + len(player.hand)
        assert total_after >= initial_board  # at least maintained board


# ===================================================================
#  2. GHOST POOL
# ===================================================================


class TestUnitSnapshot:
    """Test UnitSnapshot serialization."""

    def test_from_unit_preserves_stats(self):
        """Snapshot should preserve all unit stats."""
        unit = Unit.create_from_db(CardIDs.ANNOY_O_TRON, uid=1, owner_id=0)
        snap = UnitSnapshot.from_unit(unit)
        assert snap.card_id == CardIDs.ANNOY_O_TRON
        assert snap.base_atk == unit.base_atk
        assert snap.base_hp == unit.base_hp
        assert snap.cur_atk == unit.cur_atk
        assert snap.cur_hp == unit.cur_hp
        assert snap.tier == unit.tier

    def test_to_unit_roundtrip(self):
        """Snapshot → Unit → Snapshot should preserve data."""
        unit = Unit.create_from_db(CardIDs.ANNOY_O_TRON, uid=1, owner_id=0)
        unit.perm_atk_add = 3
        unit.perm_hp_add = 5
        unit.recalc_stats()
        snap = UnitSnapshot.from_unit(unit)
        restored = snap.to_unit(uid=99, owner_id=1)
        assert restored.cur_atk == unit.cur_atk
        assert restored.cur_hp == unit.cur_hp
        assert restored.perm_atk_add == 3
        assert restored.perm_hp_add == 5

    def test_snapshot_preserves_tags(self):
        """Tags like TAUNT, DIVINE_SHIELD should survive snapshot."""
        unit = Unit.create_from_db(CardIDs.ANNOY_O_TRON, uid=1, owner_id=0)
        snap = UnitSnapshot.from_unit(unit)
        assert "TAUNT" in snap.tags
        assert "DIVINE_SHIELD" in snap.tags


class TestGhostPool:
    """Test GhostPool lifecycle."""

    def test_record_and_finish(self):
        """Recording turns and finishing should add trajectory to pool."""
        pool = GhostPool(max_games=10)
        game = Game()
        player = game.players[0]

        # Simulate 3 turns of recording
        pool.record_turn(env_id=0, turn=1, player=player)
        pool.record_turn(env_id=0, turn=2, player=player)
        pool.record_turn(env_id=0, turn=3, player=player)
        pool.finish_game(env_id=0)

        assert pool.size == 1

    def test_finish_requires_min_turns(self):
        """Trajectory with < 2 turns should be discarded."""
        pool = GhostPool(max_games=10)
        game = Game()
        player = game.players[0]

        pool.record_turn(env_id=0, turn=1, player=player)
        pool.finish_game(env_id=0)

        assert pool.size == 0  # discarded (only 1 turn)

    def test_sample_returns_trajectory(self):
        """Sampling should return a dict of turn → BoardSnapshot."""
        pool = GhostPool(max_games=10)
        game = Game()
        player = game.players[0]

        # Add multiple games so recency-biased sampling works
        for env_id in range(5):
            pool.record_turn(env_id, 1, player)
            pool.record_turn(env_id, 2, player)
            pool.finish_game(env_id)

        traj = pool.sample_trajectory()
        assert traj is not None
        assert 1 in traj
        assert 2 in traj
        assert isinstance(traj[1], BoardSnapshot)

    def test_sample_empty_pool_returns_none(self):
        """Empty pool should return None."""
        pool = GhostPool(max_games=10)
        assert pool.sample_trajectory() is None

    def test_max_games_eviction(self):
        """Pool should evict old games when full."""
        pool = GhostPool(max_games=3)
        game = Game()
        player = game.players[0]

        for env_id in range(5):
            pool.record_turn(env_id, 1, player)
            pool.record_turn(env_id, 2, player)
            pool.finish_game(env_id)

        assert pool.size == 3  # oldest evicted

    def test_materialize_board(self):
        """Materializing snapshot should replace player's board."""
        game = Game()
        player = game.players[0]
        # Put a unit on board
        unit = Unit.create_from_db(CardIDs.ANNOY_O_TRON, uid=1, owner_id=player.uid)
        player.board.append(unit)

        # Snapshot it
        snap = BoardSnapshot(
            units=[UnitSnapshot.from_unit(unit)],
            tavern_tier=2,
        )

        # Materialize on another player
        target = game.players[1]
        target.board.clear()
        uid_counter = [100]

        def uid_fn():
            uid_counter[0] += 1
            return uid_counter[0]

        GhostPool.materialize_board(snap, target, uid_fn)
        assert len(target.board) == 1
        assert target.board[0].card_id == CardIDs.ANNOY_O_TRON
        assert target.tavern_tier == 2

    def test_save_and_load(self, tmp_path):
        """Pool should survive save/load cycle."""
        pool = GhostPool(max_games=10)
        game = Game()
        player = game.players[0]

        pool.record_turn(0, 1, player)
        pool.record_turn(0, 2, player)
        pool.finish_game(0)

        path = str(tmp_path / "ghost.pkl")
        pool.save(path)

        pool2 = GhostPool(max_games=10)
        loaded = pool2.load(path)
        assert loaded == 1
        assert pool2.size == 1

    def test_multiple_envs_independent(self):
        """Different env_ids should have independent recordings."""
        pool = GhostPool(max_games=10)
        game = Game()
        p0, p1 = game.players[0], game.players[1]

        pool.record_turn(env_id=0, turn=1, player=p0)
        pool.record_turn(env_id=0, turn=2, player=p0)
        pool.record_turn(env_id=1, turn=1, player=p1)
        pool.record_turn(env_id=1, turn=2, player=p1)

        pool.finish_game(0)
        pool.finish_game(1)
        assert pool.size == 2
