"""Tests for ES bot v2 (rule-based + evolved weights) and evolve_bot pipeline."""

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from hearthstone.engine.game import Game
from hearthstone.env.es_bot import (
    N_WEIGHTS,
    WEIGHT_NAMES,
    W_ATK,
    W_HP,
    W_TIER,
    W_UPGRADE_AGGRO,
    W_ROLL_THRESH,
    W_BUY_THRESH,
    W_BUY_EMPTY,
    W_SELL_DELTA,
    es_bot_turn,
    score_unit_es,
)
from hearthstone.env.hs_env import HearthstoneEnv
from scripts.evolve_bot import (
    Individual,
    mutate,
    play_match,
    random_individual,
    run_evolution,
    make_es_bot_fn,
    smart_bot_fn,
)


# ============================================================
# Weight vector basics
# ============================================================

def test_weight_count() -> None:
    assert N_WEIGHTS == len(WEIGHT_NAMES)
    assert N_WEIGHTS == 23


def test_score_unit_es_basic() -> None:
    """Hand-picked weights should score a unit sensibly."""
    w = np.zeros(N_WEIGHTS, dtype=np.float32)
    w[W_ATK] = 1.0
    w[W_HP] = 1.0
    w[W_TIER] = 2.0

    from hearthstone.engine.configs import CARD_DB
    # Find any card in DB
    card_id = next(iter(CARD_DB))
    data = CARD_DB[card_id]
    expected_min = data["atk"] + data["hp"] + 2 * data["tier"]

    score = score_unit_es(card_id, set(), {}, turn=1, w=w)
    assert score >= expected_min - 0.01


# ============================================================
# Bot turn integration
# ============================================================

def _make_sensible_weights() -> np.ndarray:
    """Hand-tuned weights that should produce reasonable play."""
    w = np.zeros(N_WEIGHTS, dtype=np.float32)
    w[W_ATK] = 1.0
    w[W_HP] = 1.0
    w[W_TIER] = 2.0
    w[W_UPGRADE_AGGRO] = 1.2    # upgrade when gold >= 1.2 * up_cost
    w[W_ROLL_THRESH] = 8.0      # roll if best shop < 8
    w[W_BUY_THRESH] = 3.0       # buy if score > 3
    w[W_BUY_EMPTY] = -5.0       # buy almost anything when board empty
    w[W_SELL_DELTA] = 10.0      # sell+buy if delta > 10
    return w


def test_es_bot_turn_completes() -> None:
    random.seed(0)
    np.random.seed(0)
    game = Game(max_tier=3)
    w = _make_sensible_weights()
    es_bot_turn(game, 0, w)
    assert game.players_ready[0] is True


def test_es_bot_upgrades_tavern() -> None:
    """ES bot v2 must upgrade tavern (v1 never did — that was the fatal flaw)."""
    random.seed(42)
    np.random.seed(42)
    w = _make_sensible_weights()

    # Play 10 full rounds (both sides ES bot)
    game = Game(max_tier=6)
    for _ in range(10):
        if game.game_over:
            break
        es_bot_turn(game, 0, w)
        es_bot_turn(game, 1, w)

    # After 10 turns, at least one player should have upgraded past tier 1
    max_tier = max(game.players[0].tavern_tier, game.players[1].tavern_tier)
    assert max_tier > 1, (
        f"Neither player upgraded past tier 1 in 10 turns! "
        f"tiers: {game.players[0].tavern_tier}, {game.players[1].tavern_tier}"
    )


def test_env_set_es_bot_plays_match() -> None:
    random.seed(42)
    np.random.seed(42)
    env = HearthstoneEnv(max_tier=3)
    env.set_es_bot(_make_sensible_weights())
    obs, _ = env.reset(seed=42)

    done = False
    truncated = False
    steps = 0
    while not done and not truncated and steps < 500:
        masks = np.asarray(env.action_masks(), dtype=bool)
        legal = np.flatnonzero(masks)
        if len(legal) == 0:
            break
        action = int(legal[0])
        obs, _, done, truncated, _ = env.step(action)
        steps += 1
    assert steps > 0


# ============================================================
# Evolution pipeline
# ============================================================

def test_play_match_returns_valid_result() -> None:
    rng = random.Random(7)
    a = random_individual(N_WEIGHTS, rng)
    b = random_individual(N_WEIGHTS, rng)

    fn_a = make_es_bot_fn(a.weights)
    fn_b = make_es_bot_fn(b.weights)

    result = play_match(fn_a, fn_b, seed=123, max_tier=3)
    assert result in (-1, 0, 1)


def test_mutation_changes_weights() -> None:
    rng = random.Random(11)
    parent = random_individual(N_WEIGHTS, rng)
    child = mutate(parent, rng)
    assert child.weights.shape == (N_WEIGHTS,)
    assert not np.allclose(child.weights, parent.weights)
    assert np.all(child.sigmas > 0)
    assert np.all(np.isfinite(child.sigmas))


def test_quick_evolution_runs_end_to_end(tmp_path) -> None:
    """Mini evolution: 2 gens, tiny population. Verifies full pipeline."""
    import argparse
    args = argparse.Namespace(
        generations=2,
        mu=3,
        lam=3,
        n_match=4,
        n_anchor=1,
        final_eval=0,
        seed=99,
        out_dir=str(tmp_path),
        workers=1,
        max_tier=3,
        n_hof=2,
        hof_interval=1,
        wandb=False,
        wandb_project="test",
        wandb_name=None,
        quick=False,
    )
    run_evolution(args)

    assert (tmp_path / "best.npz").exists()
    assert (tmp_path / "gen_000.npz").exists()
    assert (tmp_path / "gen_001.npz").exists()
    assert (tmp_path / "fitness_history.npy").exists()

    data = np.load(tmp_path / "best.npz")
    assert data["weights"].shape == (N_WEIGHTS,)
