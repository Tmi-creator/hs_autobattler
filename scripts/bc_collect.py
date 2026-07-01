"""Collect BC trajectories from the evolved ES bot.

Replays episodes through HearthstoneEnv. At every decision point we ask the ES
bot what its first game.step() call would be on the current state, convert that
to a 0..33 action int (the env's action space), and step the env with it.

The env handles the rest (target prompts, end-of-turn enemy play, rewards). We
record (obs, action_mask, action_int) at every step.

Output: artifacts/bc_dataset.npz with arrays:
    obs:    [N, obs_dim] float32
    masks:  [N, 34]      bool
    actions:[N]          int64

Usage:
    python scripts/bc_collect.py --episodes 1000 --weights artifacts/es_kaggle/artifacts/best.npz
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from hearthstone.env.es_bot import (
    es_bot_turn,
    _best_board_unit_idx,
    score_unit_es,
)
from hearthstone.env.hs_env import HearthstoneEnv


# ============================================================
# ES bot → single-action picker
# ============================================================

class _ESActionSnoop(Exception):
    """Sentinel: raised inside the ES bot to capture its first game.step call."""


def _action_kwargs_to_int(action_type: str, kwargs: dict) -> int:
    """Map (verb, kwargs) from game.step → 0..33 action_int from env.step."""
    if action_type == "END_TURN":
        return 0
    if action_type == "ROLL":
        return 1
    if action_type == "BUY":
        return 2 + int(kwargs.get("index", 0))
    if action_type == "SELL":
        return 9 + int(kwargs.get("index", 0))
    if action_type == "PLAY":
        # Whether spell-with-target or plain play, the first env action is
        # always 16+hand_index. Targeting (if any) is resolved on the next
        # env step via the is_targeting branch in es_pick_action.
        return 16 + int(kwargs.get("hand_index", 0))
    if action_type == "DISCOVER_CHOICE":
        return 2 + int(kwargs.get("index", 0))
    if action_type == "SWAP":
        return 26 + int(kwargs.get("index_a", 0))
    if action_type == "UPGRADE":
        return 32
    if action_type == "FREEZE":
        return 33
    return 0  # safety: end turn


def es_pick_action(env: HearthstoneEnv, weights: np.ndarray) -> int:
    """Return the action_int the ES bot would take on the current env state."""
    game = env.game
    p_idx = env.my_player_id
    player = game.players[p_idx]
    turn = game.turn_count
    mask = env.action_masks()

    # ---- Targeting mode (env is waiting for spell/magnetize target) ----
    if env.is_targeting:
        # Targets are encoded as actions 2..8 (board slots).
        target_actions = np.where(mask[2:9])[0] + 2
        if len(target_actions) == 0:
            return 0  # CANCEL_CAST during targeting

        best_idx = _best_board_unit_idx(player, weights, turn)
        if best_idx >= 0 and mask[2 + best_idx]:
            return 2 + best_idx
        return int(target_actions[0])

    # ---- Discovery mode (3 options, encoded as actions 2..4) ----
    if player.is_discovering and player.discovery and player.discovery.options:
        from hearthstone.env.es_bot import _get_board_types, _get_card_counts
        board_types = _get_board_types(player)
        card_counts = _get_card_counts(player)
        best_i, best_s = 0, -1e18
        for i, opt in enumerate(player.discovery.options):
            sc = (
                score_unit_es(opt.unit.card_id, board_types, card_counts, turn, weights)
                if opt.unit
                else 0.0
            )
            if sc > best_s:
                best_s, best_i = sc, i
        return 2 + best_i

    # ---- Normal tavern action: snoop the first game.step from es_bot_turn ----
    captured: list = []
    real_step = game.step

    def _snoop(p, action_type, **kwargs):
        captured.append((action_type, dict(kwargs)))
        raise _ESActionSnoop()

    game.step = _snoop  # type: ignore[method-assign]
    try:
        es_bot_turn(game, p_idx, weights, max_actions=1)
    except _ESActionSnoop:
        pass
    finally:
        game.step = real_step  # type: ignore[method-assign]

    if not captured:
        return 0  # ES bot has nothing to do → end turn

    action_type, kwargs = captured[0]
    aint = _action_kwargs_to_int(action_type, kwargs)

    # Last-line safety: if the snooped action is masked (e.g. ES wanted to PLAY
    # a card the env classified as needing a target with no valid target),
    # fall back to the highest-mass legal action instead of getting stuck.
    if not mask[aint]:
        legal = np.where(mask)[0]
        if len(legal) == 0:
            return 0
        return int(legal[0])

    return aint


# ============================================================
# Episode collection
# ============================================================

def collect_episode(env: HearthstoneEnv, weights: np.ndarray, seed: int):
    """Run one episode end-to-end. Returns (obs_list, mask_list, action_list, board_power)."""
    obs, _ = env.reset(seed=seed)
    obs_buf, mask_buf, act_buf = [], [], []
    done = False
    truncated = False
    while not (done or truncated):
        mask = env.action_masks()
        action = es_pick_action(env, weights)

        obs_buf.append(obs.astype(np.float32, copy=True))
        mask_buf.append(mask.astype(np.bool_, copy=True))
        act_buf.append(np.int64(action))

        obs, _, done, truncated, _ = env.step(action)

    return obs_buf, mask_buf, act_buf, env.get_board_power()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default="artifacts/es_kaggle/artifacts/best.npz")
    p.add_argument("--episodes", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-tier", type=int, default=6)
    p.add_argument("--out", default="artifacts/bc_dataset.npz")
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--use-enemy-board-obs", action="store_true")
    p.add_argument("--use-player-status-obs", action="store_true")
    args = p.parse_args()

    weights = np.load(args.weights)["weights"].astype(np.float32)
    print(f"[weights] shape={weights.shape} from {args.weights}")

    env = HearthstoneEnv(
        max_tier=args.max_tier,
        use_enemy_board_obs=args.use_enemy_board_obs,
        use_player_status_obs=args.use_player_status_obs,
    )

    all_obs, all_masks, all_acts, all_bp = [], [], [], []
    t0 = time.time()
    for ep in range(args.episodes):
        seed = args.seed + ep
        obs_buf, mask_buf, act_buf, bp = collect_episode(env, weights, seed)
        all_obs.extend(obs_buf)
        all_masks.extend(mask_buf)
        all_acts.extend(act_buf)
        all_bp.append(bp)
        if (ep + 1) % args.log_every == 0:
            elapsed = time.time() - t0
            steps = len(all_obs)
            print(
                f"[ep {ep+1}/{args.episodes}] steps={steps:,} "
                f"avg_bp={np.mean(all_bp):.1f} fps={steps/elapsed:.0f}"
            )

    obs_arr = np.stack(all_obs).astype(np.float32)
    mask_arr = np.stack(all_masks).astype(np.bool_)
    act_arr = np.array(all_acts, dtype=np.int64)
    bp_arr = np.array(all_bp, dtype=np.float32)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        obs=obs_arr,
        masks=mask_arr,
        actions=act_arr,
        board_powers=bp_arr,
    )

    print(f"[done] {len(act_arr):,} steps from {args.episodes} episodes")
    print(f"  avg_bp={bp_arr.mean():.2f} max_bp={bp_arr.max():.1f}")
    print(f"  action histogram (0..33):")
    counts = np.bincount(act_arr, minlength=34)
    for i in range(34):
        bar = "#" * int(40 * counts[i] / counts.max()) if counts.max() > 0 else ""
        print(f"    {i:2d}: {counts[i]:6d} {bar}")
    print(f"  saved: {out_path} ({out_path.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
