"""CleanRL-style PPO for Hearthstone Battlegrounds.

Single-file training loop. No SB3, no abstractions. Everything visible.

Usage:
    python scripts/train_ppo.py
    python scripts/train_ppo.py --total-timesteps 1000000 --n-envs 4
    python scripts/train_ppo.py --wandb --run-name my_experiment
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import gymnasium
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from model import (
    HSTransformerAgent,
    BIN_CENTERS,
    encode_twohot,
    decode_value,
)
from hearthstone.env.hs_env import HearthstoneEnv


# ============================================================
# Config
# ============================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    # Training
    p.add_argument("--total-timesteps", type=int, default=5_000_000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.999)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip-coef", type=float, default=0.2)
    p.add_argument("--ent-coef", type=float, default=0.04)
    p.add_argument("--ent-coef-end", type=float, default=0.01)
    p.add_argument("--ent-decay-frac", type=float, default=0.75)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--target-kl", type=float, default=None)
    # Rollout
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--n-steps", type=int, default=2048)
    p.add_argument("--n-minibatches", type=int, default=4)
    p.add_argument("--update-epochs", type=int, default=4)
    # Model
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--n-layers", type=int, default=4)
    # Env
    p.add_argument("--max-tier", type=int, default=6)
    p.add_argument("--seed", type=int, default=42)
    # Logging
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb-project", default="hs_autobattler")
    p.add_argument("--run-name", default=None)
    p.add_argument("--log-interval", type=int, default=1,
                   help="log every N updates")
    p.add_argument("--eval-interval", type=int, default=10,
                   help="eval board_power every N updates")
    p.add_argument("--save-interval", type=int, default=50,
                   help="save checkpoint every N updates")
    p.add_argument("--out-dir", default="artifacts/ppo")
    # Resume
    p.add_argument("--resume", default=None, help="path to checkpoint .pt")
    return p.parse_args()


# ============================================================
# Environment
# ============================================================

def make_env(rank: int, seed: int, max_tier: int):
    def thunk():
        env = HearthstoneEnv(max_tier=max_tier)
        env.reset(seed=seed + rank)
        return env
    return thunk


def get_action_masks(envs) -> torch.Tensor:
    """Get action masks from all envs. Returns [n_envs, n_actions] bool tensor."""
    masks = []
    for i in range(envs.num_envs):
        m = envs.envs[i].action_masks()
        masks.append(m)
    return torch.tensor(np.array(masks), dtype=torch.bool)


def get_board_powers(envs) -> list[float]:
    powers = []
    for env in envs.envs:
        powers.append(env.get_board_power())
    return powers


# ============================================================
# GAE
# ============================================================

def compute_gae(
    rewards: torch.Tensor,      # [T, N]
    values: torch.Tensor,       # [T, N]
    dones: torch.Tensor,        # [T, N]
    next_value: torch.Tensor,   # [N]
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (advantages [T, N], returns [T, N])."""
    T, N = rewards.shape
    advantages = torch.zeros_like(rewards)
    lastgaelam = torch.zeros(N, device=rewards.device)

    for t in reversed(range(T)):
        if t == T - 1:
            next_val = next_value
        else:
            next_val = values[t + 1]
        nextnonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_val * nextnonterminal - values[t]
        lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
        advantages[t] = lastgaelam

    returns = advantages + values
    return advantages, returns


# ============================================================
# Entropy decay
# ============================================================

def get_ent_coef(
    update: int,
    n_updates: int,
    ent_start: float,
    ent_end: float,
    decay_frac: float,
) -> float:
    decay_updates = int(n_updates * decay_frac)
    if update >= decay_updates:
        return ent_end
    frac = update / max(1, decay_updates)
    return ent_start + (ent_end - ent_start) * frac


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()

    # Seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    # Wandb
    run = None
    if args.wandb:
        import wandb
        run = wandb.init(
            project=args.wandb_project,
            name=args.run_name or f"ppo_{args.seed}",
            config=vars(args),
        )

    # Envs
    envs = gymnasium.vector.SyncVectorEnv(
        [make_env(i, args.seed, args.max_tier) for i in range(args.n_envs)]
    )
    n_actions = 34
    obs_dim = envs.single_observation_space.shape[0]

    # Get num_card_ids from first env
    num_card_ids = envs.envs[0].num_card_ids
    print(f"[env] obs_dim={obs_dim} n_actions={n_actions} card_ids={num_card_ids}")

    # Model
    agent = HSTransformerAgent(
        n_actions=n_actions,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        num_card_ids=num_card_ids,
    ).to(device)

    n_params = sum(p.numel() for p in agent.parameters())
    print(f"[model] {n_params:,} params")

    optimizer = torch.optim.Adam(agent.parameters(), lr=args.lr, eps=1e-5)

    # Resume
    global_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        agent.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        global_step = ckpt.get("global_step", 0)
        print(f"[resume] loaded from {args.resume}, step={global_step}")

    # Rollout storage
    batch_size = args.n_envs * args.n_steps
    minibatch_size = batch_size // args.n_minibatches
    n_updates = args.total_timesteps // batch_size
    print(f"[train] batch={batch_size} minibatch={minibatch_size} updates={n_updates}")

    obs_buf = torch.zeros((args.n_steps, args.n_envs, obs_dim), device=device)
    act_buf = torch.zeros((args.n_steps, args.n_envs), dtype=torch.long, device=device)
    logp_buf = torch.zeros((args.n_steps, args.n_envs), device=device)
    rew_buf = torch.zeros((args.n_steps, args.n_envs), device=device)
    done_buf = torch.zeros((args.n_steps, args.n_envs), device=device)
    val_buf = torch.zeros((args.n_steps, args.n_envs), device=device)
    vlogit_buf = torch.zeros((args.n_steps, args.n_envs, 255), device=device)
    mask_buf = torch.zeros((args.n_steps, args.n_envs, n_actions), dtype=torch.bool, device=device)

    # Init envs
    next_obs_np, _ = envs.reset(seed=args.seed)
    next_obs = torch.tensor(next_obs_np, dtype=torch.float32, device=device)
    next_done = torch.zeros(args.n_envs, device=device)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    for update in range(1, n_updates + 1):
        # --- Entropy decay ---
        ent_coef = get_ent_coef(
            update, n_updates, args.ent_coef, args.ent_coef_end, args.ent_decay_frac
        )

        # --- Collect rollout ---
        agent.eval()
        for step in range(args.n_steps):
            global_step += args.n_envs

            obs_buf[step] = next_obs
            done_buf[step] = next_done
            action_mask = get_action_masks(envs).to(device)
            mask_buf[step] = action_mask

            with torch.no_grad():
                action, logprob, _, value, v_logits = agent.get_action_and_value(
                    next_obs, action_mask
                )
            act_buf[step] = action
            logp_buf[step] = logprob
            val_buf[step] = value
            vlogit_buf[step] = v_logits

            next_obs_np, reward_np, terminated, truncated, infos = envs.step(
                action.cpu().numpy()
            )
            done_np = np.logical_or(terminated, truncated)
            rew_buf[step] = torch.tensor(reward_np, dtype=torch.float32, device=device)
            next_obs = torch.tensor(next_obs_np, dtype=torch.float32, device=device)
            next_done = torch.tensor(done_np, dtype=torch.float32, device=device)

        # --- GAE ---
        with torch.no_grad():
            next_value = agent.get_value(next_obs)
        advantages, returns = compute_gae(
            rew_buf, val_buf, done_buf, next_value, args.gamma, args.gae_lambda
        )

        # --- Flatten ---
        b_obs = obs_buf.reshape(-1, obs_dim)
        b_actions = act_buf.reshape(-1)
        b_logprobs = logp_buf.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_masks = mask_buf.reshape(-1, n_actions)

        # --- PPO update ---
        agent.train()
        b_inds = np.arange(batch_size)
        clipfracs = []
        pg_losses = []
        vf_losses = []
        ent_losses = []

        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb = b_inds[start:end]

                _, newlogprob, entropy, newvalue, new_vlogits = agent.get_action_and_value(
                    b_obs[mb], b_masks[mb], b_actions[mb]
                )

                logratio = newlogprob - b_logprobs[mb]
                ratio = logratio.exp()

                # Approx KL for early stopping
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean().item()
                    clipfracs.append(
                        ((ratio - 1.0).abs() > args.clip_coef).float().mean().item()
                    )

                # Normalize advantages
                mb_adv = b_advantages[mb]
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss: categorical cross-entropy (two-hot)
                bins = agent.bins
                target_twohot = encode_twohot(b_returns[mb], bins)
                vf_loss = -(target_twohot * F.log_softmax(new_vlogits, -1)).sum(-1).mean()

                ent_loss = entropy.mean()

                loss = pg_loss + args.vf_coef * vf_loss - ent_coef * ent_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

                pg_losses.append(pg_loss.item())
                vf_losses.append(vf_loss.item())
                ent_losses.append(ent_loss.item())

            # KL early stopping
            if args.target_kl is not None and approx_kl > 1.5 * args.target_kl:
                break

        # --- Logging ---
        if update % args.log_interval == 0:
            elapsed = time.time() - start_time
            fps = global_step / elapsed
            avg_reward = rew_buf.mean().item()

            print(
                f"[update {update:4d}/{n_updates}] "
                f"step={global_step:,} fps={fps:.0f} "
                f"pg={np.mean(pg_losses):.4f} vf={np.mean(vf_losses):.4f} "
                f"ent={np.mean(ent_losses):.4f} ent_coef={ent_coef:.4f} "
                f"kl={approx_kl:.4f} clip={np.mean(clipfracs):.3f} "
                f"avg_r={avg_reward:.3f}"
            )

            if run is not None:
                run.log({
                    "charts/fps": fps,
                    "charts/avg_reward": avg_reward,
                    "losses/policy": np.mean(pg_losses),
                    "losses/value": np.mean(vf_losses),
                    "losses/entropy": np.mean(ent_losses),
                    "losses/approx_kl": approx_kl,
                    "losses/clipfrac": np.mean(clipfracs),
                    "config/ent_coef": ent_coef,
                    "config/lr": optimizer.param_groups[0]["lr"],
                }, step=global_step)

        # --- Eval ---
        if update % args.eval_interval == 0:
            powers = get_board_powers(envs)
            avg_bp = np.mean(powers)
            max_bp = np.max(powers)
            print(f"  [eval] avg_board_power={avg_bp:.1f} max={max_bp:.1f}")
            if run is not None:
                run.log({
                    "eval/avg_board_power": avg_bp,
                    "eval/max_board_power": max_bp,
                }, step=global_step)

        # --- Save ---
        if update % args.save_interval == 0:
            ckpt_path = out_dir / f"ckpt_{global_step}.pt"
            torch.save({
                "model": agent.state_dict(),
                "optimizer": optimizer.state_dict(),
                "global_step": global_step,
                "args": vars(args),
            }, ckpt_path)
            print(f"  [save] {ckpt_path}")

    # Final save
    final_path = out_dir / "final.pt"
    torch.save({
        "model": agent.state_dict(),
        "optimizer": optimizer.state_dict(),
        "global_step": global_step,
        "args": vars(args),
    }, final_path)
    print(f"[done] {global_step:,} steps, saved to {final_path}")

    if run is not None:
        run.finish()
    envs.close()


if __name__ == "__main__":
    main()
