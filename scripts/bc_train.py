"""Behavior cloning pretrain on ES bot trajectories.

Loads (obs, masks, actions) from artifacts/bc_dataset.npz and trains the actor
of HSTransformerAgent via masked cross-entropy. Critic head is left untouched
(zero-init from model.py; PPO will train it from scratch with a clean optimizer).

Saves a checkpoint compatible with `train_ppo.py --resume`:
    {"model": state_dict, "global_step": 0, "args": {...}}

Usage:
    python scripts/bc_train.py --epochs 10 --batch-size 256
    python scripts/bc_train.py --epochs 10 --wandb --run-name bc_v1
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from model import HSTransformerAgent
from hearthstone.env.hs_env import HearthstoneEnv


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="artifacts/bc_dataset.npz")
    p.add_argument("--out", default="artifacts/bc/bc_pretrain.pt")
    # Train
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--val-frac", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    # Model (must match train_ppo.py for resume compatibility)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--max-tier", type=int, default=6)
    # Logging
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb-project", default="hs_autobattler")
    p.add_argument("--run-name", default=None)
    return p.parse_args()


def main():
    args = parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    # ---- Load dataset ----
    data = np.load(args.dataset)
    obs = torch.from_numpy(data["obs"]).float()
    masks = torch.from_numpy(data["masks"]).bool()
    actions = torch.from_numpy(data["actions"]).long()
    print(f"[data] {len(actions):,} samples, obs_dim={obs.shape[1]}")

    # ---- Sanity: every recorded action must be legal under its mask ----
    legal_check = masks[torch.arange(len(actions)), actions]
    illegal = (~legal_check).sum().item()
    if illegal > 0:
        print(f"[warn] {illegal}/{len(actions)} samples have action masked illegal — dropping")
        keep = legal_check
        obs, masks, actions = obs[keep], masks[keep], actions[keep]

    # ---- Train/val split ----
    n = len(actions)
    perm = torch.randperm(n)
    n_val = int(n * args.val_frac)
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]

    train_ds = TensorDataset(obs[train_idx], masks[train_idx], actions[train_idx])
    val_ds = TensorDataset(obs[val_idx], masks[val_idx], actions[val_idx])
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=(device.type == "cuda"),
    )
    print(f"[split] train={len(train_ds):,} val={len(val_ds):,}")

    # ---- Need num_card_ids from env (must match RL training!) ----
    tmp_env = HearthstoneEnv(max_tier=args.max_tier)
    num_card_ids = tmp_env.num_card_ids
    del tmp_env
    print(f"[env] num_card_ids={num_card_ids}")

    # ---- Model ----
    agent = HSTransformerAgent(
        n_actions=34,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        num_card_ids=num_card_ids,
    ).to(device)
    n_params = sum(p.numel() for p in agent.parameters())
    print(f"[model] {n_params:,} params")

    optimizer = torch.optim.AdamW(
        agent.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # ---- Wandb ----
    run = None
    if args.wandb:
        import wandb
        run = wandb.init(
            project=args.wandb_project,
            name=args.run_name or f"bc_{args.seed}",
            config=vars(args),
        )

    # ---- Train ----
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    best_val_acc = 0.0
    global_step = 0
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        agent.train()
        train_loss_sum, train_correct, train_count = 0.0, 0, 0
        for batch_obs, batch_mask, batch_act in train_loader:
            batch_obs = batch_obs.to(device, non_blocking=True)
            batch_mask = batch_mask.to(device, non_blocking=True)
            batch_act = batch_act.to(device, non_blocking=True)

            action_logits, _ = agent(batch_obs)
            action_logits = action_logits.masked_fill(~batch_mask, -1e8)
            loss = F.cross_entropy(action_logits, batch_act)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
            optimizer.step()

            train_loss_sum += loss.item() * batch_act.size(0)
            train_correct += (action_logits.argmax(-1) == batch_act).sum().item()
            train_count += batch_act.size(0)
            global_step += batch_act.size(0)

        train_loss = train_loss_sum / max(1, train_count)
        train_acc = train_correct / max(1, train_count)

        # ---- Val ----
        agent.eval()
        val_loss_sum, val_correct, val_count = 0.0, 0, 0
        with torch.no_grad():
            for batch_obs, batch_mask, batch_act in val_loader:
                batch_obs = batch_obs.to(device, non_blocking=True)
                batch_mask = batch_mask.to(device, non_blocking=True)
                batch_act = batch_act.to(device, non_blocking=True)
                action_logits, _ = agent(batch_obs)
                action_logits = action_logits.masked_fill(~batch_mask, -1e8)
                loss = F.cross_entropy(action_logits, batch_act)
                val_loss_sum += loss.item() * batch_act.size(0)
                val_correct += (action_logits.argmax(-1) == batch_act).sum().item()
                val_count += batch_act.size(0)

        val_loss = val_loss_sum / max(1, val_count)
        val_acc = val_correct / max(1, val_count)
        elapsed = time.time() - t0

        print(
            f"[ep {epoch:2d}/{args.epochs}] "
            f"train_loss={train_loss:.4f} acc={train_acc:.3f}  "
            f"val_loss={val_loss:.4f} acc={val_acc:.3f}  "
            f"({elapsed:.0f}s)"
        )

        if run is not None:
            run.log({
                "bc/train_loss": train_loss,
                "bc/train_acc": train_acc,
                "bc/val_loss": val_loss,
                "bc/val_acc": val_acc,
                "bc/epoch": epoch,
            }, step=global_step)

        # ---- Save best ----
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model": agent.state_dict(),
                "global_step": 0,
                "args": vars(args),
                "val_acc": val_acc,
            }, out_path)
            print(f"  [save] {out_path} (val_acc={val_acc:.3f})")

    print(f"[done] best val_acc={best_val_acc:.3f}, ckpt: {out_path}")
    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
