"""
Скрипт для отправки обучения на Kaggle.
Встраивает весь исходный код как base64 прямо в kernel — никаких dataset-зависимостей.

Использование:
    python scripts/kaggle_submit.py
"""

import base64
import io
import json
import os
import shutil
import zipfile
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Warning: python-dotenv not installed. Make sure environment variables are set manually.")



root_dir = Path(__file__).resolve().parent.parent
scripts_dir = root_dir / "scripts"
build_dir = root_dir / "scripts" / "_kaggle_build"

KAGGLE_USERNAME = "tmitmi1999"
KERNEL_SLUG = "hs-autobattler-training"


def _pack_project_b64() -> str:
    """Пакует src/ + scripts/ + pyproject.toml в zip, возвращает base64."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        # src/hearthstone/
        src_root = root_dir / "src"
        for fp in src_root.rglob("*"):
            if fp.is_file() and "__pycache__" not in str(fp) and not fp.suffix == ".pyc":
                zf.write(fp, str(fp.relative_to(root_dir)))

        # scripts/ (only needed files)
        for fname in [
            "__init__.py", "trans.py", "callbacks.py",
            "train.py", "train_transformer.py",
            "evaluate_pvp.py", "visualize_attention.py",
        ]:
            fp = scripts_dir / fname
            if fp.exists():
                zf.write(fp, f"scripts/{fname}")

        # pyproject.toml
        pt = root_dir / "pyproject.toml"
        if pt.exists():
            zf.write(pt, "pyproject.toml")

    data = buf.getvalue()
    print(f"[ZIP] Project packed: {len(data) / 1024:.0f} KB")
    return base64.b64encode(data).decode("ascii")


def create_kernel():
    """Создаёт self-contained Kaggle kernel со встроенным кодом."""
    kernel_dir = build_dir / "kernel"
    if kernel_dir.exists():
        shutil.rmtree(kernel_dir)
    kernel_dir.mkdir(parents=True)

    # Pack project
    project_b64 = _pack_project_b64()

    # kernel-metadata.json
    metadata = {
        "id": f"{KAGGLE_USERNAME}/{KERNEL_SLUG}",
        "title": "HS Autobattler Training",
        "code_file": "train_kaggle.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
    }
    with open(kernel_dir / "kernel-metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # === Kernel Script ===
    script = f'''#!/usr/bin/env python3
"""
HS Autobattler: MLP vs Transformer Training on Kaggle T4 GPU.
Self-contained: all source code is embedded as base64 zip.
"""

import base64
import io
import os
import shutil
import subprocess
import sys
import time
import zipfile

# === 1. Constants ===
PROJECT_B64 = "{project_b64}"
PROJECT_DIR = "/kaggle/working/project"
OUTPUT_DIR = "/kaggle/working/outputs"

# === 2. Extract + Install (module level, runs in ALL processes) ===
# Use marker to avoid re-extracting; pip install is idempotent
_marker = os.path.join(PROJECT_DIR, "pyproject.toml")
if not os.path.exists(_marker):
    print("[SETUP] Extracting embedded project...")
    os.makedirs(PROJECT_DIR, exist_ok=True)
    zip_data = base64.b64decode(PROJECT_B64)
    with zipfile.ZipFile(io.BytesIO(zip_data), "r") as zf:
        zf.extractall(PROJECT_DIR)
    print(f"[OK] Extracted to {{PROJECT_DIR}}")
    for _root, _dirs, _files in os.walk(PROJECT_DIR):
        for _fn in _files:
            _rel = os.path.relpath(os.path.join(_root, _fn), PROJECT_DIR)
            print(f"  {{_rel}}")

sys.path.insert(0, PROJECT_DIR)
sys.path.insert(0, os.path.join(PROJECT_DIR, "src"))

# pip install only in main process (children inherit installed packages)
if __name__ == "__main__":
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-e", PROJECT_DIR, "-q"],
        check=True,
    )
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "sb3-contrib", "wandb", "-q"],
        check=True,
    )
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === 3. Imports (now safe) ===
import numpy as np
import torch
from typing import cast
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv
from wandb.integration.sb3 import WandbCallback

from hearthstone.env.hs_env import HearthstoneEnv
from scripts.trans import TransformerFeaturesExtractor
from scripts.callbacks import (
    BoardPowerCallback, GameLoggerCallback, SelfPlayCallback
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[DEVICE] Using: {{DEVICE}}")
if DEVICE == "cuda":
    print(f"[GPU] {{torch.cuda.get_device_name(0)}}")

SEED = 42
N_ENVS = 4
TOTAL_TIMESTEPS = 1_500_000


def setup_determinism(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    set_random_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def mask_fn(base_env):
    return np.asarray(cast(HearthstoneEnv, base_env).action_masks(), dtype=bool)


def make_env(rank, seed=42):
    def _init():
        env = HearthstoneEnv()
        env.reset(seed=seed + rank)
        return ActionMasker(env, mask_fn)
    return _init


# =========================================
# PHASE 1: Train MLP
# =========================================
def train_mlp():
    print("\\n" + "=" * 60)
    print("PHASE 1: Training MLP Agent")
    print("=" * 60)
    setup_determinism(SEED)

    run = wandb.init(
        project="hs_autobattler_comparison",
        name="mlp_baseline",
        config={{"model": "MLP", "timesteps": TOTAL_TIMESTEPS, "n_envs": N_ENVS}},
        sync_tensorboard=True,
    )

    models_dir = os.path.join(OUTPUT_DIR, "models", "mlp", run.id)
    logs_dir = os.path.join(OUTPUT_DIR, "logs", "mlp", run.id)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    env = SubprocVecEnv([make_env(i, SEED) for i in range(N_ENVS)])

    tb_log = os.path.join(OUTPUT_DIR, "tb_logs", "mlp")
    model = MaskablePPO(
        "MlpPolicy", env, verbose=1,
        learning_rate=3e-4, gamma=0.99, batch_size=256,
        n_steps=2048, ent_coef=0.01,
        policy_kwargs=dict(net_arch=[512, 512]),
        seed=SEED, device=DEVICE,
        tensorboard_log=tb_log,
    )

    total_params = sum(p.numel() for p in model.policy.parameters() if p.requires_grad)
    print(f"[MODEL] MLP: {{total_params:,}} params")

    start = time.time()
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[
            CheckpointCallback(save_freq=100_000 // N_ENVS, save_path=models_dir, name_prefix="mlp"),
            WandbCallback(gradient_save_freq=500, verbose=0),
            GameLoggerCallback(check_freq=50000, log_dir=logs_dir),
            SelfPlayCallback(update_freq=50_000, model_save_path=models_dir),
            BoardPowerCallback(log_freq=2000),
        ],
    )
    elapsed = time.time() - start
    print(f"[MLP] Done in {{elapsed:.0f}}s ({{TOTAL_TIMESTEPS/elapsed:.0f}} steps/sec)")

    final_path = os.path.join(OUTPUT_DIR, "models", "mlp_final")
    model.save(final_path)
    run.finish()
    env.close()
    return final_path + ".zip"


# =========================================
# PHASE 2: Train Transformer
# =========================================
def train_transformer():
    print("\\n" + "=" * 60)
    print("PHASE 2: Training Transformer Agent")
    print("=" * 60)
    setup_determinism(SEED)

    run = wandb.init(
        project="hs_autobattler_comparison",
        name="transformer",
        config={{
            "model": "Transformer", "timesteps": TOTAL_TIMESTEPS,
            "n_envs": N_ENVS, "d_model": 128, "n_heads": 4, "n_layers": 4,
        }},
        sync_tensorboard=True,
    )

    models_dir = os.path.join(OUTPUT_DIR, "models", "transformer", run.id)
    logs_dir = os.path.join(OUTPUT_DIR, "logs", "transformer", run.id)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    env = SubprocVecEnv([make_env(i, SEED) for i in range(N_ENVS)])

    policy_kwargs = dict(
        features_extractor_class=TransformerFeaturesExtractor,
        features_extractor_kwargs=dict(d_model=128, n_heads=4, n_layers=4, d_context=10),
        net_arch=dict(pi=[128], vf=[128]),
    )

    tb_log = os.path.join(OUTPUT_DIR, "tb_logs", "transformer")
    model = MaskablePPO(
        MaskableActorCriticPolicy, env, verbose=1,
        learning_rate=3e-4, gamma=0.99, batch_size=256,
        n_steps=2048, ent_coef=0.01,
        policy_kwargs=policy_kwargs,
        seed=SEED, device=DEVICE,
        tensorboard_log=tb_log,
    )

    # Zero-Init (DreamerV3)
    for module in [model.policy.action_net, model.policy.value_net]:
        if hasattr(module, "weight"):
            torch.nn.init.zeros_(module.weight)
        if hasattr(module, "bias") and module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    print("[INIT] Zero-Init applied")

    ext = model.policy.features_extractor
    n_params = sum(p.numel() for p in ext.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.policy.parameters() if p.requires_grad)
    print(f"[MODEL] Transformer: {{n_params:,}} extractor, {{total_params:,}} total")

    start = time.time()
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[
            CheckpointCallback(save_freq=100_000 // N_ENVS, save_path=models_dir, name_prefix="trans"),
            WandbCallback(gradient_save_freq=500, verbose=0),
            GameLoggerCallback(check_freq=50000, log_dir=logs_dir),
            SelfPlayCallback(update_freq=50_000, model_save_path=models_dir),
            BoardPowerCallback(log_freq=2000),
        ],
    )
    elapsed = time.time() - start
    print(f"[TRANS] Done in {{elapsed:.0f}}s ({{TOTAL_TIMESTEPS/elapsed:.0f}} steps/sec)")

    final_path = os.path.join(OUTPUT_DIR, "models", "transformer_final")
    model.save(final_path)
    run.finish()
    env.close()
    return final_path + ".zip"


# =========================================
# PHASE 3: PvP
# =========================================
def evaluate_pvp(mlp_path, trans_path, n_games=200):
    print("\\n" + "=" * 60)
    print(f"PHASE 3: PvP Evaluation ({{n_games}} games each way)")
    print("=" * 60)

    mlp = MaskablePPO.load(mlp_path, device="cpu")
    trans = MaskablePPO.load(trans_path, device="cpu")

    def run_match(agent, opponent, n, seed=42):
        env = HearthstoneEnv()
        wins, losses, draws = 0, 0, 0
        for i in range(n):
            obs, _ = env.reset(seed=seed + i)
            env.set_opponent(opponent)
            done, truncated = False, False
            while not done and not truncated:
                masks = np.asarray(env.action_masks(), dtype=bool)
                action, _ = agent.predict(obs, action_masks=masks, deterministic=True)
                obs, _, done, truncated, _ = env.step(int(action))
            p0 = env.game.players[env.my_player_id]
            p1 = env.game.players[env.enemy_id]
            if p0.health > 0 and p1.health <= 0: wins += 1
            elif p0.health <= 0 and p1.health > 0: losses += 1
            else: draws += 1
        return {{"wins": wins, "losses": losses, "draws": draws, "wr": wins / n * 100}}

    print("[MATCH 1] Transformer vs MLP...")
    s1 = run_match(trans, mlp, n_games, seed=100)
    print(f"  Trans WR: {{s1['wr']:.1f}}% ({{s1['wins']}}W/{{s1['losses']}}L/{{s1['draws']}}D)")

    print("[MATCH 2] MLP vs Transformer...")
    s2 = run_match(mlp, trans, n_games, seed=200)
    print(f"  MLP WR: {{s2['wr']:.1f}}% ({{s2['wins']}}W/{{s2['losses']}}L/{{s2['draws']}}D)")

    overall_trans = (s1["wr"] + (100 - s2["wr"])) / 2
    print(f"\\n[OVERALL] Transformer: {{overall_trans:.1f}}% | MLP: {{100-overall_trans:.1f}}%")

    run = wandb.init(project="hs_autobattler_comparison", name="pvp_results")
    wandb.log({{
        "pvp/trans_wr_as_agent": s1["wr"],
        "pvp/mlp_wr_as_agent": s2["wr"],
        "pvp/trans_overall_wr": overall_trans,
    }})
    run.finish()


# =========================================
# MAIN
# =========================================
if __name__ == "__main__":
    import wandb
    wandb_key = os.environ.get('WANDB_API_KEY')
    if wandb_key:
        wandb.login(key=wandb_key)
    else:
        # Fallback to Kaggle Secrets
        from kaggle_secrets import UserSecretsClient
        try:
            user_secrets = UserSecretsClient()
            wandb.login(key=user_secrets.get_secret("wandb_api_key"))
        except Exception:
            wandb.login()
    print("[START] HS Autobattler: MLP vs Transformer Comparison")
    print(f"[CONFIG] {{TOTAL_TIMESTEPS:,}} timesteps, {{N_ENVS}} envs, device={{DEVICE}}")

    mlp_path = train_mlp()
    trans_path = train_transformer()
    evaluate_pvp(mlp_path, trans_path)

    print("\\n[ALL DONE]")
'''

    with open(kernel_dir / "train_kaggle.py", "w", encoding="utf-8") as f:
        f.write(script)

    print(f"[OK] Kernel prepared ({len(script) / 1024:.0f} KB script)")
    return kernel_dir


def push_to_kaggle():
    """Создаёт и загружает kernel на Kaggle."""
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()

    kernel_dir = create_kernel()
    print("\n[PUSH] Uploading kernel to Kaggle...")
    api.kernels_push(str(kernel_dir))
    print("[OK] Kernel pushed and running!")
    print(f"\n[URL] https://www.kaggle.com/code/{KAGGLE_USERNAME}/{KERNEL_SLUG}")


if __name__ == "__main__":
    push_to_kaggle()
