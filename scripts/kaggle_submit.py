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


def _load_env(path: Path) -> None:
    """Minimal .env loader — no external deps needed."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())


_load_env(Path(__file__).resolve().parent.parent / ".env")


root_dir = Path(__file__).resolve().parent.parent
scripts_dir = root_dir / "scripts"
build_dir = root_dir / "scripts" / "_kaggle_build"

KAGGLE_USERNAME = "tmitmi1999"
KERNEL_SLUG = "hs-autobattler-training"


def _pack_project_b64() -> str:
    """Пакует src/ + scripts/ + cpp/ + pyproject.toml в zip, возвращает base64."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        # src/hearthstone/
        src_root = root_dir / "src"
        for fp in src_root.rglob("*"):
            if fp.is_file() and "__pycache__" not in str(fp) and not fp.suffix == ".pyc":
                zf.write(fp, str(fp.relative_to(root_dir)))

        # cpp/ (source only — no build artifacts)
        cpp_root = root_dir / "cpp"
        for fp in cpp_root.rglob("*"):
            if fp.is_file() and "build" not in fp.parts and "__pycache__" not in str(fp):
                zf.write(fp, str(fp.relative_to(root_dir)))

        # scripts/ (only needed files)
        for fname in [
            "__init__.py",
            "trans.py",
            "categorical_critic.py",
            "callbacks.py",
            "train.py",
            "train_transformer.py",
            "evaluate_pvp.py",
            "visualize_attention.py",
            "generate_cpp_effects.py",
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
    wandb_key = os.environ.get("WANDB_API_KEY", "")

    # kernel-metadata.json
    metadata = {
        "id": f"{KAGGLE_USERNAME}/{KERNEL_SLUG}",
        "title": "HS Autobattler Training",
        "code_file": "train_kaggle.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "accelerator": "gpu-t4",
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
WANDB_KEY = "{wandb_key}"
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

# C++ build dir (may be built by main process, workers import from it)
CPP_BUILD = os.path.join(PROJECT_DIR, "cpp", "build")
if os.path.isdir(CPP_BUILD):
    sys.path.insert(0, CPP_BUILD)

# pip install only in main process (children inherit installed packages)
if __name__ == "__main__":
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-e", PROJECT_DIR, "-q"],
        check=True,
    )
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "sb3-contrib", "wandb", "pybind11", "-q"],
        check=True,
    )

    # === 2.5. Generate C++ effects + Build C++ combat engine ===
    import glob
    CPP_DIR = os.path.join(PROJECT_DIR, "cpp")
    _so_files = glob.glob(os.path.join(CPP_BUILD, "hs_engine_cpp*"))
    if os.path.isdir(CPP_DIR) and not _so_files:
        # Generate C++ effects from Python card_def.py
        _gen_script = os.path.join(PROJECT_DIR, "scripts", "generate_cpp_effects.py")
        if os.path.exists(_gen_script):
            print("[C++ CODEGEN] Generating effects from card_def.py...")
            subprocess.run([sys.executable, _gen_script], check=True)

        print("[C++ BUILD] Compiling combat engine...")
        os.makedirs(CPP_BUILD, exist_ok=True)
        _cmakedir = subprocess.check_output(
            [sys.executable, "-m", "pybind11", "--cmakedir"]
        ).decode().strip()
        subprocess.run([
            "cmake", "-S", CPP_DIR, "-B", CPP_BUILD,
            f"-Dpybind11_DIR={{_cmakedir}}",
            "-DCMAKE_BUILD_TYPE=Release",
        ], check=True)
        subprocess.run([
            "cmake", "--build", CPP_BUILD, "--config", "Release", "-j4",
        ], check=True)
        print("[C++ BUILD] Done!")
        # Re-add build dir now that .so exists
        if CPP_BUILD not in sys.path:
            sys.path.insert(0, CPP_BUILD)

     # === 2.7. Check GPU compatibility with PyTorch ===
    try:
        _nvsmi = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            text=True,
        ).strip()
        _major, _minor = _nvsmi.split(".")
        _sm = (int(_major), int(_minor))
        print(f"[GPU] Detected compute capability: sm_{{_major}}{{_minor}}")
        if _sm < (7, 0):
            print("[FIX] P100 detected — installing PyTorch 2.4.1+cu118 (last version with sm_60)...")
            subprocess.run([
                sys.executable, "-m", "pip", "install",
                "torch==2.4.1", "--index-url", "https://download.pytorch.org/whl/cu118",
                "-q",
            ], check=True)
            print("[FIX] Done!")
    except Exception as _e:
        print(f"[GPU] nvidia-smi check skipped: {{_e}}")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# === 3. Imports (now safe) ===
import numpy as np
import torch
from typing import cast
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv
from wandb.integration.sb3 import WandbCallback

from hearthstone.env.hs_env import HearthstoneEnv
from scripts.trans import TransformerFeaturesExtractor
from scripts.categorical_critic import CategoricalMaskablePPO, CategoricalValuePolicy
from hearthstone.env.ghost_pool import GhostPool
from scripts.callbacks import (
    BoardPowerCallback, CurriculumCallback, EntropyDecayCallback, GameLoggerCallback
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[DEVICE] Using: {{DEVICE}}")
if DEVICE == "cuda":
    print(f"[GPU] {{torch.cuda.get_device_name(0)}}")

SEED = 42
N_ENVS = 8
TOTAL_TIMESTEPS = 4_500_000


def setup_determinism(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    set_random_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def mask_fn(base_env):
    return np.asarray(cast(HearthstoneEnv, base_env).action_masks(), dtype=bool)




GHOST_POOL = GhostPool(max_games=2000)
GHOST_POOL_PATH = os.path.join(OUTPUT_DIR, "ghost_pool.pkl")

# Load previous session's boards if available
_loaded = GHOST_POOL.load(GHOST_POOL_PATH)
if _loaded > 0:
    print(f"[GHOST] Loaded {{_loaded}} games from previous session")
else:
    print("[GHOST] No previous ghost pool found, starting fresh")

def make_env(rank, seed=42):
    def _init():
        env = HearthstoneEnv()
        env.set_ghost_pool(GHOST_POOL)
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

    env = DummyVecEnv([make_env(i, SEED) for i in range(N_ENVS)])

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
            CheckpointCallback(
                save_freq=100_000 // N_ENVS,
                save_path=models_dir,
                name_prefix="mlp",
            ),
            WandbCallback(gradient_save_freq=500, verbose=0),
            GameLoggerCallback(
                check_freq=50000, log_dir=logs_dir
            ),
            CurriculumCallback(
                ghost_start_step=200_000,
                pool_preloaded=(_loaded > 0),
            ),
            BoardPowerCallback(log_freq=2000),
        ],
    )
    elapsed = time.time() - start
    print(f"[MLP] Done in {{elapsed:.0f}}s ({{TOTAL_TIMESTEPS/elapsed:.0f}} steps/sec)")

    final_path = os.path.join(OUTPUT_DIR, "models", "mlp_final")
    model.save(final_path)

    # Persist ghost pool for future sessions / transformer phase
    GHOST_POOL.save(GHOST_POOL_PATH)
    print(f"[GHOST] Saved {{GHOST_POOL.size}} games to disk")

    run.finish()
    env.close()
    return final_path + ".zip"


# =========================================
# Train Transformer (Categorical Critic + Card Embeddings)
# =========================================
def train_transformer():
    print("\\n" + "=" * 60)
    print("Training Transformer (Categorical Critic + Card Embeddings)")
    print("=" * 60)
    setup_determinism(SEED)

    # Get num_card_ids from env
    _tmp = HearthstoneEnv()
    num_card_ids = _tmp.num_card_ids
    del _tmp
    print(f"[CARDS] Card vocabulary: {{num_card_ids}} unique IDs")

    run = wandb.init(
        project="hs_autobattler_comparison",
        name="transformer_categorical",
        config={{
            "model": "Transformer+CategoricalCritic",
            "timesteps": TOTAL_TIMESTEPS,
            "n_envs": N_ENVS,
            "d_model": 128, "n_heads": 4, "n_layers": 4,
            "num_card_ids": num_card_ids,
            "critic": "two_hot_255bins",
        }},
        sync_tensorboard=True,
    )

    models_dir = os.path.join(OUTPUT_DIR, "models", "transformer", run.id)
    logs_dir = os.path.join(OUTPUT_DIR, "logs", "transformer", run.id)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    env = DummyVecEnv([make_env(i, SEED) for i in range(N_ENVS)])

    policy_kwargs = dict(
        features_extractor_class=TransformerFeaturesExtractor,
        features_extractor_kwargs=dict(
            d_model=128, n_heads=4, n_layers=4, d_context=10,
            num_card_ids=num_card_ids,
        ),
        net_arch=dict(pi=[128], vf=[128]),
    )

    tb_log = os.path.join(OUTPUT_DIR, "tb_logs", "transformer")
    model = CategoricalMaskablePPO(
        CategoricalValuePolicy, env, verbose=1,
        learning_rate=3e-4, gamma=0.999, batch_size=256,
        n_steps=2048, ent_coef=0.04,
        policy_kwargs=policy_kwargs,
        seed=SEED, device=DEVICE,
        tensorboard_log=tb_log,
    )

    # torch.compile — skip on old GPUs (P100/sm_60 not supported by triton)
    try:
        import torch._dynamo
        _sm = torch.cuda.get_device_capability()
        if _sm >= (7, 0):
            model.policy = torch.compile(model.policy)
            print(f"[COMPILE] torch.compile applied (sm_{{_sm[0]}}{{_sm[1]}})")
        else:
            print(f"[COMPILE] Skipped — GPU sm_{{_sm[0]}}{{_sm[1]}} < 7.0")
    except Exception as e:
        print(f"[COMPILE] Skipped: {{e}}")

    ext = model.policy.features_extractor
    n_params = sum(p.numel() for p in ext.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.policy.parameters() if p.requires_grad)
    print(f"[MODEL] Transformer: {{n_params:,}} extractor, {{total_params:,}} total")
    print(f"[MODEL] Critic: Categorical Two-Hot (255 bins, symlog [-20, +20])")
    print(f"[MODEL] Card Embedding: nn.Embedding({{num_card_ids}}, 64)")

    start = time.time()
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[
            CheckpointCallback(
                save_freq=100_000 // N_ENVS,
                save_path=models_dir,
                name_prefix="trans",
            ),
            WandbCallback(gradient_save_freq=500, verbose=0),
            GameLoggerCallback(
                check_freq=50000, log_dir=logs_dir
            ),
            CurriculumCallback(
                ghost_start_step=200_000,
                pool_preloaded=(_loaded > 0),
            ),
            BoardPowerCallback(log_freq=2000),
            EntropyDecayCallback(
                ent_coef_start=0.04, ent_coef_end=0.01, decay_fraction=0.75
            ),
        ],
    )
    elapsed = time.time() - start
    print(f"[TRANS] Done in {{elapsed:.0f}}s ({{TOTAL_TIMESTEPS/elapsed:.0f}} steps/sec)")

    final_path = os.path.join(OUTPUT_DIR, "models", "transformer_final")
    model.save(final_path)

    GHOST_POOL.save(GHOST_POOL_PATH)
    print(f"[GHOST] Saved {{GHOST_POOL.size}} games")

    run.finish()
    env.close()
    return final_path + ".zip"


# =========================================
# MAIN
# =========================================
if __name__ == "__main__":
    import wandb
    if WANDB_KEY:
        wandb.login(key=WANDB_KEY)
    else:
        try:
            from kaggle_secrets import UserSecretsClient
            wandb.login(key=UserSecretsClient().get_secret("wandb_api_key"))
        except Exception:
            print("[WARN] No WANDB key — logging disabled")
            os.environ["WANDB_MODE"] = "offline"
    print("[START] HS Autobattler: Transformer + Categorical Critic")
    print(f"[CONFIG] {{TOTAL_TIMESTEPS:,}} timesteps, {{N_ENVS}} envs, device={{DEVICE}}")

    train_transformer()

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
