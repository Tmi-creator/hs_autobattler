"""Submit CleanRL PPO (with optional BC pretrain) to Kaggle GPU kernel.

Embeds src/ + scripts/ + cpp/ + ES weights as base64 inside a generated kernel
script. The kernel:
    1. Builds C++ engine
    2. (optional) Runs scripts/bc_collect.py → bc_train.py
    3. Runs scripts/train_ppo.py [--resume bc_ckpt] with wandb logging

Usage:
    python scripts/kaggle_submit_ppo.py                    # BC + PPO (default)
    python scripts/kaggle_submit_ppo.py --no-bc            # PPO from scratch
    python scripts/kaggle_submit_ppo.py --bc-episodes 10000 --total-timesteps 8000000
    python scripts/kaggle_submit_ppo.py --dry-run
"""

import argparse
import base64
import io
import json
import os
import shutil
import zipfile
from pathlib import Path


def _load_env(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())


root_dir = Path(__file__).resolve().parent.parent
scripts_dir = root_dir / "scripts"
build_dir = root_dir / "scripts" / "_kaggle_build_ppo"

_load_env(root_dir / ".env")

KAGGLE_USERNAME = "tmitmi1999"
KERNEL_SLUG = "hs-autobattler-cleanrl-ppo"

ES_WEIGHTS_PATH = root_dir / "artifacts" / "es_kaggle" / "artifacts" / "best.npz"


def _pack_project_b64() -> str:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        # src/
        for fp in (root_dir / "src").rglob("*"):
            if fp.is_file() and "__pycache__" not in str(fp) and fp.suffix != ".pyc":
                zf.write(fp, str(fp.relative_to(root_dir)))
        # cpp/ (no build/, no _old/)
        for fp in (root_dir / "cpp").rglob("*"):
            if (fp.is_file() and "build" not in fp.parts
                    and "_old" not in fp.parts and "__pycache__" not in str(fp)):
                zf.write(fp, str(fp.relative_to(root_dir)))
        # scripts/ (PPO + BC pipeline)
        for fname in [
            "__init__.py",
            "model.py",
            "train_ppo.py",
            "bc_collect.py",
            "bc_train.py",
            "generate_cpp_effects.py",
        ]:
            fp = scripts_dir / fname
            if fp.exists():
                zf.write(fp, f"scripts/{fname}")
        # ES weights (best.npz, ~92 bytes payload but in zip wrapper)
        if ES_WEIGHTS_PATH.exists():
            zf.write(ES_WEIGHTS_PATH, "artifacts/es_kaggle/artifacts/best.npz")
        # pyproject.toml
        pt = root_dir / "pyproject.toml"
        if pt.exists():
            zf.write(pt, "pyproject.toml")

    data = buf.getvalue()
    print(f"[ZIP] Project packed: {len(data) / 1024:.0f} KB")
    return base64.b64encode(data).decode("ascii")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    # PPO
    p.add_argument("--total-timesteps", type=int, default=5_000_000)
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--n-steps", type=int, default=2048)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--max-tier", type=int, default=6)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval-interval", type=int, default=5)
    p.add_argument("--save-interval", type=int, default=50)
    # BC pretrain
    p.add_argument("--no-bc", action="store_true",
                   help="Skip BC pretrain stage, run PPO from scratch")
    p.add_argument("--bc-episodes", type=int, default=5000)
    p.add_argument("--bc-epochs", type=int, default=15)
    p.add_argument("--bc-batch-size", type=int, default=512)
    p.add_argument("--bc-lr", type=float, default=3e-4)
    # Submit
    p.add_argument("--dry-run", action="store_true")
    return p


def create_kernel(args: argparse.Namespace) -> Path:
    kernel_dir = build_dir / "kernel"
    if kernel_dir.exists():
        shutil.rmtree(kernel_dir)
    kernel_dir.mkdir(parents=True)

    if not args.no_bc and not ES_WEIGHTS_PATH.exists():
        raise FileNotFoundError(
            f"ES weights not found at {ES_WEIGHTS_PATH} — needed for BC pretrain. "
            "Either add the file or pass --no-bc."
        )

    project_b64 = _pack_project_b64()
    wandb_key = os.environ.get("WANDB_API_KEY", "")

    config = {
        "total_timesteps": args.total_timesteps,
        "n_envs": args.n_envs,
        "n_steps": args.n_steps,
        "lr": args.lr,
        "max_tier": args.max_tier,
        "seed": args.seed,
        "eval_interval": args.eval_interval,
        "save_interval": args.save_interval,
        "use_bc": not args.no_bc,
        "bc_episodes": args.bc_episodes,
        "bc_epochs": args.bc_epochs,
        "bc_batch_size": args.bc_batch_size,
        "bc_lr": args.bc_lr,
    }
    config_json = json.dumps(config)

    metadata = {
        "id": f"{KAGGLE_USERNAME}/{KERNEL_SLUG}",
        "title": "HS Autobattler CleanRL PPO",
        "code_file": "train_kaggle_ppo.py",
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

    script = f'''#!/usr/bin/env python3
"""HS Autobattler: CleanRL PPO (+ optional BC pretrain) on Kaggle T4 GPU."""

import base64, io, json, os, subprocess, sys, time, zipfile

PROJECT_B64 = "{project_b64}"
WANDB_KEY = "{wandb_key}"
CFG = json.loads({config_json!r})
PROJECT_DIR = "/kaggle/working/project"
OUTPUT_DIR = "/kaggle/working/artifacts/ppo"
BC_DIR = "/kaggle/working/artifacts/bc"
DATASET_PATH = os.path.join(BC_DIR, "bc_dataset.npz")
BC_CKPT = os.path.join(BC_DIR, "bc_pretrain.pt")
ES_WEIGHTS = os.path.join(PROJECT_DIR, "artifacts", "es_kaggle", "artifacts", "best.npz")
RUN_TAG = int(time.time())

# === Extract ===
_marker = os.path.join(PROJECT_DIR, "pyproject.toml")
if not os.path.exists(_marker):
    print("[SETUP] Extracting...")
    os.makedirs(PROJECT_DIR, exist_ok=True)
    zipfile.ZipFile(io.BytesIO(base64.b64decode(PROJECT_B64))).extractall(PROJECT_DIR)
    print(f"[OK] Extracted to {{PROJECT_DIR}}")

sys.path.insert(0, PROJECT_DIR)
sys.path.insert(0, os.path.join(PROJECT_DIR, "src"))
sys.path.insert(0, os.path.join(PROJECT_DIR, "scripts"))
CPP_BUILD = os.path.join(PROJECT_DIR, "cpp", "build")
if os.path.isdir(CPP_BUILD):
    sys.path.insert(0, CPP_BUILD)


def run(cmd):
    print(f"[RUN] {{' '.join(cmd)}}", flush=True)
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    # === Install deps ===
    subprocess.run([sys.executable, "-m", "pip", "install", "-e", PROJECT_DIR, "-q"], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "numpy", "pybind11", "wandb", "-q"], check=True)

    # === C++ build ===
    import glob
    CPP_DIR = os.path.join(PROJECT_DIR, "cpp")
    if os.path.isdir(CPP_DIR) and not glob.glob(os.path.join(CPP_BUILD, "hs_engine_cpp*")):
        _gen = os.path.join(PROJECT_DIR, "scripts", "generate_cpp_effects.py")
        if os.path.exists(_gen):
            subprocess.run([sys.executable, _gen], check=True)
        os.makedirs(CPP_BUILD, exist_ok=True)
        _cmake = subprocess.check_output([sys.executable, "-m", "pybind11", "--cmakedir"]).decode().strip()
        subprocess.run(["cmake", "-S", CPP_DIR, "-B", CPP_BUILD, f"-Dpybind11_DIR={{_cmake}}", "-DCMAKE_BUILD_TYPE=Release"], check=True)
        subprocess.run(["cmake", "--build", CPP_BUILD, "--config", "Release", "-j4"], check=True)
        if CPP_BUILD not in sys.path:
            sys.path.insert(0, CPP_BUILD)

    # === GPU check ===
    try:
        import torch
        print(f"[GPU] {{torch.cuda.get_device_name(0)}} ({{torch.cuda.get_device_capability()}})")
        _sm = torch.cuda.get_device_capability()
        if _sm < (7, 0):
            subprocess.run([sys.executable, "-m", "pip", "install", "torch==2.4.1", "--index-url", "https://download.pytorch.org/whl/cu118", "-q"], check=True)
    except Exception as e:
        print(f"[GPU] {{e}}")

    # === wandb ===
    try:
        import wandb
        if WANDB_KEY:
            wandb.login(key=WANDB_KEY)
        else:
            try:
                from kaggle_secrets import UserSecretsClient
                wandb.login(key=UserSecretsClient().get_secret("wandb_api_key"))
            except Exception:
                os.environ["WANDB_MODE"] = "offline"
    except ImportError:
        pass

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(BC_DIR, exist_ok=True)

    # === Stage 1: BC collect ===
    if CFG["use_bc"]:
        if not os.path.exists(ES_WEIGHTS):
            print(f"[BC] ES weights missing at {{ES_WEIGHTS}} — falling back to PPO from scratch")
            CFG["use_bc"] = False
        else:
            print(f"\\n=== [STAGE 1/3] BC trajectory collection ({{CFG['bc_episodes']}} eps) ===")
            run([
                sys.executable, os.path.join(PROJECT_DIR, "scripts", "bc_collect.py"),
                "--weights", ES_WEIGHTS,
                "--episodes", str(CFG["bc_episodes"]),
                "--max-tier", str(CFG["max_tier"]),
                "--seed", str(CFG["seed"]),
                "--out", DATASET_PATH,
                "--log-every", "100",
            ])

    # === Stage 2: BC train ===
    if CFG["use_bc"]:
        print(f"\\n=== [STAGE 2/3] BC pretrain ({{CFG['bc_epochs']}} epochs) ===")
        run([
            sys.executable, os.path.join(PROJECT_DIR, "scripts", "bc_train.py"),
            "--dataset", DATASET_PATH,
            "--out", BC_CKPT,
            "--epochs", str(CFG["bc_epochs"]),
            "--batch-size", str(CFG["bc_batch_size"]),
            "--lr", str(CFG["bc_lr"]),
            "--max-tier", str(CFG["max_tier"]),
            "--seed", str(CFG["seed"]),
            "--wandb",
            "--wandb-project", "hs_autobattler",
            "--run-name", f"bc_{{RUN_TAG}}",
        ])

    # === Stage 3: PPO ===
    stage_label = "PPO from BC" if CFG["use_bc"] and os.path.exists(BC_CKPT) else "PPO from scratch"
    print(f"\\n=== [STAGE 3/3] {{stage_label}} ===")
    ppo_cmd = [
        sys.executable, os.path.join(PROJECT_DIR, "scripts", "train_ppo.py"),
        "--wandb",
        "--wandb-project", "hs_autobattler",
        "--run-name", f"{{'ppo_from_bc' if CFG['use_bc'] else 'ppo_scratch'}}_{{RUN_TAG}}",
        "--out-dir", OUTPUT_DIR,
        "--total-timesteps", str(CFG["total_timesteps"]),
        "--n-envs", str(CFG["n_envs"]),
        "--n-steps", str(CFG["n_steps"]),
        "--lr", str(CFG["lr"]),
        "--max-tier", str(CFG["max_tier"]),
        "--seed", str(CFG["seed"]),
        "--eval-interval", str(CFG["eval_interval"]),
        "--save-interval", str(CFG["save_interval"]),
    ]
    if CFG["use_bc"] and os.path.exists(BC_CKPT):
        ppo_cmd.extend(["--resume", BC_CKPT])
    run(ppo_cmd)
    print("[DONE]")
'''

    with open(kernel_dir / "train_kaggle_ppo.py", "w", encoding="utf-8") as f:
        f.write(script)

    print(f"[OK] Kernel prepared ({len(script) / 1024:.0f} KB script)")
    return kernel_dir


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.dry_run:
        kernel_dir = create_kernel(args)
        print(f"[DRY-RUN] {kernel_dir}")
        return

    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    kernel_dir = create_kernel(args)
    print("\n[PUSH] Uploading...")
    api.kernels_push(str(kernel_dir))
    print(f"[OK] https://www.kaggle.com/code/{KAGGLE_USERNAME}/{KERNEL_SLUG}")


if __name__ == "__main__":
    main()
