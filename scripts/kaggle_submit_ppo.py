"""Submit CleanRL PPO training to Kaggle GPU kernel.

Embeds src/ + scripts/ + cpp/ as base64 inside a generated kernel script.
The kernel builds C++ engine, then runs scripts/train_ppo.py with wandb logging.

Usage:
    python scripts/kaggle_submit_ppo.py
    python scripts/kaggle_submit_ppo.py --total-timesteps 2000000 --n-envs 4
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
        # scripts/ (only what PPO needs)
        for fname in [
            "__init__.py", "model.py", "train_ppo.py", "generate_cpp_effects.py",
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


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--total-timesteps", type=int, default=5_000_000)
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--n-steps", type=int, default=2048)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--max-tier", type=int, default=6)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval-interval", type=int, default=5)
    p.add_argument("--save-interval", type=int, default=50)
    p.add_argument("--dry-run", action="store_true")
    return p


def create_kernel(args: argparse.Namespace) -> Path:
    kernel_dir = build_dir / "kernel"
    if kernel_dir.exists():
        shutil.rmtree(kernel_dir)
    kernel_dir.mkdir(parents=True)

    project_b64 = _pack_project_b64()
    wandb_key = os.environ.get("WANDB_API_KEY", "")

    ppo_args = {
        "total_timesteps": args.total_timesteps,
        "n_envs": args.n_envs,
        "n_steps": args.n_steps,
        "lr": args.lr,
        "max_tier": args.max_tier,
        "seed": args.seed,
        "eval_interval": args.eval_interval,
        "save_interval": args.save_interval,
    }
    ppo_args_json = json.dumps(ppo_args)

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
"""HS Autobattler: CleanRL PPO on Kaggle T4 GPU."""

import base64, io, json, os, subprocess, sys, time, zipfile

PROJECT_B64 = "{project_b64}"
WANDB_KEY = "{wandb_key}"
PPO_ARGS = json.loads({ppo_args_json!r})
PROJECT_DIR = "/kaggle/working/project"
OUTPUT_DIR = "/kaggle/working/artifacts/ppo"

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

    # === Run PPO ===
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    cmd = [
        sys.executable, os.path.join(PROJECT_DIR, "scripts", "train_ppo.py"),
        "--wandb",
        "--wandb-project", "hs_autobattler",
        "--run-name", f"cleanrl_ppo_{{int(time.time())}}",
        "--out-dir", OUTPUT_DIR,
        "--total-timesteps", str(PPO_ARGS["total_timesteps"]),
        "--n-envs", str(PPO_ARGS["n_envs"]),
        "--n-steps", str(PPO_ARGS["n_steps"]),
        "--lr", str(PPO_ARGS["lr"]),
        "--max-tier", str(PPO_ARGS["max_tier"]),
        "--seed", str(PPO_ARGS["seed"]),
        "--eval-interval", str(PPO_ARGS["eval_interval"]),
        "--save-interval", str(PPO_ARGS["save_interval"]),
    ]
    print(f"[RUN] {{' '.join(cmd)}}")
    subprocess.run(cmd, check=True)
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
