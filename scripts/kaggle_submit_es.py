"""Submit ES-bot evolution to Kaggle as a self-contained CPU kernel.

Embeds src/, cpp/ sources, scripts/evolve_bot.py and friends as base64 inside
a generated ``train_kaggle_es.py`` script. No dataset dependencies. The kernel:

  1. Extracts the embedded project zip.
  2. Pip-installs the project + numpy + pybind11.
  3. Generates + builds the C++ combat engine (same as main submit).
  4. Runs ``scripts.evolve_bot.run_evolution`` with the baked-in args.
  5. Saves ``artifacts/es_bot/{preset}/best.npz`` + fitness history as kernel output.

Usage:
    python scripts/kaggle_submit_es.py                         # defaults (full, 50 gens)
    python scripts/kaggle_submit_es.py --preset micro --generations 100
    python scripts/kaggle_submit_es.py --preset full --generations 50 --mu 25 --lam 25
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


_load_env(Path(__file__).resolve().parent.parent / ".env")

root_dir = Path(__file__).resolve().parent.parent
scripts_dir = root_dir / "scripts"
build_dir = root_dir / "scripts" / "_kaggle_build_es"

KAGGLE_USERNAME = "tmitmi1999"
KERNEL_SLUG = "hs-autobattler-es-evolution"


def _pack_project_b64() -> str:
    """Packs src/ + cpp/ sources + the subset of scripts needed for ES evolution."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        # src/hearthstone/
        src_root = root_dir / "src"
        for fp in src_root.rglob("*"):
            if fp.is_file() and "__pycache__" not in str(fp) and fp.suffix != ".pyc":
                zf.write(fp, str(fp.relative_to(root_dir)))

        # cpp/ (source only — no build artifacts)
        cpp_root = root_dir / "cpp"
        for fp in cpp_root.rglob("*"):
            if (
                fp.is_file()
                and "build" not in fp.parts
                and "_old" not in fp.parts
                and "__pycache__" not in str(fp)
            ):
                zf.write(fp, str(fp.relative_to(root_dir)))

        # scripts/ — only what ES evolution needs
        for fname in [
            "__init__.py",
            "evolve_bot.py",
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


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Submit ES evolution to Kaggle")
    p.add_argument("--generations", type=int, default=500)
    p.add_argument("--mu", type=int, default=50)
    p.add_argument("--lam", type=int, default=50)
    p.add_argument("--n-match", type=int, default=40)
    p.add_argument("--n-anchor", type=int, default=8)
    p.add_argument("--n-hof", type=int, default=0,
                   help="games vs Hall of Fame per agent (0=disabled)")
    p.add_argument("--hof-interval", type=int, default=10)
    p.add_argument("--bench-interval", type=int, default=20,
                   help="benchmark vs SmartBot every N gens (wandb)")
    p.add_argument("--final-eval", type=int, default=500)
    p.add_argument("--max-tier", type=int, default=6)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=0,
                   help="0 = use cpu_count() inside the kernel at runtime")
    p.add_argument("--dry-run", action="store_true",
                   help="build the kernel script but don't push to Kaggle")
    p.add_argument("--no-internet", dest="enable_internet", action="store_false",
                   default=True,
                   help="disable kernel internet (pip install will fail)")
    return p


def create_kernel(args: argparse.Namespace) -> Path:
    """Generate the Kaggle kernel directory from the current args."""
    kernel_dir = build_dir / "kernel"
    if kernel_dir.exists():
        shutil.rmtree(kernel_dir)
    kernel_dir.mkdir(parents=True)

    project_b64 = _pack_project_b64()
    wandb_key = os.environ.get("WANDB_API_KEY", "")

    metadata = {
        "id": f"{KAGGLE_USERNAME}/{KERNEL_SLUG}",
        "title": "HS Autobattler ES Evolution",
        "code_file": "train_kaggle_es.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": False,
        "enable_internet": bool(args.enable_internet),
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
    }
    with open(kernel_dir / "kernel-metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    evo_args = {
        "generations": args.generations,
        "mu": args.mu,
        "lam": args.lam,
        "n_match": args.n_match,
        "n_anchor": args.n_anchor,
        "n_hof": args.n_hof,
        "hof_interval": args.hof_interval,
        "final_eval": args.final_eval,
        "max_tier": args.max_tier,
        "seed": args.seed,
        "workers": args.workers,
        "bench_interval": args.bench_interval,
    }
    evo_args_json = json.dumps(evo_args)

    script = f'''#!/usr/bin/env python3
"""HS Autobattler: ES Bot Evolution on Kaggle CPU kernel.

Self-contained: project source is embedded as base64 zip below.
Runs scripts.evolve_bot.run_evolution with pre-baked hyperparameters.
"""

import base64
import io
import json
import os
import shutil
import subprocess
import sys
import time
import types
import zipfile

# === 1. Constants (baked in at submit time) ===
PROJECT_B64 = "{project_b64}"
EVO_ARGS = json.loads({evo_args_json!r})
WANDB_KEY = "{wandb_key}"
PROJECT_DIR = "/kaggle/working/project"
OUTPUT_DIR = "/kaggle/working/outputs"
ARTIFACTS_DIR = "/kaggle/working/artifacts"

# === 2. Extract project (module level so workers also see unzipped files) ===
_marker = os.path.join(PROJECT_DIR, "pyproject.toml")
if not os.path.exists(_marker):
    print("[SETUP] Extracting embedded project...")
    os.makedirs(PROJECT_DIR, exist_ok=True)
    zip_data = base64.b64decode(PROJECT_B64)
    with zipfile.ZipFile(io.BytesIO(zip_data), "r") as zf:
        zf.extractall(PROJECT_DIR)
    print(f"[OK] Extracted to {{PROJECT_DIR}}")

sys.path.insert(0, PROJECT_DIR)
sys.path.insert(0, os.path.join(PROJECT_DIR, "src"))

CPP_BUILD = os.path.join(PROJECT_DIR, "cpp", "build")
if os.path.isdir(CPP_BUILD):
    sys.path.insert(0, CPP_BUILD)

# === 3. Main-process-only setup: pip install + C++ build ===
if __name__ == "__main__":
    print("[SETUP] Installing project + deps...")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-e", PROJECT_DIR, "-q"],
        check=True,
    )
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "numpy", "pybind11", "wandb", "-q"],
        check=True,
    )

    # C++ combat engine ------------------------------------------------
    import glob
    CPP_DIR = os.path.join(PROJECT_DIR, "cpp")
    _so_files = glob.glob(os.path.join(CPP_BUILD, "hs_engine_cpp*"))
    if os.path.isdir(CPP_DIR) and not _so_files:
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
        if CPP_BUILD not in sys.path:
            sys.path.insert(0, CPP_BUILD)

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# === 4. wandb login ===
if __name__ == "__main__":
    use_wandb = True
    try:
        import wandb
        if WANDB_KEY:
            wandb.login(key=WANDB_KEY)
        else:
            try:
                from kaggle_secrets import UserSecretsClient
                wandb.login(key=UserSecretsClient().get_secret("wandb_api_key"))
            except Exception as _e:
                print(f"[WARN] No WANDB key ({{_e}}); running wandb offline")
                os.environ["WANDB_MODE"] = "offline"
    except ImportError:
        print("[WARN] wandb not installed; skipping")
        use_wandb = False

# === 5. Run evolution ===
if __name__ == "__main__":
    from scripts.evolve_bot import run_evolution

    workers = EVO_ARGS.get("workers", 0) or max(1, (os.cpu_count() or 2) - 1)
    print(f"[ES] Using {{workers}} workers ({{os.cpu_count()}} CPUs available)")

    args = types.SimpleNamespace(
        generations=EVO_ARGS["generations"],
        mu=EVO_ARGS["mu"],
        lam=EVO_ARGS["lam"],
        n_match=EVO_ARGS["n_match"],
        n_anchor=EVO_ARGS["n_anchor"],
        n_hof=EVO_ARGS.get("n_hof", 0),
        hof_interval=EVO_ARGS.get("hof_interval", 10),
        final_eval=EVO_ARGS["final_eval"],
        max_tier=EVO_ARGS["max_tier"],
        seed=EVO_ARGS["seed"],
        out_dir=ARTIFACTS_DIR,
        workers=workers,
        wandb=use_wandb,
        wandb_project="hs_autobattler_es",
        bench_interval=EVO_ARGS.get("bench_interval", 20),
        wandb_name=f"kaggle_es_v2_{{int(time.time())}}",
        quick=False,
    )

    print(f"[START] HS Autobattler ES evolution")
    print(f"[CONFIG] {{json.dumps(EVO_ARGS, indent=2)}}")
    start = time.time()
    run_evolution(args)
    elapsed = time.time() - start
    print(f"[DONE] Evolution finished in {{elapsed/60:.1f}} min")

    # List saved artifacts
    if os.path.isdir(ARTIFACTS_DIR):
        print(f"[OUTPUT] Artifacts saved in {{ARTIFACTS_DIR}}")
        for fn in sorted(os.listdir(ARTIFACTS_DIR)):
            fp = os.path.join(ARTIFACTS_DIR, fn)
            print(f"  {{fn}}: {{os.path.getsize(fp)}} bytes")
'''

    script_path = kernel_dir / "train_kaggle_es.py"
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(script)

    print(f"[OK] Kernel prepared ({len(script) / 1024:.0f} KB script)")
    return kernel_dir


def push_to_kaggle(args: argparse.Namespace) -> None:
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()

    kernel_dir = create_kernel(args)
    print("\n[PUSH] Uploading kernel to Kaggle...")
    api.kernels_push(str(kernel_dir))
    print("[OK] Kernel pushed and running!")
    print(f"\n[URL] https://www.kaggle.com/code/{KAGGLE_USERNAME}/{KERNEL_SLUG}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.dry_run:
        kernel_dir = create_kernel(args)
        print(f"[DRY-RUN] Kernel built at {kernel_dir}")
        print(f"[DRY-RUN] Review scripts/_kaggle_build_es/kernel/train_kaggle_es.py "
              f"before pushing.")
        return
    push_to_kaggle(args)


if __name__ == "__main__":
    main()
