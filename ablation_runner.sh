#!/usr/bin/env bash
# Automated Ablation Study Runner for Hearthstone Battlegrounds RL Agent.
# Designed for servers with 2x A100 GPUs and multi-core CPUs.
#
# Runs 6 experiments in parallel pairs (one on GPU 0, one on GPU 1)
# to evaluate all architectural extensions and the genetics-based BC pretraining.
#
# Usage:
#   chmod +x ablation_runner.sh
#   ./ablation_runner.sh

set -e

echo "=========================================================="
echo "    Hearthstone Battlegrounds RL: Ablation Runner         "
echo "=========================================================="

# Export Wandb key for automated beautiful charting
export WANDB_API_KEY="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"

# Export CUDA and NVIDIA library paths to prevent CUDA driver loading failures in docker/virtualized environments
export LD_LIBRARY_PATH="/usr/local/cuda/compat:/usr/local/cuda/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:$LD_LIBRARY_PATH"

# 1. System & Python dependency installation with uv
echo ">>> [1/5] Setting up virtual environment with uv..."
if ! command -v uv &> /dev/null; then
    echo "    - uv not found, installing via pip..."
    pip install uv --quiet
fi

uv venv .venv
source .venv/bin/activate

echo ">>> Installing Python requirements..."
uv pip install pybind11
# Install PyTorch compiled with CUDA 11.8 to match older server GPU drivers
uv pip install --force-reinstall --no-cache torch --index-url https://download.pytorch.org/whl/cu118
uv pip install -e .

# 2. Verify CUDA availability
echo ">>> [2/5] Checking CUDA availability..."
CUDA_OK=$(python -c "import torch; print(torch.cuda.is_available())")
if [ "$CUDA_OK" != "True" ]; then
    echo "=========================================================="
    echo "  CRITICAL ERROR: PyTorch cannot initialize CUDA!"
    echo "  - torch.cuda.is_available() returned False."
    echo "  - Please inspect the output of:"
    echo "    python -c 'import torch; torch.cuda.init()'"
    echo "  - Make sure your NVIDIA drivers are active and accessible."
    echo "=========================================================="
    exit 1
fi
echo "    - CUDA is available! Found GPU devices."

# 3. Compile C++ engine
echo ">>> [3/5] Compiling accelerated C++ combat engine..."
python scripts/generate_cpp_effects.py
cmake -S cpp -B cpp/build \
      -DCMAKE_BUILD_TYPE=Release \
      -DPYTHON_EXECUTABLE="$(which python)" \
      -Dpybind11_DIR="$(python -m pybind11 --cmakedir)"
cmake --build cpp/build --config Release -j$(nproc)

# 4. Behavior Cloning pre-train dataset generation (if needed for the genetics run)
echo ">>> [4/5] Checking BC/Genetics dataset..."
mkdir -p artifacts/bc
if [ ! -f artifacts/bc/bc_pretrain.pt ]; then
    echo "    - BC pretrain checkpoint not found. Collecting and training BC model first..."
    python scripts/bc_collect.py --episodes 5000 --weights artifacts/es_kaggle/artifacts/best.npz --out artifacts/bc/bc_dataset.npz
    python scripts/bc_train.py --dataset artifacts/bc/bc_dataset.npz --out artifacts/bc/bc_pretrain.pt --epochs 15
else
    echo "    - Found existing BC pretrain checkpoint at artifacts/bc/bc_pretrain.pt"
fi

# 4. Determine environment scaling
N_CORES=$(nproc)
# Each of the 2 concurrent training runs gets half of the available CPU cores (min 4)
N_ENVS=$(( N_CORES / 2 ))
if [ "$N_ENVS" -lt 4 ]; then
    N_ENVS=4
fi

echo ">>> [5/5] Starting Ablation matrix..."
echo "    - CPU cores: $N_CORES (assigning n-envs=$N_ENVS per parallel run)"
echo "    - Total experiments: 6 (running in 3 concurrent rounds on GPU 0 and GPU 1)"
echo "    - Charts will be synced to Wandb project: hs_autobattler"

# Shared model architecture parameters for "Full" runs (with periodic checkpoint save every 25 updates)
ARCH_FLAGS="--d-model 256 --n-heads 8 --n-layers 6 --memory-size 8 --use-enemy-board-obs --use-player-status-obs --use-summary-tokens --use-memory --save-interval 25"

# -------------------------------------------------------------------------
# ROUND 1: Full Scratch vs. Full BC (Genetics Pre-trained)
# -------------------------------------------------------------------------
echo "=========================================================="
echo " ROUND 1: Full Scratch vs. Full BC (Genetics)             "
echo "=========================================================="

# Run 1 on GPU 0: Full architecture from scratch (10M steps)
CUDA_VISIBLE_DEVICES=0 python scripts/train_ppo.py \
    --n-envs "$N_ENVS" \
    --total-timesteps 10000000 \
    $ARCH_FLAGS \
    --wandb \
    --run-name "full_scratch" &
PID1=$!

# Run 2 on GPU 1: Full architecture initialized with BC from ES Bot (10M steps)
CUDA_VISIBLE_DEVICES=1 python scripts/train_ppo.py \
    --n-envs "$N_ENVS" \
    --total-timesteps 10000000 \
    --resume artifacts/bc/bc_pretrain.pt \
    $ARCH_FLAGS \
    --wandb \
    --run-name "full_bc_genetics" &
PID2=$!

# Wait for both to finish before starting Round 2
wait $PID1
wait $PID2
echo ">>> Round 1 complete."

# -------------------------------------------------------------------------
# ROUND 2: Baseline vs. Ablation: No Memory
# -------------------------------------------------------------------------
echo "=========================================================="
echo " ROUND 2: Baseline Scratch vs. Ablation (No Memory)      "
echo "=========================================================="

# Run 3 on GPU 0: Base PPO (no ST, no M, no EB, no PS)
CUDA_VISIBLE_DEVICES=0 python scripts/train_ppo.py \
    --n-envs "$N_ENVS" \
    --total-timesteps 10000000 \
    --d-model 256 --n-heads 8 --n-layers 6 \
    --save-interval 25 \
    --wandb \
    --run-name "base_scratch" &
PID3=$!

# Run 4 on GPU 1: Full minus Memory (ST + EB + PS active, Memory off)
CUDA_VISIBLE_DEVICES=1 python scripts/train_ppo.py \
    --n-envs "$N_ENVS" \
    --total-timesteps 10000000 \
    --d-model 256 --n-heads 8 --n-layers 6 \
    --use-enemy-board-obs --use-player-status-obs --use-summary-tokens \
    --save-interval 25 \
    --wandb \
    --run-name "ablation_no_memory" &
PID4=$!

wait $PID3
wait $PID4
echo ">>> Round 2 complete."

# -------------------------------------------------------------------------
# ROUND 3: Ablations: No Enemy Board vs. No Player Status
# -------------------------------------------------------------------------
echo "=========================================================="
echo " ROUND 3: Ablation (No Enemy Board) vs. Ablation (No Status)"
echo "=========================================================="

# Run 5 on GPU 0: Full minus Enemy Board snapshot (ST + M + PS active)
CUDA_VISIBLE_DEVICES=0 python scripts/train_ppo.py \
    --n-envs "$N_ENVS" \
    --total-timesteps 10000000 \
    --d-model 256 --n-heads 8 --n-layers 6 --memory-size 8 \
    --use-player-status-obs --use-summary-tokens --use-memory \
    --save-interval 25 \
    --wandb \
    --run-name "ablation_no_enemy_board" &
PID5=$!

# Run 6 on GPU 1: Full minus Player Status history (ST + M + EB active)
CUDA_VISIBLE_DEVICES=1 python scripts/train_ppo.py \
    --n-envs "$N_ENVS" \
    --total-timesteps 10000000 \
    --d-model 256 --n-heads 8 --n-layers 6 --memory-size 8 \
    --use-enemy-board-obs --use-summary-tokens --use-memory \
    --save-interval 25 \
    --wandb \
    --run-name "ablation_no_status" &
PID6=$!

wait $PID5
wait $PID6
echo "=========================================================="
echo "    All Ablation Studies Completed Successfully!          "
echo "=========================================================="
