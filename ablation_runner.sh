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

# 1. System & Python dependency installation
echo ">>> [1/4] Checking dependencies..."
pip install pybind11 --quiet
pip install -e . --quiet

# 2. Compile C++ engine
echo ">>> [2/4] Compiling accelerated C++ combat engine..."
python scripts/generate_cpp_effects.py
cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Release
cmake --build cpp/build --config Release -j$(nproc)

# 3. Behavior Cloning pre-train dataset generation (if needed for the genetics run)
echo ">>> [3/4] Checking BC/Genetics dataset..."
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

echo ">>> [4/4] Starting Ablation matrix..."
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
