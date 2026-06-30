#!/usr/bin/env bash
# Auto-setup and train script for Hearthstone RL agent.
# Optimized for high-performance multi-GPU/multi-core VM deployments (e.g. 2x A100).
#
# Usage:
#   chmod +x setup_and_train.sh
#   ./setup_and_train.sh
#   ./setup_and_train.sh --wandb --run-name my_a100_run

# Exit immediately if any command fails
set -e

echo "============================================="
echo "  Hearthstone Battlegrounds RL Auto-Setup    "
echo "============================================="

# 1. System Dependencies (GCC/G++, CMake)
if [ -f /etc/debian_version ]; then
    echo ">>> [1/4] Installing system dependencies (gcc, g++, cmake)..."
    sudo apt-get update -y
    sudo apt-get install -y build-essential cmake python3-dev
else
    echo ">>> [1/4] Non-Debian system detected. Please ensure gcc/g++ and cmake are installed."
fi

# 2. Python package setup
echo ">>> [2/4] Installing Python requirements..."
pip install --upgrade pip
pip install pybind11
pip install -e .

# 3. C++ Combat Engine compilation
echo ">>> [3/4] Generating C++ effects & compiling engine..."
python scripts/generate_cpp_effects.py
cmake -S cpp -B cpp/build \
      -DCMAKE_BUILD_TYPE=Release \
      -DPYTHON_EXECUTABLE="$(which python)" \
      -Dpybind11_DIR="$(python -m pybind11 --cmakedir)"
cmake --build cpp/build --config Release -j$(nproc)

# 4. Run PPO Training
# Automatically detect CPU cores to maximize parallelism
N_CORES=$(nproc)
# Ensure at least 8 envs, up to CPU core count
if [ "$N_CORES" -lt 8 ]; then
    N_ENVS=8
else
    N_ENVS=$N_CORES
fi

# Set Wandb API key for automatic logging
export WANDB_API_KEY="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"

echo ">>> [4/4] Starting PPO Training..."
echo "    - Parallel environments (envs): $N_ENVS"
echo "    - High-Capacity Architecture Enabled: d_model=256, n_heads=8, n_layers=6"
echo "    - Memory Stack Enabled: size=8"
echo "    - Symmetrical Obs: Enemy Board Snapshot & Player Status History"
echo "    - Wandb Logging: Enabled (Project: hs_autobattler)"

# Run PPO training, allowing additional args override via "$@"
python scripts/train_ppo.py \
    --n-envs "$N_ENVS" \
    --total-timesteps 20000000 \
    --d-model 256 \
    --n-heads 8 \
    --n-layers 6 \
    --memory-size 8 \
    --use-enemy-board-obs \
    --use-player-status-obs \
    --use-summary-tokens \
    --use-memory \
    --wandb \
    "$@"
