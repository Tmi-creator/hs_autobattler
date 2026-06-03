# Hearthstone Battlegrounds RL Environment

High-performance Hearthstone Battlegrounds simulator and RL training stack. The
current main path is a custom Python engine with optional C++ combat, a
Gymnasium environment, a standalone Set Transformer actor-critic, ES bot
pretraining data, and a CleanRL-style PPO loop.

## Architecture

```text
Game Engine (Python tavern + Python/C++ combat)
    |
Gymnasium Environment
  - flat Box(1036) observation
  - 34 discrete tavern actions
  - dynamic legal-action masks
    |
HSTransformerAgent (scripts/model.py)
  - DecomposedEncoder for card id / continuous / binary / type features
  - FiLM global context modulation
  - GTrXL gated residual blocks
  - PMA multi-seed aggregation
  - actor logits + symlog two-hot categorical critic
    |
CleanRL-style PPO (scripts/train_ppo.py)
  - AsyncVectorEnv
  - masked categorical policy
  - entropy decay
  - target_kl early stopping
  - optional resume from BC checkpoint
```

Older SB3 files still exist for reference and compatibility, but they are no
longer the primary training route.

## Current Features

* **200+ cards** through the declarative `CardDef` system and generated C++ effects.
* **Rewritten C++ combat engine** with pybind11 bindings and a batched numpy path. Current benchmarks are ~90-95k combats/sec on complex cases and 270k+ combats/sec on simple cases, about **800x faster than the pure Python combat baseline**; Python combat remains the fallback.
* **Standalone transformer agent** in `scripts/model.py`, with no SB3 dependency.
* **Categorical critic** using DreamerV3-style symlog two-hot targets over 255 bins.
* **Critic-detached encoder path** so value loss updates the critic head without corrupting actor representations.
* **ES Bot v2** in `src/hearthstone/env/es_bot.py`: 23 evolved weights, Hall of Fame support in `scripts/evolve_bot.py`, and saved weights expected at `artifacts/es_kaggle/artifacts/best.npz`.
* **Behavior cloning infrastructure** via `scripts/bc_collect.py` and `scripts/bc_train.py`.
* **Kaggle PPO submission flow** in `scripts/kaggle_submit_ppo.py`, with optional BC collection/train before PPO.
* **Ghost pool and MC Oracle infrastructure** in the environment. MC Oracle code is present, but the current CleanRL reward path is still round outcome + terminal reward; dense oracle reward is tracked as follow-up work in theory docs.

## Project Structure

### `src/hearthstone/engine/` - Game Engine

* `card_def.py` - single source of truth for cards and declarative effects.
* `game.py` - main game loop and phase management.
* `event_system.py` - event bus with trigger priorities.
* `combat.py` - Python combat resolution.
* `tavern.py` - buy/sell/roll/upgrade/freeze/discover logic.
* `entities.py` - units, players, spells, and buff scopes.
* `pool.py` - minion and spell pool management.
* `auras.py` - position-dependent aura recalculation.

### `src/hearthstone/env/` - RL Environment

* `hs_env.py` - Gymnasium wrapper, observation encoding, masks, rewards, ghost/oracle hooks.
* `es_bot.py` - evolved heuristic bot used for BC data.
* `ghost_pool.py` - recorded board trajectories for self-play.
* `smart_bot.py` - score-based baseline opponent.

### `scripts/` - Training and Tooling

* `model.py` - current standalone `HSTransformerAgent`.
* `train_ppo.py` - current CleanRL-style PPO loop.
* `bc_collect.py` - collect ES bot trajectories as `(obs, mask, action)`.
* `bc_train.py` - masked cross-entropy actor pretrain.
* `kaggle_submit_ppo.py` - self-contained Kaggle kernel generator for BC + PPO.
* `evolve_bot.py` - `(mu+lambda)` ES trainer for `es_bot.py`.
* `generate_cpp_effects.py` - generate C++ effects from `CardDef`.
* `legacy/` - legacy SB3-era training files (trans, categorical_critic, train, evaluate_pvp, visualize_attention), integration tests, and the archived Python combat simulator.

### `cpp/` - C++ Combat Engine

* `src/combat.cpp` - combat hot path.
* `src/generated_effects.cpp` - generated card effects.
* `src/profiler.cpp` - optional profiler counters.
* `bindings/pybind_module.cpp` - `fast_combat()` / `fast_combat_batch()` bindings.

### `theory/` - Design Documents

* `INDEX.md` - current map of theory docs and status.
* `CLAUDE.md` - main training pipeline design: ES -> BC -> PPO, RMCTS, 8-player path.
* `transformer.md` - transformer architecture and current implementation notes.
* `battle_predictor_design.md` - deferred combat predictor design.
* `reward_design_analysis.md` - reward design notes and ablation plan.

## Installation

```bash
pip install -e .
```

### C++ Engine

```bash
pip install pybind11
python scripts/generate_cpp_effects.py
cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Release
cmake --build cpp/build
```

## Training

```bash
# Evolve ES bot weights
python scripts/evolve_bot.py --generations 500 --out-dir artifacts/es_kaggle/artifacts

# Collect behavior-cloning data from ES bot
python scripts/bc_collect.py --episodes 5000 --weights artifacts/es_kaggle/artifacts/best.npz

# Train BC actor checkpoint
python scripts/bc_train.py --epochs 15 --batch-size 512

# Fine-tune with PPO from BC checkpoint
python scripts/train_ppo.py --resume artifacts/bc/bc_pretrain.pt --total-timesteps 5000000

# Submit BC + PPO pipeline to Kaggle
python scripts/kaggle_submit_ppo.py
```

## Tests

```bash
$env:PYTHONPATH = "src"
pytest tests -q
```

`tests/` currently collects 826 tests in this checkout. Full repository
collection also sees `scripts/test_trans_integration.py`, which depends on the
legacy SB3 path.

## Current Status

**Implemented:**

- [x] CardDef-based card database and generated C++ effects.
- [x] Python engine with optional rewritten C++ combat acceleration (~800x faster than pure Python combat in benchmarks).
- [x] Gymnasium environment with 34 actions and action masks.
- [x] Standalone Set Transformer actor-critic in `scripts/model.py`.
- [x] CleanRL-style PPO in `scripts/train_ppo.py`.
- [x] AsyncVectorEnv rollout collection.
- [x] Symlog two-hot categorical critic.
- [x] Critic-detached encoder path and `target_kl=0.03`.
- [x] ES bot evolution and BC data/training scripts.
- [x] Kaggle BC + PPO submission script.

**Next:**

- [ ] Run BC pretrain on the current ES weights and compare PPO from BC vs PPO from scratch.
- [ ] Wire ghost pool curriculum into the current CleanRL PPO path.
- [ ] Decide whether MC Oracle dense reward should be enabled in the current reward function.
- [ ] Revisit Battle Predictor only if C++ MC Oracle becomes the bottleneck or COMBAT_CTX is needed.

---
Created by Tmi-creator.
