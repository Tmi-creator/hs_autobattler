# Hearthstone Battlegrounds RL Environment

High-performance RL environment for training agents in **Hearthstone: Battlegrounds** with a custom game engine (Python + C++), Set Transformer architecture, and modern training pipeline.

## Architecture

```
Game Engine (Python + C++ combat)
    ↓
Gymnasium Environment (obs/action/reward)
    ↓
Set Transformer Encoder
  • DecomposedEncoder (card embedding + continuous + binary + types)
  • FiLM (global context modulation)
  • GTrXL Gating (stable PPO training)
  • PMA (multi-seed aggregation)
    ↓
MaskablePPO (Categorical Critic + MC Oracle reward)
```

## Features

* **200+ cards** implemented via declarative CardDef system with auto-generated C++ effects
* **C++ combat engine** — 7,400+ combats/sec (20x faster than Python), pybind11
* **Card Embeddings** — `nn.Embedding(202, 64)` learned per-card representations
* **Categorical Critic** — Symlog Two-Hot (255 bins, cross-entropy) from DreamerV3
* **MC Oracle** — C++ engine as dense reward (PBRS): 20 combats per action for instant board evaluation
* **Ghost Pool** — zero-inference self-play via recorded board trajectories with recency bias
* **Action Masking** — dynamic masking of illegal actions via sb3-contrib MaskablePPO

## Project Structure

### `src/hearthstone/engine/` — Game Engine
* **`card_def.py`** — Single source of truth for all cards. Declarative EffectDef system (40+ effect types)
* **`game.py`** — Main game loop, phase management
* **`event_system.py`** — Event bus with trigger priorities
* **`combat.py`** — Combat resolution (Cleave, Divine Shield, Poison, Reborn, Immediate Attack, Auras)
* **`tavern.py`** — Tavern phase (buy/sell/roll/upgrade/freeze/discover)
* **`entities.py`** — Unit, Player, Spell with 4-scope buff system
* **`pool.py`** — Card pool management per tier
* **`auras.py`** — Position-dependent aura recalculation

### `src/hearthstone/env/` — RL Environment
* **`hs_env.py`** — Gymnasium wrapper, observation encoding, MC Oracle, reward function
* **`ghost_pool.py`** — Historical board replay for self-play
* **`smart_bot.py`** — Score-based heuristic opponent

### `scripts/` — Training & Analysis
* **`trans.py`** — Set Transformer feature extractor (DecomposedEncoder, FiLM, GTrXL, PMA)
* **`categorical_critic.py`** — Symlog Two-Hot categorical value head
* **`train_transformer.py`** — Transformer PPO training pipeline
* **`callbacks.py`** — WandB logging, curriculum, entropy decay, board power tracking
* **`kaggle_submit.py`** — Self-contained Kaggle kernel with embedded source
* **`generate_cpp_effects.py`** — Auto-generate C++ effects from CardDef

### `cpp/` — C++ Combat Engine
* **`src/combat.cpp`** — Full combat loop with POD structs, memcpy batch, GIL release
* **`src/generated_effects.cpp`** — Auto-generated from card_def.py
* **`bindings/pybind_module.cpp`** — `fast_combat()` and `fast_combat_batch()`

### `theory/` — Design Documents
* **`todo.md`** — Original MARL-GPT roadmap
* **`claude.md`** — Training pipeline design (ES → BC → PPO → Battle Predictor)
* **`battle_predictor_design.md`** — Combat outcome predictor architecture
* **`transformer_scaling_research.md`** — Scaling analysis from 60+ papers
* **`reward_design_analysis.md`** — Reward function design and experiments

## Installation

```bash
pip install -e .
```

### C++ Engine (optional, for faster combat)
```bash
pip install pybind11
python scripts/generate_cpp_effects.py
cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Release
cmake --build cpp/build
```

## Training

```bash
# Transformer with Categorical Critic + MC Oracle
python scripts/train_transformer.py

# Submit to Kaggle (T4 GPU)
python scripts/kaggle_submit.py
```

## Tests

```bash
pytest tests/ -q  # 709 tests
```

## Current Status

**Implemented:**
- [x] 200+ cards with declarative CardDef system
- [x] C++ combat engine (20x speedup)
- [x] Set Transformer with card embeddings
- [x] Categorical Critic (DreamerV3 Two-Hot)
- [x] MC Oracle dense reward (PBRS)
- [x] Ghost pool self-play with recency bias
- [x] Kaggle training pipeline

**In Progress:**
- [ ] Breaking board_power plateau via MC Oracle
- [ ] ES bot for Behavioral Cloning pretrain
- [ ] Battle Predictor (neural combat outcome predictor)
- [ ] Curriculum learning (tier-based progressive complexity)

---
Created by Tmi-creator.
