# Hearthstone Battlegrounds RL Environment

This project is a high-performance environment for training Reinforcement Learning (RL) agents in **Hearthstone: Battlegrounds**.

At its core lies a fully custom game engine written in Python that simulates key game mechanics (Tavern phase, Combat phase, Triplets, complex event chains) without requiring the original game client to run. The project is designed with a focus on simulation speed for efficient sample collection during model training.

## Features

* **Custom Engine:** Full simulation of Hearthstone logic with zero Unity dependencies.
* **Event-Driven Architecture:** The `event_system` handles complex trigger priorities (Deathrattles, Battlecries, "When damaged", Auras) just like the real game.
* **Gym Interface:** The environment is wrapped in the standard `gymnasium` (OpenAI Gym) interface, allowing the use of modern RL algorithms (PPO, DQN, SAC).
* **Complex Combat:** Honest combat simulation: attack order, Cleave, Divine Shield, Poisonous/Venomous, Reborn, Immediate Attack (Pirates), and dynamic Auras.
* **Economy & Meta:** Logic for buying/selling, freezing the board, rerolling, collecting triplets, and upgrading the tavern.

## Project Structure

The project is divided into the simulation core, the training environment, and execution scripts.

### `src/hearthstone/engine` — Core
The "Brain" of the game. Logic is modularized as follows:

* **`game.py`**: Main orchestrator. Manages the game loop and phase switching.
* **`event_system.py`**: Event Bus. The heart of the engine, managing the trigger queue.
* **`combat.py`**: Combat phase logic. Target selection, damage calculation, death processing, and token spawning.
* **`tavern.py`**: Recruitment phase logic (Bob's Tavern). Economy and card management.
* **`entities.py`**: Base classes (`Unit`, `Player`, `Card`).
* **`auras.py`**: System for recalculating static buffs (Wipe & Reapply logic).
* **`pool.py`**: Minion pool management (tiers, copy counts).
* **`configs.py` / `enums.py`**: Configurations, Card IDs, and Tag enumerations.

### `src/hearthstone/env` — RL Environment
* **`hs_env.py`**: Wraps the game in the Gym format. Converts the game state into an Observation Space vector and maps discrete agent actions to engine calls.

### `scripts` — Execution & Training
* **`train.py`**: Agent training pipeline (based on Stable Baselines 3).
* **`test.py`**: Trained model validation and win-rate metric collection.
### `tests` - Tests for different core mechanics (not updated)
* **`run_simulation.py`**: Debug script for verifying mechanics (combat simulation, tavern scenarios).

## Installation

Requires Python 3.9+.

1. **Clone the repository:**
   ```bash
   git clone <repository_url>
   cd hs_autobattler
   ```

2. **Install dependencies:**
   The project uses `pyproject.toml`.
   ```bash
   pip install -e .
   ```

## Usage

### Training the Model
Launch PPO training (hyperparameter config inside):
```bash
python scripts/train.py
```

### Testing Mechanics
Launch simulation (Debug mode) to verify mechanics correctness
```bash
python tests/run_simulation.py
```
more tests places in tests/... (not updated right now)

## Current Status (Development)

The project is under active development.

**Implemented:**
- [x] Basic Game Loop (Tavern <-> Combat)
- [x] Event System and Queues
- [x] Combat Mechanics (Cleave, Shield, Reborn, Venomous, Immediate Attack)
- [x] Aura System (Stateless recalc)
- [x] Gymnasium Integration
- [x] Economy (Buy, Sell, Reroll, Triplets)

**Planned:**
- [ ] Magnetism for Mechs
- [ ] Full Card Set Implementation (currently ~40% of the base set)
- [ ] New Model for learning with Transformers

---
Created by Tmi-creator.