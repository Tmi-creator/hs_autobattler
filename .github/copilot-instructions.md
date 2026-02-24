# Project Context: hs_autobattler
This is a custom, highly optimized headless simulator for Hearthstone: Battlegrounds, built from scratch in Python to train Reinforcement Learning (RL) agents. It has ZERO dependencies on the official game client. Performance and strict determinism are critical.

## Architecture & Directory Structure
- `src/hearthstone/engine/`: The core game logic. Treat this as the "physics engine".
  - `event_system.py`: The heart of the game. ALL triggers (Deathrattles, Battlecries, Auras) go through the `EventManager` queue.
  - `combat.py`: Honest combat simulation (attack order, cleave, divine shield, overkill).
  - `entities.py`: Data models (`Unit`, `Player`, `Spell`).
  - `auras.py`: Stateless recalculation of board auras.
- `src/hearthstone/env/`: The RL environment.
  - `hs_env.py`: Gymnasium wrapper. Translates engine state to fixed-size observation vectors and discrete actions.
- `scripts/`: Execution logic (`train.py`, `test.py` with Stable Baselines 3 / MaskablePPO).

## Core Invariants & Rules You MUST Follow

1. **The Event System is King:**
   - NEVER hardcode immediate effects for cards in `combat.py` or `tavern.py`.
   - All card effects MUST be implemented as `TriggerDef` in `effects.py` or `spells.py` and processed via `EventManager`.
   - Be extremely careful with `EntityRef` and `PosRef`. Units can die and change positions; always resolve their current state via `EffectContext`.

2. **Auras are Stateless:**
   - Do not permanently modify a unit's base stats for auras (e.g., Murloc Warleader, Dire Wolf Alpha). 
   - Auras are completely wiped and recalculated using `recalculate_board_auras()` whenever board state changes.

3. **RL Observation Space Strictness:**
   - If you add a new keyword, mechanic, or unit type to the engine, you MUST update the feature vector sizes in `hs_env.py` (`self.entity_features`, `self.observation_space`). 
   - A mismatch in vector sizes will fatally crash the PPO agent.

4. **Code Style & Tools:**
   - We use `ruff` for linting and formatting. Do not use `black`, `flake8`, or `isort`.
   - Strict Python typing is mandatory (`mypy` with `strict=True` mindset). Use `Optional`, `Dict`, `List`, `Tuple` correctly.
   - No magic numbers. Use `enums.py` (`CardIDs`, `Tags`, `UnitType`).

5. **When refactoring or adding mechanics:**
   - Think about edge cases: Does this break Golden versions? Does this break when triggering twice (e.g., Avenge)?
   - If you modify core engine mechanics, suggest writing or updating a test in `tests/`.