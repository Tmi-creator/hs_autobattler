# Theory Documents Index

## Architecture & Implementation
- [transformer.md](transformer.md) — Set Transformer architecture: DecomposedEncoder, FiLM, GTrXL, PMA
- [adding_cards_guide.md](adding_cards_guide.md) — How to add new cards via CardDef system

## Training Pipeline
- [claude.md](claude.md) — **Main design doc**: full training pipeline (ES → BC → PPO), RMCTS, 8-player, transformer refinements
- [reward_design_analysis.md](reward_design_analysis.md) — Reward function design: round outcome, MC Oracle, experiment configs

## Research & Scaling
- [transformer_scaling_research.md](transformer_scaling_research.md) — 60+ paper survey: entity encoding, scaling laws, SAINT, SPR, curriculum
- [battle_predictor_design.md](battle_predictor_design.md) — Battle Predictor + Positioning Module architecture
- [todo.md](todo.md) — Original MARL-GPT roadmap with paper references

## Reports (don't modify)
- [project_report.md](project_report.md) — project report
- report.tex / report.pdf — LaTeX report (don't modify)
- MARL_GPT_main.pdf / MARL_GPT_Appendix.pdf — reference paper

## Current Status (April 2026)

**Implemented:**
- 200+ cards, C++ combat engine (~95k combats/sec), pybind11 bindings (numpy zero-copy input)
- Set Transformer: DecomposedEncoder + FiLM + GTrXL (gated residual) + Multi-Seed PMA + Card Embeddings (nn.Embedding, d=64)
- Categorical Critic (Symlog Two-Hot, 255 bins, cross-entropy loss)
- **SB3 removed** — migrated to CleanRL-style single-file PPO (`scripts/train_ppo.py` + `scripts/model.py`). 2x FPS improvement (250 → 500+ on Kaggle P100). Board power broke past previous plateau.
- ES Bot v2 (`src/hearthstone/env/es_bot.py`): rule-based priority loop with 23 evolved weights, (mu+lambda)-ES evolution (`scripts/evolve_bot.py`). 93.8% winrate vs Smart Bot (500 gen, Kaggle). Hall of Fame + elitism supported.
- Entropy Decay (linear ent_coef 0.04 → 0.01)
- Ghost Pool self-play infrastructure
- Tier-based curriculum learning

**Key results:**
- CleanRL PPO from scratch: board_power 0 → 15+ (and still growing), broke through SB3 plateau
- ES Bot: 93.8% vs Smart Bot, avg_board_power=26.2, avg_max_tier=4.4
- CleanRL 2x faster than SB3 (500 vs 250 FPS on same hardware)

**Next steps:**
- BC Pretrain: train transformer via cross-entropy on ES bot trajectories (scripts/train_bc.py)
- BC → PPO transition: load BC weights, fine-tune with PPO
- Battle Predictor: deferred (C++ MC Oracle at 95k/sec is fast enough for now)
- 8-player FFA environment with ghost pool
- AsyncVectorEnv for additional 1.5-2x FPS boost
