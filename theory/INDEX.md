# Theory Documents Index

## Architecture & Implementation

- [transformer.md](transformer.md) - Set Transformer architecture and the current `scripts/model.py` implementation.
- [adding_cards_guide.md](adding_cards_guide.md) - How to add cards via the `CardDef` system.

## Training Pipeline

- [CLAUDE.md](CLAUDE.md) - **Main design doc**: ES -> BC -> PPO, RMCTS, 8-player path, transformer refinements.
- [reward_design_analysis.md](reward_design_analysis.md) - Reward function design: round outcome, MC Oracle, experiment configs.

## Research & Scaling

- [transformer_scaling_research.md](transformer_scaling_research.md) - 60+ paper survey: entity encoding, scaling laws, SAINT, SPR, curriculum.
- [battle_predictor_design.md](battle_predictor_design.md) - Battle Predictor + Positioning Module architecture.
- [todo.md](todo.md) - Original MARL-GPT roadmap with paper references.

## Reports (do not modify)

- [project_report.md](project_report.md) - project report.
- `report.tex` / `report.pdf` - LaTeX report.
- `MARL_GPT_main.pdf` / `MARL_GPT_Appendix.pdf` - reference paper.

## Current Status (May 2026 working tree)

**Latest repo direction:**

- `617b844 update theory + submit new ppo` added the current standalone transformer model, CleanRL PPO loop, and Kaggle PPO submission path.
- `0aab30c update cmake` added CMake switches for the C++ combat/profiling build.
- The uncommitted working tree extends that path with AsyncVectorEnv, BC collect/train scripts, PPO resume from BC checkpoints, and critic-detached value learning.

**Implemented:**

- 200+ cards through `CardDef`, generated C++ effects, and Python fallback logic.
- Rewritten C++ combat engine with pybind11 batched calls; current benchmark notes are ~90-95k combats/sec on complex cases and 270k+ on simple cases, about 800x faster than the pure Python combat baseline.
- Set Transformer in `scripts/model.py`: `DecomposedEncoder` + FiLM + GTrXL gated residuals + Multi-Seed PMA + dynamic card embeddings.
- Symlog two-hot categorical critic in `scripts/model.py`; `scripts/categorical_critic.py` is legacy SB3-era code.
- CleanRL-style PPO in `scripts/train_ppo.py`: masked categorical policy, AsyncVectorEnv, entropy decay 0.04 -> 0.01, `target_kl=0.03`.
- Critic-detached encoder path: value loss trains the critic head without sending gradients through shared actor representations.
- ES Bot v2 in `src/hearthstone/env/es_bot.py`: 23 evolved weights, Hall of Fame + elitism in `scripts/evolve_bot.py`.
- BC infrastructure: `scripts/bc_collect.py` generates `(obs, mask, action)` from ES bot decisions; `scripts/bc_train.py` trains the actor with masked cross-entropy.
- Kaggle BC + PPO wrapper in `scripts/kaggle_submit_ppo.py`.
- Ghost pool and MC Oracle hooks in `HearthstoneEnv`.
- `tests/` collects 826 tests with `PYTHONPATH=src`.

**Current caveats:**

- The active reward in `HearthstoneEnv.step()` is still action penalty + round outcome + terminal reward. MC Oracle methods exist and cache winrate, but dense oracle reward is not wired into the current CleanRL reward path.
- Legacy SB3-era files and the Python combat simulator have been moved to the `legacy/` directory. They are useful as references, not the main training path.
- Full repository test collection now passes 100% cleanly without requiring any legacy `stable_baselines3` or `sb3_contrib` dependencies.
- BC scripts are implemented, but the main doc still treats the Kaggle BC run and PPO-from-BC comparison as pending experiment work.

**Next steps:**

- Run BC collection/training on `artifacts/es_kaggle/artifacts/best.npz`.
- Compare PPO from scratch vs PPO resumed from `artifacts/bc/bc_pretrain.pt`.
- Wire ghost-pool curriculum into the current CleanRL PPO path.
- Decide whether to enable MC Oracle dense reward in `HearthstoneEnv.step()`.
- Defer Battle Predictor until C++ MC Oracle is too slow or a COMBAT_CTX token is needed.
