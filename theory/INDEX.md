# Theory Documents Index

## Architecture & Implementation
- [transformer.md](transformer.md) — Current Set Transformer architecture (trans.py): DecomposedEncoder, FiLM, GTrXL, PMA
- [adding_cards_guide.md](adding_cards_guide.md) — How to add new cards via CardDef system

## Training Pipeline
- [claude.md](claude.md) — **Main design doc**: full training pipeline (ES → BC → PPO → Battle Predictor), RMCTS, 8-player, transformer refinements
- [reward_design_analysis.md](reward_design_analysis.md) — Reward function design: round outcome, MC Oracle, experiment configs

## Research & Scaling
- [transformer_scaling_research.md](transformer_scaling_research.md) — 60+ paper survey: entity encoding, scaling laws, SAINT, SPR, curriculum
- [battle_predictor_design.md](battle_predictor_design.md) — Battle Predictor + Positioning Module architecture
- [todo.md](todo.md) — Original MARL-GPT roadmap with paper references

## Reports (don't modify)
- [project_report.md](project_report.md) — курсовая / project report
- report.tex / report.pdf — LaTeX report (don't modify)
- MARL_GPT_main.pdf / MARL_GPT_Appendix.pdf — reference paper

## Current Status (April 2026)

**Implemented:**
- Card Embeddings (nn.Embedding, d=64)
- Categorical Critic (Symlog Two-Hot, 255 bins)
- MC Oracle (C++ engine as PBRS dense reward)
- Entropy Decay callback
- 200+ cards, C++ combat engine

**Next steps:**
- Evaluate MC Oracle results (running on Kaggle)
- If plateau persists → ES bot + BC pretrain
- Curriculum learning (tier-based)
- Battle Predictor (neural, if MC Oracle too slow)
