# Полный обзор: как улучшить трансформер для HS:BG

## Context

Research из 60+ источников: AlphaStar, SAINT, DreamerV3, Metamon (Pokemon), TFT (Riot), Set Transformer, scaling laws papers, auxiliary tasks, curriculum learning, и др. Цель: что конкретно применить, чтобы трансформер масштабировался с ростом complexity/compute.

---

## 1. ENTITY REPRESENTATION — самое важное

### 1.1 Card Embedding (КРИТИЧНО)

**Проблема**: card_id как normalized float (0.42 = "Scallywag"). При 200 картах — 1 нейрон на 200 вариантов.

**Что делают другие**:
- **AlphaStar**: unit_type → learned embedding (250+ типов юнитов), dim=256
- **Card2Vec** (Hearthstone): word2vec на co-occurrence данных, 100-dim embeddings
- **Entity Embeddings** (Guo & Berkhahn, 2016): `nn.Embedding` обучается jointly. Dim = min(50, num_categories/2). Для 200 карт → **32-dim embedding**
- **Metamon (Pokemon)**: game-specific vocabulary → tokenizer → embedding

**Решение**: `nn.Embedding(num_cards, 32)`, обучается через backprop вместе с policy. При BC pretrain — выучит "какие карты хорошие", при PPO — подстроится.

**Альтернатива — Card2Vec pretrain**: нагенерить 100k игр ES-ботом, записать co-occurrence карт на доске, обучить word2vec-style embeddings, инициализировать `nn.Embedding` этими векторами. Это **warm start** — embeddings уже знают "мурлоки ходят вместе".

### 1.2 Mechanic Category Embedding

Добавить embedding функциональной роли: "DR: summon", "DR: buff", "Aura: neighbors", "Scaling: on play", "Economy: gold". Dim = 16. Это inductive bias — ускоряет обучение синергий.

### 1.3 AlphaStar-style Entity Feature Vector

AlphaStar кодирует каждый юнит как:
- unit_type: embedding
- health/shields: **one-hot of sqrt(value)** (не raw float!)
- alliance/owner: embedding
- binary keywords
- build_progress: float

Для BG entity vector (~60-80 dim):
- card_id: 32-dim embedding
- ATK, HP: symlog floats (уже есть)
- tier: 6-dim embedding или one-hot
- types: multi-hot (уже есть)
- keywords: binary flags (уже есть)
- mechanic_category: 16-dim embedding (новое)
- is_golden, is_token: binary (уже есть)

---

### 1.4 MARL-GPT Positional Encoding (AAMAS 2026)

MARL-GPT добавляет **4 positional embeddings** к каждому feature:

```
res_i = tok_i + emb_indx + emb_team + emb_attr + emb_time
```

- `emb_attr` — embedding для каждого **типа атрибута** (ATK, HP, tier отдельно). Более гранулярно чем наш DecomposedEncoder
- `emb_team` — аналог нашего zone embedding (board/hand/store)
- `emb_indx` — индекс entity внутри зоны
- `emb_time` — temporal step (для memory/history)

**Что взять**: `emb_attr` для per-feature embeddings и `emb_time` когда добавим memory для 8-player.

### 1.5 MARL-GPT Critic & Policy Loss

- **Categorical critic** (bins + cross-entropy, НЕ MSE) — подтверждает наш Two-Hot plan
- **Combined loss**: `L = advantage × log π + BC_cross_entropy` — advantage + BC в одном loss
- **Conservative regularization**: штраф на Q-values для unlikely actions → q_min
- **Online fine-tune**: actor frozen → critic pretrains online → together. Обратный порядок от нашего — стоит попробовать оба

Ref: [MARL-GPT, AAMAS 2026](https://anonymous.4open.science/r/marl-gpt-20365)

---

## 2. ENCODER ARCHITECTURE

### 2.1 Текущее vs AlphaStar

| Компонент | Текущий | AlphaStar | Рекомендация |
|-----------|---------|-----------|-------------|
| Entity embed | Linear(38, 128) decomposed | Linear(features, 256) | ✅ Decomposed лучше |
| Transformer | 4 layers, 4 heads, d=128 | 3 layers, 2 heads, d=256 | Увеличить d до 192-256 |
| Aggregation | PMA (K=4 seeds) | Mean pool + per-entity | ✅ PMA лучше mean |
| Context | FiLM + [GLOBAL_CTX] | Scalar features embedded | ✅ FiLM + [GLOBAL_CTX] лучше |
| Gating | GTrXL (b_z=-3) | Нет (но GTrXL paper = 2019) | ✅ Gating нужен для PPO |

**Вывод**: текущая архитектура **лучше** AlphaStar в нескольких аспектах (decomposed encoder, PMA, FiLM, gating). Слабое место — card representation (float vs embedding).

### 2.2 Pointer Network для Entity Selection

AlphaStar использует pointer network для выбора юнитов: query из decoder dot-product с entity embeddings → softmax → select.

У тебя уже есть state machine (action → target). Pointer network — следующий шаг если/когда перейдёшь на явный autoregressive actor. Пока state machine работает.

### 2.3 SAINT — Sub-Action Factorization (2025)

Новейшая работа. Три стадии:
1. FiLM conditioning: sub-action embeddings модулируются global state
2. Transformer self-attention over sub-actions (captures dependencies)
3. Parallel decoding each sub-action

Работает на action spaces до **16M+ discrete actions**. Outperforms autoregressive и independent baselines.

**Применимость**: высокая, когда перейдёшь на compound actions. Пока state machine достаточно.

---

## 3. SCALING LAWS — модель скорее всего слишком маленькая

### AlphaZero Scaling Laws (ICLR 2023)

**Elo масштабируется как power law от числа параметров.** Почти одинаковые exponents для Connect Four и Pentago. **Большие модели более sample efficient** — учатся быстрее при том же числе samples.

**Ключевой вывод**: опубликованные game AI модели **significantly undersized** для их compute budgets. Оптимальный размер модели vs compute = **как в NLP (Chinchilla scaling)**.

### Metamon (Pokemon, RLC 2025)

Clear scaling: 200M > 50M > 15M params. С 1M human + 4M self-play battles.

### Для твоего проекта

Текущие 500K params при 200 картах — **скорее всего undersized**. Рекомендация:

| Compute budget | d_model | n_layers | Params | FPS estimate |
|----------------|---------|----------|--------|-------------|
| Текущий | 128 | 4 | 500K | 250 |
| Средний | 192 | 6 | 1.5M | ~170 |
| Большой | 256 | 8 | 4M | ~120 |

**FPS можно компенсировать**: async envs (+2-4x), fp16 (+1.5x), gradient accumulation.

---

## 4. TRAINING TECHNIQUES — что реально помогает

### 4.1 Potential-Based Reward Shaping (PBRS)

**Единственная теоретически безопасная форма reward shaping** (preserves optimal policy).

```python
shaped_reward = gamma * Phi(s') - Phi(s)
# Phi = Battle Predictor win probability
```

Это ровно то, что мы спроектировали как `r_predictor = predictor(board_after) - predictor(board_before)`. Формально это PBRS с Phi = predictor. **Наш дизайн теоретически обоснован.**

### 4.2 Curriculum Learning

**FastCuRL (2025)**: достигает того же performance с **50% training resources** через progressive complexity.

Для BG — естественный curriculum:
1. Tier 1-2 карты only, 2 players, short games
2. Tier 1-3, 4 players
3. Tier 1-5, 8 players
4. Full game

**Конкретно**: ограничить `CardPool` по тирам на ранних этапах обучения. Разблокировать тиры по мере роста winrate (>55%).

### 4.3 Auxiliary Tasks (ВАЖНО для трансформера)

**Forward dynamics prediction** — consistently лучший auxiliary task. Предсказать `state(t+1) | state(t), action(t)`.

Для BG: encoder предсказывает "как изменится board_power после покупки этого юнита?" Это заставляет encoder учить **причинно-следственные связи**, а не только корреляции.

**Оптимальный вес**: 0.1-1.0x от main RL loss. С decay к 0 за 75% обучения (как мы планировали).

**2024 RLJ paper**: auxiliary tasks помогают БОЛЬШЕ всего в sparse-reward, non-visual environments — **именно наш случай**.

### 4.4 Self-Predictive Representations (SPR)

Encoder предсказывает **свои собственные future latent states**: `predict h(t+k) from h(t)`. **+55% improvement на Atari 100k.** С fast C++ engine — генерация pretraining data дёшево.

Можно использовать для **pretrain encoder** до PPO: нагенерить 1M game states ES-ботом, обучить encoder через SPR, затем fine-tune с PPO.

### 4.5 BC → PPO Transition (PostBC insight)

**Проблема**: стандартный BC убивает exploration — policy становится слишком уверенной. PPO fine-tune не может исправить.

**Решение (PostBC)**: после BC pretrain, **искусственно повысить entropy** в uncertain states. Конкретно: `ent_coef=0.02-0.05` на первых 200k steps PPO, затем decay к 0.01. (Уже в нашем плане!)

### 4.6 Population-Based Training (PBT)

**PBT обнаружил**: entropy coefficient должен быть **HIGH early, LOW late**. Этот schedule критичен для PPO, но hard to hand-tune.

Если есть compute — PBT для hyperparameter search (lr, ent_coef, clip_range, gamma). Если нет — использовать known-good schedules из literature.

---

## 5. WHAT NOT TO DO

### Decision Transformer — НЕ использовать

Paper "When to Prefer DT" (2023): DT проигрывает PPO при **high stochasticity** (combat random) и **online access** (мы можем генерить данные). DT не умеет stitching suboptimal trajectories. **Для BG: PPO строго лучше.**

### GNNs — НЕ нужны

AlphaGateau (NeurIPS 2024) показал GNN > CNN для шахмат, но **transformer self-attention уже IS pairwise relational reasoning**. GAT добавляет explicit edge features — overkill для BG где все entities в одном "графе" (доска).

### World Model — НЕ нужна

DreamerV3/V4 учат world model для imagination. Но **C++ engine = perfect world model**. Dynamics prediction полезна только как **auxiliary task**, не как замена simulation.

---

## 6. RIOT GAMES (TFT) VALIDATION

Riot сделали для TFT ровно то, что мы планируем:
1. Обучили **neural net предсказывать combat outcomes** (= наш Battle Predictor)
2. Использовали как **fast evaluation** для RL training
3. Decoupled shop-phase RL от expensive combat sim

**Это прямая валидация нашего подхода от production game studio.**

---

## 7. КОНКРЕТНЫЙ ПЛАН ДЕЙСТВИЙ (обновлённый)

### Tier 1: Сделать сейчас (до серьёзного обучения)

| # | Что | Зачем | Усилия |
|---|-----|-------|--------|
| 1 | **Card Embedding** (nn.Embedding, d=32) | Критично для 200 карт | 1-2 дня |
| 2 | **Mechanic Category Embedding** (d=16) | Ускоряет обучение синергий | 1 день |
| 3 | **Curriculum Learning** (tier 1-2 → full) | -50% training resources (FastCuRL) | 1 день |
| 4 | **Symlog Two-Hot Critic** (255 bins) | Бимодальные rewards | 2-3 дня |

### Tier 2: При увеличении compute

| # | Что | Зачем | Усилия |
|---|-----|-------|--------|
| 5 | **Scale d_model 128→192, n_layers 4→6** | Scaling laws — модель undersized | 1 день |
| 6 | **SPR pretraining** (encoder predicts own futures) | +55% на Atari 100k | 3-4 дня |
| 7 | **Card2Vec warm start** (word2vec на ES gameplay) | Better card embedding init | 2 дня |
| 8 | **Forward dynamics auxiliary task** | Best aux task по literature | 2 дня |

### Tier 3: Advanced (после основного pipeline)

| # | Что | Зачем | Усилия |
|---|-----|-------|--------|
| 9 | **SAINT sub-action factorization** | Compound actions | 1 неделя |
| 10 | **PBT для hyperparameters** | Automated tuning | 3-4 дня |
| 11 | **Async envs (Sample Factory)** | Faster training | 1 неделя |

### Gaps identified by Gemini verification

| Gap | Решение | Когда |
|-----|---------|-------|
| **Hero ID embedding** | `nn.Embedding(num_heroes, d_hero)` → FiLM conditioning | Tier 1 (вместе с card embeddings) |
| **Explicit Memory (POMDP)** | LSTM/GRU поверх transformer output для N последних наблюдений | При переходе на 8-player |
| **SAINT / explicit autoregressive** | State machine уже делает то же самое. Формализовать позже | Только если >50 action types |

### Новые статьи (найдены Gemini Deep Research)

| Paper | Link | Relevance |
|-------|------|-----------|
| Mastering HS with OSFP (Xiao 2023) | [2303.05197](https://arxiv.org/abs/2303.05197) | 73.6% WR, beat top-10 streamer. Прямой конкурент. |
| ByteRL vulnerabilities (2024) | [2404.16689](https://arxiv.org/abs/2404.16689) | BC highly exploitable → нужен ghost pool + SPR |
| Dota Underlords NP-completeness | [2007.05020](https://arxiv.org/abs/2007.05020) | Formal proof autobattler = NP-hard. Для intro статьи |
| Riot TFT GDC slides (Cao Ran) | [PDF](https://media.gdcvault.com/gdc2023/Slides/Simulating++Teamfight+Tactics_Cao_Ran.pdf) | Конкретные слайды архитектуры Duo AI |

### Tier 4: NOT TO DO

- Decision Transformer (stochastic combat kills it)
- GNNs (transformer handles it)
- Full world model (C++ engine = perfect model)
- Board position encoding (авторасстановщик отдельно)

---

## 8. COMPETITIVE LANDSCAPE

**Автобаттлеры**: практически **нет академических работ**. Существуют:
- hearthstone-ai (peter1591): C++ MCTS + NN, 403-value flat vector, 7 features per minion
- TFTMuZeroAgent: Sampled MuZero + transformer, 63 discrete actions
- BGSimulator: no published architecture
- Super Auto Pets AI: simple gym envs

**Твой проект заполняет реальный пробел** в литературе. Ни у кого нет:
- Transformer + Set Attention для autobattler
- Battle Predictor + Positioning Module
- ES → BC → PPO pipeline для card games
- Systematic ablation study на autobattler

---

## Ключевые источники

| Paper | Venue | Link | Relevance |
|-------|-------|------|-----------|
| AlphaStar (2019) | Nature | [nature.com](https://www.nature.com/articles/s41586-019-1724-z) | Entity encoder, pointer network — gold standard |
| SAINT (2025) | arXiv | [2505.12109](https://arxiv.org/abs/2505.12109) | Sub-action factorization with FiLM |
| DreamerV3 (2023) | JMLR | [2301.04104](https://arxiv.org/abs/2301.04104) | Symlog, two-hot critic |
| AlphaZero Scaling Laws (2023) | ICLR | [2210.00849](https://arxiv.org/abs/2210.00849) | Power law scaling, models are undersized |
| FastCuRL (2025) | arXiv | [2502.15168](https://arxiv.org/abs/2502.15168) | Curriculum = 50% less compute |
| Metamon / Pokemon (2025) | RLC | [2504.04395](https://arxiv.org/abs/2504.04395) | Pokemon RL, clear model scaling |
| SPR (2021) | NeurIPS | [2007.05929](https://arxiv.org/abs/2007.05929) | Self-predictive representations, +55% |
| Riot TFT GDC (2023) | GDC | [GDC Vault](https://gdcvault.com/play/1028851/) | Combat outcome predictor = our Battle Predictor |
| PBRS (Ng et al., 1999) | ICML | [PDF](https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/NgHaradaRussell-shaping-ICML1999.pdf) | Potential-based reward shaping — теоретическое обоснование нашего r_predictor |
| Relational Deep RL (2018) | arXiv | [1806.01830](https://arxiv.org/abs/1806.01830) | Self-attention IS relational reasoning |
| Set Transformer (2019) | ICML | [1810.00825](https://arxiv.org/abs/1810.00825) | SAB + PMA = our architecture |
| Entity Embeddings (2016) | arXiv | [1604.06737](https://arxiv.org/abs/1604.06737) | nn.Embedding > normalized float |
| DreamerV3 tricks for PPO (2023) | NeurIPS | [2310.17805](https://arxiv.org/abs/2310.17805) | Symlog two-hot for PPO specifically |
| GTrXL (2019) | arXiv | [1910.06764](https://arxiv.org/abs/1910.06764) | Gated residuals for RL stability |
| Scaling Laws for Agents (2025) | ICML | [2411.04434](https://arxiv.org/abs/2411.04434) | Scaling laws transfer to RL |

---

## Files to modify

- `scripts/trans.py` — Card embedding, mechanic embedding, d_model/n_layers scaling, SPR head, forward dynamics head
- `src/hearthstone/env/hs_env.py` — Raw card_id int in obs, mechanic_category in entity vector
- `src/hearthstone/engine/card_def.py` — mechanic_category field on CardDef
- `src/hearthstone/engine/enums.py` — MechanicCategory enum
- `scripts/train_transformer.py` — Curriculum learning (tier-limited CardPool), PBT integration

## Verification

1. `pytest tests/` — all pass
2. Card embeddings: train 100k steps, visualize embedding clusters (t-SNE). Cards of same tribe should cluster.
3. Scaling: compare d=128 vs d=192 learning curves at 200 cards
4. Curriculum: compare full-game training vs tier-progressive curriculum
5. Ablation: each component on/off, measure winrate vs ES bot
