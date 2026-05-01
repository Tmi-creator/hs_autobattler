# HS Autobattler: Training Pipeline & Architecture Design

## Context

Среда дорабатывается (карты, UPGRADE/FREEZE, аура) — это механическая работа. Главный вопрос: **когда среда будет готова (80-100+ карт), как на ней учить?** Этот документ описывает полный training pipeline и архитектурные решения.

Связанные docs:
- `theory/battle_predictor_design.md` — Battle Predictor, Positioning Module
- `theory/todo.md` — оригинальный roadmap

Актуальные числа производительности (April 2026):
- C++ combat: ~90-95,000 combats/sec on complex cases, 270k+ on simple cases (pybind11 numpy path), about 800x faster than pure Python combat
- MLP training: ~1,200 FPS (DummyVecEnv, SB3) — deprecated
- Transformer training (SB3): ~250 FPS (DummyVecEnv) — **deprecated, replaced by CleanRL**
- Transformer training (CleanRL): ~500 FPS SyncVectorEnv → **AsyncVectorEnv switch (current)**, 750-1000 FPS expected
- ES bot evolution: ~2ms/game, 500 gen in ~20 min on Kaggle CPU (4 workers). Best: gen 499, fitness 0.85, weights in `artifacts/es_kaggle/artifacts/best.npz`
- BC trajectory collection: ~1,200 steps/sec single-threaded (es_pick_action via game.step snoop)

---

## 1. Полный Training Pipeline

```
Этап 0: ES Evolution ──────────────► Evolved Bot (W*)         ✅ DONE
  │  (mu+lambda, 500 поколений)       fitness=0.85 vs Smart Bot
  │  rule-based + 23 evolved weights   board_power≈25, tier≈4.3
  │  artifacts/es_kaggle/artifacts/best.npz
  │
  └──► Этап 1: BC Pretrain ─────────► BC-инициализированный Actor    ✅ INFRA DONE
       │   (cross-entropy от ES bot)   scripts/bc_collect.py + bc_train.py
       │                                training run pending (Kaggle)
       ▼
       Этап 2: PPO Fine-tune ───────► Trained Agent              ✅ INFRA DONE
         - CleanRL PPO (AsyncVectorEnv, 750-1000 FPS expected)
         - Categorical Critic + entropy decay
         - Critic detached from encoder (value loss не отравляет actor representations)
         - target_kl=0.03 (early-stop если политика дрейфует)
         - --resume from BC checkpoint (model-only, fresh PPO optimizer)
         - Ghost pool curriculum (70% ghost / 30% bot)        ← TODO
         - MC Oracle dense reward (95k combats/sec)           ← TODO
                │
                ▼
       Этап 3: Self-Play Iterations
         - Лучшая модель vs копии → ghost pool
         - Опционально: RMCTS для complex decisions
         - Повторять 3-4 пока растёт winrate
```

### Стабилизация PPO (April 2026 fixes)

После первого full PPO ран (cleanrl_ppo_1776448844, 5M steps) увидели плато на board_power 20-25 (vs ES бот 32+) с диагнозом:

1. **Plateau** — настоящая причина: PPO с нуля на 200 карт × 34 actions не нащупает синергии. Лекарство = BC pretrain (см. выше).
2. **Critic вытекает в encoder** — общий энкодер для actor/critic, vf_coef=0.5 → шум value loss портит представления стола. **Fix**: `critic(features.detach())` в [scripts/model.py:325](scripts/model.py#L325).
3. **KL drift** — approx_kl рос с 0.005 до 0.02 без early-stop (target_kl=None). **Fix**: target_kl=0.03 default.
4. **AsyncVectorEnv** — env.step параллельно с inference, +50-100% FPS. На Linux/Kaggle через fork, на Windows — spawn (pickle issues возможны, но мы тренируем на Kaggle).

### Почему именно такой порядок

- **Чистый PPO from scratch** уже показывает рост (board_power 0→15+ на CleanRL), но BC pretrain ускорит cold start
- **BC pretrain** перескакивает через early training за часы: агент стартует уже умея "покупай по синергиям, апгрейдь на curve"
- **Battle Predictor отложен**: C++ MC Oracle даёт ground-truth winrate за 0.2ms (20 combats at 95k/sec). Neural predictor нужен только для COMBAT_CTX embedding в observation
- **ES bot** масштабируется автоматически: добавил карты → перезапустил эволюцию (~20 мин) → новый оптимальный bot

### Что изменилось vs оригинальный план

1. **SB3 убран** — заменён CleanRL-style PPO (`scripts/train_ppo.py` + `scripts/model.py`). 2x FPS, пробил plateau по board_power.
2. **Battle Predictor стал необязательным** — MC Oracle достаточно быстр для dense reward. Predictor нужен только для embedding-based RMCTS leaf evaluation.
3. **ES bot v2** — rule-based priority loop (не deepcopy lookahead). 93.8% vs Smart Bot, ~2ms/game.

---

## 2. Reward Design

### Структура

```python
# Базовый сигнал (не затухает)
r_combat = damage_dealt × 0.2 - damage_taken × 0.2
r_terminal = ±10.0  (win/loss)

# Battle Predictor dense reward (между действиями таверны)
r_predictor = predictor(board_after, ghost) - predictor(board_before, ghost)

# Shaping (затухает за 2M steps)
decay = max(0, 1 - steps / 2_000_000)
r_economy = decay × board_power_delta × 0.05
r_triple = decay × triple_formed × 2.0

reward = r_combat + r_terminal + r_predictor + r_economy + r_triple
```

### Принцип затухания

Shaping нужен на старте (ускоряет обучение), но вреден в конце (агент эксплуатирует промежуточные метрики вместо побед). Predictor reward НЕ затухает — он калиброван и отражает реальный winrate.

### Нужна аблация

Запустить 4 конфигурации:
1. Только terminal (sparse)
2. Terminal + combat damage
3. Terminal + combat + predictor
4. Full (всё включено)

---

## 3. RMCTS — простым языком

### Что это

Перед каждым действием в таверне, вместо "спросить PPO", делаем:
1. **Клонируем** состояние игры (board, hand, store, gold, pool)
2. На клоне **пробуем последовательности**: buy→play→sell→buy→end_turn
3. В конце каждой последовательности **спрашиваем Battle Predictor**: "кто победит?"
4. **Выбираем** первое действие из лучшей последовательности
5. Делаем его в реальной игре

### Проблема рандома (ROLL)

Если в симуляции нажали ROLL — магазин обновится случайно. Одна и та же кнопка "BUY slot 0" после разных ROLL = покупка разных карт.

**Решение — determinization**: фиксируем seed CardPool заранее. Запускаем 8 параллельных деревьев, каждое с *разным* фиксированным "миром". **Усредняем visit counts** по всем 8 мирам. Это НЕ оверфит под один seed — наоборот, если действие хорошее в 7 из 8 миров, оно робастно. Если только в 1 из 8 — оно зависит от удачного ROLL и будет отфильтровано. Стандартный подход: Information Set MCTS / Determinized UCT.

### Когда полезен

| Ситуация | RMCTS помогает? | Почему |
|----------|-----------------|--------|
| Sell→Buy→Triple цепочка | Да | PPO не видит reward от трипла до его завершения |
| Upgrade timing | Да | Отложенный reward (лучший магазин через ход) |
| Простая покупка "лучшее из shop" | Нет | PPO+Predictor справляется |
| Позиционирование | Нет | Positioning Module лучше |

### Скорость

| Режим | Время per decision | Feasible? |
|-------|-------------------|-----------|
| Depth-1 (все действия × Predictor) | ~30ms | Всегда |
| Depth-3, 128 sims | ~770ms | Для training |
| Full depth, 512 sims | ~5s | Только offline |

**Рекомендация**: начать с depth-1 (по сути "попробуй все действия, оцени Predictor'ом"). Если недостаточно — depth-3 для ключевых ходов (4-8).

### Prerequisites

1. Battle Predictor (обязательно)
2. `Game.tavern_snapshot()` + `Game.restore_from_snapshot()` — клонирование полного состояния
3. `Game.step_fast()` — шаг без obs encoding (только мутация состояния)

---

## 4. Переход на 8 игроков — как решать opponent modeling

Это реально сложная задача. В 1v1 с ghost pool оппонент полностью известен. В 8-player FFA:
- Видим **одного** оппонента за ход (дерёмся с ним)
- Остальных 6 видим **раз в ~7 ходов** каждого
- Между наблюдениями оппонент меняет доску 5-7 раз

### Уровень 1: Архетип-классификатор (самый простой)

Не пытаемся угадать точную доску. Классифицируем архетип: "мехи", "мурлоки", "зоопарк".

```
Увидели Murloc Warleader 3 хода назад
  → P(murlocs) = 0.7, P(beasts) = 0.1, P(mixed) = 0.2
  → "готовься к ядовитым мурлокам"
```

**Реализация**: оффлайн co-occurrence матрица из ES-логов. Для каждой пары карт: `P(card_j | card_i seen)`. В онлайне: Bayesian update после каждого боя.

**Плюсы**: zero inference, тривиальная реализация, уже полезно для стратегических решений (покупать контр-карты).

### Уровень 2: Board strength prediction (средняя сложность)

Предсказать точную доску нереально — слишком много вариантов. Но можно предсказать **скаляр или embedding** силы доски.

**Вариант A (скалярный)**: предсказать `board_power(t+k) | board_power(t), archetype, tier`. Простая regression, точно обучится. Даёт: "на ходу 8 у мурлочника будет board_power ≈ 25".

**Вариант B (embedding)**: предсказать `board_embedding(t+k)` — вектор из encoder'а Battle Predictor. Не конкретные юниты, а "сжатая суть" доски. Обучается через MSE на embedding'ах из Predictor.

**Вариант A+B → Battle Predictor**: предсказанный embedding/power скармливается в Predictor вместо точной доски. Predictor оценивает "мой стол vs predicted_embedding оппонента". Не идеально, но лучше чем ничего.

**Данные**: те же ES-траектории. Supervised, простой loss.

### Уровень 3: Distributional planning

Генерируем **распределение** вероятных досок, смотрим по нему.

```
7 оппонентов × K=5 possible board strengths each
Для каждого: Battle Predictor(my_board, sampled_board_embedding)
→ distribution of outcomes across all opponents
```

Не обязательно worst-case — скорее **expected value с учётом uncertainty**. Если мы не уверены в силе оппонента, выбираем стратегию, которая хороша по всему распределению (robust optimization).

**Взвешивание**: начать с простого expected value. Если агент систематически проигрывает сильным оппонентам — добавить CVaR (conditional value at risk) или risk-averse weighting. Нужен эксперимент.

K=5 × 7 opponents = 35 Predictor calls per action (~3.5ms). Feasible.

### Рекомендуемый путь

1. **Начать с 1v1 + ghost pool** (уже есть)
2. **Добавить 8-player env** с ghost pool (8 записанных траекторий). Технически это 8 ghost trajectories в пуле, каждый ход дерёмся с одним
3. **Уровень 1** (архетип-классификатор) — добавить вектор `P(archetype)` в observation
4. **Уровень 2** (trajectory prediction) — если архетип недостаточно
5. **Уровень 3** — research territory, если всё остальное plateau'd

### ES bot в контексте 8-player

ES scoring function — жадный оптимизатор своей доски, **без инфы об оппонентах**: `score = Σ w_i × Δattr_i`. Оценивает только собственную дельту состояния. В 8-player это нормально — ES строит лучшую доску "в вакууме", а PPO потом учится адаптироваться к оппонентам.

Для 8-player ghost pool: ES запускается в **8-player round-robin**. 8 ботов (возможно с разными W-векторами) играют друг против друга. Побочный продукт: 8 траекторий на игру, все попадают в ghost pool. Разные W → разные стратегии → разнообразный пул.

### 8-player env: конкретная механика

```python
class BG8PlayerEnv:
    """8-player FFA через ghost pool"""
    
    # 8 ghost trajectories (7 opponents + 1 agent)
    # Каждый ход:
    #   1. Agent играет таверну (PPO decisions)
    #   2. Выбираем оппонента для боя (round-robin или random)
    #   3. Берём ghost_board[opponent_id][turn] для боя
    #   4. Combat → damage → update HP всех 8 игроков
    #   5. "Убитые" игроки (HP≤0) выбывают
    #   6. Last man standing = winner
    
    # Observation расширяется:
    #   + HP всех 8 игроков
    #   + last_seen_board[opponent_id] (может быть устаревшим)
    #   + turns_since_seen[opponent_id]
    #   + archetype_probs[opponent_id] (если Уровень 1)
```

---

## 5. Transformer Refinements

### 5.1 Symexp Two-Hot Categorical Critic

**Зачем**: MSE-критик усредняет "выиграл +8" и "проиграл -6" в бесполезный "+1". Categorical critic выучивает распределение.

| Параметр | Значение |
|----------|----------|
| Bins | 255, uniformly in symlog[-20, +20] |
| Encoding | Two-hot: интерполяция между ближайшими бинами |
| Loss | Cross-entropy вместо MSE |
| Decoding | `V = symexp(Σ softmax(logits) × bin_centers)` |

**Текущая реализация**: categorical critic встроен прямо в `scripts/model.py` (`value_logits [B, 255]`) и обучается в `scripts/train_ppo.py` через two-hot cross-entropy. Старый SB3-вариант через `MaskableActorCriticPolicy`/`MaskablePPO.train()` остался только в legacy `scripts/categorical_critic.py`.

### 5.2 Battle Predictor → [COMBAT_CTX] Token

Добавить второй special token (аналог [GLOBAL_CTX]). Predictor embedding проецируется в d_model через zero-init MLP и участвует в self-attention.

Почему token, а не FiLM: FiLM = одинаковая модуляция всех entity. Token = per-entity selective interaction через attention (Taunt-юнит может обращать внимание на combat context, спелл в руке — нет).

До Predictor: передавать нули → zero-init = identity.

### 5.3 Auxiliary Heads (train-only)

| Head | Target | Откуда данные | Вес |
|------|--------|---------------|-----|
| Per-unit survival | P(survive) | Battle Predictor MC | 0.5 |
| Future tier | tier +3 turns | rollout buffer retrospective | 0.1 |
| Win probability | P(win) | Battle Predictor | 0.3 |

Decay к 0 за 75% обучения.

### 5.4 BC Pretrain → PPO Transition  ✅ INFRA DONE

**Текущая реализация** (April 2026):

`scripts/bc_collect.py` — генерация датасета. Стратегия: monkey-patch `game.step` внутри `es_bot_turn` через `_ESActionSnoop`, перехватываем первый вызов, конвертируем `(verb, kwargs)` в `action_int` 0..33, выполняем через `env.step` и записываем `(obs, mask, action_int)`. Корректно обрабатывает `is_targeting` (spell-with-target разворачивается в две action: PLAY → target slot) и `is_discovering` (3 опции скорятся ES-функцией). Скорость ~1,200 шагов/сек.

`scripts/bc_train.py` — cross-entropy на masked actor logits. Critic head не обучается (zero-init сохраняется для PPO). AdamW + grad clip. Сохраняет чекпоинт `{model, global_step=0, args}`, совместимый с `train_ppo --resume`.

`scripts/train_ppo.py` — `--resume` теперь толерантен к чекпоинтам без optimizer state ([train_ppo.py:217-228](scripts/train_ppo.py#L217-L228)): для BC pretrain создаётся свежий Adam.

**Команды на Kaggle:**
```bash
python scripts/bc_collect.py --episodes 5000 --weights artifacts/es_kaggle/artifacts/best.npz
python scripts/bc_train.py --epochs 15 --batch-size 512 --wandb
python scripts/train_ppo.py --resume artifacts/bc/bc_pretrain.pt --total-timesteps 5000000
```

**Caveat про action distribution**: ES бот всегда играет первую карту в руке (`hand_index=0`), поэтому в датасете действие `16` (PLAY hand[0]) сильно доминирует, а `17..25` — почти нули. Также SWAP (26..31) и FREEZE (33) ES бот не использует. PPO потом дообучит эти ветви через ε-exploration, но BC даст быстрый старт только по экономике/покупкам/UPGRADE/discovery.

**Возможные улучшения** (если BC даст слабый старт):
- Дать ES боту randomized weights ε≠0 при сборе → разнообразие траекторий
- Сэмплировать `hand_index` после shuffle при PLAY → агент увидит все слоты
- Добавить small entropy bonus в BC loss → не схлопываться в argmax

### 5.5 Percentile Return Normalization

`adv / S`, где `S = max(Perc95 - Perc05, 1)` с EMA (decay=0.99). Compute S один раз per rollout.

### 5.6 EMA Target Network

`θ_target = (1-τ)·θ_target + τ·θ_online`, τ=0.005. Target values для GAE bootstrap.

### Порядок имплементации

1. Percentile Norm (standalone, 1 день)
2. Categorical Critic (2-3 дня)
3. EMA Target (1-2 дня)
4. BC Pipeline (2-3 дня)
5. [COMBAT_CTX] (после Battle Predictor)
6. Auxiliary Heads (после Battle Predictor)

---

## 6. Performance: что реально является bottleneck

**Текущие числа (CleanRL, April 2026):**
- CleanRL PPO: ~500 FPS на SyncVectorEnv → **AsyncVectorEnv switch**, ожидаем 750-1000 FPS на Kaggle P100/T4
- Было: SB3 250 FPS → **2x speedup от удаления SB3 overhead** (unified forward, no ActionMasker, no callback dispatch)
- Board power пробил SB3-plateau (10→20-25), но плато на ~25 — лечится BC pretrain

**Текущий bottleneck = env.step() + action_masks() + obs encoding** (Python).
Model inference на T4 ~0.5-1ms, env.step ~0.1-0.3ms.

**Что ещё можно выжать:**
1. ~~AsyncVectorEnv~~ ✅ done (April 2026). Env.step параллелится с inference через worker процессы.
2. **torch.compile + fp16** — ещё 1.5-2x на GPU inference
3. **Больше envs** (16-32) — лучшая GPU утилизация при batched inference
4. **На мощном GPU (A100/4090)** — inference 3-5x быстрее T4. Реалистично 2000-5000 FPS.
5. **Multi-GPU (DDP)** — 4x A100 = ~10000 FPS. 5M steps за 8 минут.
6. **C++ obs encoding** — перенести `_get_obs()` в pybind. Полезно только если env.step станет >50% total time.
7. C++ таверна — overkill на текущем этапе. Имеет смысл только при FPS >5000 когда env.step доминирует.

---

## 7. Путь к статье

### Contribution

Систематическое эмпирическое исследование архитектурных компонентов для autobattler RL. В литературе не существует.

### Эксперименты

1. Аблация: {MLP} vs {Transformer-minimal} vs {+FiLM} vs {+GTrXL} vs {+PMA} vs {Full} vs {+Predictor}
2. Масштабирование: gap MLP vs Transformer при 40→80→100+ картах
3. Training pipeline: pure PPO vs BC+PPO vs BC+PPO+Predictor
4. Reward ablation: sparse vs dense vs predictor-augmented
5. Attention heatmaps: визуализация детекции синергий

### Venue

BG сложнее Pokemon по комбинаторике. Pokemon: action space ~9, одна битва за ход. BG: action space реально огромный (авторегрессионный: тип×источник×цель), compound multi-step decisions, positioning, stochastic combat, economy management. Методологически богаче.

Pokemon paper (arxiv 2504.04395) опубликован на **RLC 2025** (Reinforcement Learning Conference). Затем их фреймворк стал основой **NeurIPS 2025 PokéAgent Challenge**. RLC — не A/A*, но солидная venue.

Если pipeline (ES+BC+PPO+Predictor+Positioning) даёт convincing gap и аблация показывает вклад каждого компонента:

- **Реалистично**: IEEE CoG, RLC, FDG — full paper
- **Амбициозно**: NeurIPS workshop / competition track (по примеру PokéAgent)
