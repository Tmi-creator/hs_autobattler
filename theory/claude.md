# HS Autobattler: Training Pipeline & Architecture Design

## Context

Среда дорабатывается (карты, UPGRADE/FREEZE, аура) — это механическая работа. Главный вопрос: **когда среда будет готова (80-100+ карт), как на ней учить?** Этот документ описывает полный training pipeline и архитектурные решения.

Связанные docs:
- `theory/battle_predictor_design.md` — Battle Predictor, Positioning Module
- `theory/todo.md` — оригинальный roadmap

Актуальные числа производительности:
- C++ combat: 7,412 combats/sec
- MLP training: ~1,200 FPS (DummyVecEnv)
- Transformer training: ~250 FPS (DummyVecEnv)

---

## 1. Полный Training Pipeline

```
Этап 0: ES Evolution ──────────────► Evolved Bot (W*)
  │  (mu+lambda, 50 поколений)        + 200k записанных игр
  │
  ├──► Этап 1: Battle Predictor ────► Trained Predictor
  │    (supervised на MC combat data)   embedding + P(win) + survival
  │
  └──► Этап 2: BC Pretrain ─────────► BC-инициализированный Actor
       (cross-entropy от ES bot)
                │
                ▼
       Этап 3: PPO Fine-tune ───────► Trained Agent
         - Ghost pool curriculum (70% ghost / 30% bot)
         - Predictor как dense reward
         - Categorical Critic + EMA target
         - Percentile return normalization
                │
                ▼
       Этап 4: Self-Play Iterations
         - Лучшая модель vs копии → ghost pool
         - Опционально: RMCTS для complex decisions
         - Повторять 3-4 пока растёт winrate
```

### Почему именно такой порядок

- **Чистый PPO from scratch** на 100+ картах обречён: комбинаторика таверны слишком велика, агент потратит миллионы шагов на обучение "покупай юнитов"
- **BC pretrain** перескакивает через cold start за часы
- **Battle Predictor** даёт dense reward между действиями таверны (PPO видит эффект каждой покупки, не только конец хода)
- **ES bot** масштабируется автоматически: добавил карты → перезапустил эволюцию → новый оптимальный bot

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

**SB3**: subclass `MaskableActorCriticPolicy` (override value_net, predict_values, evaluate_actions) + subclass `MaskablePPO.train()` (replace MSE loss).

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

### 5.4 BC Pretrain → PPO Transition

```
BC: cross-entropy от ES-bot, lr=1e-4, ~20 epochs
    dummy critic target=0 (vf_coef=0.01)
    
Переход:
    PPO lr=3e-5 → warmup до 1e-4 за 50k steps
    Freeze features_extractor на 100k steps
    target_kl=0.02, ent_coef=0.02→0.01
```

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

При 250 FPS transformer на DummyVecEnv:
- **Bottleneck = model inference**, не Python env
- C++ tavern НЕ поможет (env шаги быстрые, inference медленный)
- Что поможет:
  1. **Async envs** (Sample Factory / custom) — убрать sync overhead, N_envs буст
  2. **ONNX/TorchScript** для inference — 2-3x speedup
  3. **Mixed precision** (fp16) — 1.5-2x на GPU inference
  4. **Больше envs** — текущий DummyVecEnv ограничен single-threaded Python

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
