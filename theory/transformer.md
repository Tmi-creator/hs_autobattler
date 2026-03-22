# MARL-GPT Карточка Трансформера (Hearthstone Autobattler)

Этот документ отслеживает реализацию архитектуры **MARL-GPT**, адаптированной для проекта симулятора автобатлера (Hearthstone Battlegrounds). 

## 🏗️ Что уже реализовано (`scripts/trans.py`)

Архитектура трансформера успешно переписана из текстового генератора в агента, принимающего состояния марковского процесса (Dec-POMDP) и выдающего вероятности действий.

### 1. Dense Observation Encoder (Смена парадигмы с EAV)
В отличие от MARL-GPT, где каждый атрибут является отдельным токеном (что полезно для зоопарка разных сред, но избыточно здесь), мы реализовали **Плотную репрезентацию объектов (Dense Entity Representation)**:
* Каждая сущность (карта в Таверне, юнит на столе) подается как единый плотный вектор признаков `val` размерностью `d_features` (например, все 35 полей).
* Это решает **Binding Problem** на аппаратном уровне: нейросети больше не нужно тратить первые слои на то, чтобы связать ХП и Атаку одного существа, так как они сшиты изначально.
* Вычислительная сложность падает: вместо $N \approx 1000$ (разбитых токенов фичей), у нас $N \approx 30$ (карт на доске в сумме).

**В состав токена прибавляются позиционные эмбеддинги:**
  1. **`emb_team`**: Идентификатор зоны (доска игрока, раздача таверны, рука, доска врага).
  2. **`emb_time`**: Временной шаг (история состояний для учета частичной наблюдаемости).
*(`emb_pos` удален, так как на этапе таверны стол является неупорядоченным множеством (Set). Позиционирование будет решаться как отдельный Action/Heuristic в конце хода)*.

### 2. Core (TransformerBlock)
Ядро трансформера — стандартный Pre-LN блок Self-Attention, дополненный стабилизирующими механизмами:
* **`TransformerBlock`**: Self-Attention без позиционного энкодинга → **Pure Set Transformer** (permutation-equivariant). При $N \sim 30$ сложность $O(N^2) \approx 900$ — быстрее чем ISAB.
* **`GatedResidual` (GTrXL)**: GRU-подобный шлюз вместо тупого `x + f(x)`:
  * $z = \sigma(W_z \cdot [x, y] + b_z)$ → gate
  * $r = \sigma(W_r \cdot [x, y] + b_r)$ → reset
  * $h = \tanh(W_h \cdot [r \odot x, y])$ → candidate
  * $out = (1 - z) \odot x + z \odot h$
  * **Identity Init**: $b_z = -3 \Rightarrow z \approx 0.05$ на старте. Трансформер фактически *выключен*, агент учит базу через линейные слои. Градиенты постепенно открывают шлюз по мере обучения. Это предотвращает ранний коллапс PPO с трансформером.
  * Опционально: `use_gating=False` для отключения (классический residual).
* **Padding Mask**: пустые слоты (`is_present == 0`) получают `-inf` в attention scores и полностью исключаются из self-attention, а не только из пулинга.
* **`FFN_SwiGLU`**: Feed-Forward сеть с Swish-Gated Linear Unit (SiLU).
* **`RMSNorm`**: Root Mean Square Normalization вместо LayerNorm.

### 3. FiLM (Feature-wise Linear Modulation)
Разработан генератор динамического контекста. Вместо слабой аддитивной конкатенации `[X, C]`, глобальный стейт `C` (содержащий тир таверны, количество золота, ХП игрока) используется для генерации гиперпараметров $\gamma$ и $\beta$. Они мультипликативно воздействуют (модулируют) эмбеддинги `X`, позволяя аппаратно "выключать" или "усиливать" целые пространства признаков локальных карт до того, как они попадут в `Set Transformer`.
* **Zero-Initialization Trick (Identity Start)**: Последний слой `FiLMGenerator` инициализируется строго нулями, а формула модуляции изменена на $X' = X \cdot (\gamma + 1.0) + \beta$. Это гарантирует, что на старте обучения (когда $\gamma=0, \beta=0$) FiLM работает как функция тождества ($X' = X \cdot 1 + 0 = X$), предотвращая зануление признаков и затухание градиентов на первых эпохах.

### 4. Пулинг
* **`PMA` (Pooling by Multihead Attention)**: обучаемый Seed-вектор $S$ выступает как Query, токены стола — как Key/Value. Динамически извлекает информацию, критичную для оценки состояния, вместо тупого `mean()` который уничтожает найденные синергии между картами. Опционально: `use_pma=False` для fallback на masked mean pooling.
* Actor/Critic головы **управляются SB3** (`MaskableActorCriticPolicy`), standalone ActorHead/DiscreteCriticHead удалены как мёртвый код.

### 5. SB3 Интеграция (`TransformerFeaturesExtractor`)
Реализована **модульная** интеграция с `MaskablePPO` через `BaseFeaturesExtractor`. Без изменений `hs_env.py` / `train.py`:
* Парсит `Box(1009,)` → `val[B,27,37]` + `team_id` + `context`
* Прогоняет через `Encoder → FiLM → GatedTransformer → PMA → [B, d_model]`
* SB3 подключает Actor/Critic MLP поверх
* Отдельный скрипт: `scripts/train_transformer.py`

### 6. Утилиты
* **`symlog(x)`**: $\text{sign}(x) \cdot \ln(|x| + 1)$ из DreamerV3. Безопасно сжимает HP ~50000 до ~10.8. **Включен по умолчанию** (`use_symlog=True`).

---

## 🚀 План дальнейшей работы (TODO)

- [x] **Интеграция со средой** → `TransformerFeaturesExtractor` парсит плоский `Box(1009,)` без изменения среды.
- [x] **Цикл обучения** → `train_transformer.py` с `MaskablePPO` + `TransformerFeaturesExtractor`.
- [x] **Архитектурная стабильность** → GatedResidual (GTrXL) + PMA + padding mask в attention.
- [x] **Cleanup** → Удалены `MARLGPT`, `ActorHead`, `DiscreteCriticHead` (мёртвый код). Symlog включен по умолчанию. PMA нормализация исправлена.
- [ ] **Сбор экспертных данных**: пайплайн Behavior Cloning / учитель-бот.
- [ ] **Сложные механики внимания**: MVP Detection, Spatial Encoding из `ideas.md`.

