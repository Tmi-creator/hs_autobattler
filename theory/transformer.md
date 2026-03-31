# MARL-GPT Карточка Трансформера (Hearthstone Autobattler)

Этот документ отслеживает реализацию архитектуры **MARL-GPT**, адаптированной для проекта симулятора автобатлера (Hearthstone Battlegrounds). 

## 🏗️ Что уже реализовано (`scripts/trans.py`)

Архитектура трансформера успешно переписана из текстового генератора в агента, принимающего состояния марковского процесса (Dec-POMDP) и выдающего вероятности действий.

### 1. DecomposedEncoder (Dense Entity Representation)
Каждая сущность подается как единый плотный вектор, разделённый на 4 семантические группы:
1. **Card ID** → `nn.Embedding(202, 64)` — обучаемый per-card вектор. Каждая из 200 карт получает уникальное представление в латентном пространстве. Похожие карты (мурлоки, мехи) кластеризуются через backprop.
2. **Continuous (4)**: cost, tier, ATK, HP → symlog → Linear → d_model
3. **Binary (20)**: is_present, is_spell, keywords, effect flags → Linear → d_model
4. **Types (11)**: one-hot расовых типов → Linear → d_model

Результаты суммируются аддитивно + **Zone Embedding** (`emb_team`).

*(`emb_pos` удален — стол = неупорядоченное множество. Позиционирование через отдельный авторасстановщик).*

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
* Actor/Critic головы **управляются SB3** (`CategoricalValuePolicy`), используют `CategoricalMaskablePPO`.

### 5. Categorical Critic (Symlog Two-Hot)
Вместо MSE на скаляр V(s), критик выдаёт распределение вероятностей на 255 bins в symlog-пространстве [-20, +20]. Two-hot encoding, cross-entropy loss. Из DreamerV3.
* Bounded gradients (не взрывается от outliers)
* Мультимодальность (может выучить "или +100, или -100")
* `scripts/categorical_critic.py`: `CategoricalValuePolicy` + `CategoricalMaskablePPO`

### 6. SB3 Интеграция (`TransformerFeaturesExtractor`)
Реализована **модульная** интеграция с `CategoricalMaskablePPO` через `BaseFeaturesExtractor`:
* Парсит `Box(1036,)` → `val[B,27,38]` + `team_id` + `context`
* Прогоняет через `Encoder → FiLM → GatedTransformer → PMA → [B, d_model]`
* Card embeddings извлекаются из val[..., 2] как raw int → nn.Embedding lookup
* Categorical Critic + Actor MLP поверх
* Отдельный скрипт: `scripts/train_transformer.py`

### 6. Утилиты
* **`symlog(x)`**: $\text{sign}(x) \cdot \ln(|x| + 1)$ из DreamerV3. Безопасно сжимает HP ~50000 до ~10.8. **Включен по умолчанию** (`use_symlog=True`).

---

## 🚀 План дальнейшей работы (TODO)

- [x] **Интеграция со средой** → `TransformerFeaturesExtractor` парсит плоский `Box(1009,)` без изменения среды.
- [x] **Цикл обучения** → `train_transformer.py` с `MaskablePPO` + `TransformerFeaturesExtractor`.
- [x] **Архитектурная стабильность** → GatedResidual (GTrXL) + PMA + padding mask в attention.
- [x] **Cleanup** → Удалены `MARLGPT`, `ActorHead`, `DiscreteCriticHead` (мёртвый код). Symlog включен по умолчанию.
- [x] **Card Embeddings** → `nn.Embedding(202, 64)` вместо normalized float card_id.
- [x] **Categorical Critic** → Symlog Two-Hot 255 bins, cross-entropy loss.
- [x] **MC Oracle** → C++ engine как dense PBRS reward (20 combats per action).
- [x] **Entropy Decay** → ent_coef 0.04 → 0.01 linear decay.
- [ ] **Сбор экспертных данных**: пайплайн Behavior Cloning / учитель-бот.
- [ ] **Battle Predictor**: neural combat outcome predictor (может заменить MC Oracle).
- [ ] **Curriculum Learning**: tier-based progressive complexity.

