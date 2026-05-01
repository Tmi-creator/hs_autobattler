# MARL-GPT Карточка Трансформера (Hearthstone Autobattler)

Этот документ отслеживает реализацию архитектуры **MARL-GPT**, адаптированной для проекта симулятора автобатлера (Hearthstone Battlegrounds). 

## 🏗️ Что уже реализовано (`scripts/model.py`)

Архитектура трансформера успешно переписана из текстового генератора в standalone actor-critic агента, принимающего плоское состояние среды и выдающего logits действий + categorical value logits. Актуальная реализация находится в `scripts/model.py`; `scripts/trans.py` остался legacy SB3-wrapper.

### 1. DecomposedEncoder (Dense Entity Representation)
Каждая сущность подается как единый плотный вектор, разделённый на 4 семантические группы:
1. **Card ID** → `nn.Embedding(num_card_ids, d_model // 2)` — обучаемый per-card вектор. `num_card_ids` берется из `HearthstoneEnv`, поэтому модель не завязана на старое фиксированное число `202`. Похожие карты (мурлоки, мехи) могут кластеризоваться через backprop.
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
* Actor/Critic головы встроены в `HSTransformerAgent` (`scripts/model.py`). Actor: MLP → 34 logits. Critic: MLP → 255 bins (two-hot categorical).

### 5. Categorical Critic (Symlog Two-Hot)
Вместо MSE на скаляр V(s), критик выдаёт распределение вероятностей на 255 bins в symlog-пространстве [-20, +20]. Two-hot encoding, cross-entropy loss. Из DreamerV3.
* Bounded gradients (не взрывается от outliers)
* Мультимодальность (может выучить "или +100, или -100")
* Two-hot utilities: `encode_twohot()`, `decode_value()`, `BIN_CENTERS` в `scripts/model.py`

### 6. Training Infrastructure (CleanRL)
**SB3-путь больше не основной (April/May 2026).** Миграция на CleanRL-style single-file PPO дала 2x FPS (250→500) и пробила старый plateau по board_power.

Текущие файлы:
* `scripts/model.py` — standalone `HSTransformerAgent(nn.Module)`. Единый `forward()`: obs → (action_logits, value_logits). Нет `BaseFeaturesExtractor`, нет раздельных extractor/mlp/heads.
* `scripts/train_ppo.py` — CleanRL PPO loop. Rollout collection → GAE → minibatch PPO update. Action masking через `logits[~mask] = -1e8`.
* `scripts/train_ppo.py` использует `AsyncVectorEnv`, `target_kl=0.03`, entropy decay 0.04→0.01 и умеет `--resume` из BC checkpoint без optimizer state.
* Critic в `scripts/model.py` теперь получает `features.detach()`: value loss обучает critic head, но не портит encoder features, которыми пользуется actor.
* `scripts/bc_collect.py` и `scripts/bc_train.py` — текущая BC-инфраструктура: сбор `(obs, mask, action)` с ES bot и masked cross-entropy pretrain actor'а.
* `scripts/trans.py` — **legacy**, содержит SB3-обёртку `TransformerFeaturesExtractor`. Building blocks (encoder, attention, etc.) продублированы в `model.py`.
* `scripts/categorical_critic.py` — **legacy**, two-hot utils переехали в `model.py`.

### 7. Утилиты
* **`symlog(x)`**: $\text{sign}(x) \cdot \ln(|x| + 1)$ из DreamerV3. Безопасно сжимает HP ~50000 до ~10.8. **Включен по умолчанию** (`use_symlog=True`).

---

## Plan / TODO

- [x] **Интеграция со средой** → парсинг плоского `Box(1036,)` obs
- [x] **Архитектурная стабильность** → GatedResidual (GTrXL) + PMA + padding mask
- [x] **Card Embeddings** → `nn.Embedding(num_card_ids, d_model // 2)`
- [x] **Categorical Critic** → Symlog Two-Hot 255 bins, cross-entropy loss
- [x] **MC Oracle infra** → C++ engine oracle methods в `HearthstoneEnv`; dense reward wiring остаётся отдельным TODO
- [x] **Entropy Decay** → ent_coef 0.04 → 0.01 linear decay
- [x] **SB3 → CleanRL** → standalone `HSTransformerAgent` в `scripts/model.py`, PPO loop в `scripts/train_ppo.py`. 2x FPS, пробил старый plateau.
- [x] **ES Bot** → rule-based priority loop с 23 эволюционными весами, 93.8% vs Smart Bot
- [x] **BC Pretrain infra** → `scripts/bc_collect.py` + `scripts/bc_train.py`, masked CE от ES bot действий
- [x] **BC → PPO transition infra** → `train_ppo.py --resume` принимает model-only BC checkpoint и создает свежий Adam
- [x] **AsyncVectorEnv** → текущий rollout backend в `scripts/train_ppo.py`
- [x] **Critic detach** → `critic(features.detach())`, value loss не течет в actor encoder
- [ ] **BC experiment**: прогнать BC на текущих ES weights и сравнить PPO-from-BC vs PPO-from-scratch
- [ ] **Ghost curriculum**: подключить ghost pool к CleanRL PPO pipeline
- [ ] **MC Oracle reward**: решить, включать ли dense oracle reward в `HearthstoneEnv.step()`
- [ ] **Battle Predictor**: deferred, MC Oracle достаточно быстр. Нужен только для COMBAT_CTX token embedding.

