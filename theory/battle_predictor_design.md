# Battle Predictor: Архитектура и Интеграция

## 1. Battle Analyzer: нужен или нет?

### Аргумент "за" (твой)

Battle Analyzer на полных логах боя может выучить тончайшие зависимости:
- "Ривендер рядом с таунтом умирает от Cleave" — позиционная уязвимость
- "Ривендер первым на доске = первый атакует = не успевает удвоить deathrattle"
- "Divine Shield + Poison на позиции 1 — вытягивает taunt-удар, теряя shield впустую"
- Сложные цепочки: "Scallywag deathrattle → token с Immediate Attack → trigger Deflect-o-Bot → Divine Shield"

Supervised регрессия на outcome (`P(win), E[dmg]`) **может не выучить эти нюансы**, потому что один и тот же outcome (win 60%) может получаться из совершенно разных боевых динамик.

### Контраргумент и его ограничение

На первый взгляд кажется: агенту достаточно знать *что* он выиграет, не *почему*. Для решения "купить юнита" — да, хватит дельты winrate.

Но для **позиционирования** — нет. Predictor, обученный на outcomes, скажет "эта доска = 60% win". Но он не скажет "swap позиций 2↔5 превращает 60% в 75%, потому что Cleave больше не убивает Ривендера". Чтобы агент это обнаружил через чистый Predictor, ему нужно перебрать все перестановки и сравнить winrate — это O(N!) вызовов Predictor на каждый ход.

Battle Analyzer, понимающий *динамику* боя, мог бы дать более богатый embedding, из которого агент напрямую извлекает позиционные зависимости. Это ускоряет обучение свапов и positioning.

### Практический компромисс

Analyzer как полноценный "учитель для дистилляции" — overkill. Но идея **обогащения Predictor** позиционной информацией — ценная. Два пути:

1. **Predictor + позиционные пертурбации в данных**: при генерации обучающих данных для каждой пары досок дополнительно генерировать K=10 случайных перестановок позиций → тот же бой с разными расстановками. Predictor выучит, что порядок юнитов на доске влияет на outcome. Дёшево и не требует Analyzer.

2. **Auxiliary head "unit survival"**: добавить к Predictor вспомогательную голову, предсказывающую для каждого юнита P(survive_combat). Это заставляет encoder учить позиционные зависимости (юнит рядом с таунтом под Cleave умирает чаще). По сути — "мини-Analyzer" без полных боевых логов.

**Вывод**: начинаем с Predictor + позиционные пертурбации + auxiliary survival head. Если positioning остаётся слабым — тогда думаем про полноценный Analyzer. Порог решения: если winrate агента растёт, но swap accuracy нет — значит Predictor не даёт достаточно позиционного сигнала.

---

## 2. Battle Predictor: что именно предсказываем

### Вариант A: Скалярный (простой)
```
Input:  board_A (7 units), board_B (7 units)
Output: P(win_A), P(draw), P(loss_A), E[damage_A], E[damage_B]
```
5 скаляров. MSE или cross-entropy loss. Просто, быстро, интерпретируемо.

### Вариант B: Distributional (лучше)
```
Output: histogram[damage_A] (256 bins), histogram[damage_B] (256 bins)
```
Полное распределение урона, не только матожидание. Агент может отличить "стабильные 5 урона" от "50/50 между 0 и 10". **Критично при низком HP** — агент должен минимизировать variance, а не maximizировать expected value.

### Вариант C: Embedding (мощнее, но сложнее интегрировать)
```
Output: dense vector (d=64 или 128)
```
Predictor выдаёт не скаляр, а embedding боя. Из него отдельные головки предсказывают win/loss/damage. Но сам embedding можно скормить агенту как часть observation — агент получает "интуицию" о бое, которую сам интерпретирует.

**Рекомендация**: Вариант A — слишком мало информации (5 скаляров не передают shape распределения). Начинать с **Варианта B** — distributional output даёт агенту понимание risk/variance с первого дня. Вариант C — только если нужен transfer embedding в obs агента.

---

## 3. Идея: "на ходу 5 у чувака вот такой стол, мы его победим?"

Это ключевая идея, и она глубже, чем кажется. Разберём по слоям.

### Слой 1: Что мы знаем об оппоненте

В реальном BG (и в текущей среде с ghost pool):
- Мы **видим** доску оппонента после каждого боя
- Мы **не знаем** что он купит до следующего боя
- Мы **можем оценить** его примерную траекторию по прошлым наблюдениям

В 1v1 с ghost pool ситуация проще: ghost trajectory — записанная последовательность досок, мы знаем доску оппонента на ходу N.

### Проблема 8-player FFA

В реальном BG на 8 человек всё радикально сложнее:
- Мы видим **одного** оппонента за ход (дерёмся с ним)
- Остальных 6 видим только когда дерёмся с ними — **раз в ~7 ходов каждого**
- Между наблюдениями оппонент меняет доску 5-7 раз
- Мы играем в **угадайку**: по одному snapshot'у 3 хода назад предсказать текущую доску

Пайплайн для FFA максимально неочевидный. Варианты:
1. **Bayesian update**: после каждого боя обновляем posterior на архетип оппонента → предсказываем его вероятную доску через ES-модель ("если он играет мехов на тире 4, его доска примерно такая")
2. **Trajectory prediction**: маленькая сеть, обученная на ES-траекториях, предсказывает `board(t+k) | board(t)` — "если 3 хода назад у него было вот это, сейчас у него примерно вот это"
3. **Pessimistic Predictor**: не пытаемся угадать точную доску — вместо этого предсказываем worst-case winrate против **распределения** вероятных досок оппонента

Это отдельный research вопрос. Для начала: **работаем в 1v1 с ghost pool**, где оппонент известен. FFA — следующий этап.

Но для обучения самого Battle Predictor 8-player не нужен — ему нужны просто пары досок.

### Слой 2: Откуда брать пары досок для обучения

Вот тут идея с генетикой становится конкретной:

```
┌─────────────────────────────────────────────┐
│ 1. Запускаем ES эволюцию (50 поколений)     │
│    Evolved bots играют друг с другом         │
│                                             │
│ 2. Записываем ВСЕ промежуточные состояния:  │
│    turn=1: bot_A_board, bot_B_board          │
│    turn=2: bot_A_board, bot_B_board          │
│    ...                                       │
│    turn=15: bot_A_board, bot_B_board         │
│                                             │
│ 3. Для каждой пары (board_A, board_B):      │
│    Прогоняем через C++ combat K=200 раз      │
│    Записываем: wins_A, draws, wins_B,        │
│                damage_distribution            │
│                                             │
│ 4. Датасет:                                  │
│    (board_A, board_B, turn, tier_A, tier_B)  │
│    → (P(win), P(draw), P(loss),              │
│       E[dmg_A], E[dmg_B])                    │
└─────────────────────────────────────────────┘
```

### Слой 3: Почему доски от ES лучше, чем случайные доски

Случайные доски — плохие обучающие данные:

1. **Нереалистичные комбинации.** Случайная доска может содержать 3 Golden Deflect-o-Bot + 4 Murloc Warleader. Такое никогда не случится в реальной игре. Predictor потратит capacity на моделирование несуществующих ситуаций.

2. **Нет прогрессии.** На ходу 3 доска выглядит иначе, чем на ходу 12. Случайная генерация не учитывает экономику: на ходу 3 у тебя 1-2 юнита Tier 1, а не 7 Golden Tier 5.

3. **Нет синергий.** ES-боты оптимизируют scoring function, которая включает tribal synergy. Их доски содержат реалистичные комбинации (3 мурлока + Warleader, а не 7 случайных рас). Predictor учится на том, что он реально увидит в бою.

Доски от ES — это **on-distribution** данные. Predictor учится оценивать именно те ситуации, которые возникают в реальной игре.

### Слой 4: Как это превращается в reward signal

```python
class BattlePredictor:
    """Обученный offline, замороженный при PPO-обучении."""
    
    def predict(self, board_a, board_b) -> dict:
        # board_a, board_b → encoded → forward pass
        return {
            "win_prob": 0.72,
            "expected_damage_dealt": 6.3,
            "expected_damage_taken": 2.1,
        }

# В env.step():
def _compute_predictor_reward(self, action):
    ghost_board = self._get_current_ghost_board()
    
    pred_before = self.predictor.predict(my_board, ghost_board)
    # ... execute action ...
    pred_after = self.predictor.predict(my_board, ghost_board)
    
    # Dense reward: как изменился прогноз после действия
    delta_win = pred_after["win_prob"] - pred_before["win_prob"]
    return delta_win * self.predictor_reward_scale
```

Важный нюанс: **ghost_board не меняется** между действиями таверны (это записанная доска оппонента). Меняется только наша доска. Поэтому delta_win чисто отражает эффект нашего действия.

---

## 4. Полный пайплайн генерации данных

```
Фаза 1: ES Эволюция
├── Запуск: 100 агентов × 50 поколений × 100 игр/fitness
├── Побочный продукт: ~500k полных игр с записанными досками
├── Результат: evolved_bot (оптимальный W*)
└── Время: 2-3 дня (параллелится)

Фаза 2: Data Collection
├── Evolved bot vs evolved bot:     100k игр
├── Evolved bot vs random:          50k игр  (лёгкие примеры)
├── Evolved bot vs smart_bot:       50k игр  (разнообразие стилей)
├── Результат: ~200k игр × ~12 ходов = ~2.4M пар досок
└── Время: часы (C++ combat fast)

Фаза 3: Monte Carlo Labeling
├── Для каждой пары (board_A, board_B):
│   └── C++ combat × 200 runs → statistics
├── 2.4M пар × 200 runs = 480M боёв
├── При ~95,000 боёв/сек (C++ engine, April 2026) = ~1.4 часа
├── (параллелится по ядрам: 8 cores → ~12 мин)
└── Результат: labeled dataset

Фаза 4: Train Predictor
├── Архитектура: small transformer (2 layers, d=64)
│   или MLP (512, 256, 128)
├── Input: concat(encode(board_A), encode(board_B), turn, tier_A, tier_B)
├── Output: P(win), E[dmg_dealt], E[dmg_taken]
├── Loss: cross-entropy (win/loss/draw) + MSE (damage)
├── Обучение: 50-100 epochs
└── Время: часы на GPU
```

### Entity Encoder: один на всех

Predictor, PPO-агент и Positioning Module используют **один и тот же** Entity Encoder (DecomposedEncoder из trans.py). Он кодирует сырые фичи юнита (ATK, HP, tags, types) в плотный токен.

Пайплайн:
1. Обучаем Predictor (supervised на combat data) → encoder учится понимать юнитов
2. Берём веса encoder из Predictor → инициализируем encoder PPO-агента (transfer learning)
3. PPO-агент стартует с encoder, который уже "понимает" силу юнитов, синергии, позиционные эффекты

Это не обязательно — можно обучать PPO-агент с нуля. Но transfer learning из Predictor = бесплатный warm start.

```
Board A (7 units) → DecomposedEncoder → 7 токенов → Self-Attention → PMA → emb_A
Board B (7 units) → DecomposedEncoder → 7 токенов → Self-Attention → PMA → emb_B
                                                                          ↓
                    concat(emb_A, emb_B, turn, tier_A, tier_B) → Cross-Attention / MLP
                                                                          ↓
                                                                   battle_embedding
                                                                          ↓
                                                              prediction heads
```

---

## 5. Оценка сложности

| Компонент | Усилия | Зависимости | Приоритет |
|-----------|--------|-------------|-----------|
| ES Evolution | 3-4 дня | Готовая среда с 80+ картами | Высокий |
| Game Recording (логирование досок) | 1 день | ES bot | Высокий |
| Monte Carlo Labeling pipeline | 1 день | C++ engine + записанные доски | Высокий |
| Battle Predictor model + training | 2-3 дня | Labeled dataset | Высокий |
| Интеграция в env как reward | 1 день | Trained predictor | Средний |
| Интеграция в obs как embedding | 2 дня | Trained predictor | Низкий (позже) |
| BC Pretrain от ES bot | 1 день | ES bot + PPO agent | Высокий |
| PPO Fine-tune | настройка, дни | Всё выше | — |

**Итого: ~2 недели** от готовой среды до обученного PPO агента с Battle Predictor reward.

---

## 6. Потенциальные проблемы и решения

### Ранние тиры: ~50% winrate — не баг, а фича
На ходу 1-3 доски крошечные (1-2 юнита), бои действительно близки к 50/50. Но это **не значит**, что Predictor бесполезен:

- Predictor, выдающий 52% вместо 48%, сигнализирует "твоя доска чуть сильнее среднего" — это уже полезный reward signal
- Если мы знаем результаты оппонента против других (в FFA) и видим 50%, это значит "мы на уровне" — тоже информация
- Разница между 45% и 55% на раннем ходу — это разница между "ты потеряешь 3-5 HP" и "ты нанесёшь 3-5 HP"

**Не нужно** взвешивать predictor reward по ходу. Predictor выдаёт калиброванные вероятности — если он говорит 50%, значит бой реально ~50/50, и это честный сигнал. Занижать его вес — значит терять информацию.

### Distribution shift: нужно ли дообучать Predictor?

Скорее нет. Predictor оценивает **бой между двумя досками** — это чисто функция от юнитов, позиций и механик. Если Predictor обучен на достаточно разнообразных досках (ES bot vs bot, vs random, vs smart_bot, + позиционные пертурбации), он покрывает пространство возможных досок хорошо.

PPO-агент может строить доски, которых ES не строил (новые комбинации, pivot-стратегии). Но бой между любыми двумя досками подчиняется тем же правилам. Predictor не моделирует "стиль игры" — он моделирует "кто победит при данной расстановке".

Если хочется подстраховаться — RMCTS-данные (50-200 боёв на конец хода) дают бесплатный fine-tuning dataset. Но это скорее "приятный бонус", чем необходимость.

### Reward hacking: маловероятен

Adversarial examples маловероятны на on-distribution данных. Агент строит доски из того же пула карт с теми же механиками. Он не может создать OOD-вход для Predictor.

Ground truth combat (особенно через RMCTS с N=50-200 боёв) всегда остаётся в reward — Predictor дополняет, не заменяет.

---

## 7. Predictor + RMCTS: как они дополняют друг друга

Ты упомянул замену одного боя на RMCTS с несколькими боями для получения distribution. Это хорошо сочетается с Predictor:

```
Фаза таверны (между боями):
  Агент делает действие (buy/sell/play)
  → Battle Predictor оценивает: "winrate вырос на +3%"     ← dense reward, O(1)
  → Агент делает следующее действие
  → ...
  → END_TURN

Фаза боя (конец хода):
  RMCTS: N=50-200 боёв через C++ engine                    ← ground truth distribution
  → Реальный reward: damage dealt/taken (averaged)
  → Distribution: variance, tail risk
```

**Predictor и RMCTS решают разные задачи:**
- **Predictor** = intra-turn signal ("эта покупка улучшила позицию") — нужен между действиями таверны, когда бой ещё не произошёл
- **RMCTS** = inter-turn ground truth ("этот ход в целом дал +5 damage") — даёт low-variance оценку после хода

Они не конкурируют. Predictor полезен именно потому, что между действиями бой не запускается. RMCTS полезен потому, что один бой — слишком шумный сигнал.

**Бонус**: RMCTS-данные можно использовать для **дообучения Predictor**. Каждый RMCTS-запуск = 50-200 боёв с ground truth. Это бесплатный fine-tuning dataset, который автоматически отслеживает distribution shift.

---

## 8. Predictor как основа Positioning Module

Одна нейросеть для всего (buy/sell/play/swap) — перегруз. Свапы — это отдельная задача с отдельной структурой: "переставь юнитов так, чтобы максимизировать winrate". Решение: **отдельный модуль позиционирования**, построенный поверх Predictor.

### Архитектура

```
                    ┌──────────────────────────────┐
                    │  Positioning Module           │
                    │                              │
                    │  Input: my_board (7 units)    │
                    │         enemy_board (ghost)   │
                    │                              │
                    │  Генерирует K перестановок    │
                    │  Для каждой → Predictor       │
                    │  Выбирает лучшую              │
                    │                              │
                    │  Output: optimal positions    │
                    └──────────────────────────────┘
```

### Два варианта реализации

**A. Greedy Search через Predictor (простой, baseline)**
```python
def find_best_positioning(my_board, enemy_board, predictor):
    best_score = predictor.predict(my_board, enemy_board).win_prob
    best_board = my_board.copy()
    
    # Жадные свапы: пробуем все пары, берём лучший
    improved = True
    while improved:
        improved = False
        for i in range(len(board)):
            for j in range(i+1, len(board)):
                board[i], board[j] = board[j], board[i]
                score = predictor.predict(board, enemy_board).win_prob
                if score > best_score:
                    best_score = score
                    best_board = board.copy()
                    improved = True
                else:
                    board[i], board[j] = board[j], board[i]  # revert
    
    return best_board
```
O(N^2) вызовов Predictor per iteration, ~2-3 итерации. Для N=7 это ~60-90 forward passes. Если Predictor лёгкий (MLP) — микросекунды.

**B. Learned Positioner (мощнее)**
```
my_board (7 units) → Encoder → 7 token embeddings
                                       ↓
                            Permutation head (attention-based)
                                       ↓
                              optimal ordering [3,0,5,1,6,2,4]
```
Обучается через **REINFORCE**: reward = Predictor(reordered_board, enemy) - Predictor(original_board, enemy). Predictor — замороженный, даёт gradient-free reward signal.

**Рекомендация**: начать с варианта A (greedy search). Если bottleneck — скорость (60-90 Predictor calls на каждый END_TURN), перейти на B.

### Когда вызывать Positioning Module

```
Фаза таверны:
  PPO Agent: buy, sell, play        ← основной агент
  После END_TURN (до боя):
    Positioning Module: swap         ← отдельный модуль
  Фаза боя:
    RMCTS: N боёв → reward
```

PPO Agent вообще не видит swap actions в своём action space. Он решает **что** купить/продать/сыграть. Positioning Module решает **куда** поставить. Чистое разделение ответственности.

### Плюсы разделения
1. PPO Agent не тратит capacity на позиционирование — action space меньше, обучение быстрее
2. Positioning Module оптимизирует напрямую winrate через Predictor — чёткий objective
3. Можно итерировать модули независимо
4. Positioning легко бенчмаркать: "greedy search +X% winrate vs random positioning"

---

## 9. Predictor: embedding output

Predictor возвращает **embedding**, а не только скаляры. Из embedding отдельные головки предсказывают конкретные метрики.

```
Board A → Encoder → emb_A ─┐
                            ├→ Cross-Attention → battle_embedding (d=128)
Board B → Encoder → emb_B ─┘          │
                                       ├→ Head 1: P(win), P(draw), P(loss)
                                       ├→ Head 2: damage distribution (histogram)
                                       ├→ Head 3: per-unit P(survive)  [auxiliary]
                                       └→ raw embedding → в obs агента (опционально)
```

**Зачем embedding, а не только скаляры:**
- Embedding содержит сжатую информацию о динамике боя (какие юниты сильны, какие уязвимы)
- Можно скормить embedding в PPO-агента как часть observation — агент получает "интуицию" о бое
- Per-unit survival head заставляет encoder учить позиционные зависимости
- Один и тот же encoder переиспользуется в Positioning Module

---

## 10. Open Questions

- [ ] Distributional output (histogram) vs scalar для damage: стоит ли усложнение? → **гипотеза: да, нужно тестить**
- [ ] Greedy positioning vs learned positioner: при каком N юнитов greedy становится слишком медленным?
- [ ] Сколько позиционных пертурбаций (K) генерировать при обучении Predictor? K=10? K=50?
- [ ] Как передавать Predictor embedding в PPO agent: concat к obs или через FiLM-модуляцию?
