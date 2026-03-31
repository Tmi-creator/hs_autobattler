# Reward Design для HS:BG Autobattler — Глубокий анализ

## Контекст проекта

RL-агент для Hearthstone Battlegrounds. ~200 карт, Set Transformer encoder (DecomposedEncoder + FiLM + GTrXL + PMA), MaskablePPO с Discrete(34) action space. 1v1 через ghost pool. Categorical Critic (Two-Hot 255 bins) только что внедрён.

## Текущая проблема: Reward Hacking (Sell-Cycle Exploit)

Агент обнаружил, что продажа всех юнитов → покупка спеллов → продажа → roll → repeat генерирует стабильный поток маленьких положительных rewards. Он уходит в бой с пустой доской и проигрывает, но сумма per-action rewards превышает штраф за проигрыш.

### Текущая reward function:
```
Per-action (внутри хода таверны):
  - Buy triple (3-я копия): +2.5
  - Buy pair (2-я копия): +0.5
  - Play Triplet Reward spell (S999): +3.0
  - Sell: 0
  - Roll: 0

При END_TURN (после боя):
  - damage_dealt * 0.2
  - -damage_taken * 0.2
  - Win: +5.0
  - Loss: -5.0
```

### Почему агент эксплуатирует:
- 10-20 actions/turn × маленький positive reward per action > 1 combat penalty per turn
- Sell → gold → buy spell → instant +3.0 → repeat
- Terminal ±5.0 слишком слабый относительно суммы промежуточных наград

---

## Рассмотренные варианты

### Вариант A: Только terminal reward (чистый sparse)

```python
reward = 0.0
if game_over:
    reward = 15.0 if win else -15.0
```

**Плюсы**: невозможно эксплуатировать; единственный honest signal
**Минусы**: catastrophically sparse. 150+ actions за игру, reward на последнем. PPO с рандомных весов не сможет присвоить credit конкретным действиям. Gradient signal будет равномерно "размазан" по всем actions, включая бессмысленные.

**Оценка**: не работает без BC pretrain.

### Вариант B: PBRS через board_power

```python
r_shaped = gamma * board_power_after - board_power_before
```

**Плюсы**: теоретически доказано что не меняет optimal policy (Ng 1999); sell-cycle штрафуется (продажа = падение power)
**Минусы**: 
- **Штрафует upgrade таверны** (тратишь gold, board_power не растёт)
- **Штрафует sell для трипла** (временно убираешь юнита с доски)
- **Штрафует greedy economy** (играешь на лейт, сознательно жертвуя early game)
- board_power — грубая метрика, не учитывает синергии
- По сути навязывает "жадную" стратегию: "каждое действие должно увеличить силу стола"

**Оценка**: ломает стратегическую глубину игры.

### Вариант C: Per-turn reward (только при END_TURN)

```python
if action_type == "END_TURN":
    reward = damage_dealt * 0.3 - damage_taken * 0.3
    if win: reward += 15.0
    if loss: reward -= 15.0
else:
    reward = 0.0  # ВСЕ действия таверны — ноль
```

**Плюсы**: sell-cycle ломается (нет per-action rewards); damage feedback сохраняется
**Минусы**:
- **Credit assignment problem**: 10-15 actions в ходе, reward на последнем (END_TURN). Какое именно действие привело к победе/проигрышу? PPO будет медленно это разбирать.
- **Damage shaping конфликтует с greedy economy**: апнуть таверну на ходу 4 = слабее на ходу 4 = проиграть бой = negative reward. Но это может быть ПРАВИЛЬНОЕ решение для лейтгейма.
- **В 1v1 greedy до 1 HP + вынос в лейте = валидная стратегия**, но damage_taken наказывает за неё.

**Оценка**: лучше чем A и B, но credit assignment и damage bias — реальные проблемы.

### Вариант D: Terminal-only + BC pretrain (отложенный фикс)

```python
reward = 0.0
if game_over:
    reward = 15.0 if win else -15.0
```
Но агент стартует не с нуля, а с BC pretrain от ES bot. Уже умеет покупать/продавать/играть нормально. PPO только оптимизирует.

**Плюсы**: honest signal, BC решает cold start, нет shaping для эксплуатации
**Минусы**: требует ES bot + BC pipeline (ещё не реализовано); при PPO fine-tune с pure terminal agentмможет "забыть" BC поведение и деградировать

**Оценка**: правильный долгосрочный ответ, но требует инфраструктуры.

### Вариант E: Battle Predictor как dense reward (будущий фикс)

```python
# Между действиями таверны:
r_predictor = predictor(my_board_after, ghost_board).win_prob 
            - predictor(my_board_before, ghost_board).win_prob
# Это Potential-Based Reward Shaping где Phi = P(win game)

# При END_TURN: реальный combat result
```

**Плюсы**: PBRS с честным потенциалом (P(win game), не board_power); учитывает синергии, позиционирование, всё; доказанно safe (Ng 1999)
**Минусы**: Battle Predictor ещё не обучен; требует ES данные для обучения

**Оценка**: лучший вариант, но зависит от другой инфраструктуры.

---

## Ключевая дилемма: Credit Assignment

Фундаментальная проблема autobattler RL: **между действиями и их результатом — длинная цепочка**.

```
Turn 4: Buy Murloc Warleader (action #47)
Turn 4: Buy Murloc Tidehunter (action #49)
Turn 4: END_TURN → combat
Turn 5: Buy another murloc (action #62)
...
Turn 8: Murloc army wins fight decisively (reward!)
```

Reward за бой на ходу 8 нужно аттрибутировать покупке Warleader на ходу 4 (40+ actions назад). PPO с gamma=0.99 и 40 шагов: discount = 0.99^40 = 0.67 — сигнал ослабляется на 33%.

Но хуже: между action #47 и reward было 15 других действий (rolls, sells, plays). PPO должен понять что именно action #47 привёл к reward, а не action #48 (roll) или #50 (sell). Это **multi-step credit assignment** — одна из самых сложных проблем в RL.

### Как это решается в литературе:

1. **AlphaStar**: MCTS + policy iteration. MCTS делает look-ahead и присваивает credit через tree backup. Дорого.

2. **Metamon (Pokemon)**: Offline RL на миллионах human trajectories. Credit assignment через большую модель (200M params) на огромных данных.

3. **DreamerV3**: World model для imagination. Imagination rollouts дают dense signal. Но у нас уже есть perfect world model (C++ engine).

4. **BC pretrain + PPO fine-tune**: BC даёт reasonable starting policy, PPO только корректирует. Credit assignment проще потому что изменения относительно малы.

5. **Battle Predictor PBRS**: Dense reward на каждом action, основанный на предсказании game outcome. Превращает sparse terminal в dense signal.

---

## Вопросы для обсуждения

1. **Вариант C (per-turn only) как временное решение**: достаточно ли для обучения пока не готов ES bot / Battle Predictor? Или credit assignment problem слишком тяжёлый?

2. **Damage shaping**: стоит ли вообще включать damage_dealt/damage_taken? Или даже per-turn damage — это bias против greedy economy?

3. **Размер terminal reward**: ±5 явно мало. ±15? ±50? ±100? Как калибровать относительно gamma и episode length?

4. **Gamma**: текущий 0.99. Может стоит 0.999 для лучшего propagation terminal reward? С categorical critic это может быть стабильно.

5. **Есть ли промежуточный вариант**: что-то между "per-action shaping (exploitable)" и "pure terminal (too sparse)"?

6. **Entropy coefficient**: текущий 0.01. При sparse rewards нужен больше (0.02-0.05) чтобы агент не коллапсировал в одну стратегию от первого случайного success.

7. **n_steps**: текущий 2048. При sparse rewards больше n_steps = больше data per update = лучше credit assignment. 4096? 8192?

---

## Предложенный experiment plan

Запустить 3 конфигурации параллельно на Kaggle и сравнить:

### Config 1: Per-turn damage + terminal ±15
```python
END_TURN: damage_dealt * 0.3 - damage_taken * 0.3
game_over: ±15.0
all other actions: 0.0
```

### Config 2: Terminal-only ±25
```python
game_over: ±25.0
all other actions: 0.0
```

### Config 3: Terminal ±15 + gamma=0.999 + ent_coef=0.03
```python
game_over: ±15.0
all other actions: 0.0
gamma: 0.999  (вместо 0.99)
ent_coef: 0.03 (вместо 0.01)
n_steps: 4096 (вместо 2048)
```

Метрики: avg_board_power, win_rate vs smart_bot, value_loss stability, entropy trajectory.

---

## Принятое решение: Config 1 (реализовано)

Реализован Config 1 "Round Outcomes + Penalty" в `hs_env.py`:
- Все per-action positive rewards удалены (triple +2.5, pair +0.5, spell +3.0)
- Действия таверны: -0.005 (микро-штраф), END_TURN: 0 (бесплатно)
- Round outcome: +1.0 (победа), -1.0 (поражение), 0 (ничья)
- Terminal: ±50.0
- gamma: 0.999, ent_coef: 0.04, n_steps: 4096

Config 2 и 3 сохранены ниже для будущих stress-тестов.

### Config 2: "Damage Annealing" (будущий stress-test)
```python
# hs_env.py changes:
# Tavern actions: reward = 0.0
# END_TURN: reward = (damage_dealt * 0.1 - damage_taken * 0.1) * decay
#   decay = max(0.0, 1.0 - current_timestep / (total_timesteps * 0.5))
# Game over: ±50.0
# Hyperparams: gamma=0.999, n_steps=4096, ent_coef=0.04
```

### Config 3: "Pure Sparse" (будущий stress-test)
```python
# hs_env.py changes:
# ALL actions including END_TURN: reward = 0.0
# Game over ONLY: ±15.0
# Hyperparams: gamma=0.9995, n_steps=8192, ent_coef=0.05
```
