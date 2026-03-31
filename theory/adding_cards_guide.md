# Гайд: добавление новых карт

## Быстрый чеклист

1. `enums.py` — добавить `CardIDs.NEW_CARD = "XXX"` (и токен если есть)
2. `card_def.py` — добавить `CardDef(...)` в `ALL_CARDS` (с `EffectDef` если есть эффект)
3. `python scripts/generate_cpp_effects.py` — перегенерировать C++
4. `cmake --build cpp/build` — пересобрать C++
5. Написать тест в `tests/test_card_def.py` или `tests/test_coverage_gaps.py`

Всё. Больше ничего трогать не нужно — `CARD_DB`, `TRIGGER_REGISTRY`, C++ effects генерируются автоматически.

---

## Подробно

### Шаг 1: enums.py

```python
class CardIDs(str, Enum):
    # --- TIER 2 ---
    FREEDEALING_GAMBLER = "201"
    SHELL_COLLECTOR = "202"
    SEWER_RAT = "203"
    MY_NEW_CARD = "204"          # ← добавить

    # --- TOKENS ---
    ...
    MY_NEW_TOKEN = "t007"        # ← если карта спавнит токен
```

**Нумерация**: `"XYY"` где X = тир, YY = порядковый номер. Токены: `"tNNN"`.

### Шаг 2: card_def.py

Найти секцию нужного тира в `ALL_CARDS` и добавить `CardDef`:

```python
# Vanilla карта (без эффектов)
CardDef(
    CardIDs.MY_NEW_CARD, "My New Card", 2, 3, 4,
    [UnitType.BEAST],
    tags={Tags.TAUNT},
),

# Карта с DR
CardDef(
    CardIDs.MY_NEW_CARD, "My New Card", 2, 3, 4,
    [UnitType.MECH],
    deathrattle=True,
    effects=[DeathrattleSummon(token_id=CardIDs.MY_NEW_TOKEN, count=2)],
),

# Токен (в секции TOKENS)
CardDef(
    CardIDs.MY_NEW_TOKEN, "My Token", 1, 1, 1,
    [UnitType.MECH],
    is_token=True,
),
```

### Доступные EffectDef

| EffectDef | Что делает | Пример карты |
|-----------|-----------|-------------|
| `DeathrattleSummon(token_id, count)` | DR: спавн токенов | Cord Puller, Harmless Bonehead |
| `DeathrattleSummonWithTag(token_id, count, tag)` | DR: спавн + доп. тег | Twilight Hatchling (IMMEDIATE_ATTACK) |
| `BattlecryAddSpell(spell_id, count)` | BC: дать спелл | Razorfen Geomancer, Shell Collector |
| `BattlecryMakeGolden()` | BC: сделать себя золотым | Aureate Laureate |
| `BattlecrySpellDiscount(amount)` | BC: скидка на спелл | Ominous Seer |
| `BattlecryModifyMechanic(mechanic, atk, hp)` | BC: глобальный бафф механики | Dune Dweller |
| `BattlecryConsumeShopUnit()` | BC: сожрать юнита из магазина | Picky Eater |
| `OnFriendlyPlayType(type, atk, hp)` | При игре своего юнита типа X: бафф | Swampstriker |
| `OnFriendlyPlayTypeDamageHero(type, hero_dmg, atk, hp)` | То же + урон герою | Wrath Weaver |
| `OnFriendlyDeathBuff(atk, hp)` | При смерти союзника: бафф (combat) | Rot Hide Gnoll |
| `StartOfCombatBuffSelfByTier()` | SoC: +тир/+тир | Misfit Dragonling |
| `SellAddSpell(spell_id, count)` | При продаже: дать спелл | Minted Corsair |
| `SellGetRandomUnit(tier)` | При продаже: рандом юнит из пула | River Skipper |

### Кастомный эффект

Если карта не ложится ни в один EffectDef:

```python
# 1. Написать функцию в card_def.py
def _my_custom_effect(ctx: EffectContext, event: Event, trigger_uid: int) -> None:
    ...

# 2. Использовать CustomEffect
CardDef(
    CardIDs.MY_CARD, "My Card", 3, 5, 5,
    [UnitType.DRAGON],
    effects=[CustomEffect(trigger_defs=[
        TriggerDef(
            event_type=EventType.MINION_DAMAGED,
            condition=some_condition,
            effect=_my_custom_effect,
            name="My Card Trigger",
        )
    ])],
),

# 3. Если эффект combat-relevant — написать C++ вручную в cpp/src/custom_effects.cpp
```

### Аура-карты

Ауры (Dire Wolf Alpha, Murloc Warleader) — **не через EffectDef**, а через `auras.py`:

```python
# auras.py
def _my_aura(source: Unit, board: List[Unit], idx: int) -> None:
    bonus = 2 if source.is_golden else 1
    for i, unit in enumerate(board):
        if i == idx:
            continue
        if UnitType.BEAST in unit.types:
            unit.aura_atk_add += bonus

AURA_REGISTRY[CardIDs.MY_AURA_CARD] = _my_aura
```

C++ аура: `cpp/src/auras.cpp` — аналогичная функция.

---

## Что генерируется автоматически

| Артефакт | Источник | Когда |
|----------|----------|-------|
| `CARD_DB` (configs.py) | `card_def.py: build_card_db()` | При импорте |
| `TRIGGER_REGISTRY` (effects.py) | `card_def.py: build_trigger_registry()` | При импорте |
| `cpp/include/generated_card_ids.h` | `generate_cpp_effects.py` | Ручной запуск |
| `cpp/src/generated_effects.cpp` | `generate_cpp_effects.py` | Ручной запуск |

## Какие эффекты попадают в C++

Только **combat-relevant** (те, что стреляют во время боя):
- `DeathrattleSummon` / `DeathrattleSummonWithTag`
- `OnFriendlyDeathBuff`
- `StartOfCombatBuffSelfByTier`
- `OnFriendlyPlayType` / `OnFriendlyPlayTypeDamageHero`

**Не попадают** (только Python, таверна):
- `BattlecryAddSpell`, `BattlecryMakeGolden`, `BattlecrySpellDiscount`
- `BattlecryModifyMechanic`, `BattlecryConsumeShopUnit`
- `SellAddSpell`, `SellGetRandomUnit`

## Что НЕ нужно трогать

- `configs.py` — CARD_DB генерируется из card_def
- `effects.py` — TRIGGER_REGISTRY генерируется из card_def
- `cpp/include/types.h` — подключает generated_card_ids.h
- `cpp/src/effects.cpp` — заменён на generated_effects.cpp
- `cpp_bridge.py` — маппинг автоматический из enum order

## Новый EffectDef

Если нужен **новый тип** декларативного эффекта (не кастомный, а переиспользуемый):

1. Добавить `@dataclass class MyNewEffect(EffectDef)` в card_def.py
2. Добавить factory `_make_my_new_effect(...)` в card_def.py
3. Добавить `elif isinstance(eff, MyNewEffect)` в `build_trigger_registry()`
4. Если combat-relevant — добавить генерацию в `generate_cpp_effects.py`
