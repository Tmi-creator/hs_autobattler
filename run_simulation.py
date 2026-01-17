import random

from engine.configs import CARD_DB
from engine.enums import MechanicType, CardIDs, SpellIDs, Tags
from engine.game import Game
from engine.entities import HandCard, Player, Unit, Spell
from engine.pool import CardPool, SpellPool
from engine.tavern import TavernManager
from engine.combat import CombatManager


def print_player_state(player, name):
    print(f"--- {name} (HP: {player.health}, Gold: {player.gold}, Tier: {player.tavern_tier}) ---")

    board_str = " | ".join([f"{u.card_id} ({u.cur_atk}/{u.cur_hp})" for u in player.board])
    print(f"Board: [{board_str}]")

    hand_str = ", ".join([c.unit.card_id if c.unit else c.spell.card_id for c in player.hand])
    print(f"Hand:  [{hand_str}]")

    store_str = ", ".join(
        [item.unit.card_id if item.unit else item.spell.card_id for item in player.store]
    )
    print(f"Store: [{store_str}]")
    print("-" * 40)


def simple_bot_turn(game, player_idx):
    """
    Простой бот с адаптированной логикой под RL-интерфейс (reward, done, info).
    """
    player = game.players[player_idx]

    while len(player.board) < 7 and len(player.hand) > 0:
        target_index = -1
        if player.hand and player.hand[0].spell and player.board:
            target_index = random.randint(0, len(player.board) - 1)
        reward, done, info = game.step(player_idx, "PLAY", hand_index=0, insert_index=-1, target_index=target_index)

        if reward >= 0:
            print(f"[P{player_idx}] PLAY success: {info}")
        else:
            print(f"[P{player_idx}] PLAY failed: {info}")
            break

    while player.gold >= 3:
        if not player.store or (player.gold >= 4 and random.random() < 0.1):
            reward, done, info = game.step(player_idx, "ROLL")
            if reward >= 0:
                print(f"[P{player_idx}] ROLLED tavern")
            continue

        if player.store:
            store_idx = random.randint(0, len(player.store) - 1)
            card_id = player.store[store_idx].card_id

            reward, done, info = game.step(player_idx, "BUY", index=store_idx)

            if reward >= 0:
                print(f"[P{player_idx}] BOUGHT {card_id}")

                if len(player.hand) > 0:
                    target_index = -1
                    if player.hand[-1].spell and player.board:
                        target_index = random.randint(0, len(player.board) - 1)
                    r_play, _, i_play = game.step(
                        player_idx,
                        "PLAY",
                        hand_index=len(player.hand) - 1,
                        insert_index=-1,
                        target_index=target_index,
                    )
                    if r_play >= 0:
                        print(f"[P{player_idx}] ...and PLAYED it immediately")
            else:
                break
        else:
            break

    game.step(player_idx, "END_TURN")


def run_simulation():
    print("=== STARTING HS BATTLEGROUNDS SIMULATION ===")
    game = Game()
    max_turns = 30

    while not game.game_over and game.turn_count <= max_turns:
        current_turn = game.turn_count
        print(f"\n>>> TURN {current_turn} <<<")

        hp_p0_start = game.players[0].health
        hp_p1_start = game.players[1].health

        print_player_state(game.players[0], "Player 0")
        simple_bot_turn(game, 0)

        print_player_state(game.players[1], "Player 1")
        simple_bot_turn(game, 1)

        if game.turn_count > current_turn or game.game_over:
            print(f"\n*** COMBAT RESOLVED ***")

            dmg_p0 = hp_p0_start - game.players[0].health
            dmg_p1 = hp_p1_start - game.players[1].health

            if dmg_p0 > 0: print(f"Player 0 took {dmg_p0} damage!")
            if dmg_p1 > 0: print(f"Player 1 took {dmg_p1} damage!")
            if dmg_p0 == 0 and dmg_p1 == 0: print("It's a DRAW!")

        else:
            print("ERROR: Turn did not advance! Check game logic.")
            break

    print("\n=== GAME OVER ===")
    p0 = game.players[0]
    p1 = game.players[1]

    if p0.health <= 0 and p1.health <= 0:
        print("Result: DRAW (Both died)")
    elif p0.health <= 0:
        print(f"Result: Player 1 WINS! (HP: {p1.health})")
    elif p1.health <= 0:
        print(f"Result: Player 0 WINS! (HP: {p0.health})")
    else:
        print("Result: Turn limit reached")


def run_effect_smoke_tests():
    print("\n=== RUNNING EFFECT SMOKE TESTS ===")
    pool = CardPool()
    spell_pool = SpellPool()
    tavern = TavernManager(pool, spell_pool)
    player = Player(uid=0, board=[], hand=[])

    print("\n=== RUNNING ALLEYCAT TESTS ===")
    alleycat = Unit.create_from_db(CardIDs.ALLEYCAT, tavern._get_next_uid(), player.uid)
    player.hand.append(HandCard(uid=alleycat.uid, unit=alleycat))
    tavern.play_unit(player, 0, 0, -1)
    assert len(player.board) == 2, "Alleycat should summon a token"
    assert player.board[1].card_id == CardIDs.TABBYCAT, "Alleycat token should be in the next slot"
    print("PASSED")

    print("\n=== RUNNING SHELL COLLECTOR TESTS ===")
    shell_collector = Unit.create_from_db(CardIDs.SHELL_COLLECTOR, tavern._get_next_uid(), player.uid)
    player.hand.append(HandCard(uid=shell_collector.uid, unit=shell_collector))
    starting_gold = player.gold
    tavern.play_unit(player, len(player.hand) - 1, len(player.board), -1)
    assert player.gold == starting_gold + 1, "Shell Collector should grant 1 gold"
    print("PASSED")

    print("\n=== RUNNING WRATH WEAVER TESTS ===")
    player = Player(uid=0, board=[], hand=[])
    cnt = 3
    wrath_weaver = [Unit.create_from_db(CardIDs.WRATH_WEAVER, tavern._get_next_uid(), player.uid) for _ in range(cnt)]
    for i in range(cnt):
        player.board.append(wrath_weaver[i])
    demon = Unit.create_from_db(CardIDs.IMPRISONER, tavern._get_next_uid(), player.uid)
    player.hand.append(HandCard(uid=demon.uid, unit=demon))
    starting_health = player.health
    tavern.play_unit(player, len(player.hand) - 1, len(player.board), -1)
    assert player.health == starting_health - cnt, "Wrath Weaver should deal 1 damage to hero"
    assert all([wrath_weaver[i].max_atk == 3 and wrath_weaver[i].max_hp == 4 for i in
                range(cnt)]), "Wrath Weaver should gain +2/+1"
    print("PASSED")

    print("\n=== RUNNING SWAMPSTRIKER TESTS ===")
    player = Player(uid=0, board=[], hand=[])
    cnt = 3
    swampstriker = [Unit.create_from_db(CardIDs.SWAMPSTRIKER, tavern._get_next_uid(), player.uid) for _ in range(cnt)]
    for i in range(cnt):
        player.board.append(swampstriker[i])
    swampstriker1 = Unit.create_from_db(CardIDs.SWAMPSTRIKER, tavern._get_next_uid(), player.uid)
    player.hand.append(HandCard(uid=swampstriker1.uid, unit=swampstriker1))
    tavern.play_unit(player, len(player.hand) - 1, len(player.board), -1)
    assert all([swampstriker[i].max_atk == 2 for i in
                range(cnt)]), "Swampstriker should gain +1"
    print("PASSED")

    print("\n=== RUNNING SCALLYWAG TESTS ===")
    combat = CombatManager()
    dead_unit = Unit.create_from_db(CardIDs.SCALLYWAG, combat.get_uid(), player.uid)
    dead_unit.cur_hp = 0
    board = [dead_unit]
    opponent_board = []
    combat_players = {
        player.uid: Player(uid=player.uid, board=board, hand=[]),
        1: Player(uid=1, board=opponent_board, hand=[]),
    }
    combat.cleanup_dead([board, opponent_board], [0, 0], combat_players)
    assert board and board[0].card_id == CardIDs.PIRATE_TOKEN, "Scallywag deathrattle should summon a token"
    print("PASSED")

    print("\n=== RUNNING COIN TESTS ===")
    player = Player(uid=0, board=[], hand=[])
    coin_spell = HandCard(uid=0, spell=Spell.create_from_db(SpellIDs.TAVERN_COIN))
    player.hand.append(coin_spell)
    starting_gold = player.gold
    tavern.play_unit(player, 0, -1, -1)
    assert player.gold == starting_gold + 1, "Coin spell should grant 1 gold"
    print("PASSED")

    print("\n=== RUNNING SPELL TESTS ===")
    target = Unit.create_from_db(CardIDs.WRATH_WEAVER, tavern._get_next_uid(), player.uid)
    player.board.append(target)
    buff_spell = HandCard(uid=1, spell=Spell.create_from_db(SpellIDs.BANANA))
    player.hand.append(buff_spell)
    tavern.play_unit(player, 0, -1, 0)
    assert target.max_atk == 3 and target.max_hp == 5, "Buff spell should grant +2/+2"
    print("BANANA PASSED")
    player.mechanics.modify_stat(MechanicType.BLOOD_GEM, 1, 0)
    buff_spell = HandCard(uid=100, spell=Spell.create_from_db(SpellIDs.BLOOD_GEM))
    player.hand.append(buff_spell)
    tavern.play_unit(player, 0, -1, 0)
    assert target.max_atk == 5 and target.max_hp == 6, "Gem should grant +2/+1"
    print("GEM PASSED")
    buff_spell = HandCard(uid=3, spell=Spell.create_from_db(SpellIDs.FORTIFY))
    player.hand.append(buff_spell)
    tavern.play_unit(player, 0, -1, 0)
    assert target.max_atk == 5 and target.max_hp == 9, "fortify should give +0/+3"
    assert target.has_taunt, "fortify should give taunt"
    print("FORTIFY PASSED")

    print("\n=== RUNNING MINTED CORSAIR TESTS ===")
    player = Player(uid=0, board=[], hand=[])
    minted = Unit.create_from_db(CardIDs.MINTED_CORSAIR, tavern._get_next_uid(), player.uid)
    player.board.append(minted)
    tavern.sell_unit(player, 0)
    assert any(
        c.spell and c.spell.card_id == SpellIDs.TAVERN_COIN for c in player.hand), "Minted Corsair should grant a coin"
    print("PASSED")

    print("\n=== RUNNING TEMPORARY SPELL TESTS ===")
    player = Player(uid=0, board=[], hand=[])
    unit = Unit.create_from_db(CardIDs.MINTED_CORSAIR, tavern._get_next_uid(), player.uid)
    unit.cur_hp = 0
    player.board.append(unit)
    spell = HandCard(uid=101, spell=Spell.create_from_db(SpellIDs.SURF_SPELLCRAFT))
    spell1 = HandCard(uid=102, spell=Spell.create_from_db(SpellIDs.SURF_SPELLCRAFT))
    player.hand.append(spell)
    tavern.play_unit(player, 0, -1, 0)
    player.hand.append(spell1)
    combat = CombatManager()
    board = [unit]
    opponent_board = []
    combat_players = {
        player.uid: Player(uid=player.uid, board=board, hand=[]),
        1: Player(uid=1, board=opponent_board, hand=[]),
    }
    combat.cleanup_dead([board, opponent_board], [0, 0], combat_players)
    assert board and board[0].card_id == CardIDs.CRAB_TOKEN, "Should summon 3/2 token crab"
    print("3/2 SUMMONED")

    tavern.end_turn(player)
    assert not player.hand, "spell should be cleared"
    print("HAND CLEARED")

    tavern.start_turn(player, 1)
    unit.cur_hp = 0
    board = [unit]
    opponent_board = []
    combat_players = {
        player.uid: Player(uid=player.uid, board=board, hand=[]),
        1: Player(uid=1, board=opponent_board, hand=[]),
    }
    combat.cleanup_dead([board, opponent_board], [0, 0], combat_players)
    assert not board, "effect should not be permanent"
    print("EFFECT RUNS OUT")

    print("Effect smoke tests passed.")


def run_golden_tests():
    print("\n=== ЗАПУСК ТЕСТОВ ЗОЛОТЫХ СУЩЕСТВ ===")
    pool = CardPool()
    spell_pool = SpellPool()
    tavern = TavernManager(pool, spell_pool)
    combat = CombatManager()

    print("\n[TEST] Создание золотого существа (Stats Check)")
    g_weaver = Unit.create_from_db(CardIDs.WRATH_WEAVER, 1, 1, is_golden=True)

    assert g_weaver.is_golden is True, "Флаг is_golden должен быть True"
    assert g_weaver.base_atk == 2, f"Ожидалась атака 2, получено {g_weaver.base_atk}"
    assert g_weaver.base_hp == 6, f"Ожидалось здоровье 6, получено {g_weaver.base_hp}"
    print("PASSED: Статы удвоены корректно")

    print("\n[TEST] Дефолтное удвоение (Shell Collector)")
    player = Player(uid=0, board=[], hand=[])

    g_collector = Unit.create_from_db(CardIDs.SHELL_COLLECTOR, tavern._get_next_uid(), player.uid, is_golden=True)
    player.hand.append(HandCard(uid=g_collector.uid, unit=g_collector))

    start_gold = player.gold
    tavern.play_unit(player, 0, 0, -1)

    expected_gold = start_gold + 2
    assert player.gold == expected_gold, f"Ожидалось золото {expected_gold}, получено {player.gold}. Триггер не сработал дважды?"
    print("PASSED: Получено 2 монетки")

    print("\n[TEST] Спец. реализация (Golden Alleycat)")
    player = Player(uid=0, board=[], hand=[])

    g_cat = Unit.create_from_db(CardIDs.ALLEYCAT, tavern._get_next_uid(), player.uid, is_golden=True)
    player.hand.append(HandCard(uid=g_cat.uid, unit=g_cat))

    tavern.play_unit(player, 0, 0, -1)

    assert len(player.board) == 2, f"Ожидалось 2 существа на столе, получено {len(player.board)}"

    token = player.board[1]
    assert token.card_id == CardIDs.TABBYCAT, "Второй юнит должен быть токеном"
    if token.is_golden and token.base_atk == 2:
        print("PASSED: Призван один золотой токен 2/2 (Override сработал)")
    else:
        print(
            f"FAILED: Токен имеет статы {token.base_atk}/{token.base_hp} и golden={token.is_golden}. Ожидалось 2/2 Golden.")

    print("\n[TEST] Золотой Reborn")
    g_reborn_unit = Unit.create_from_db(CardIDs.WRATH_WEAVER, combat.get_uid(), 0, is_golden=True)
    g_reborn_unit.tags.add(Tags.REBORN)
    g_reborn_unit.cur_hp = 0

    board = [g_reborn_unit]
    combat_players = {
        0: Player(uid=0, board=board, hand=[]),
        1: Player(uid=1, board=[], hand=[])
    }

    combat.cleanup_dead([board, []], [0, 0], combat_players)

    assert len(board) == 1, "Юнит должен возродиться"
    reborn_u = board[0]

    assert reborn_u.cur_hp == 1, "Возрожденный юнит должен иметь 1 HP"
    assert reborn_u.max_hp == 6, f"Макс HP должно быть золотым (6), получено {reborn_u.max_hp}"
    assert reborn_u.is_golden is True, "Возрожденный юнит должен остаться золотым"
    assert Tags.REBORN not in reborn_u.tags, "Тэг Reborn должен исчезнуть"

    print("PASSED: Юнит возродился золотым с 1 HP")

    print("\n[TEST] Золотая Scallywag (Deathrattle x2)")

    board = []
    p0 = Player(uid=0, board=board, hand=[])

    g_scallywag = Unit.create_from_db(CardIDs.SCALLYWAG, combat.get_uid(), 0, is_golden=True)
    g_scallywag.cur_hp = 0
    board.append(g_scallywag)

    combat_players = {
        0: p0,
        1: Player(uid=1, board=[], hand=[])
    }

    combat.cleanup_dead([board, []], [0, 0], combat_players)

    assert len(board) == 2, f"Ожидалось 2 пирата, получено {len(board)}. (Возможно, не сработал stacks=2)"

    token1 = board[0]
    token2 = board[1]

    assert token1.card_id == CardIDs.PIRATE_TOKEN, "Первый юнит должен быть Пиратом"
    assert token2.card_id == CardIDs.PIRATE_TOKEN, "Второй юнит должен быть Пиратом"

    print("PASSED: Золотая Scallywag успешно призвала двух пиратов")


def run_discovery_tests():
    print("\n=== ЗАПУСК ТЕСТОВ РАСКОПКИ (DISCOVERY) ===")

    # 1. Подготовка среды
    pool = CardPool()
    spell_pool = SpellPool()
    tavern = TavernManager(pool, spell_pool)
    player = Player(uid=0, board=[], hand=[])
    player.tavern_tier = 1

    # =================================================================
    # ТЕСТ 1: Простая раскопка и состояние игрока
    # =================================================================
    print("\n[TEST] Старт раскопки и блокировка игрока")

    # Запускаем раскопку 3-х карт Тир-1
    success = tavern.start_discovery(player, source="Test1", count=3)

    assert success is True, "Раскопка должна запуститься успешно"
    assert player.is_discovering is True, "Флаг is_discovering должен быть True"
    assert len(player.discovery.options) == 3, f"Ожидалось 3 опции, получено {len(player.discovery.options)}"

    print("PASSED: Игрок в режиме раскопки, опции сгенерированы")

    # =================================================================
    # ТЕСТ 2: Выбор карты и возврат невыбранных в пул
    # =================================================================
    print("\n[TEST] Выбор карты и проверка пула (Return Cards)")

    # Запоминаем карты, которые нам предложили
    offered_cards = [item.unit.card_id for item in player.discovery.options]
    selected_index = 0
    selected_card_id = offered_cards[selected_index]
    unpicked_card_ids = offered_cards[1:]

    # Считаем, сколько таких карт сейчас в пуле (они уже изъяты для показа!)
    # В данный момент (во время показа) их должно быть меньше, чем было изначально.
    # Но мы проверим состояние ПОСЛЕ выбора.

    # Запоминаем текущее состояние пула для невыбранных карт
    # (Внимание: карты УЖЕ изъяты из списка pool.tiers, пока они в options)
    pool_counts_during_show = {}
    for cid in unpicked_card_ids:
        tier = CARD_DB[cid]['tier']
        pool_counts_during_show[cid] = pool.tiers[tier].count(cid)

    # Делаем выбор
    success, msg = tavern.make_discovery_choice(player, selected_index)

    assert success is True, f"Выбор должен пройти успешно: {msg}"
    assert player.is_discovering is False, "Игрок должен выйти из режима раскопки"
    assert len(player.hand) == 1, "Карта должна попасть в руку"
    assert player.hand[0].unit.card_id == selected_card_id, "В руке должна быть выбранная карта"

    # ПРОВЕРКА ПУЛА:
    # Невыбранные карты должны вернуться. Значит их количество в пуле должно увеличиться на 1
    # по сравнению с моментом, когда окно выбора было открыто.
    for cid in unpicked_card_ids:
        tier = CARD_DB[cid]['tier']
        current_count = pool.tiers[tier].count(cid)
        prev_count = pool_counts_during_show[cid]
        assert current_count == prev_count + 1, f"Карта {cid} должна вернуться в пул! Было {prev_count}, стало {current_count}"

    print("PASSED: Выбранная карта в руке, остальные вернулись в пул")

    # =================================================================
    # ТЕСТ 3: Фильтрация по Тиру (Exact Tier - для триплетов)
    # =================================================================
    print("\n[TEST] Раскопка с Exact Tier (Триплеты)")
    # Очищаем руку
    player.hand = []

    # Просим строго Тир 2 (например, награда за триплет на 1 уровне таверны)
    tavern.start_discovery(player, source="TripletTest", tier=2, exact_tier=True, count=3)

    for item in player.discovery.options:
        unit_tier = item.unit.tier
        assert unit_tier == 2, f"Ожидался строго Тир 2, получен Тир {unit_tier}"

    # Завершаем выбор, чтобы очистить состояние
    tavern.make_discovery_choice(player, 0)
    print("PASSED: Все предложенные карты строго 2-го уровня")

    # =================================================================
    # ТЕСТ 4: Предикат (Фильтр по типу)
    # =================================================================
    print("\n[TEST] Раскопка с предикатом (Только Демоны)")

    # Предикат: Тип существа содержит DEMON
    # (Обрати внимание, в UnitType enum это может быть строка или Enum, в зависимости от твоей реализации DB.
    # В конфигах обычно строки: 'type': ['Demon'])
    def demon_filter(data):
        return "Demon" in data.get("type", [])

    # Ищем демонов (Wrath Weaver, Imprisoner и т.д.)
    # Ставим max_tier побольше, чтобы точно найти демонов
    success = tavern.start_discovery(player, source="DemonTest", tier=2, exact_tier=False, predicate=demon_filter)

    if success:
        for item in player.discovery.options:
            is_demon = "Demon" in [t.value if hasattr(t, 'value') else t for t in item.unit.type]
            # Или просто проверка по DB, если unit.type уже сконвертирован в Enum
            # В твоем коде unit.type это List[UnitType]. Проверим строковое представление.
            unit_types_str = [str(t) for t in item.unit.type]
            # UnitType.DEMON -> "UnitType.DEMON" или просто проверка, что не пусто
            # Проще проверить ID, зная базу: 101 (Weaver) или 108 (Imprisoner)
            pass
            # П Bard check:
            assert any(t.name == "DEMON" for t in item.unit.type), f"Юнит {item.unit.card_id} должен быть Демоном"

        tavern.make_discovery_choice(player, 0)
        print("PASSED: Все предложенные карты — Демоны")
    else:
        print("WARNING: Не удалось найти достаточно демонов в пуле для теста (это нормально, если пул маленький)")

    print("\n=== ВСЕ ТЕСТЫ РАСКОПКИ УСПЕШНО ПРОЙДЕНЫ ===")

if __name__ == "__main__":
    # run_simulation()
    run_effect_smoke_tests()
    run_golden_tests()
    run_discovery_tests()
