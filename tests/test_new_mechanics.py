import unittest
from hearthstone.engine.combat import CombatManager
from hearthstone.engine.entities import Player, Unit
from hearthstone.engine.enums import CardIDs, Tags, UnitType
from hearthstone.engine.event_system import EventType, Event, EntityRef, PosRef, Zone


class TestNewMechanics(unittest.TestCase):
    def setUp(self):
        self.combat = CombatManager()
        self.uid_counter = 1000

    def _make_unit(self, card_id, owner_id):
        self.uid_counter += 1
        return Unit.create_from_db(card_id, self.uid_counter, owner_id)

    def test_spawn_of_nzoth_deathrattle(self):
        """
        Тест Порождения Н'Зота:
        Проверяем, что при смерти он баффает остальных существ +1/+1.
        """
        print("\n[TEST] Spawn of N'Zoth (Global Buff)")

        # Стол: [Спавн (мертв), Кот, Кот]
        p0 = Player(uid=0, board=[], hand=[])

        spawn = self._make_unit(CardIDs.SPAWN_OF_NZOTH, 0)
        cat1 = self._make_unit(CardIDs.TABBYCAT, 0)  # 1/1
        cat2 = self._make_unit(CardIDs.TABBYCAT, 0)  # 1/1

        p0.board = [spawn, cat1, cat2]

        # Симулируем, что Спавн умер
        spawn.cur_hp = 0

        # Подготовка комбата
        combat_players = {0: p0, 1: Player(uid=1, board=[], hand=[])}

        # Запускаем очистку трупов (она триггерит хрипы)
        # boards, attack_indices, players
        self.combat.cleanup_dead([p0.board, []], [0, 0], combat_players)

        # Проверки
        self.assertNotIn(spawn, p0.board, "Спавн должен исчезнуть со стола")

        # Коты должны стать 2/2 (получили +1/+1 в combat layer)
        print(f"Cat1 Stats: {cat1.cur_atk}/{cat1.cur_hp} (Expected 2/2)")
        self.assertEqual(cat1.combat_atk_add, 1)
        self.assertEqual(cat1.combat_hp_add, 1)
        self.assertEqual(cat1.cur_atk, 2)
        print("PASSED: Spawn buffed everyone")

    def test_kaboom_bot_damage(self):
        """
        Тест Взрывоопасного бота:
        Проверяем нанесение 4 ед. урона врагу при смерти.
        """
        print("\n[TEST] Kaboom Bot (Bomb Damage)")

        p0 = Player(uid=0, board=[], hand=[])
        kaboom = self._make_unit(CardIDs.KABOOM_BOT, 0)
        p0.board = [kaboom]

        p1 = Player(uid=1, board=[], hand=[])
        # Враг: 0 атаки, 10 здоровья
        dummy = self._make_unit(CardIDs.ANNOY_O_TRON, 1)
        dummy.cur_atk = 0
        dummy.max_hp = 10
        dummy.cur_hp = 10
        dummy.perm_hp_add = 8
        dummy.tags.discard(Tags.DIVINE_SHIELD)  # Снимаем щит для чистоты теста урона
        p1.board = [dummy]

        # Убиваем бота
        kaboom.cur_hp = 0

        combat_players = {0: p0, 1: p1}
        self.combat.cleanup_dead([p0.board, p1.board], [0, 0], combat_players)

        print(f"Enemy Dummy HP: {dummy.cur_hp} (Expected 6)")
        self.assertEqual(dummy.cur_hp, 6, "Бот должен нанести 4 урона (10 - 4 = 6)")
        print("PASSED: Kaboom Bot dealt damage")

    def test_kaboom_bot_golden(self):
        """
        Тест Золотого Взрывоопасного бота:
        Должен нанести 4 урона ДВАЖДЫ (сняв щит первым тиком и убив вторым, если хп мало).
        """
        print("\n[TEST] Golden Kaboom Bot (Double Trigger)")

        p0 = Player(uid=0, board=[], hand=[])
        # Создаем золотого вручную
        g_kaboom = self._make_unit(CardIDs.KABOOM_BOT, 0)
        g_kaboom.is_golden = True
        g_kaboom.cur_hp = 0
        p0.board = [g_kaboom]

        p1 = Player(uid=1, board=[], hand=[])
        # Враг с щитом и 3 хп
        target = self._make_unit(CardIDs.ANNOY_O_TRON, 1)
        target.cur_hp = 3
        target.tags.add(Tags.DIVINE_SHIELD)
        p1.board = [target]

        combat_players = {0: p0, 1: p1}
        self.combat.cleanup_dead([p0.board, p1.board], [0, 0], combat_players)

        # Логика:
        # 1-й тик (4 урона) -> Попадает в Divine Shield -> Щит лопается, урон 0.
        # 2-й тик (4 урона) -> Попадает в тушку -> 3 - 4 = -1 HP.

        print(f"Target HP: {target.cur_hp}, Has Shield: {target.has_divine_shield}")
        self.assertFalse(target.has_divine_shield, "Щит должен быть сбит")
        self.assertTrue(target.cur_hp <= 0, "Юнит должен умереть от второго взрыва")
        print("PASSED: Golden bot fired twice")

    def test_deflect_o_bot_reset(self):
        """
        СЛОЖНЫЙ ТЕСТ: Дефлектобот + Реборн Механизм.
        Сценарий:
        1. У Дефлектобота сбит щит.
        2. Рядом умирает Механизм с Reborn.
        3. Механизм возрождается (Summon).
        4. Дефлектобот должен получить +1 Атаку и вернуть Щит.
        """
        print("\n[TEST] Deflect-o-Bot Logic (Reborn Interaction)")

        p0 = Player(uid=0, board=[], hand=[])

        # 1. Deflect-o-Bot (3/2) без щита
        deflecto = self._make_unit(CardIDs.DEFLECT_O_BOT, 0)
        deflecto.tags.discard(Tags.DIVINE_SHIELD)  # Имитируем, что щит уже сбили
        base_atk = deflecto.cur_atk

        # 2. Механизм с Reborn (умирающий)
        # Используем Annoy-o-Tron (Mech) и даем ему Reborn
        dying_mech = self._make_unit(CardIDs.ANNOY_O_TRON, 0)
        dying_mech.tags.add(Tags.REBORN)
        dying_mech.cur_hp = 0  # Он умер в бою

        p0.board = [deflecto, dying_mech]

        combat_players = {0: p0, 1: Player(uid=1, board=[], hand=[])}

        print(f"BEFORE: Deflecto Shield: {deflecto.has_divine_shield}, Atk: {deflecto.cur_atk}")

        # Запускаем процессинг смертей
        # Это вызовет:
        # 1. Удаление dying_mech
        # 2. Триггер Reborn -> Summon (призыв)
        # 3. Триггер Deflecto на Summon -> Buff
        self.combat.cleanup_dead([p0.board, []], [0, 0], combat_players)

        print(f"AFTER:  Deflecto Shield: {deflecto.has_divine_shield}, Atk: {deflecto.cur_atk}")

        # Проверки
        self.assertTrue(deflecto.has_divine_shield, "Дефлектобот должен восстановить щит!")
        self.assertEqual(deflecto.cur_atk, base_atk + 2, "Дефлектобот должен получить +2 к атаке")

        # Проверяем, что токен реально призвался
        self.assertEqual(len(p0.board), 2, "На столе должен появиться возрожденный механизм")
        token = p0.board[1]
        self.assertIn(UnitType.MECH, token.types, "Призванный токен должен быть механизмом")

        print("PASSED: Deflect-o-Bot successfully reset off Reborn")


if __name__ == '__main__':
    unittest.main()
