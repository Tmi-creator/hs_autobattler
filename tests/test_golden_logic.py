import unittest
from hearthstone.engine.pool import CardPool, SpellPool
from hearthstone.engine.tavern import TavernManager
from hearthstone.engine.entities import Player, Unit, HandCard, Spell
from hearthstone.engine.enums import CardIDs, SpellIDs


class TestGoldenLogic(unittest.TestCase):
    def setUp(self):
        self.pool = CardPool()
        self.spell_pool = SpellPool()
        self.tavern = TavernManager(self.pool, self.spell_pool)
        # Создаем игрока
        self.player = Player(uid=1, board=[], hand=[])
        self.player.tavern_tier = 1
        self.card_id = CardIDs.WRATH_WEAVER  # 1/3 Demon

    def test_triplet_stats_segregation(self):
        """
        Проверяет, что при слиянии:
        - Перманентные баффы остаются перманентными.
        - Временные баффы остаются временными.
        """
        # 1. Юнит в руке (+2/+2 Perm)
        u1 = Unit.create_from_db(self.card_id, 1, self.player.uid)
        u1.perm_atk_add = 2
        u1.perm_hp_add = 2
        u1.recalc_stats()
        self.player.hand.append(HandCard(uid=u1.uid, unit=u1))

        # 2. Юнит на столе (+3/+0 Turn)
        u2 = Unit.create_from_db(self.card_id, 2, self.player.uid)
        u2.turn_atk_add = 3
        u2.recalc_stats()
        self.player.board.append(u2)

        # 3. Юнит на столе (Чистый)
        u3 = Unit.create_from_db(self.card_id, 3, self.player.uid)
        self.player.board.append(u3)

        # Вызываем слияние
        self.tavern._check_triplet(self.player, self.card_id)

        # ПРОВЕРКИ
        self.assertEqual(len(self.player.board), 0, "Стол должен быть пуст")
        self.assertEqual(len(self.player.hand), 1, "В руке должно быть 1 золотое существо")

        golden = self.player.hand[0].unit
        self.assertTrue(golden.is_golden)

        # База золотого: 2/6
        # Perm Bonus: +2/+2
        # Turn Bonus: +3/+0

        # Проверяем поля напрямую
        self.assertEqual(golden.perm_atk_add, 2)
        self.assertEqual(golden.turn_atk_add, 3)

        # Проверяем итоговые статы сейчас
        # Atk: 2(base) + 2(perm) + 3(turn) = 7
        self.assertEqual(golden.cur_atk, 7)

        # Эмуляция конца хода
        golden.reset_turn_layer()
        golden.recalc_stats()

        # Turn bonus должен исчезнуть
        # Atk: 2(base) + 2(perm) = 4
        self.assertEqual(golden.cur_atk, 4)
        print("✅ test_triplet_stats_segregation passed")

    def test_triplet_reward_tier(self):
        """
        Проверяет получение награды и фиксацию Тира.
        """
        # Даем 3 копии в руку
        for i in range(3):
            u = Unit.create_from_db(self.card_id, i + 10, self.player.uid)
            self.player.hand.append(HandCard(uid=u.uid, unit=u))

        # Игрок на 2 тире таверны
        self.player.tavern_tier = 2

        # Слияние
        self.tavern._check_triplet(self.player, self.card_id)

        # В руке должен быть Золотой юнит, но ПОКА НЕТ награды (она дается при розыгрыше)
        self.assertEqual(len(self.player.hand), 1)
        golden_card = self.player.hand[0]

        # Апаем таверну до 4 (симуляция стратегии "leveling")
        self.player.tavern_tier = 4

        # Разыгрываем золотого
        self.tavern.play_unit(self.player, 0)

        # Теперь на столе золотой, а в руке НАГРАДА
        self.assertEqual(len(self.player.board), 1)
        self.assertEqual(len(self.player.hand), 1)

        reward_card = self.player.hand[0]
        self.assertIsNotNone(reward_card.spell)
        self.assertEqual(reward_card.spell.card_id, SpellIDs.TRIPLET_REWARD)

        # Проверяем "запеченный" тир
        # Логика: При розыгрыше (play_unit) берется min(6, tavern_tier + 1)
        # Мы были на 4 тире -> Должен быть 5
        recorded_tier = reward_card.spell.params.get('tier')
        self.assertEqual(recorded_tier, 5, f"Ожидался Тир 5, записан {recorded_tier}")
        print("✅ test_triplet_reward_tier passed")

    def test_play_reward_starts_discovery(self):
        """
        Проверяет, что розыгрыш награды запускает правильную раскопку.
        """
        # Создаем спелл-награду вручную с Тиром 6
        spell = Spell.create_from_db(SpellIDs.TRIPLET_REWARD)
        spell.params['tier'] = 1
        self.player.hand.append(HandCard(uid=999, spell=spell))

        # Разыгрываем (индекс 0)
        success, msg = self.tavern.play_unit(self.player, 0)
        print(msg)
        self.assertTrue(success)
        self.assertTrue(self.player.is_discovering)
        self.assertTrue(self.player.discovery.is_exact_tier)
        self.assertEqual(self.player.discovery.discover_tier, 1)
        print("✅ test_play_reward_starts_discovery passed")


if __name__ == '__main__':
    unittest.main()