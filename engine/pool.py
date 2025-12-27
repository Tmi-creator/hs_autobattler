import random
from typing import Dict, List
from .configs import TIER_COPIES, CARD_DB


class CardPool:
    def __init__(self):
        # Структура: {1: ['101', '101'...], 2: ['201', ...]}
        self.tiers: Dict[int, List[str]] = {}
        self._initialize_pool()

    def _initialize_pool(self):
        """Заполняет пул картами согласно конфигу TIER_COPIES"""
        for t in TIER_COPIES.keys():
            self.tiers[t] = []

        for card_id, data in CARD_DB.items():
            if data.get('is_token', False):
                continue

            tier = data['tier']
            count = TIER_COPIES.get(tier, 0)

            self.tiers[tier].extend([card_id] * count)

    def draw_cards(self, count: int, max_tier: int) -> List[str]:
        """
        Достает N карт. Вероятность зависит от кол-ва карт в тирах.
        """
        drawn_cards = []

        available_tiers = [t for t in self.tiers.keys() if t <= max_tier]

        for _ in range(count):
            weights = [len(self.tiers[t]) for t in available_tiers]

            chosen_tier = random.choices(available_tiers, weights=weights, k=1)[0]

            card_index = random.randrange(len(self.tiers[chosen_tier]))
            card_id = self.tiers[chosen_tier].pop(card_index)

            drawn_cards.append(card_id)

        return drawn_cards

    def return_cards(self, card_ids: List[str]):
        """Возвращает карты обратно в пул (при продаже или реролле)"""
        for cid in card_ids:
            if cid in CARD_DB:
                if CARD_DB[cid].get('is_token', False):
                    continue
                tier = CARD_DB[cid]['tier']
                self.tiers[tier].append(cid)
