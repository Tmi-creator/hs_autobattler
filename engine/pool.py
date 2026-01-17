import random
from typing import Dict, List, Callable

from .configs import TIER_COPIES, CARD_DB, SPELL_DB
from .enums import CardIDs, SpellIDs


class CardPool:
    def __init__(self):
        # Структура: {1: ['101', '101'...], 2: ['201', ...]}
        self.tiers: Dict[int, List[CardIDs]] = {}
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

    def return_cards(self, card_ids: List[CardIDs]):
        """Возвращает карты обратно в пул (при продаже или реролле)"""
        for cid in card_ids:
            if cid in CARD_DB:
                if CARD_DB[cid].get('is_token', False):
                    continue
                tier = CARD_DB[cid]['tier']
                self.tiers[tier].append(cid)

    def draw_discovery_cards(self, count: int, tier: int, exact_tier: bool = False,
                             predicate: Callable[[dict], bool] = None) -> List[str]:
        """
        Выбирает count УНИКАЛЬНЫХ карт для раскопки и временно изымает их из пула.
        """
        candidates = []

        search_tiers = []
        for t in self.tiers.keys():
            if exact_tier:
                if t == tier:
                    search_tiers.append(t)
            else:
                if t <= tier:
                    search_tiers.append(t)

        for t in search_tiers:
            unique_ids_in_pool = set(self.tiers[t])
            for card_id in unique_ids_in_pool:
                data = CARD_DB.get(card_id)
                if not data:
                    continue
                if predicate and not predicate(data):
                    continue

                candidates.append(card_id)

        if not candidates:
            return []
        k = min(len(candidates), count)
        chosen_ids = random.sample(candidates, k)
        for cid in chosen_ids:
            c_tier = CARD_DB[cid]['tier']
            if cid in self.tiers[c_tier]:
                self.tiers[c_tier].remove(cid)

        return chosen_ids


class SpellPool:
    def __init__(self):
        self.tiers: Dict[int, List[str]] = {}
        self._initialize_pool()

    def _initialize_pool(self):
        for spell_id, data in SPELL_DB.items():
            tier = data["tier"]
            self.tiers.setdefault(tier, []).append(spell_id)

    def draw_spells(self, count: int, max_tier: int) -> List[SpellIDs]:
        drawn_spells = []
        available_tiers = [t for t in self.tiers.keys() if t <= max_tier]
        if not available_tiers:
            return drawn_spells

        for _ in range(count):
            chosen_tier = random.choice(available_tiers)
            spell_id = random.choice(self.tiers[chosen_tier])
            drawn_spells.append(spell_id)
        return drawn_spells
