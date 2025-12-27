from typing import Tuple
from .entities import Player, Unit, HandCard
from .configs import TAVERN_SLOTS, COST_BUY, COST_REROLL, TIER_UPGRADE_COSTS
from .enums import UnitType


class TavernManager:
    def __init__(self, pool):
        self.pool = pool
        self._uid_counter = 1000

    def _get_next_uid(self):
        self._uid_counter += 1
        return self._uid_counter

    def start_turn(self, player: Player, turn_number: int):
        """
        Логика начала хода (Фаза вербовки):
        1. Восстановить/увеличить золото.
        2. Снизить стоимость улучшения таверны.
        3. Обновить магазин (с учетом заморозки).
        """
        max_gold = min(10, 3 + turn_number - 1)
        player.gold = max_gold

        if player.up_cost > 0 and turn_number != 1:
            player.up_cost -= 1

        frozen_units = [u for u in player.store if u.is_frozen]

        not_frozen_ids = [u.card_id for u in player.store if not u.is_frozen]
        self.pool.return_cards(not_frozen_ids)

        player.store.clear()

        for u in frozen_units:
            u.is_frozen = False
            player.store.append(u)

        self._fill_tavern(player)

    def roll_tavern(self, player: Player):
        """Платное обновление (1 золотой). Игнорирует заморозку (сбрасывает всё)."""
        if player.gold < COST_REROLL:
            return False, "Not enough gold"

        player.gold -= COST_REROLL

        all_ids = [u.card_id for u in player.store]
        self.pool.return_cards(all_ids)

        player.store.clear()

        self._fill_tavern(player)

        return True, "Rolled"

    def _fill_tavern(self, player: Player):
        """Вспомогательный метод: добивает магазин до максимума карт"""
        slots_total = TAVERN_SLOTS.get(player.tavern_tier)
        slots_needed = slots_total - len(player.store)

        if slots_needed > 0:
            new_ids = self.pool.draw_cards(slots_needed, player.tavern_tier)
            for cid in new_ids:
                new_unit = Unit.create_from_db(cid, self._get_next_uid(), player.uid)
                player.store.append(new_unit)

    def upgrade_tavern(self, player: Player) -> Tuple[bool, str]:
        """Повышение уровня таверны"""
        if player.tavern_tier >= 6:
            return False, "Max tier reached"

        cost = player.up_cost

        if player.gold < cost:
            return False, "Not enough gold"

        player.gold -= cost

        player.tavern_tier += 1

        next_cost = TIER_UPGRADE_COSTS.get(player.tavern_tier + 1, 0)
        player.up_cost = next_cost

        return True, f"Upgraded to Tier {player.tavern_tier}"

    def toggle_freeze(self, player: Player) -> Tuple[bool, str]:
        """Заморозить/Разморозить весь магазин"""

        all_frozen = all(u.is_frozen for u in player.store)
        if all_frozen:
            for u in player.store:
                u.is_frozen = False
            return True, "Unfrozen"
        else:
            for u in player.store:
                u.is_frozen = True
            return True, "Frozen"

    def buy_unit(self, player: Player, store_index: int) -> Tuple[bool, str]:
        if store_index < 0 or store_index >= len(player.store):
            return False, "Invalid index"
        if player.gold < COST_BUY:
            return False, "Not enough gold"
        if len(player.hand) >= 10:
            return False, "Hand is full"

        unit = player.store.pop(store_index)

        unit.is_frozen = False

        player.gold -= COST_BUY
        hand_card = HandCard(uid=unit.uid, unit=unit)
        player.hand.append(hand_card)

        return True, f"Bought {unit.card_id}"

    def sell_unit(self, player: Player, board_index: int) -> Tuple[bool, str]:
        if board_index < 0 or board_index >= len(player.board):
            return False, "Invalid index"

        unit = player.board.pop(board_index)
        player.gold += 1

        self.pool.return_cards([str(unit.card_id)])

        return True, "Sold unit"

    def play_unit(self, player: Player, hand_index: int, insert_index: int, target_index: int = -1) -> Tuple[bool, str]:
        """
        Разыгрывает карту из руки на стол в конкретную позицию.
        Args:
            hand_index: Индекс карты в руке.
            insert_index: Индекс на столе, куда поставить существо (0 - слева, len(board) - справа).
            target_index: Индекс цели для Battlecry (если нужен).
        """
        if hand_index < 0 or hand_index >= len(player.hand):
            return False, "Invalid hand index"

        hand_card = player.hand[hand_index]

        if hand_card.spell:
            return False, "Spells not implemented yet"

        unit = hand_card.unit

        if len(player.board) >= 7:
            return False, "Board is full"

        player.hand.pop(hand_index)

        real_index = insert_index
        if real_index < 0: real_index = 0
        if real_index > len(player.board): real_index = len(player.board)

        player.board.insert(real_index, unit)

        self._resolve_battlecry(player, unit, real_index, target_index)

        return True, "Played unit"

    def _resolve_battlecry(self, player: Player, unit: Unit, unit_index: int, target_index: int):
        cid = unit.card_id

        # 107: Shell Collector (Дает монетку)
        if cid == "107":
            player.gold += 1

        # 102: Alleycat
        # Призывает 1/1 кота. Токен должен появиться СПРАВА от основного кота.
        elif cid == "102":
            if len(player.board) < 7:
                token = Unit.create_from_db("102t", self._get_next_uid(), player.uid)
                spawn_index = unit_index + 1
                player.board.insert(spawn_index, token)

        # 101: Wrath Weaver (Заклинатель гнева)
        # Триггер: "После того как вы разыграли демона"
        if "101" in [u.card_id for u in player.board]:
            if UnitType.DEMON in unit.type and unit.card_id != "101":
                for u in player.board:
                    if u.card_id == "101":
                        player.health -= 1
                        u.cur_atk += 2
                        u.cur_hp += 1
                        u.max_atk += 2
                        u.max_hp += 1
        # Болотный налетчик
        # Триггер: "После того как вы разыграли мурлока"
        if "104" in [u.card_id for u in player.board]:
            if UnitType.MURLOC in unit.type:
                for u in player.board:
                    if u.card_id == "104" and u.uid != unit.uid:
                        u.cur_atk += 1
                        u.max_atk += 1

    def swap_units(self, player: Player, index_a: int, index_b: int) -> Tuple[bool, str]:
        """
        Меняет местами двух существ на столе.
        """
        board_len = len(player.board)

        if not (0 <= index_a < board_len) or not (0 <= index_b < board_len):
            return False, "Invalid indices"

        if index_a == index_b:
            return False, "Same index"

        player.board[index_a], player.board[index_b] = player.board[index_b], player.board[index_a]

        return True, "Swapped"
