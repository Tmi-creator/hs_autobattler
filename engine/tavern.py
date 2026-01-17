from typing import Dict, Tuple
from .entities import Player, Unit, HandCard, Spell, StoreItem
from .configs import TAVERN_SLOTS, COST_BUY, COST_REROLL, TIER_UPGRADE_COSTS, SPELLS_PER_ROLL
from .effects import TRIGGER_REGISTRY, GOLDEN_TRIGGER_REGISTRY
from .event_system import EntityRef, Event, EventManager, EventType, PosRef, TriggerInstance, Zone
from .enums import UnitType
from .spells import SPELL_TRIGGER_REGISTRY, SPELLS_REQUIRE_TARGET


class TavernManager:
    def __init__(self, pool, spell_pool):
        self.pool = pool
        self.spell_pool = spell_pool
        self._uid_counter = 1000
        self.event_manager = EventManager(TRIGGER_REGISTRY, GOLDEN_TRIGGER_REGISTRY)

    def _get_next_uid(self):
        self._uid_counter += 1
        return self._uid_counter

    def start_turn(self, player: Player, turn_number: int) -> None:
        """
        Логика начала хода (Фаза вербовки):
        1. Восстановить/увеличить золото.
        2. Снизить стоимость улучшения таверны.
        3. Обновить магазин (с учетом заморозки).
        """
        for unit in player.board:
            unit.reset_turn_layer()
            unit.restore_stats()
        self.event_manager.process_event(
            Event(event_type=EventType.START_OF_TURN,
                  source_pos=PosRef(side=player.uid, zone=Zone.HERO, slot=0),
                  ), {player.uid: player}, self._get_next_uid,
        )
        max_gold = min(10, 3 + turn_number - 1)
        player.gold = max_gold + player.gold_next_turn
        player.gold_next_turn = 0

        if player.up_cost > 0 and turn_number != 1:
            player.up_cost -= 1

        frozen_items = [item for item in player.store if item.is_frozen]

        not_frozen_units = [item.unit.card_id for item in player.store if not item.is_frozen and item.unit]
        self.pool.return_cards(not_frozen_units)

        player.store.clear()

        for item in frozen_items:
            item.is_frozen = False
            player.store.append(item)

        self._fill_tavern(player)

    def roll_tavern(self, player: Player) -> tuple[bool, str]:
        """Платное обновление (1 золотой). Игнорирует заморозку (сбрасывает всё)."""
        if player.gold < COST_REROLL:
            return False, "Not enough gold"

        player.gold -= COST_REROLL

        all_unit_ids = [item.unit.card_id for item in player.store if item.unit]
        self.pool.return_cards(all_unit_ids)

        player.store.clear()

        self._fill_tavern(player)

        return True, "Rolled"

    def _fill_tavern(self, player: Player) -> None:
        """Вспомогательный метод: добивает магазин до максимума карт"""
        slots_total = TAVERN_SLOTS.get(player.tavern_tier)
        current_units = sum(1 for item in player.store if item.unit)
        slots_needed = slots_total - current_units

        if slots_needed > 0:
            new_ids = self.pool.draw_cards(slots_needed, player.tavern_tier)
            for cid in new_ids:
                new_unit = self._make_unit(player, cid)
                player.store.append(StoreItem(unit=new_unit))
        cnt_spells = len([u for u in player.store if u.spell])
        if cnt_spells >= SPELLS_PER_ROLL:
            return
        spell_ids = self.spell_pool.draw_spells(SPELLS_PER_ROLL, player.tavern_tier)
        for spell_id in spell_ids:
            spell = Spell.create_from_db(spell_id)
            player.store.append(StoreItem(spell=spell))

    def _make_unit(self, player: Player, cid: str) -> Unit:
        unit = Unit.create_from_db(cid, self._get_next_uid(), player.uid)
        # TODO: дописать сюда бафф элементалей через ивент
        return unit

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

        all_frozen = all(item.is_frozen for item in player.store)
        if all_frozen:
            for item in player.store:
                item.is_frozen = False
            return True, "Unfrozen"
        else:
            for item in player.store:
                item.is_frozen = True
            return True, "Frozen"

    def buy_unit(self, player: Player, store_index: int) -> Tuple[bool, str]:
        if store_index < 0 or store_index >= len(player.store):
            return False, "Invalid index"
        if len(player.hand) >= 10:
            return False, "Hand is full"

        item = player.store[store_index]

        if item.unit:
            if player.gold < COST_BUY:
                return False, "Not enough gold"
            item = player.store.pop(store_index)
            item.is_frozen = False
            player.gold -= COST_BUY
            hand_card = HandCard(uid=item.unit.uid, unit=item.unit)
            player.hand.append(hand_card)
            return True, f"Bought {item.unit.card_id}"

        if item.spell:
            cost = max(0, item.spell.cost - player.spell_discount)
            if player.gold < cost:
                return False, "Not enough gold"
            item = player.store.pop(store_index)
            item.is_frozen = False
            player.gold -= cost
            player.spell_discount = 0
            hand_card = HandCard(uid=self._get_next_uid(), spell=item.spell)
            player.hand.append(hand_card)
            return True, f"Bought {item.spell.card_id}"

        return False, "Empty slot"

    def sell_unit(self, player: Player, board_index: int) -> Tuple[bool, str]:
        if board_index < 0 or board_index >= len(player.board):
            return False, "Invalid index"

        unit = player.board[board_index]
        source = EntityRef(uid=unit.uid)
        source_pos = PosRef(side=player.uid, zone=Zone.BOARD, slot=board_index)
        event = Event(
            event_type=EventType.MINION_SOLD,
            source=source,
            source_pos=source_pos,
        )
        players_by_uid: Dict[int, Player] = {player.uid: player}
        self.event_manager.process_event(event, players_by_uid, self._get_next_uid)

        unit = player.board.pop(board_index)
        player.gold += 1

        self.pool.return_cards([unit.card_id])

        return True, "Sold unit"

    def play_unit(self, player: Player, hand_index: int, insert_index: int = -1, target_index: int = -1) -> Tuple[
        bool, str]:
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
            return self._cast_spell(player, hand_index, target_index)

        unit = hand_card.unit

        if len(player.board) >= 7:
            return False, "Board is full"
        if insert_index == -1:
            insert_index = len(player.board)
        if insert_index < 0 or insert_index > len(player.board):
            return False, "Invalid Index"
        player.hand.pop(hand_index)

        player.board.insert(insert_index, unit)

        self._resolve_battlecry(player, unit, insert_index, target_index)

        return True, "Played unit"

    def _cast_spell(self, player: Player, hand_index: int, target_index: int) -> Tuple[bool, str]:
        hand_card = player.hand[hand_index]
        spell = hand_card.spell
        if not spell:
            return False, "No spell to cast"

        if spell.card_id in SPELLS_REQUIRE_TARGET and not (0 <= target_index < len(player.board)):
            return False, "Invalid target"

        trigger_defs = SPELL_TRIGGER_REGISTRY.get(spell.card_id)
        if not trigger_defs:
            return False, f"Unknown spell effect {spell.effect}"
        trigger_def = trigger_defs[0]
        trigger = TriggerInstance(trigger_def=trigger_def, trigger_uid=0)
        source_pos = PosRef(side=player.uid, zone=Zone.HAND, slot=hand_index)
        target_ref = None
        if 0 <= target_index < len(player.board):
            target_ref = EntityRef(uid=player.board[target_index].uid)
        event = Event(
            event_type=EventType.SPELL_CAST,
            source=None,
            target=target_ref,
            source_pos=source_pos,
        )
        players_by_uid: Dict[int, Player] = {player.uid: player}
        self.event_manager.process_event(event, players_by_uid, self._get_next_uid, extra_triggers=[trigger])
        player.hand.pop(hand_index)
        return True, f"Cast {spell.card_id}"

    def _resolve_battlecry(self, player: Player, unit: Unit, unit_index: int, target_index: int):
        source = EntityRef(uid=unit.uid)
        source_pos = PosRef(side=player.uid, zone=Zone.BOARD, slot=unit_index)
        target_ref = None
        target_pos = None
        if 0 <= target_index < len(player.board):
            target_unit = player.board[target_index]
            target_ref = EntityRef(uid=target_unit.uid)
            target_pos = PosRef(side=player.uid, zone=Zone.BOARD, slot=target_index)
        event = Event(
            event_type=EventType.MINION_PLAYED,
            source=source,
            target=target_ref,
            source_pos=source_pos,
            target_pos=target_pos,
        )
        players_by_uid: Dict[int, Player] = {player.uid: player}
        self.event_manager.process_event(event, players_by_uid, self._get_next_uid)

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

    def end_turn(self, player: Player) -> None:
        player.hand[:] = [
            hc for hc in player.hand
            if not (hc.spell is not None and hc.spell.is_temporary)
        ]
