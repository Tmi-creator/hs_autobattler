from .entities import Unit, Player
from .configs import UnitType
import random


class Combat_Manager:
    def __init__(self):
        self.uid = 10000

    def get_uid(self):
        self.uid += 1
        return self.uid

    def resolve_combat(self, player_1, player_2):
        board_1 = [u.combat_copy() for u in player_1.board]
        board_2 = [u.combat_copy() for u in player_2.board]

        boards = [board_1, board_2]

        # 2. Фаза "Start of Combat" (Начало боя)

        if len(board_1) > len(board_2):
            attacker_player_idx = 0
        elif len(board_2) > len(board_1):
            attacker_player_idx = 1
        else:
            attacker_player_idx = random.choice([0, 1])

        attack_indices = [0, 0]

        while True:
            if not board_1 and not board_2:
                return "DRAW", 0
            if not board_1:
                damage = sum(u.tier for u in board_2) + player_2.tavern_tier
                return "LOSE", -damage
            if not board_2:
                damage = sum(u.tier for u in board_1) + player_1.tavern_tier
                return "WIN", damage

            attacker_board = boards[attacker_player_idx]
            defender_board = boards[1 - attacker_player_idx]

            if attack_indices[attacker_player_idx] >= len(attacker_board):
                attack_indices[attacker_player_idx] = 0

            attacker_idx = attack_indices[attacker_player_idx]
            attacker_unit = attacker_board[attacker_idx]
            num_attacks = 1
            if attacker_unit.has_windfury:
                num_attacks += 1
            for i in range(num_attacks):
                taunts = [u for u in defender_board if u.has_taunt]
                if taunts:
                    target = random.choice(taunts)
                else:
                    target = random.choice(defender_board)

                self.perform_attack(attacker_unit, target)

                self.cleanup_dead(boards, attack_indices)

                if not attacker_unit.is_alive:
                    break
                if not board_1 and not board_2:
                    return "DRAW", 0
                if not board_1:
                    damage = sum(u.tier for u in board_2) + player_2.tavern_tier
                    return "LOSE", -damage
                if not board_2:
                    damage = sum(u.tier for u in board_1) + player_1.tavern_tier
                    return "WIN", damage

            if attacker_unit.is_alive:
                attack_indices[attacker_player_idx] += 1

            attacker_player_idx = 1 - attacker_player_idx

    def perform_attack(self, attacker, target):
        """
        Реализация атаки существа со всеми доп механиками
        """
        dmg_to_target = attacker.cur_atk
        dmg_to_attacker = target.cur_atk
        if dmg_to_target > 0 and target.has_divine_shield:
            target.has_divine_shield = False
        else:
            target.cur_hp -= dmg_to_target
            if attacker.has_poisonous or attacker.has_venomous:
                target.cur_hp = 0
                attacker.has_venomous = False
        if dmg_to_attacker > 0 and attacker.has_divine_shield:
            attacker.has_divine_shield = False
        else:
            attacker.cur_hp -= dmg_to_attacker
            if target.has_poisonous or target.has_venomous:
                attacker.cur_hp = 0
                target.has_venomous = False

    def get_spawns(self, unit, owner_id):
        """Возвращает список Unit, которые должны появиться после смерти"""
        spawns = []

        # 1. Обработка Deathrattle (по ID карты)
        # Пример: Scallywag id="103" призывает пирата "103t"
        if unit.card_id == "103":
            token = Unit.create_from_db("103t", self.get_uid(), owner_id)
            spawns.append(token)

        elif unit.card_id == "108":  # Imprisoner
            imp = Unit.create_from_db("108t", self.get_uid(), owner_id)
            spawns.append(imp)

        if unit.has_reborn:
            reborn_unit = Unit.create_from_db(unit.card_id, self.get_uid(), owner_id)
            reborn_unit.cur_hp = 1
            reborn_unit.has_reborn = False
            spawns.append(reborn_unit)

        return spawns

    def cleanup_dead(self, boards, attack_indices):
        """
        Чистим стол при смерти существ и двигаем индекс атаки куда надо
        """
        for p_idx in range(2):
            board = boards[p_idx]
            i = 0
            while i < len(board):
                unit = board[i]

                if not unit.is_alive:
                    new_units = self.get_spawns(unit, unit.owner_id)

                    board.pop(i)

                    if i < attack_indices[p_idx]:
                        attack_indices[p_idx] -= 1

                    units_added = 0
                    for new_u in new_units:
                        if len(board) < 7:
                            board.insert(i + units_added, new_u)
                            units_added += 1
                        else:
                            break

                    if i < attack_indices[p_idx]:
                        attack_indices[p_idx] += units_added
                    i += units_added

                else:
                    i += 1
