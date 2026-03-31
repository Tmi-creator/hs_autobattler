from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Dict, List

from .entities import Unit

if TYPE_CHECKING:
    from .enums import UnitType

# Aura function: aura source, board and source index
AuraEffectFn = Callable[[Unit, List[Unit], int], None]


# Register: CardID -> Aura function
# Currently no T1 cards have combat auras. Rot Hide Gnoll's
# "Has +1 Attack for each friendly minion that died" is handled
# via MINION_DIED trigger in effects.py (buff_combat), not as an aura.
AURA_REGISTRY: Dict[str, AuraEffectFn] = {}


def _adjacent_buff_aura(atk: int, hp: int) -> AuraEffectFn:
    """Factory: create an aura that buffs adjacent units."""
    def _aura(source: Unit, board: list, idx: int) -> None:
        bonus_atk = atk * (2 if source.is_golden else 1)
        bonus_hp = hp * (2 if source.is_golden else 1)
        if idx > 0:
            board[idx - 1].aura_atk_add += bonus_atk
            board[idx - 1].aura_hp_add += bonus_hp
        if idx < len(board) - 1:
            board[idx + 1].aura_atk_add += bonus_atk
            board[idx + 1].aura_hp_add += bonus_hp
    return _aura


def _type_buff_aura(unit_type: UnitType, atk: int, hp: int) -> AuraEffectFn:
    """Factory: create an aura that buffs all other friendly units of a type."""
    def _aura(source: Unit, board: list, idx: int) -> None:
        bonus_atk = atk * (2 if source.is_golden else 1)
        bonus_hp = hp * (2 if source.is_golden else 1)
        for i, unit in enumerate(board):
            if i == idx:
                continue
            if unit_type in unit.types:
                unit.aura_atk_add += bonus_atk
                unit.aura_hp_add += bonus_hp
    return _aura


def recalculate_board_auras(board: List[Unit]) -> None:
    for unit in board:
        unit.reset_aura_layer()

    for i, unit in enumerate(board):
        if unit.card_id in AURA_REGISTRY:
            AURA_REGISTRY[unit.card_id](unit, board, i)
        for attached_layer in (unit.attached_perm, unit.attached_turn, unit.attached_combat):
            for effect_id, count in attached_layer.items():
                if effect_id in AURA_REGISTRY:
                    for _ in range(count):
                        AURA_REGISTRY[effect_id](unit, board, i)

    for unit in board:
        unit.recalc_stats()
