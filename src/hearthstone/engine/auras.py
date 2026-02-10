from typing import List, Callable, Dict
from .entities import Unit
from .enums import CardIDs, UnitType

# Aura function: aura source, board and source index
AuraEffectFn = Callable[[Unit, List[Unit], int], None]


def _dire_wolf_alpha_aura(source: Unit, board: List[Unit], idx: int):
    """Neighbours gain +1\+0(Golden: +2\+0)"""
    bonus = 1 if not source.is_golden else 2

    if idx > 0:
        board[idx - 1].aura_atk_add += bonus
    if idx < len(board) - 1:
        board[idx + 1].aura_atk_add += bonus


def _murloc_warleader_aura(source: Unit, board: List[Unit], idx: int):
    """Other murlocs gain +2/+0 (Golden: +4/+0)"""
    atk_bonus = 2 if not source.is_golden else 4

    for i, unit in enumerate(board):
        if i == idx: continue  # dont buff self
        if UnitType.MURLOC in unit.types:
            unit.aura_atk_add += atk_bonus


def _southsea_captain_aura(source: Unit, board: List[Unit], idx: int):
    """Other pirates gain +1/+1 (Golden: +2/+2)"""
    bonus = 1 if not source.is_golden else 2

    for i, unit in enumerate(board):
        if i == idx: continue  # dont buff self
        if UnitType.PIRATE in unit.types:
            unit.aura_atk_add += bonus
            unit.aura_hp_add += bonus


# Register: CardID -> Aura function
AURA_REGISTRY: Dict[str, AuraEffectFn] = {
    CardIDs.DIRE_WOLF_ALPHA: _dire_wolf_alpha_aura,
    CardIDs.MURLOC_WARLEADER: _murloc_warleader_aura,
    CardIDs.SOUTHSEA_CAPTAIN: _southsea_captain_aura,
}


def recalculate_board_auras(board: List[Unit]):
    for unit in board:
        unit.reset_aura_layer()

    for i, unit in enumerate(board):
        if unit.card_id in AURA_REGISTRY:
            AURA_REGISTRY[unit.card_id](unit, board, i)

    for unit in board:
        unit.recalc_stats()
