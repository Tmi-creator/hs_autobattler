from __future__ import annotations

from typing import Callable, Dict, List

from .entities import Unit

# Aura function: aura source, board and source index
AuraEffectFn = Callable[[Unit, List[Unit], int], None]


# Register: CardID -> Aura function
# Currently no T1 cards have combat auras. Rot Hide Gnoll's
# "Has +1 Attack for each friendly minion that died" is handled
# via MINION_DIED trigger in effects.py (buff_combat), not as an aura.
AURA_REGISTRY: Dict[str, AuraEffectFn] = {}


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
