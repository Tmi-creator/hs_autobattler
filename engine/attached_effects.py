from typing import Dict, List, Optional

EFFECT_ID_TO_INDEX: Dict[str, int] = {
    "E_DR_CRAB32": 0,
}

N_EFFECTS = len(EFFECT_ID_TO_INDEX)
EFFECT_INDEX_TO_ID: List[Optional[str]] = [None] * N_EFFECTS
for effect_id, index in EFFECT_ID_TO_INDEX.items():
    EFFECT_INDEX_TO_ID[index] = effect_id


def make_effect_counts() -> List[int]:
    return [0] * N_EFFECTS
