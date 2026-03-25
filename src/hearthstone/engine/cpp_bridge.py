"""
cpp_bridge.py — Mapping tables and lazy loader for C++ combat engine.

Converts Python enums (UnitType, Tags, CardIDs) to C++ bitset integers
and provides a single get_cpp_engine() entry point.

Tables are auto-generated from enums.py — no need to update manually
when adding new cards/tags/types. The bit layout matches
cpp/include/types.h (positional index → bit).
"""
from __future__ import annotations

from .enums import CardIDs, Tags, UnitType

# =============================================================
# UnitType → C++ TypeBitset (uint16_t)
# Bit position = index in enum definition order.
# Must match cpp/include/types.h UnitTypes namespace.
# =============================================================
TYPE_TO_BIT: dict[UnitType, int] = {
    ut: 1 << i for i, ut in enumerate(UnitType)
}

# =============================================================
# Tags → C++ TagBitset (uint32_t)
# Bit position = index in enum definition order.
# Must match cpp/include/types.h Tags namespace.
# =============================================================
TAG_TO_BIT: dict[Tags, int] = {
    tag: 1 << i for i, tag in enumerate(Tags)
}

# =============================================================
# CardIDs (Python str) → C++ int16_t
# Rules (mirror cpp/include/types.h CardID namespace):
#   - Numeric IDs ("101", "207") → int(value)
#   - Token IDs ("102t", "103t")  → 900 + sequential index
# Unknown / unmatchable → excluded (will map to 0 at lookup).
# =============================================================
_next_token_id = 901
CARD_ID_MAP: dict[str, int] = {}
for _card in CardIDs:
    _val: str = _card.value
    if _val.endswith("t"):
        CARD_ID_MAP[_card] = _next_token_id
        _next_token_id += 1
    else:
        try:
            CARD_ID_MAP[_card] = int(_val)
        except ValueError:
            pass  # skip non-numeric, non-token IDs


# =============================================================
# Lazy import of compiled C++ module
# =============================================================
_cpp_engine = None
_cpp_init_done = False
_effects_registered = False


def get_cpp_engine():
    """
    Returns the hs_engine_cpp module if available, None otherwise.
    Caches successful loads. Retries if previously failed (allows
    test files to add sys.path entries after initial failure).
    """
    global _cpp_engine, _cpp_init_done, _effects_registered
    if _cpp_engine is not None:
        return _cpp_engine

    try:
        import hs_engine_cpp  # type: ignore[import-not-found]
        if not _effects_registered:
            hs_engine_cpp.register_all_effects()
            _effects_registered = True
        _cpp_engine = hs_engine_cpp
        if not _cpp_init_done:
            print(
                f"[C++] Engine loaded "
                f"(CombatState = {hs_engine_cpp.get_state_size()} bytes)"
            )
            _cpp_init_done = True
    except ImportError:
        if not _cpp_init_done:
            print("[C++] Engine not found — falling back to Python combat")
            _cpp_init_done = True

    return _cpp_engine
