"""Combat mechanics tests — pure pytest, no unittest.

Covers: Spawn of N'Zoth DR, Kaboom Bot (normal + golden), Deflect-o-Bot + Reborn.

NOTE: All mechanics in this file (SPAWN_OF_NZOTH, KABOOM_BOT, DEFLECT_O_BOT)
are not present in the current patch and are skipped.
"""

from __future__ import annotations

import pytest


# ===================================================================
#  SPAWN OF N'ZOTH
# ===================================================================


@pytest.mark.skip(reason="SPAWN_OF_NZOTH deathrattle not in current patch")
class TestSpawnOfNzoth:
    """Spawn of N'Zoth DR: give your minions +1/+1 combat buff."""

    def test_deathrattle_buffs_surviving_allies(self) -> None:
        pass


# ===================================================================
#  KABOOM BOT
# ===================================================================


@pytest.mark.skip(reason="KABOOM_BOT deathrattle not in current patch")
class TestKaboomBot:
    """Kaboom Bot DR: deal 4 damage to a random enemy."""

    def test_deals_4_damage_to_enemy(self) -> None:
        pass

    def test_golden_fires_twice(self) -> None:
        pass


# ===================================================================
#  DEFLECT-O-BOT + REBORN
# ===================================================================


@pytest.mark.skip(reason="DEFLECT_O_BOT trigger not in current patch")
class TestDeflectOBot:
    """Deflect-o-Bot gains +2 Atk & DS when a friendly Mech is summoned."""

    def test_gains_shield_and_atk_from_reborn_mech(self) -> None:
        pass
