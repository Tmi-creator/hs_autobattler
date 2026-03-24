"""Combat mechanics tests — pure pytest, no unittest.

Covers: Spawn of N'Zoth DR, Kaboom Bot (normal + golden), Deflect-o-Bot + Reborn.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Dict, List, Tuple

from hearthstone.engine.combat import CombatManager
from hearthstone.engine.entities import Player, Unit
from hearthstone.engine.enums import CardIDs, Tags, UnitType

if TYPE_CHECKING:
    pass


# ===================================================================
#  SPAWN OF N'ZOTH
# ===================================================================


class TestSpawnOfNzoth:
    """Spawn of N'Zoth DR: give your minions +1/+1 combat buff."""

    def test_deathrattle_buffs_surviving_allies(
        self,
        combat_players: Callable[..., Tuple[Dict[int, Player], List[List[Unit]], CombatManager]],
    ) -> None:
        players, boards, cm = combat_players(
            [CardIDs.SPAWN_OF_NZOTH, CardIDs.TABBYCAT, CardIDs.TABBYCAT],
            [],  # enemy empty — we just test cleanup_dead
        )

        # Kill spawn manually
        spawn = boards[0][0]
        spawn.cur_hp = 0

        cm.cleanup_dead(boards, [0, 0], players)

        # Spawn removed
        assert spawn not in boards[0]

        # Both cats get +1/+1 in combat layer → 2/2
        for cat in boards[0]:
            assert cat.combat_atk_add == 1
            assert cat.combat_hp_add == 1
            assert cat.cur_atk == 2


# ===================================================================
#  KABOOM BOT
# ===================================================================


class TestKaboomBot:
    """Kaboom Bot DR: deal 4 damage to a random enemy."""

    def test_deals_4_damage_to_enemy(
        self,
        combat_players: Callable[..., Tuple[Dict[int, Player], List[List[Unit]], CombatManager]],
    ) -> None:
        players, boards, cm = combat_players(
            [CardIDs.KABOOM_BOT],
            [CardIDs.ANNOY_O_TRON],
        )

        bot = boards[0][0]
        bot.cur_hp = 0

        dummy = boards[1][0]
        # Remove divine shield for clean damage check
        dummy.tags.discard(Tags.DIVINE_SHIELD)
        # Give dummy 10 HP so it survives
        dummy.perm_hp_add = 8
        dummy.recalc_stats()
        dummy.restore_stats()
        hp_before = dummy.cur_hp  # should be 10

        cm.cleanup_dead(boards, [0, 0], players)

        # 4 damage dealt
        assert dummy.cur_hp == hp_before - 4

    def test_golden_fires_twice(
        self,
        combat_players: Callable[..., Tuple[Dict[int, Player], List[List[Unit]], CombatManager]],
    ) -> None:
        players, boards, cm = combat_players(
            [CardIDs.KABOOM_BOT],
            [CardIDs.ANNOY_O_TRON],
        )

        bot = boards[0][0]
        bot.is_golden = True
        bot.cur_hp = 0

        target = boards[1][0]
        # DS + 3 HP: first tick pops shield, second tick deals 4 > 3 → dead
        target.cur_hp = 3
        target.tags.add(Tags.DIVINE_SHIELD)

        cm.cleanup_dead(boards, [0, 0], players)

        # First tick: DS popped. Second tick: 3 - 4 = -1
        assert not target.has_divine_shield
        assert target.cur_hp <= 0


# ===================================================================
#  DEFLECT-O-BOT + REBORN
# ===================================================================


class TestDeflectOBot:
    """Deflect-o-Bot gains +2 Atk & DS when a friendly Mech is summoned."""

    def test_gains_shield_and_atk_from_reborn_mech(
        self,
        combat_players: Callable[..., Tuple[Dict[int, Player], List[List[Unit]], CombatManager]],
    ) -> None:
        players, boards, cm = combat_players(
            [CardIDs.DEFLECT_O_BOT, CardIDs.ANNOY_O_TRON],
            [],
        )

        deflecto = boards[0][0]
        # Simulate shield already popped
        deflecto.tags.discard(Tags.DIVINE_SHIELD)
        base_atk = deflecto.cur_atk

        # Dying mech with Reborn
        dying_mech = boards[0][1]
        dying_mech.tags.add(Tags.REBORN)
        dying_mech.cur_hp = 0

        cm.cleanup_dead(boards, [0, 0], players)

        # Reborn summons a new Mech → triggers Deflect-o-Bot
        assert deflecto.has_divine_shield, "Deflect-o-Bot must restore divine shield"
        assert deflecto.cur_atk == base_atk + 2, "Deflect-o-Bot must gain +2 attack"

        # Reborn token actually appeared
        assert len(boards[0]) == 2
        token = boards[0][1]
        assert UnitType.MECH in token.types
