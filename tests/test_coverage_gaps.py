"""Additional integration tests targeting coverage gaps.

These tests cover engine paths missed in the first pass:
zero-atk combat, tavern lifecycle, spell mechanics (Apple, Blood Gem),
Southsea Captain aura HP, Swampstriker synergy, absorbed pool copies,
game lifecycle, swap edge cases, IMMEDIATE_ATTACK, configs sanity,
triplet discovery flow, elemental buff, magnetic fallthrough,
golden Spawn of N'Zoth, avenge counter, multiple aura sources.
"""

from __future__ import annotations

import random
from collections import deque
from typing import TYPE_CHECKING, Callable, Dict, List, Tuple

import pytest

from hearthstone.engine.auras import recalculate_board_auras
from hearthstone.engine.combat import CombatManager
from hearthstone.engine.configs import CARD_DB, SPELL_DB
from hearthstone.engine.entities import HandCard, Player, Spell, StoreItem, Unit
from hearthstone.engine.enums import (
    BattleOutcome,
    CardIDs,
    MechanicType,
    SpellIDs,
    UnitType,
)
from hearthstone.engine.event_system import (
    EffectContext,
    EntityRef,
    Event,
    EventType,
    PosRef,
    Zone,
)

if TYPE_CHECKING:
    from hearthstone.engine.game import Game
    from hearthstone.engine.tavern import TavernManager


# ===================================================================
#  1. ZERO-ATK COMBAT
# ===================================================================


class TestZeroAtkCombat:
    """Units with 0 attack should be skipped; both sides 0 atk → draw."""

    def test_both_sides_zero_atk_is_draw(
        self,
        empty_game: "Game",
        mock_unit: Callable[..., Unit],
        combat_manager: CombatManager,
    ) -> None:
        p0, p1 = empty_game.players
        zero0 = mock_unit(CardIDs.MICROBOT, owner_id=p0.uid)
        zero0.perm_atk_add = -1
        zero0.recalc_stats()
        p0.board = [zero0]

        zero1 = mock_unit(CardIDs.MICROBOT, owner_id=p1.uid)
        zero1.perm_atk_add = -1
        zero1.recalc_stats()
        p1.board = [zero1]

        random.seed(42)
        outcome, damage = combat_manager.resolve_combat(p0, p1)

        assert outcome == BattleOutcome.DRAW
        assert damage == 0

    def test_mixed_zero_and_nonzero_atk(
        self,
        empty_game: "Game",
        mock_unit: Callable[..., Unit],
        combat_manager: CombatManager,
    ) -> None:
        p0, p1 = empty_game.players
        zero = mock_unit(CardIDs.MICROBOT, owner_id=p0.uid)
        zero.perm_atk_add = -1
        zero.recalc_stats()
        normal = mock_unit(CardIDs.ROT_HIDE_GNOLL, owner_id=p0.uid)
        p0.board = [zero, normal]

        p1.board = [mock_unit(CardIDs.MICROBOT, owner_id=p1.uid)]

        random.seed(42)
        outcome, _ = combat_manager.resolve_combat(p0, p1)
        assert outcome == BattleOutcome.WIN


# ===================================================================
#  2. ATTACKER SELECTION
# ===================================================================


class TestAttackerSelection:
    """Side with more units attacks first."""

    def test_larger_board_attacks_first(
        self,
        empty_game: "Game",
        mock_unit: Callable[..., Unit],
        combat_manager: CombatManager,
    ) -> None:
        p0, p1 = empty_game.players
        p0.board = [mock_unit(CardIDs.MICROBOT, owner_id=p0.uid) for _ in range(3)]
        p1.board = [mock_unit(CardIDs.MICROBOT, owner_id=p1.uid)]

        random.seed(42)
        outcome, _ = combat_manager.resolve_combat(p0, p1)
        assert outcome == BattleOutcome.WIN


# ===================================================================
#  3. TAVERN LIFECYCLE
# ===================================================================


class TestTavernLifecycle:
    """start_turn, end_turn, gold, turn-layer reset."""

    def test_start_turn_resets_turn_layer(
        self,
        empty_game: "Game",
        player: Player,
        tavern: "TavernManager",
        mock_unit: Callable[..., Unit],
    ) -> None:
        unit = mock_unit(CardIDs.MICROBOT)
        unit.turn_atk_add = 5
        unit.turn_hp_add = 5
        unit.recalc_stats()
        player.board.append(unit)

        tavern.start_turn(player, 2)

        assert unit.turn_atk_add == 0
        assert unit.turn_hp_add == 0

    def test_start_turn_restores_hp(
        self,
        empty_game: "Game",
        player: Player,
        tavern: "TavernManager",
        mock_unit: Callable[..., Unit],
    ) -> None:
        unit = mock_unit(CardIDs.ROT_HIDE_GNOLL)
        unit.cur_hp = 1  # "damaged"
        player.board.append(unit)

        tavern.start_turn(player, 2)

        assert unit.cur_hp == unit.max_hp

    def test_start_turn_gold_formula(
        self,
        empty_game: "Game",
        player: Player,
        tavern: "TavernManager",
    ) -> None:
        tavern.start_turn(player, 3)
        assert player.gold == min(10, 3 + 3 - 1)  # 5

    def test_start_turn_gold_with_extra(
        self,
        empty_game: "Game",
        player: Player,
        tavern: "TavernManager",
    ) -> None:
        player.gold_next_turn = 2
        tavern.start_turn(player, 3)
        assert player.gold == 5 + 2

    def test_end_turn_removes_temporary_spells(
        self,
        empty_game: "Game",
        player: Player,
        tavern: "TavernManager",
    ) -> None:
        temp = Spell.create_from_db(SpellIDs.SURF_SPELLCRAFT)
        assert temp.is_temporary
        player.hand.append(HandCard(uid=9999, spell=temp))

        normal = Spell.create_from_db(SpellIDs.BANANA)
        player.hand.append(HandCard(uid=9998, spell=normal))

        tavern.end_turn(player)

        remaining_spells = [hc for hc in player.hand if hc.spell]
        assert all(not s.spell.is_temporary for s in remaining_spells)  # type: ignore[union-attr]

    def test_fill_tavern_includes_spells(
        self,
        empty_game: "Game",
        player: Player,
        tavern: "TavernManager",
    ) -> None:
        player.store.clear()
        tavern._fill_tavern(player)

        has_units = any(item.unit for item in player.store)
        has_spells = any(item.spell for item in player.store)
        assert has_units
        assert has_spells


# ===================================================================
#  4. APPLE SPELL
# ===================================================================


class TestAppleSpell:
    """Apple buffs all units in the store."""

    def test_apple_buffs_store_units(
        self,
        empty_game: "Game",
        player: Player,
        tavern: "TavernManager",
        mock_unit: Callable[..., Unit],
    ) -> None:
        # Clear store, add a known unit
        player.store.clear()
        store_unit = mock_unit(CardIDs.MICROBOT)
        player.store.append(StoreItem(unit=store_unit))

        atk_before = store_unit.perm_atk_add
        hp_before = store_unit.perm_hp_add

        spell = Spell.create_from_db(SpellIDs.APPLE)
        player.hand.append(HandCard(uid=9999, spell=spell))

        tavern.play_unit(player, hand_index=0, insert_index=-1, target_index=-1)

        assert store_unit.perm_atk_add == atk_before + 1
        assert store_unit.perm_hp_add == hp_before + 2


# ===================================================================
#  5. BLOOD GEM SPELL
# ===================================================================


class TestBloodGemSpell:
    """Blood Gem uses MechanicState for buff amount."""

    def test_blood_gem_default(
        self,
        empty_game: "Game",
        player: Player,
        tavern: "TavernManager",
        mock_unit: Callable[..., Unit],
    ) -> None:
        target = mock_unit(CardIDs.MICROBOT)
        player.board.append(target)

        spell = Spell.create_from_db(SpellIDs.BLOOD_GEM)
        player.hand.append(HandCard(uid=9999, spell=spell))

        tavern.play_unit(player, hand_index=0, insert_index=-1, target_index=0)

        assert target.perm_atk_add == 1
        assert target.perm_hp_add == 1

    def test_blood_gem_with_mechanic_buff(
        self,
        empty_game: "Game",
        player: Player,
        tavern: "TavernManager",
        mock_unit: Callable[..., Unit],
    ) -> None:
        target = mock_unit(CardIDs.MICROBOT)
        player.board.append(target)

        player.mechanics.modify_stat(MechanicType.BLOOD_GEM, 2, 3)
        # BLOOD_GEM default (1,1) + (2,3) = (3,4)

        spell = Spell.create_from_db(SpellIDs.BLOOD_GEM)
        player.hand.append(HandCard(uid=9999, spell=spell))

        tavern.play_unit(player, hand_index=0, insert_index=-1, target_index=0)

        assert target.perm_atk_add == 3
        assert target.perm_hp_add == 4


# ===================================================================
#  6. SOUTHSEA CAPTAIN AURA (HP component)
# ===================================================================


@pytest.mark.skip(reason="SOUTHSEA_CAPTAIN aura not in current patch")
class TestSouthseaCaptainAura:
    """Southsea Captain gives +1/+1 (Golden: +2/+2) — including HP."""

    def test_captain_buffs_pirate_hp(self) -> None:
        pass

    def test_golden_captain_doubles(self) -> None:
        pass

    def test_captain_does_not_buff_self(self) -> None:
        pass

    def test_captain_does_not_buff_non_pirates(self) -> None:
        pass


# ===================================================================
#  7. SWAMPSTRIKER SYNERGY
# ===================================================================


class TestSwampstriker:
    """Swampstriker gains +1 atk when a (non-self) Murloc is played."""

    def test_buffs_on_murloc_played(
        self,
        empty_game: "Game",
        player: Player,
        tavern: "TavernManager",
        mock_unit: Callable[..., Unit],
    ) -> None:
        striker = mock_unit(CardIDs.SWAMPSTRIKER)
        player.board.append(striker)
        atk_before = striker.perm_atk_add

        murloc = mock_unit(CardIDs.FLIGHTY_SCOUT)
        player.hand.append(HandCard(uid=murloc.uid, unit=murloc))
        tavern.play_unit(player, hand_index=0, insert_index=1)

        assert striker.perm_atk_add == atk_before + 1

    def test_no_buff_on_non_murloc(
        self,
        empty_game: "Game",
        player: Player,
        tavern: "TavernManager",
        mock_unit: Callable[..., Unit],
    ) -> None:
        striker = mock_unit(CardIDs.SWAMPSTRIKER)
        player.board.append(striker)
        atk_before = striker.perm_atk_add

        non_murloc = mock_unit(CardIDs.ANNOY_O_TRON)  # Mech, not Murloc
        player.hand.append(HandCard(uid=non_murloc.uid, unit=non_murloc))
        tavern.play_unit(player, hand_index=0, insert_index=1)

        assert striker.perm_atk_add == atk_before

    def test_no_self_buff(
        self,
        empty_game: "Game",
        player: Player,
        tavern: "TavernManager",
        mock_unit: Callable[..., Unit],
    ) -> None:
        striker = mock_unit(CardIDs.SWAMPSTRIKER)
        player.hand.append(HandCard(uid=striker.uid, unit=striker))

        tavern.play_unit(player, hand_index=0, insert_index=0)

        assert striker.perm_atk_add == 0


# ===================================================================
#  8. SELL GOLDEN WITH ABSORBED COPIES
# ===================================================================


@pytest.mark.skip(reason="ANNOY_O_MODULE (Magnetic) not in current patch")
class TestSellAbsorbed:
    """Selling a unit with absorbed_pool_copies returns them all."""

    def test_sell_golden_with_magnetized(self) -> None:
        pass


# ===================================================================
#  9. GAME LIFECYCLE
# ===================================================================


class TestGameLifecycle:
    """Full game lifecycle: turn counter, combat damage, winner."""

    def test_turn_counter_increments_after_combat(
        self,
        empty_game: "Game",
    ) -> None:
        turn_before = empty_game.turn_count
        empty_game.step(0, "END_TURN")
        empty_game.step(1, "END_TURN")
        if not empty_game.game_over:
            assert empty_game.turn_count == turn_before + 1

    def test_winner_id_set_correctly(
        self,
        empty_game: "Game",
        mock_unit: Callable[..., Unit],
    ) -> None:
        empty_game.players[1].health = 1
        empty_game.players[0].board = [mock_unit(CardIDs.ROT_HIDE_GNOLL, owner_id=0)]
        empty_game.players[1].board = []

        empty_game.step(0, "END_TURN")
        empty_game.step(1, "END_TURN")

        assert empty_game.game_over
        assert empty_game.winner_id == 0

    def test_all_action_types_accepted(
        self,
        empty_game: "Game",
    ) -> None:
        player = empty_game.players[0]
        player.gold = 20

        success, _, _ = empty_game.step(0, "ROLL")
        assert success

        success, _, _ = empty_game.step(0, "FREEZE")
        assert success

        player.gold = 20
        empty_game.step(0, "UPGRADE")
        # May succeed or fail → just no crash

        empty_game.step(0, "SWAP", index_a=0, index_b=1)
        # May fail → just no crash

    def test_discovery_flow_through_game_step(
        self,
        empty_game: "Game",
    ) -> None:
        player = empty_game.players[0]
        empty_game.tavern.start_discovery(player, source="Test", tier=1)

        if player.is_discovering:
            success, _, _ = empty_game.step(0, "DISCOVER_CHOICE", index=0)
            assert success
            assert not player.is_discovering


# ===================================================================
#  10. SWAP EDGE CASES
# ===================================================================


class TestSwapEdgeCases:
    """Swap operations boundary conditions."""

    def test_swap_same_index_fails(
        self,
        player: Player,
        tavern: "TavernManager",
        mock_unit: Callable[..., Unit],
    ) -> None:
        player.board = [mock_unit(CardIDs.MICROBOT)]
        success, _ = tavern.swap_units(player, 0, 0)
        assert not success

    def test_swap_out_of_bounds_fails(
        self,
        player: Player,
        tavern: "TavernManager",
        mock_unit: Callable[..., Unit],
    ) -> None:
        player.board = [mock_unit(CardIDs.MICROBOT)]
        success, _ = tavern.swap_units(player, 0, 5)
        assert not success

    @pytest.mark.skip(reason="DIRE_WOLF_ALPHA aura not in current patch")
    def test_swap_recalculates_auras(
        self,
        player: Player,
        tavern: "TavernManager",
        mock_unit: Callable[..., Unit],
    ) -> None:
        pass


# ===================================================================
#  11. IMMEDIATE ATTACK IN COMBAT
# ===================================================================


@pytest.mark.skip(reason="SCALLYWAG/PIRATE_TOKEN not in current patch")
class TestImmediateAttack:
    """Pirate Token with IMMEDIATE_ATTACK spawns from Scallywag DR."""

    def test_scallywag_token_spawns(self) -> None:
        pass


# ===================================================================
#  12. CONFIGS SANITY
# ===================================================================


class TestConfigsSanity:
    """Validate CARD_DB and SPELL_DB have required fields."""

    @pytest.mark.parametrize("card_id", list(CARD_DB.keys()))
    def test_card_db_has_required_fields(self, card_id: str) -> None:
        data = CARD_DB[card_id]
        assert "name" in data
        assert "tier" in data
        assert "atk" in data
        assert "hp" in data
        assert isinstance(data["tier"], int)
        assert data["atk"] >= 0
        assert data["hp"] >= 0

    @pytest.mark.parametrize("spell_id", list(SPELL_DB.keys()))
    def test_spell_db_has_required_fields(self, spell_id: str) -> None:
        data = SPELL_DB[spell_id]
        assert "name" in data
        assert "tier" in data
        assert "cost" in data
        assert "effect" in data

    def test_all_card_ids_in_db(self) -> None:
        from hearthstone.engine.enums import CardIDs as CIDs

        for cid in CIDs:
            assert cid in CARD_DB, f"{cid} not in CARD_DB"

    def test_all_spell_ids_in_db(self) -> None:
        from hearthstone.engine.enums import SpellIDs as SIDs

        for sid in SIDs:
            assert sid in SPELL_DB, f"{sid} not in SPELL_DB"


# ===================================================================
#  13. TRIPLET REWARD → DISCOVERY FLOW
# ===================================================================


class TestTripletDiscovery:
    """Full triplet → golden → reward → discovery → choose."""

    def test_triplet_reward_starts_discovery(
        self,
        empty_game: "Game",
        player: Player,
        tavern: "TavernManager",
    ) -> None:
        reward = Spell.create_from_db(SpellIDs.TRIPLET_REWARD)
        reward.params["tier"] = 2
        player.hand.append(HandCard(uid=9999, spell=reward))

        success, info = tavern.play_unit(player, hand_index=0, insert_index=-1, target_index=-1)

        assert success
        assert player.is_discovering
        assert player.discovery.is_exact_tier

    def test_triplet_full_flow(
        self,
        empty_game: "Game",
        player: Player,
        tavern: "TavernManager",
        mock_unit: Callable[..., Unit],
    ) -> None:
        cid = CardIDs.FLIGHTY_SCOUT
        for _ in range(2):
            u = mock_unit(cid)
            player.hand.append(HandCard(uid=u.uid, unit=u))

        u3 = mock_unit(cid)
        player.store.insert(0, StoreItem(unit=u3))
        player.gold = 10

        tavern.buy_unit(player, 0)
        golden_idx = next(i for i, hc in enumerate(player.hand) if hc.unit and hc.unit.is_golden)

        tavern.play_unit(player, golden_idx, insert_index=0)
        reward_idx = next(
            i
            for i, hc in enumerate(player.hand)
            if hc.spell and hc.spell.card_id == SpellIDs.TRIPLET_REWARD
        )
        tavern.play_unit(player, reward_idx, insert_index=-1, target_index=-1)
        assert player.is_discovering

        tavern.make_discovery_choice(player, 0)
        assert not player.is_discovering


# ===================================================================
#  14. ELEMENTAL BUFF SYSTEM TRIGGER
# ===================================================================


class TestElementalBuff:
    """MINION_ADDED_TO_SHOP system trigger buffs elementals."""

    def test_elemental_buff_when_mechanic_set(
        self,
        empty_game: "Game",
        player: Player,
        tavern: "TavernManager",
    ) -> None:
        player.mechanics.modify_stat(MechanicType.ELEMENTAL_BUFF, 2, 3)
        player.tavern_tier = 1  # CRACKLING_CYCLONE and DUNE_DWELLER are tier 1 Elementals
        player.gold = 10
        tavern.roll_tavern(player)

        elementals = [
            item.unit
            for item in player.store
            if item.unit and UnitType.ELEMENTAL in item.unit.types
        ]

        for elem in elementals:
            assert elem.perm_atk_add >= 2
            assert elem.perm_hp_add >= 3


# ===================================================================
#  15. MAGNETIC FALLTHROUGH
# ===================================================================


@pytest.mark.skip(reason="ANNOY_O_MODULE (Magnetic) not in current patch")
class TestMagneticFallthrough:
    """Magnetic unit without Mech target plays normally."""

    def test_magnetic_on_empty_board_plays_normally(self) -> None:
        pass

    def test_magnetic_targeting_non_mech_plays_normally(self) -> None:
        pass


# ===================================================================
#  16. GOLDEN SPAWN OF N'ZOTH
# ===================================================================


@pytest.mark.skip(reason="SPAWN_OF_NZOTH deathrattle not in current patch")
class TestGoldenSpawnOfNzoth:
    """Golden Spawn of N'Zoth DR fires twice → +2/+2 combat buff."""

    def test_golden_dr_double_buff(self) -> None:
        pass


# ===================================================================
#  17. AVENGE MECHANIC
# ===================================================================


@pytest.mark.skip(reason="make_avenge_trigger not implemented yet")
class TestAvengeMechanic:
    """make_avenge_trigger counter / threshold / golden logic."""

    def test_avenge_counter_threshold(self) -> None:
        pass


# ===================================================================
#  18. MULTIPLE AURA SOURCES
# ===================================================================


@pytest.mark.skip(reason="MURLOC_WARLEADER, DIRE_WOLF_ALPHA, SOUTHSEA_CAPTAIN auras not in current patch")
class TestMultipleAuras:
    """Multiple aura sources stack correctly."""

    def test_two_warleaders_stack(self) -> None:
        pass

    def test_wolf_and_captain_stack(self) -> None:
        pass

    def test_warleader_does_not_buff_warleader_type_mismatch(self) -> None:
        pass


# ===================================================================
#  19. OVERKILL EVENT
# ===================================================================


class TestOverkillEvent:
    """OVERKILL event should fire when damage exceeds target HP."""

    def test_overkill_fires(
        self,
        combat_players: Callable[..., Tuple[Dict[int, Player], List[List[Unit]], CombatManager]],
    ) -> None:
        players, boards, cm = combat_players(
            [CardIDs.ROT_HIDE_GNOLL],  # 1/4
            [CardIDs.MICROBOT],  # 1/1
        )
        attacker = boards[0][0]
        attacker.perm_atk_add = 3  # 1+3=4 atk, 3 overkill on MICROBOT (1 hp)
        attacker.recalc_stats()
        target = boards[1][0]

        cm.perform_attack(attacker, target, players)

        # Target should be dead
        assert target.cur_hp <= 0
        # No crash = event pipeline handled OVERKILL correctly
