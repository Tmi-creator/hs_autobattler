"""Tests for entity data models: Unit, Spell, Player, EconomyState, MechanicState.

Covers: create_from_db, recalc_stats, magnetize_from, combat_copy,
reset_*_layer, property accessors, StoreItem, HandCard.
"""

from __future__ import annotations

import pytest

from hearthstone.engine.entities import (
    EconomyState,
    HandCard,
    MechanicState,
    Player,
    Spell,
    StoreItem,
    Unit,
)
from hearthstone.engine.enums import CardIDs, MechanicType, SpellIDs, Tags, UnitType

# ===================================================================
#  1. UNIT CREATION
# ===================================================================


class TestUnitCreation:
    """Unit.create_from_db factory."""

    def test_create_valid_unit(self) -> None:
        unit = Unit.create_from_db(CardIDs.TABBYCAT, uid=1, owner_id=0)
        assert unit.card_id == CardIDs.TABBYCAT
        assert unit.uid == 1
        assert unit.owner_id == 0

    def test_create_golden_doubles_base_stats(self) -> None:
        normal = Unit.create_from_db(CardIDs.ALLEYCAT, uid=1, owner_id=0, is_golden=False)
        golden = Unit.create_from_db(CardIDs.ALLEYCAT, uid=2, owner_id=0, is_golden=True)

        assert golden.base_atk == normal.base_atk * 2
        assert golden.base_hp == normal.base_hp * 2

    def test_create_preserves_tags(self) -> None:
        unit = Unit.create_from_db(CardIDs.ANNOY_O_TRON, uid=1, owner_id=0)
        assert Tags.DIVINE_SHIELD in unit.tags
        assert Tags.TAUNT in unit.tags

    def test_create_preserves_types(self) -> None:
        unit = Unit.create_from_db(CardIDs.ANNOY_O_TRON, uid=1, owner_id=0)
        assert UnitType.MECH in unit.types

    def test_create_invalid_id_raises(self) -> None:
        with pytest.raises(ValueError, match="not found"):
            Unit.create_from_db("INVALID_999", uid=1, owner_id=0)

    def test_create_sets_cur_equal_max(self) -> None:
        unit = Unit.create_from_db(CardIDs.MOLTEN_ROCK, uid=1, owner_id=0)
        assert unit.cur_atk == unit.max_atk
        assert unit.cur_hp == unit.max_hp

    def test_create_sets_tier(self) -> None:
        unit = Unit.create_from_db(CardIDs.MOLTEN_ROCK, uid=1, owner_id=0)
        assert unit.tier == 2  # Molten Rock is tier 2


# ===================================================================
#  2. SPELL CREATION
# ===================================================================


class TestSpellCreation:
    """Spell.create_from_db factory."""

    def test_create_valid_spell(self) -> None:
        spell = Spell.create_from_db(SpellIDs.BANANA)
        assert spell.card_id == SpellIDs.BANANA
        assert spell.cost == 3

    def test_create_invalid_id_raises(self) -> None:
        with pytest.raises(ValueError, match="not found"):
            Spell.create_from_db("INVALID_SPELL")

    def test_spell_has_params(self) -> None:
        spell = Spell.create_from_db(SpellIDs.BANANA)
        assert "atk" in spell.params
        assert "hp" in spell.params

    def test_temporary_spell_flag(self) -> None:
        temp = Spell.create_from_db(SpellIDs.SURF_SPELLCRAFT)
        assert temp.is_temporary

        normal = Spell.create_from_db(SpellIDs.BANANA)
        assert not normal.is_temporary

    def test_triplet_reward_not_temporary(self) -> None:
        spell = Spell.create_from_db(SpellIDs.TRIPLET_REWARD)
        assert not spell.is_temporary


# ===================================================================
#  3. STAT RECALCULATION
# ===================================================================


class TestRecalcStats:
    """Unit.recalc_stats preserves missing HP, clamps correctly."""

    def test_buff_preserves_damage(self) -> None:
        """If unit has taken 2 damage, adding +3 HP should keep 2 HP missing."""
        unit = Unit.create_from_db(CardIDs.MOLTEN_ROCK, uid=1, owner_id=0)  # 4/7
        # Take 2 damage
        unit.cur_hp = unit.max_hp - 2  # 5
        old_missing = unit.max_hp - unit.cur_hp  # 2

        unit.perm_hp_add += 3
        unit.recalc_stats()

        assert unit.max_hp == 7 + 3  # 10
        assert unit.cur_hp == 10 - old_missing  # 8

    def test_negative_hp_clamps_to_zero(self) -> None:
        unit = Unit.create_from_db(CardIDs.TABBYCAT, uid=1, owner_id=0)
        unit.cur_hp = -5
        unit.recalc_stats()
        assert unit.cur_hp == 0

    def test_restore_stats_heals_to_full(self) -> None:
        unit = Unit.create_from_db(CardIDs.MOLTEN_ROCK, uid=1, owner_id=0)
        unit.cur_hp = 1
        unit.restore_stats()
        assert unit.cur_hp == unit.max_hp

    def test_all_layers_contribute_to_max(self) -> None:
        unit = Unit.create_from_db(CardIDs.TABBYCAT, uid=1, owner_id=0)  # base 1/1
        unit.perm_atk_add = 1
        unit.turn_atk_add = 2
        unit.combat_atk_add = 3
        unit.aura_atk_add = 4
        unit.recalc_stats()
        assert unit.max_atk == 1 + 1 + 2 + 3 + 4  # 11
        assert unit.cur_atk == 11

    def test_hp_cannot_exceed_max(self) -> None:
        unit = Unit.create_from_db(CardIDs.TABBYCAT, uid=1, owner_id=0)
        unit.recalc_stats()
        # cur_hp should never exceed max_hp
        assert unit.cur_hp <= unit.max_hp


# ===================================================================
#  4. LAYER RESETS
# ===================================================================


class TestLayerResets:
    """reset_aura_layer, reset_turn_layer, reset_combat_layer."""

    def test_reset_aura_layer(self) -> None:
        unit = Unit.create_from_db(CardIDs.TABBYCAT, uid=1, owner_id=0)
        unit.aura_atk_add = 5
        unit.aura_hp_add = 3
        unit.reset_aura_layer()
        assert unit.aura_atk_add == 0
        assert unit.aura_hp_add == 0

    def test_reset_turn_layer_clears_attached(self) -> None:
        unit = Unit.create_from_db(CardIDs.TABBYCAT, uid=1, owner_id=0)
        unit.turn_atk_add = 3
        unit.turn_hp_add = 2
        unit.attached_turn = {"SOME_EFFECT": 1}
        unit.reset_turn_layer()
        assert unit.turn_atk_add == 0
        assert unit.turn_hp_add == 0
        assert unit.attached_turn == {}

    def test_reset_combat_layer_clears_avenge(self) -> None:
        unit = Unit.create_from_db(CardIDs.TABBYCAT, uid=1, owner_id=0)
        unit.combat_atk_add = 10
        unit.combat_hp_add = 10
        unit.avenge_counter = 3
        unit.attached_combat = {"COMBAT_EFFECT": 2}
        unit.reset_combat_layer()
        assert unit.combat_atk_add == 0
        assert unit.combat_hp_add == 0
        assert unit.avenge_counter == 0
        assert unit.attached_combat == {}


# ===================================================================
#  5. MAGNETIZE
# ===================================================================


class TestMagnetize:
    """Unit.magnetize_from merging mechanics."""

    def test_magnetize_adds_stats(self) -> None:
        target = Unit.create_from_db(CardIDs.ANNOY_O_TRON, uid=1, owner_id=0)
        source = Unit.create_from_db(CardIDs.ANNOY_O_MODULE, uid=2, owner_id=0)

        atk_before = target.cur_atk
        hp_before = target.cur_hp

        target.magnetize_from(source)

        assert target.cur_atk == atk_before + source.base_atk
        assert target.cur_hp == hp_before + source.base_hp

    def test_magnetize_transfers_tags_except_magnetic(self) -> None:
        target = Unit.create_from_db(CardIDs.ANNOY_O_TRON, uid=1, owner_id=0)
        target.tags.discard(Tags.DIVINE_SHIELD)

        source = Unit.create_from_db(CardIDs.ANNOY_O_MODULE, uid=2, owner_id=0)
        target.magnetize_from(source)

        assert Tags.DIVINE_SHIELD in target.tags
        assert Tags.MAGNETIC not in target.tags

    def test_magnetize_normal_source_one_pool_copy(self) -> None:
        target = Unit.create_from_db(CardIDs.ANNOY_O_TRON, uid=1, owner_id=0)
        source = Unit.create_from_db(CardIDs.ANNOY_O_MODULE, uid=2, owner_id=0)
        target.magnetize_from(source)
        assert target.absorbed_pool_copies.get(source.card_id, 0) == 1

    def test_magnetize_golden_source_three_pool_copies(self) -> None:
        target = Unit.create_from_db(CardIDs.ANNOY_O_TRON, uid=1, owner_id=0)
        source = Unit.create_from_db(CardIDs.ANNOY_O_MODULE, uid=2, owner_id=0, is_golden=True)
        target.magnetize_from(source)
        assert target.absorbed_pool_copies.get(source.card_id, 0) == 3

    def test_magnetize_adds_trigger_stacks(self) -> None:
        target = Unit.create_from_db(CardIDs.ANNOY_O_TRON, uid=1, owner_id=0)
        source = Unit.create_from_db(CardIDs.ANNOY_O_MODULE, uid=2, owner_id=0)
        target.magnetize_from(source)
        # normal source → stacks=1
        assert target.attached_perm.get(source.card_id, 0) == 1

    def test_magnetize_golden_source_double_trigger_stacks(self) -> None:
        target = Unit.create_from_db(CardIDs.ANNOY_O_TRON, uid=1, owner_id=0)
        source = Unit.create_from_db(CardIDs.ANNOY_O_MODULE, uid=2, owner_id=0, is_golden=True)
        target.magnetize_from(source)
        assert target.attached_perm.get(source.card_id, 0) == 2

    def test_magnetize_merges_absorbed_pool_copies(self) -> None:
        target = Unit.create_from_db(CardIDs.ANNOY_O_TRON, uid=1, owner_id=0)
        source = Unit.create_from_db(CardIDs.ANNOY_O_MODULE, uid=2, owner_id=0)
        source.absorbed_pool_copies["EXTRA_CARD"] = 2

        target.magnetize_from(source)

        assert target.absorbed_pool_copies.get("EXTRA_CARD", 0) == 2


# ===================================================================
#  6. COMBAT COPY
# ===================================================================


class TestCombatCopyEntities:
    """Unit/Player.combat_copy isolation."""

    def test_unit_combat_copy_resets_combat_and_aura(self) -> None:
        unit = Unit.create_from_db(CardIDs.TABBYCAT, uid=1, owner_id=0)
        unit.combat_atk_add = 10
        unit.combat_hp_add = 10
        unit.aura_atk_add = 5
        unit.aura_hp_add = 5

        copy = unit.combat_copy()

        assert copy.combat_atk_add == 0
        assert copy.combat_hp_add == 0
        assert copy.aura_atk_add == 0
        assert copy.aura_hp_add == 0

    def test_unit_combat_copy_preserves_perm_and_turn(self) -> None:
        unit = Unit.create_from_db(CardIDs.TABBYCAT, uid=1, owner_id=0)
        unit.perm_atk_add = 3
        unit.turn_hp_add = 5

        copy = unit.combat_copy()

        assert copy.perm_atk_add == 3
        assert copy.turn_hp_add == 5

    def test_unit_combat_copy_independent_tags(self) -> None:
        unit = Unit.create_from_db(CardIDs.ANNOY_O_TRON, uid=1, owner_id=0)
        copy = unit.combat_copy()
        copy.tags.clear()

        assert Tags.DIVINE_SHIELD in unit.tags

    def test_unit_combat_copy_independent_attached(self) -> None:
        unit = Unit.create_from_db(CardIDs.TABBYCAT, uid=1, owner_id=0)
        unit.attached_perm["EFFECT_A"] = 1

        copy = unit.combat_copy()
        copy.attached_perm["EFFECT_B"] = 2

        assert "EFFECT_B" not in unit.attached_perm

    def test_player_combat_copy_deep_copies_board(self) -> None:
        player = Player(uid=0, board=[], hand=[])
        unit = Unit.create_from_db(CardIDs.TABBYCAT, uid=1, owner_id=0)
        player.board = [unit]

        cp = player.combat_copy()
        cp.board[0].cur_hp = 0

        assert player.board[0].cur_hp > 0

    def test_player_combat_copy_deep_copies_economy(self) -> None:
        player = Player(uid=0, board=[], hand=[])
        player.gold = 10

        cp = player.combat_copy()
        cp.gold = 0

        assert player.gold == 10


# ===================================================================
#  7. ECONOMY STATE
# ===================================================================


class TestEconomyState:
    """EconomyState.new_turn gold/up_cost calculations."""

    def test_new_turn_gold_calculation(self) -> None:
        eco = EconomyState()
        eco.new_turn(3)  # gold = min(10, 3+3-1) = 5
        assert eco.gold == 5

    def test_new_turn_gold_caps_at_10(self) -> None:
        eco = EconomyState()
        eco.new_turn(10)  # gold = min(10, 3+10-1) = min(10, 12) = 10
        assert eco.gold == 10

    def test_new_turn_gold_next_turn_added(self) -> None:
        eco = EconomyState()
        eco.gold_next_turn = 3
        eco.new_turn(3)  # gold = 5 + 3 = 8
        assert eco.gold == 8
        assert eco.gold_next_turn == 0  # consumed

    def test_new_turn_up_cost_decreases(self) -> None:
        eco = EconomyState()
        eco.up_cost = 5
        eco.new_turn(2)  # not turn 1 → up_cost -= 1
        assert eco.up_cost == 4

    def test_new_turn_first_turn_no_decrease(self) -> None:
        eco = EconomyState()
        eco.up_cost = 5
        eco.new_turn(1)
        assert eco.up_cost == 5

    def test_new_turn_zero_up_cost_stays_zero(self) -> None:
        eco = EconomyState()
        eco.up_cost = 0
        eco.new_turn(3)
        assert eco.up_cost == 0


# ===================================================================
#  8. MECHANIC STATE
# ===================================================================


class TestMechanicState:
    """MechanicState.modify_stat/get_stat."""

    def test_modify_stat_adds(self) -> None:
        ms = MechanicState()
        ms.modify_stat(MechanicType.BLOOD_GEM, 2, 3)
        atk, hp = ms.get_stat(MechanicType.BLOOD_GEM)
        # Default (1,1) + (2,3) = (3,4)
        assert (atk, hp) == (3, 4)

    def test_get_stat_default_blood_gem(self) -> None:
        ms = MechanicState()
        assert ms.get_stat(MechanicType.BLOOD_GEM) == (1, 1)

    def test_get_stat_default_elemental_buff(self) -> None:
        ms = MechanicState()
        assert ms.get_stat(MechanicType.ELEMENTAL_BUFF) == (0, 0)


# ===================================================================
#  9. PROPERTY ACCESSORS
# ===================================================================


class TestPropertyAccessors:
    """StoreItem.card_id, HandCard.card_id."""

    def test_store_item_unit_card_id(self) -> None:
        unit = Unit.create_from_db(CardIDs.TABBYCAT, uid=1, owner_id=0)
        assert StoreItem(unit=unit).card_id == CardIDs.TABBYCAT

    def test_store_item_spell_card_id(self) -> None:
        spell = Spell.create_from_db(SpellIDs.BANANA)
        assert StoreItem(spell=spell).card_id == SpellIDs.BANANA

    def test_store_item_empty_card_id(self) -> None:
        assert StoreItem().card_id == ""

    def test_hand_card_unit_card_id(self) -> None:
        unit = Unit.create_from_db(CardIDs.TABBYCAT, uid=1, owner_id=0)
        assert HandCard(uid=1, unit=unit).card_id == CardIDs.TABBYCAT

    def test_hand_card_spell_card_id(self) -> None:
        spell = Spell.create_from_db(SpellIDs.BANANA)
        assert HandCard(uid=1, spell=spell).card_id == SpellIDs.BANANA

    def test_hand_card_empty_card_id(self) -> None:
        assert HandCard(uid=1).card_id == "NO ID"


# ===================================================================
#  10. PLAYER PROPERTIES
# ===================================================================


class TestPlayerProperties:
    """Player property accessors delegate to EconomyState."""

    def test_gold_property(self) -> None:
        p = Player(uid=0, board=[], hand=[])
        p.gold = 7
        assert p.gold == 7
        assert p.economy.gold == 7

    def test_tavern_tier_property(self) -> None:
        p = Player(uid=0, board=[], hand=[])
        p.tavern_tier = 3
        assert p.tavern_tier == 3
        assert p.economy.tavern_tier == 3

    def test_spell_discount_property(self) -> None:
        p = Player(uid=0, board=[], hand=[])
        p.spell_discount = 2
        assert p.spell_discount == 2

    def test_gold_next_turn_property(self) -> None:
        p = Player(uid=0, board=[], hand=[])
        p.gold_next_turn = 5
        assert p.gold_next_turn == 5
        assert p.economy.gold_next_turn == 5

    def test_up_cost_property(self) -> None:
        p = Player(uid=0, board=[], hand=[])
        p.up_cost = 8
        assert p.up_cost == 8

    def test_store_property_is_economy_store(self) -> None:
        p = Player(uid=0, board=[], hand=[])
        assert p.store is p.economy.store

    def test_is_discovering_default_false(self) -> None:
        p = Player(uid=0, board=[], hand=[])
        assert not p.is_discovering

    def test_is_discovering_active(self) -> None:
        p = Player(uid=0, board=[], hand=[])
        p.discovery.is_active = True
        assert p.is_discovering


# ===================================================================
#  11. UNIT BOOLEAN PROPERTIES
# ===================================================================


class TestUnitBooleanProperties:
    """has_taunt, has_divine_shield, etc."""

    def test_all_boolean_properties(self) -> None:
        unit = Unit.create_from_db(CardIDs.TABBYCAT, uid=1, owner_id=0)
        # Tabbycat has no special tags
        assert not unit.has_taunt
        assert not unit.has_divine_shield
        assert not unit.has_windfury
        assert not unit.has_poisonous
        assert not unit.has_venomous
        assert not unit.has_reborn
        assert not unit.has_cleave
        assert not unit.has_stealth
        assert not unit.has_immediate_attack
        assert not unit.has_magnetic

    def test_is_alive(self) -> None:
        unit = Unit.create_from_db(CardIDs.TABBYCAT, uid=1, owner_id=0)
        assert unit.is_alive
        unit.cur_hp = 0
        assert not unit.is_alive
        unit.cur_hp = -5
        assert not unit.is_alive
