"""Tests for the RL environment (hs_env.py).

Covers: observation space shape/range, action masks for every phase,
action mapping, step cycle, entity encoding, bot turn, auto-positioning.

This is the SINGLE BIGGEST coverage gap — 834 lines of hs_env.py had ZERO tests.
"""

from __future__ import annotations

import numpy as np
import pytest

from hearthstone.engine.entities import HandCard, Spell, StoreItem, Unit
from hearthstone.engine.enums import CardIDs, SpellIDs, Tags
from hearthstone.env.hs_env import HearthstoneEnv


@pytest.fixture()
def env() -> HearthstoneEnv:
    """Fresh HearthstoneEnv instance."""
    return HearthstoneEnv()


# ===================================================================
#  1. OBSERVATION SPACE
# ===================================================================


class TestObservationSpace:
    """Observation shape and value-range invariants."""

    def test_reset_returns_correct_obs_shape(self, env: HearthstoneEnv) -> None:
        obs, info = env.reset(seed=42)
        assert obs.shape == env.observation_space.shape

    def test_obs_values_within_bounds(self, env: HearthstoneEnv) -> None:
        obs, _ = env.reset(seed=42)
        assert np.all(obs >= 0.0), f"Min value: {obs.min()}"
        assert np.all(obs <= 1.0 + 1e-6), f"Max value: {obs.max()}"

    def test_obs_dtype_is_float32(self, env: HearthstoneEnv) -> None:
        obs, _ = env.reset(seed=42)
        assert obs.dtype == np.float32

    def test_obs_shape_stable_across_resets(self, env: HearthstoneEnv) -> None:
        obs1, _ = env.reset(seed=1)
        obs2, _ = env.reset(seed=2)
        assert obs1.shape == obs2.shape

    def test_obs_after_step_has_correct_shape(self, env: HearthstoneEnv) -> None:
        env.reset(seed=42)
        obs, reward, done, truncated, info = env.step(0)  # END_TURN
        if not done:
            assert obs.shape == env.observation_space.shape

    def test_obs_global_features_count(self, env: HearthstoneEnv) -> None:
        """First 7 floats are global features."""
        obs, _ = env.reset(seed=42)
        # gold/MAX_GOLD, tier/MAX_TIER, health/MAX_HP, up_cost/10, discount/MAX, discover, target
        assert len(obs) >= 7


# ===================================================================
#  2. ACTION MASKS
# ===================================================================


class TestActionMasks:
    """Action mask correctness for default / discovery / targeting phases."""

    def test_masks_shape_and_dtype(self, env: HearthstoneEnv) -> None:
        env.reset(seed=42)
        masks = env.action_masks()
        assert masks.shape == (32,)
        assert masks.dtype == np.bool_

    def test_end_turn_always_valid_in_default_phase(self, env: HearthstoneEnv) -> None:
        env.reset(seed=42)
        masks = env.action_masks()
        assert masks[0]  # END_TURN always valid

    def test_roll_requires_gold(self, env: HearthstoneEnv) -> None:
        env.reset(seed=42)
        player = env.game.players[env.my_player_id]
        player.gold = 0
        masks = env.action_masks()
        assert not masks[1]

    def test_roll_available_with_gold(self, env: HearthstoneEnv) -> None:
        env.reset(seed=42)
        player = env.game.players[env.my_player_id]
        player.gold = 10
        masks = env.action_masks()
        assert masks[1]

    def test_buy_masks_depend_on_store_and_gold(self, env: HearthstoneEnv) -> None:
        env.reset(seed=42)
        player = env.game.players[env.my_player_id]
        player.gold = 10
        store_len = len(player.store)
        masks = env.action_masks()
        # Slots with items + enough gold → buyable
        for i in range(store_len):
            item = player.store[i]
            if item.unit:
                assert masks[2 + i], f"Store unit slot {i} should be buyable"
        # Slots beyond store → not buyable
        for i in range(store_len, 7):
            assert not masks[2 + i]

    def test_sell_masks_depend_on_board(self, env: HearthstoneEnv) -> None:
        env.reset(seed=42)
        player = env.game.players[env.my_player_id]
        board_len = len(player.board)
        masks = env.action_masks()
        for i in range(board_len):
            assert masks[9 + i]
        for i in range(board_len, 7):
            assert not masks[9 + i]

    def test_play_empty_hand_all_invalid(self, env: HearthstoneEnv) -> None:
        env.reset(seed=42)
        player = env.game.players[env.my_player_id]
        player.hand.clear()
        masks = env.action_masks()
        for i in range(10):
            assert not masks[16 + i]

    def test_discovery_phase_masks(self, env: HearthstoneEnv) -> None:
        """During discovery, only option slots (2-4) should be valid."""
        env.reset(seed=42)
        player = env.game.players[env.my_player_id]
        env.game.tavern.start_discovery(player, source="Test", tier=1, count=3)
        masks = env.action_masks()

        # Only discovery options
        num_options = len(player.discovery.options)
        for i in range(num_options):
            assert masks[2 + i]
        # Nothing else
        assert not masks[0]  # END_TURN
        assert not masks[1]  # ROLL
        for i in range(num_options + 2, 32):
            assert not masks[i]

    def test_targeting_phase_masks_board_targets(self, env: HearthstoneEnv) -> None:
        """During spell targeting, only board units should be valid targets."""
        env.reset(seed=42)
        player = env.game.players[env.my_player_id]

        unit = Unit.create_from_db(CardIDs.TABBYCAT, 9999, player.uid)
        player.board.append(unit)

        spell = Spell.create_from_db(SpellIDs.BANANA)
        player.hand.append(HandCard(uid=9998, spell=spell))

        env.is_targeting = True
        env.pending_spell_hand_index = len(player.hand) - 1
        env.pending_target_kind = "SPELL"

        masks = env.action_masks()

        # Board slot 0 (action 2) → valid target
        assert masks[2]
        # Cancel NOT valid when targets exist
        assert not masks[0]

    def test_magnetize_targeting_only_mechs(self, env: HearthstoneEnv) -> None:
        """Magnetize targeting should only allow Mech targets."""
        env.reset(seed=42)
        player = env.game.players[env.my_player_id]

        cat = Unit.create_from_db(CardIDs.TABBYCAT, 9999, player.uid)
        mech = Unit.create_from_db(CardIDs.ANNOY_O_TRON, 9998, player.uid)
        player.board = [cat, mech]

        env.is_targeting = True
        env.pending_target_kind = "MAGNETIZE"
        env.pending_spell_hand_index = 0

        masks = env.action_masks()

        assert not masks[2]  # cat is Beast → invalid
        assert masks[3]  # mech is Mech → valid

    def test_max_actions_forces_end_turn_only(self, env: HearthstoneEnv) -> None:
        """After max_actions_in_turn, only END_TURN should be valid."""
        env.reset(seed=42)
        env.actions_in_turn = env.max_actions_in_turn

        masks = env.action_masks()

        assert masks[0]
        assert not any(masks[1:])


# ===================================================================
#  3. STEP CYCLE
# ===================================================================


class TestStepCycle:
    """step() action processing and return values."""

    def test_step_returns_five_values(self, env: HearthstoneEnv) -> None:
        env.reset(seed=42)
        result = env.step(0)
        assert len(result) == 5
        obs, reward, done, truncated, info = result
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert isinstance(truncated, bool)

    def test_step_roll_deducts_gold(self, env: HearthstoneEnv) -> None:
        env.reset(seed=42)
        player = env.game.players[env.my_player_id]
        gold_before = player.gold
        env.step(1)  # ROLL
        assert player.gold == gold_before - 1

    def test_step_buy_works(self, env: HearthstoneEnv) -> None:
        env.reset(seed=42)
        player = env.game.players[env.my_player_id]
        player.gold = 10
        hand_before = len(player.hand)
        store_has_unit = any(item.unit for item in player.store)

        if store_has_unit:
            env.step(2)  # BUY slot 0
            # Hand may grow (or trigger triplet)
            assert len(player.hand) >= hand_before or any(u.is_golden for u in player.board)

    def test_step_sell_removes_from_board(self, env: HearthstoneEnv) -> None:
        env.reset(seed=42)
        player = env.game.players[env.my_player_id]
        unit = Unit.create_from_db(CardIDs.TABBYCAT, 9999, player.uid)
        player.board.append(unit)
        board_before = len(player.board)

        env.step(9)  # SELL slot 0

        assert len(player.board) == board_before - 1

    def test_step_truncated_on_max_steps(self, env: HearthstoneEnv) -> None:
        env.reset(seed=42)
        env.steps_taken = env.max_steps_per_episode - 1
        _, _, _, truncated, _ = env.step(0)  # END_TURN
        assert truncated

    def test_step_targeting_spell_flow(self, env: HearthstoneEnv) -> None:
        """PLAY targeted spell → WAIT_FOR_TARGET → select target."""
        env.reset(seed=42)
        player = env.game.players[env.my_player_id]

        cat = Unit.create_from_db(CardIDs.TABBYCAT, 9999, player.uid)
        player.board.append(cat)

        spell = Spell.create_from_db(SpellIDs.BANANA)
        player.hand.append(HandCard(uid=9998, spell=spell))

        hand_idx = len(player.hand) - 1
        env.step(16 + hand_idx)  # PLAY card → enters targeting

        assert env.is_targeting
        assert env.pending_spell_hand_index == hand_idx

    def test_step_cancel_targeting(self, env: HearthstoneEnv) -> None:
        """Cancel targeting via action 0 while in targeting phase."""
        env.reset(seed=42)
        env.is_targeting = True
        env.pending_spell_hand_index = 0

        _, reward, _, _, _ = env.step(0)  # Cancel

        assert not env.is_targeting
        assert env.pending_spell_hand_index is None
        assert reward == pytest.approx(-0.01)


# ===================================================================
#  4. ENTITY ENCODING
# ===================================================================


class TestEntityEncoding:
    """_encode_single_entity vector correctness."""

    def test_empty_slot_all_zeros(self, env: HearthstoneEnv) -> None:
        env.reset(seed=42)
        vec = env._encode_single_entity(StoreItem())
        assert all(v == 0.0 for v in vec)

    def test_unit_is_present_flag(self, env: HearthstoneEnv) -> None:
        env.reset(seed=42)
        unit = Unit.create_from_db(CardIDs.TABBYCAT, 9999, 0)
        vec = env._encode_single_entity(unit)
        assert vec[0] == 1.0  # is_present
        assert vec[1] == 0.0  # not a spell

    def test_spell_is_spell_flag(self, env: HearthstoneEnv) -> None:
        env.reset(seed=42)
        spell = Spell.create_from_db(SpellIDs.BANANA)
        item = HandCard(uid=9999, spell=spell)
        vec = env._encode_single_entity(item)
        assert vec[0] == 1.0  # is_present
        assert vec[1] == 1.0  # is_spell

    def test_entity_vector_length(self, env: HearthstoneEnv) -> None:
        env.reset(seed=42)
        unit = Unit.create_from_db(CardIDs.ANNOY_O_TRON, 9999, 0)
        vec = env._encode_single_entity(unit)
        assert len(vec) == env.entity_features

    def test_targeting_selected_flag(self, env: HearthstoneEnv) -> None:
        env.reset(seed=42)
        env.is_targeting = True
        env.pending_spell_hand_index = 2

        spell = Spell.create_from_db(SpellIDs.BANANA)
        item = HandCard(uid=9999, spell=spell)

        # is_selected should be 1.0 when index matches and zone is HAND
        vec = env._encode_single_entity(item, index_in_zone=2, zone_type="HAND")
        assert vec[25] == 1.0

    def test_targeting_not_selected_wrong_index(self, env: HearthstoneEnv) -> None:
        env.reset(seed=42)
        env.is_targeting = True
        env.pending_spell_hand_index = 2

        spell = Spell.create_from_db(SpellIDs.BANANA)
        item = HandCard(uid=9999, spell=spell)

        vec = env._encode_single_entity(item, index_in_zone=0, zone_type="HAND")
        assert vec[25] == 0.0

    def test_frozen_store_item(self, env: HearthstoneEnv) -> None:
        env.reset(seed=42)
        unit = Unit.create_from_db(CardIDs.TABBYCAT, 9999, 0)
        item = StoreItem(unit=unit, is_frozen=True)
        vec = env._encode_single_entity(item)
        assert vec[5] == 1.0  # is_frozen


# ===================================================================
#  5. _can_play_card EDGE CASES
# ===================================================================


class TestCanPlayCard:
    """Playability checks for hand cards."""

    def test_targeted_spell_empty_board_cannot_play(self, env: HearthstoneEnv) -> None:
        env.reset(seed=42)
        player = env.game.players[env.my_player_id]
        player.board.clear()

        spell = Spell.create_from_db(SpellIDs.BANANA)  # needs target
        player.hand.append(HandCard(uid=9999, spell=spell))

        assert not env._can_play_card(player, len(player.hand) - 1)

    def test_unit_full_board_cannot_play(self, env: HearthstoneEnv) -> None:
        env.reset(seed=42)
        player = env.game.players[env.my_player_id]
        player.board = [Unit.create_from_db(CardIDs.TABBYCAT, i, player.uid) for i in range(7)]

        unit = Unit.create_from_db(CardIDs.TABBYCAT, 999, player.uid)
        player.hand.append(HandCard(uid=999, unit=unit))

        assert not env._can_play_card(player, len(player.hand) - 1)

    def test_untargeted_spell_can_play(self, env: HearthstoneEnv) -> None:
        env.reset(seed=42)
        player = env.game.players[env.my_player_id]

        spell = Spell.create_from_db(SpellIDs.TAVERN_COIN)  # no target needed
        player.hand.append(HandCard(uid=9999, spell=spell))

        assert env._can_play_card(player, len(player.hand) - 1)

    def test_out_of_range_index_returns_false(self, env: HearthstoneEnv) -> None:
        env.reset(seed=42)
        player = env.game.players[env.my_player_id]
        assert not env._can_play_card(player, 99)


# ===================================================================
#  6. AUTO POSITIONING
# ===================================================================


class TestAutoPosition:
    """_auto_position_board heuristic sorting."""

    def test_cleave_positioned_first(self, env: HearthstoneEnv) -> None:
        env.reset(seed=42)
        player = env.game.players[env.my_player_id]

        cat = Unit.create_from_db(CardIDs.TABBYCAT, 9999, player.uid)
        cleaver = Unit.create_from_db(CardIDs.TABBYCAT, 9998, player.uid)
        cleaver.tags.add(Tags.CLEAVE)
        cleaver.perm_atk_add = 5
        cleaver.recalc_stats()

        player.board = [cat, cleaver]
        env._auto_position_board(player)

        assert player.board[0].uid == cleaver.uid  # Cleave first

    def test_poison_positioned_early(self, env: HearthstoneEnv) -> None:
        env.reset(seed=42)
        player = env.game.players[env.my_player_id]

        normal = Unit.create_from_db(CardIDs.MOLTEN_ROCK, 9999, player.uid)
        poison = Unit.create_from_db(CardIDs.TABBYCAT, 9998, player.uid)
        poison.tags.add(Tags.POISONOUS)

        player.board = [normal, poison]
        env._auto_position_board(player)

        assert player.board[0].uid == poison.uid  # Poison first

    def test_low_atk_taunt_positioned_last(self, env: HearthstoneEnv) -> None:
        env.reset(seed=42)
        player = env.game.players[env.my_player_id]

        # Molten Rock: 4/7 Taunt, atk<5 → penalty (-2000)
        tank = Unit.create_from_db(CardIDs.MOLTEN_ROCK, 9999, player.uid)
        attacker = Unit.create_from_db(CardIDs.TABBYCAT, 9998, player.uid)
        attacker.perm_atk_add = 5
        attacker.recalc_stats()

        player.board = [tank, attacker]
        env._auto_position_board(player)

        assert player.board[0].uid == attacker.uid

    def test_empty_board_no_crash(self, env: HearthstoneEnv) -> None:
        env.reset(seed=42)
        player = env.game.players[env.my_player_id]
        player.board = []
        env._auto_position_board(player)  # Should not crash


# ===================================================================
#  7. SIMPLE BOT
# ===================================================================


class TestSimpleBot:
    """Simple bot completes a turn without crashing."""

    def test_simple_bot_completes_turn(self, env: HearthstoneEnv) -> None:
        env.reset(seed=42)
        env._simple_bot_turn(env.enemy_id)
        # If we got here, bot didn't crash

    def test_simple_bot_ends_with_ready(self, env: HearthstoneEnv) -> None:
        env.reset(seed=42)
        env._simple_bot_turn(env.enemy_id)
        # Bot should have called END_TURN → player is ready
        assert env.game.players_ready[env.enemy_id]


# ===================================================================
#  8. BOARD POWER CALCULATION
# ===================================================================


class TestBoardPower:
    """_calculate_board_power heuristic."""

    def test_empty_board_zero_power(self, env: HearthstoneEnv) -> None:
        env.reset(seed=42)
        player = env.game.players[env.my_player_id]
        player.board = []
        assert env._calculate_board_power(player) == 0.0

    def test_stronger_board_higher_power(self, env: HearthstoneEnv) -> None:
        env.reset(seed=42)
        player = env.game.players[env.my_player_id]

        weak_unit = Unit.create_from_db(CardIDs.TABBYCAT, 9999, player.uid)  # 1/1
        player.board = [weak_unit]
        weak_power = env._calculate_board_power(player)

        strong_unit = Unit.create_from_db(CardIDs.MOLTEN_ROCK, 9998, player.uid)  # 4/7
        player.board = [strong_unit]
        strong_power = env._calculate_board_power(player)

        assert strong_power > weak_power

    def test_ds_increases_power(self, env: HearthstoneEnv) -> None:
        env.reset(seed=42)
        player = env.game.players[env.my_player_id]

        no_ds = Unit.create_from_db(CardIDs.TABBYCAT, 9999, player.uid)
        player.board = [no_ds]
        power_no_ds = env._calculate_board_power(player)

        with_ds = Unit.create_from_db(CardIDs.TABBYCAT, 9998, player.uid)
        with_ds.tags.add(Tags.DIVINE_SHIELD)
        player.board = [with_ds]
        power_ds = env._calculate_board_power(player)

        assert power_ds > power_no_ds


# ===================================================================
#  9. DECODE ACTION
# ===================================================================


class TestDecodeAction:
    """_decode_action_for_engine mapping."""

    @pytest.mark.parametrize(
        "action, expected_type",
        [
            (1, "ROLL"),
            (2, "BUY"),
            (9, "SELL"),
            (16, "PLAY"),
            (26, "SWAP"),
        ],
        ids=["roll", "buy", "sell", "play", "swap"],
    )
    def test_decode_action_types(
        self, env: HearthstoneEnv, action: int, expected_type: str
    ) -> None:
        action_type, kwargs = env._decode_action_for_engine(action)
        assert action_type == expected_type

    def test_decode_buy_index(self, env: HearthstoneEnv) -> None:
        _, kwargs = env._decode_action_for_engine(5)  # BUY slot 3
        assert kwargs["index"] == 3

    def test_decode_sell_index(self, env: HearthstoneEnv) -> None:
        _, kwargs = env._decode_action_for_engine(12)  # SELL slot 3
        assert kwargs["index"] == 3

    def test_decode_swap_indices(self, env: HearthstoneEnv) -> None:
        _, kwargs = env._decode_action_for_engine(28)  # SWAP 2<->3
        assert kwargs["index_a"] == 2
        assert kwargs["index_b"] == 3

    def test_decode_unknown_action(self, env: HearthstoneEnv) -> None:
        action_type, _ = env._decode_action_for_engine(99)
        assert action_type == "UNKNOWN"
