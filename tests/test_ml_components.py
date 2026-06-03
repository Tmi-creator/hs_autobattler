"""Tests for ML components: Categorical Critic, MC Oracle, Card Embeddings, Rewards."""

import numpy as np
import pytest
import torch

from hearthstone.env.hs_env import HearthstoneEnv

try:
    import stable_baselines3
    has_sb3 = True
except ImportError:
    has_sb3 = False

try:
    import sb3_contrib
    has_sb3_contrib = True
except ImportError:
    has_sb3_contrib = False

try:
    import wandb
    has_wandb = True
except ImportError:
    has_wandb = False


# ===================================================================
#  1. CATEGORICAL CRITIC (symlog, two-hot, encode/decode)
# ===================================================================


@pytest.mark.skipif(not has_sb3_contrib, reason="sb3_contrib is not installed")
class TestSymlog:
    """Test symlog/symexp transformations."""

    def test_symlog_zero(self):
        from legacy.categorical_critic import symlog
        assert symlog(torch.tensor(0.0)).item() == pytest.approx(0.0)

    def test_symlog_positive(self):
        from legacy.categorical_critic import symlog
        # symlog(x) = sign(x) * log(1 + |x|)
        x = torch.tensor(100.0)
        expected = torch.log1p(torch.tensor(100.0)).item()
        assert symlog(x).item() == pytest.approx(expected)

    def test_symlog_negative(self):
        from legacy.categorical_critic import symlog
        x = torch.tensor(-50.0)
        expected = -torch.log1p(torch.tensor(50.0)).item()
        assert symlog(x).item() == pytest.approx(expected)

    def test_symexp_inverse(self):
        from legacy.categorical_critic import symlog, symexp
        values = torch.tensor([-100.0, -1.0, 0.0, 1.0, 50.0, 1000.0])
        roundtrip = symexp(symlog(values))
        assert torch.allclose(roundtrip, values, atol=1e-4)

    def test_symlog_compresses_large_values(self):
        from legacy.categorical_critic import symlog
        # 50000 → ~10.8, keeps things manageable
        result = symlog(torch.tensor(50000.0))
        assert result.item() < 12.0
        assert result.item() > 10.0


@pytest.mark.skipif(not has_sb3_contrib, reason="sb3_contrib is not installed")
class TestTwoHotEncoding:
    """Test two-hot encoding and decoding for categorical critic."""

    def test_encode_exact_bin(self):
        from legacy.categorical_critic import encode_twohot, BIN_CENTERS
        # Target exactly at a bin center should be ~one-hot
        target = torch.tensor([0.0])  # symlog(0) = 0, should hit center bin
        twohot = encode_twohot(target, BIN_CENTERS)
        assert twohot.shape == (1, 255)
        assert twohot.sum().item() == pytest.approx(1.0, abs=1e-5)

    def test_encode_between_bins(self):
        from legacy.categorical_critic import encode_twohot, BIN_CENTERS
        # Target between bins → two non-zero entries summing to 1
        target = torch.tensor([5.7])
        twohot = encode_twohot(target, BIN_CENTERS)
        nonzero = (twohot > 0).sum().item()
        assert nonzero == 2
        assert twohot.sum().item() == pytest.approx(1.0, abs=1e-5)

    def test_encode_extreme_positive(self):
        from legacy.categorical_critic import encode_twohot, BIN_CENTERS
        # Very large value → clamped to last bin
        target = torch.tensor([999999.0])
        twohot = encode_twohot(target, BIN_CENTERS)
        assert twohot.sum().item() == pytest.approx(1.0, abs=1e-5)

    def test_encode_extreme_negative(self):
        from legacy.categorical_critic import encode_twohot, BIN_CENTERS
        target = torch.tensor([-999999.0])
        twohot = encode_twohot(target, BIN_CENTERS)
        assert twohot.sum().item() == pytest.approx(1.0, abs=1e-5)

    def test_decode_roundtrip(self):
        from legacy.categorical_critic import encode_twohot, decode_value, BIN_CENTERS
        # Encode → decode should approximately recover the original
        original = torch.tensor([3.0, -2.0, 0.5, 10.0])
        twohot = encode_twohot(original, BIN_CENTERS)
        # Use twohot as "perfect logits" (log of twohot)
        logits = torch.log(twohot + 1e-10)
        decoded = decode_value(logits, BIN_CENTERS)
        assert torch.allclose(decoded, original, atol=0.5)

    def test_batch_encoding(self):
        from legacy.categorical_critic import encode_twohot, BIN_CENTERS
        targets = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        twohot = encode_twohot(targets, BIN_CENTERS)
        assert twohot.shape == (5, 255)
        # Each row sums to 1
        row_sums = twohot.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones(5), atol=1e-5)


# ===================================================================
#  2. CARD EMBEDDINGS
# ===================================================================


class TestCardEmbeddings:
    """Test that card_id is passed as raw integer for nn.Embedding lookup."""

    @pytest.fixture
    def env(self):
        e = HearthstoneEnv()
        e.reset(seed=42)
        return e

    def test_card_id_is_raw_integer(self, env):
        """card_id (obs index 2 in each entity) should be a raw integer, not normalized."""
        obs, _ = env.reset(seed=42)
        # First board entity starts at offset 7 (after global), card_id at +2
        ef = env.entity_features
        board_card_id = obs[7 + 2]
        # Should be 0 (empty) or an integer > 1 (not 0.xx normalized)
        assert board_card_id == 0.0 or board_card_id >= 1.0

    def test_card_id_in_store(self, env):
        """Store entities should have non-zero card_ids."""
        obs, _ = env.reset(seed=42)
        ef = env.entity_features
        store_offset = 7 + (7 + 10) * ef  # after global + board + hand
        # Check first store slot
        store_card_id = obs[store_offset + 2]
        is_present = obs[store_offset + 0]
        if is_present > 0.5:
            assert store_card_id >= 1.0, "Present store item should have card_id >= 1"

    def test_num_card_ids_matches_db(self, env):
        """num_card_ids should cover all cards + spells + padding."""
        from hearthstone.engine.configs import CARD_DB, SPELL_DB
        total_unique = len(set(CARD_DB.keys()) | set(SPELL_DB.keys()))
        assert env.num_card_ids == total_unique + 1  # +1 for padding id 0

    @pytest.mark.skipif(not has_sb3, reason="stable_baselines3 is not installed")
    def test_transformer_embedding_shape(self):
        """TransformerFeaturesExtractor should have card embedding table."""
        from legacy.trans import TransformerFeaturesExtractor
        env = HearthstoneEnv()
        ext = TransformerFeaturesExtractor(
            env.observation_space,
            d_model=64, n_heads=2, n_layers=2, d_context=10,
            num_card_ids=env.num_card_ids,
        )
        assert ext.encoder.emb_card.num_embeddings == env.num_card_ids
        assert ext.encoder.emb_card.embedding_dim == 32  # d_model // 2

    @pytest.mark.skipif(not has_sb3, reason="stable_baselines3 is not installed")
    def test_transformer_forward_with_embeddings(self):
        """Forward pass should work with card embeddings."""
        from legacy.trans import TransformerFeaturesExtractor
        env = HearthstoneEnv()
        obs, _ = env.reset(seed=42)
        ext = TransformerFeaturesExtractor(
            env.observation_space,
            d_model=64, n_heads=2, n_layers=2, d_context=10,
            num_card_ids=env.num_card_ids,
        )
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            out = ext(obs_t)
        assert out.shape == (1, 64)
        assert torch.isfinite(out).all()


# ===================================================================
#  3. REWARD FUNCTION
# ===================================================================


class TestRewardFunction:
    """Test the reward function: round outcome, action penalty, terminal, MC Oracle."""

    @pytest.fixture
    def env(self):
        e = HearthstoneEnv()
        e.reset(seed=42)
        return e

    def test_roll_gives_action_penalty(self, env):
        """ROLL should give -0.005 action penalty."""
        masks = env.action_masks()
        if masks[1]:  # ROLL available
            _, reward, _, _, _ = env.step(1)
            assert reward == pytest.approx(-0.005)

    def test_end_turn_gives_round_outcome(self, env):
        """END_TURN should give ±1 for round outcome (or 0 for draw)."""
        _, reward, done, _, _ = env.step(0)  # END_TURN
        if not done:
            assert reward in [pytest.approx(1.0), pytest.approx(-1.0), pytest.approx(0.0)]

    def test_no_positive_per_action_rewards(self, env):
        """BUY should NOT give positive triple/pair rewards (old exploit)."""
        masks = env.action_masks()
        for action in range(2, 9):  # BUY actions
            if masks[action]:
                obs, reward, _, _, _ = env.step(action)
                # Reward should be oracle delta (can be pos/neg) + penalty
                # But NOT +2.5 or +0.5 fixed bonus
                assert reward < 2.0, f"Suspiciously high reward {reward} for BUY"
                break

    def test_terminal_reward_magnitude(self, env):
        """Terminal reward should be ±100."""
        # Play until game over
        for _ in range(2000):
            masks = env.action_masks()
            valid = np.where(masks)[0]
            action = valid[0]
            _, reward, done, trunc, _ = env.step(int(action))
            if done or trunc:
                if done:
                    assert abs(reward) >= 99.0, f"Terminal reward {reward} too small"
                break

    def test_upgrade_gives_only_penalty(self, env):
        """UPGRADE (action 32) should give -0.005 action penalty, no oracle."""
        # Give enough gold for upgrade
        player = env.game.players[env.my_player_id]
        player.gold = 10
        masks = env.action_masks()
        if masks[32]:  # UPGRADE
            _, reward, _, _, _ = env.step(32)
            assert reward == pytest.approx(-0.005)


# ===================================================================
#  4. MC ORACLE
# ===================================================================


class TestMCOracle:
    """Test MC Oracle dense reward mechanism."""

    @pytest.fixture
    def env(self):
        e = HearthstoneEnv()
        e.reset(seed=42)
        return e

    def test_oracle_cached_winrate_initialized(self, env):
        """Oracle cached winrate should start at 0.5."""
        assert env._oracle_cached_wr == 0.5

    def test_oracle_seed_changes(self, env):
        """Oracle seed should change after evaluation."""
        initial_seed = env._oracle_seed
        env._oracle_prepare_ghost()
        if env._oracle_ghost_flat is not None:
            player = env.game.players[env.my_player_id]
            env._oracle_eval_winrate(player)
            assert env._oracle_seed != initial_seed

    def test_oracle_without_cpp_returns_half(self, env):
        """Without C++ engine, oracle should return 0.5 (no delta)."""
        from hearthstone.engine.cpp_bridge import get_cpp_engine
        cpp = get_cpp_engine()
        if cpp is None:
            # No C++ available — oracle should be inert
            player = env.game.players[env.my_player_id]
            wr = env._oracle_eval_winrate(player)
            assert wr == 0.5

    def test_oracle_reward_is_delta_based(self, env):
        """Oracle reward should be (wr_after - wr_before) * scale."""
        env._oracle_cached_wr = 0.4
        # Mock: next eval returns 0.6
        original_eval = env._oracle_eval_winrate
        env._oracle_eval_winrate = lambda p: 0.6
        player = env.game.players[env.my_player_id]
        reward = env._oracle_reward(player)
        assert reward == pytest.approx((0.6 - 0.4) * 10.0)
        env._oracle_eval_winrate = original_eval

    def test_oracle_reset_on_new_episode(self, env):
        """Oracle state should reset on env.reset()."""
        env._oracle_cached_wr = 0.8
        env._oracle_seed = 999
        env.reset(seed=123)
        assert env._oracle_cached_wr == 0.5
        assert env._oracle_seed != 999


# ===================================================================
#  5. ENTROPY DECAY CALLBACK
# ===================================================================


@pytest.mark.skipif(not has_wandb, reason="wandb is not installed")
class TestEntropyDecayCallback:
    """Test entropy coefficient decay."""

    def test_decay_start(self):
        from legacy.callbacks import EntropyDecayCallback
        cb = EntropyDecayCallback(ent_coef_start=0.04, ent_coef_end=0.01, decay_fraction=0.75)

        # Simulate: at progress=0, ent_coef should be 0.04
        class MockModel:
            _current_progress_remaining = 1.0  # progress = 0
            ent_coef = 0.04
        cb.model = MockModel()
        cb._on_step()
        assert cb.model.ent_coef == pytest.approx(0.04)

    def test_decay_midway(self):
        from legacy.callbacks import EntropyDecayCallback
        cb = EntropyDecayCallback(ent_coef_start=0.04, ent_coef_end=0.01, decay_fraction=0.75)

        class MockModel:
            _current_progress_remaining = 0.5  # progress = 0.5
            ent_coef = 0.04
        cb.model = MockModel()
        cb._on_step()
        # At progress=0.5 with decay_fraction=0.75: t = 0.5/0.75 = 0.667
        expected = 0.04 + 0.667 * (0.01 - 0.04)  # = 0.04 - 0.02 = 0.02
        assert cb.model.ent_coef == pytest.approx(expected, abs=0.002)

    def test_decay_end(self):
        from legacy.callbacks import EntropyDecayCallback
        cb = EntropyDecayCallback(ent_coef_start=0.04, ent_coef_end=0.01, decay_fraction=0.75)

        class MockModel:
            _current_progress_remaining = 0.0  # progress = 1.0
            ent_coef = 0.04
        cb.model = MockModel()
        cb._on_step()
        assert cb.model.ent_coef == pytest.approx(0.01)

    def test_decay_after_fraction(self):
        from legacy.callbacks import EntropyDecayCallback
        cb = EntropyDecayCallback(ent_coef_start=0.04, ent_coef_end=0.01, decay_fraction=0.75)

        class MockModel:
            _current_progress_remaining = 0.2  # progress = 0.8 > 0.75
            ent_coef = 0.04
        cb.model = MockModel()
        cb._on_step()
        # After decay_fraction → stays at ent_coef_end
        assert cb.model.ent_coef == pytest.approx(0.01)
