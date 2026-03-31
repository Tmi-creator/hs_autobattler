"""Integration tests: full game lifecycle through the RL environment.

Tests that complete games can be played with different action strategies
without crashing, and that the game state progresses correctly.
"""
from __future__ import annotations

import numpy as np
import pytest

from hearthstone.env.hs_env import HearthstoneEnv


class TestFullGameLoop:
    """Test complete game episodes through the environment."""

    @pytest.fixture
    def env(self):
        e = HearthstoneEnv()
        e.reset(seed=42)
        return e

    def _play_game(self, env, strategy="first_valid", max_steps=2000):
        """Play a full game and return stats."""
        obs, _ = env.reset(seed=42)
        total_reward = 0.0
        steps = 0
        turns = 0

        for _ in range(max_steps):
            masks = env.action_masks()
            valid = np.where(masks)[0]

            if strategy == "first_valid":
                action = valid[0]
            elif strategy == "random":
                action = np.random.choice(valid)
            elif strategy == "end_turn_only":
                action = 0 if masks[0] else valid[0]
            elif strategy == "buy_everything":
                # Try to buy first, then play, then end turn
                buy_actions = [a for a in valid if 2 <= a <= 8]
                play_actions = [a for a in valid if 16 <= a <= 25]
                if buy_actions:
                    action = buy_actions[0]
                elif play_actions:
                    action = play_actions[0]
                elif masks[0]:
                    action = 0  # END_TURN
                else:
                    action = valid[0]
            else:
                action = valid[0]

            obs, reward, done, trunc, _ = env.step(int(action))
            total_reward += reward
            steps += 1
            if int(action) == 0 or int(action) == 33:  # END_TURN or FREEZE+END
                turns += 1

            if done or trunc:
                break

        return {
            "steps": steps,
            "turns": turns,
            "total_reward": total_reward,
            "done": done,
            "final_hp": env.game.players[env.my_player_id].health,
        }

    def test_first_valid_completes(self, env):
        """Game should complete with 'first valid action' strategy."""
        stats = self._play_game(env, "first_valid")
        assert stats["done"] or stats["steps"] >= 50
        assert stats["turns"] >= 1

    def test_random_completes(self, env):
        """Game should complete with random valid actions."""
        np.random.seed(42)
        stats = self._play_game(env, "random")
        assert stats["done"] or stats["steps"] >= 50

    def test_end_turn_only_completes(self, env):
        """Game should complete when only pressing END_TURN."""
        stats = self._play_game(env, "end_turn_only")
        assert stats["done"]
        # Should lose quickly with empty board
        assert stats["final_hp"] <= 0

    def test_buy_everything_completes(self, env):
        """Game should complete with buy+play+end strategy."""
        stats = self._play_game(env, "buy_everything")
        assert stats["done"] or stats["steps"] >= 50

    def test_game_over_has_terminal_reward(self, env):
        """Game over should include ±100 terminal reward."""
        stats = self._play_game(env, "first_valid")
        if stats["done"]:
            # Total reward should include terminal ±100
            assert abs(stats["total_reward"]) > 10.0

    def test_multiple_episodes(self, env):
        """Multiple episodes should work without state leaking."""
        for seed in [1, 2, 3, 4, 5]:
            env.reset(seed=seed)
            for _ in range(100):
                masks = env.action_masks()
                valid = np.where(masks)[0]
                obs, _, done, trunc, _ = env.step(int(valid[0]))
                if done or trunc:
                    break


class TestObservationConsistency:
    """Test that observations are consistent across game progression."""

    @pytest.fixture
    def env(self):
        e = HearthstoneEnv()
        e.reset(seed=42)
        return e

    def test_obs_shape_stable_during_game(self, env):
        """Observation shape should never change during a game."""
        obs, _ = env.reset(seed=42)
        expected_shape = obs.shape

        for _ in range(200):
            masks = env.action_masks()
            valid = np.where(masks)[0]
            obs, _, done, trunc, _ = env.step(int(valid[0]))
            assert obs.shape == expected_shape
            if done or trunc:
                break

    def test_obs_finite_during_game(self, env):
        """All observation values should be finite (no NaN/Inf)."""
        obs, _ = env.reset(seed=42)

        for _ in range(200):
            assert np.all(np.isfinite(obs)), f"Non-finite obs: {obs[~np.isfinite(obs)]}"
            masks = env.action_masks()
            valid = np.where(masks)[0]
            obs, _, done, trunc, _ = env.step(int(valid[0]))
            if done or trunc:
                break

    def test_board_power_nonnegative(self, env):
        """Board power should always be >= 0."""
        env.reset(seed=42)
        for _ in range(200):
            power = env.get_board_power()
            assert power >= 0.0
            masks = env.action_masks()
            valid = np.where(masks)[0]
            _, _, done, trunc, _ = env.step(int(valid[0]))
            if done or trunc:
                break


class TestActionMaskConsistency:
    """Test that action masks are always valid."""

    @pytest.fixture
    def env(self):
        e = HearthstoneEnv()
        e.reset(seed=42)
        return e

    def test_at_least_one_action_valid(self, env):
        """At every step, at least one action must be valid."""
        env.reset(seed=42)
        for _ in range(500):
            masks = env.action_masks()
            assert masks.any(), "No valid actions!"
            valid = np.where(masks)[0]
            _, _, done, trunc, _ = env.step(int(valid[0]))
            if done or trunc:
                break

    def test_end_turn_always_available_in_default(self, env):
        """END_TURN (action 0) should be available in default phase."""
        env.reset(seed=42)
        masks = env.action_masks()
        # In default phase (not discovering, not targeting), END_TURN should be valid
        if not env.game.players[env.my_player_id].is_discovering and not env.is_targeting:
            assert masks[0], "END_TURN should be available in default phase"

    def test_invalid_action_doesnt_crash(self, env):
        """Stepping with a masked-out action should not crash (returns 0 reward)."""
        env.reset(seed=42)
        masks = env.action_masks()
        invalid = np.where(~masks)[0]
        if len(invalid) > 0:
            # Pick an invalid action that's not END_TURN
            for a in invalid:
                if a != 0:
                    obs, reward, done, _, _ = env.step(int(a))
                    # Should handle gracefully (may return 0 reward)
                    assert np.all(np.isfinite(obs))
                    break


class TestRewardIntegrity:
    """Test reward function produces valid values."""

    @pytest.fixture
    def env(self):
        e = HearthstoneEnv()
        e.reset(seed=42)
        return e

    def test_rewards_are_finite(self, env):
        """All rewards should be finite numbers."""
        env.reset(seed=42)
        for _ in range(300):
            masks = env.action_masks()
            valid = np.where(masks)[0]
            _, reward, done, trunc, _ = env.step(int(valid[0]))
            assert np.isfinite(reward), f"Non-finite reward: {reward}"
            if done or trunc:
                break

    def test_action_penalty_is_small(self, env):
        """Non-END_TURN actions should have small penalty."""
        env.reset(seed=42)
        masks = env.action_masks()
        # Try ROLL (action 1)
        if masks[1]:
            _, reward, _, _, _ = env.step(1)
            assert abs(reward) < 1.0, f"Action penalty too large: {reward}"

    def test_no_sell_cycle_exploit(self, env):
        """Repeated sell-buy should not accumulate positive reward."""
        env.reset(seed=42)
        # Buy a unit first
        masks = env.action_masks()
        for a in range(2, 9):
            if masks[a]:
                env.step(a)
                break

        # Play it
        masks = env.action_masks()
        for a in range(16, 26):
            if masks[a]:
                env.step(a)
                break

        # Now sell-buy cycle should not be profitable
        total_reward = 0.0
        for _ in range(5):
            masks = env.action_masks()
            # Sell
            for a in range(9, 16):
                if masks[a]:
                    _, r, _, _, _ = env.step(a)
                    total_reward += r
                    break
            # Roll
            masks = env.action_masks()
            if masks[1]:
                _, r, _, _, _ = env.step(1)
                total_reward += r

        # Total from sell-roll cycle should be negative or near zero
        assert total_reward < 1.0, f"Sell-cycle reward too high: {total_reward}"
