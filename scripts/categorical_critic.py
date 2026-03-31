"""
Symlog Two-Hot Categorical Critic for MaskablePPO.

Replaces the scalar MSE critic with a distributional critic that outputs
a probability distribution over 255 bins in symlog space. Trained with
cross-entropy loss (bounded gradients) instead of MSE (exploding gradients).

Based on DreamerV3 (Hafner et al., 2023): https://arxiv.org/abs/2301.04104
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.ppo_mask.ppo_mask import MaskablePPO
from stable_baselines3.common.type_aliases import Schedule


# ============================================================
# Symlog Two-Hot Utilities
# ============================================================

NUM_BINS = 255
SYMLOG_MIN = -20.0
SYMLOG_MAX = 20.0

# Pre-compute bin centers (uniform in symlog space)
BIN_CENTERS = torch.linspace(SYMLOG_MIN, SYMLOG_MAX, NUM_BINS)


def symlog(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.log1p(torch.abs(x))


def symexp(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1.0)


def encode_twohot(target: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
    """Encode scalar targets as two-hot vectors over bin centers.

    Args:
        target: [B] raw scalar targets (returns)
        bins: [K] bin center values in symlog space

    Returns:
        [B, K] two-hot encoded probability vectors (sum to 1)
    """
    # Transform target to symlog space
    z = symlog(target)
    # Clamp to bin range
    z = z.clamp(bins[0], bins[-1])

    # Find the bin index below z
    # bins is sorted, so searchsorted gives the insertion point
    idx = torch.searchsorted(bins, z) - 1
    idx = idx.clamp(0, len(bins) - 2)

    # Interpolation weights
    lo = bins[idx]
    hi = bins[idx + 1]
    w_hi = (z - lo) / (hi - lo + 1e-8)
    w_lo = 1.0 - w_hi

    # Build two-hot vector
    twohot = torch.zeros(target.shape[0], len(bins), device=target.device)
    twohot.scatter_(1, idx.unsqueeze(1), w_lo.unsqueeze(1))
    twohot.scatter_(1, (idx + 1).unsqueeze(1), w_hi.unsqueeze(1))

    return twohot


def decode_value(logits: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
    """Decode bin logits to scalar value.

    Args:
        logits: [B, K] raw logits from value network
        bins: [K] bin center values in symlog space

    Returns:
        [B] scalar values in original space
    """
    probs = F.softmax(logits, dim=-1)
    symlog_value = (probs * bins.to(logits.device)).sum(dim=-1)
    return symexp(symlog_value)


# ============================================================
# Categorical Value Policy
# ============================================================


class CategoricalValuePolicy(MaskableActorCriticPolicy):
    """MaskableActorCriticPolicy with Two-Hot Categorical value head."""

    def __init__(self, *args, **kwargs):
        # Store bins before _build is called
        self._bin_centers: torch.Tensor  # will be registered as buffer
        super().__init__(*args, **kwargs)

    def _build(self, lr_schedule: Schedule) -> None:
        super()._build(lr_schedule)
        # Replace scalar value_net with categorical head
        latent_dim = self.mlp_extractor.latent_dim_vf
        self.value_net = nn.Linear(latent_dim, NUM_BINS)
        # Zero-init output weights for stable start (DreamerV3 trick)
        nn.init.zeros_(self.value_net.weight)
        nn.init.zeros_(self.value_net.bias)
        # Register bin centers as buffer (moves to correct device automatically)
        self.register_buffer("_bin_centers", BIN_CENTERS.clone())
        # Rebuild optimizer to include new value_net params
        self.optimizer = self.optimizer_class(
            self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs
        )

    def forward(self, obs, deterministic=False, action_masks=None):
        """Override forward to decode categorical values to scalar."""
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)

        # Decode categorical value to scalar
        value_logits = self.value_net(latent_vf)
        values = decode_value(value_logits, self._bin_centers).unsqueeze(-1)

        distribution = self._get_action_dist_from_latent(latent_pi)
        if action_masks is not None:
            distribution.apply_masking(action_masks)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        return actions, values, log_prob

    def predict_values(self, obs, **kwargs) -> torch.Tensor:
        """Return decoded scalar values [B, 1] for GAE computation."""
        features = self.extract_features(obs, self.vf_features_extractor)
        latent_vf = self.mlp_extractor.forward_critic(features)
        logits = self.value_net(latent_vf)
        return decode_value(logits, self._bin_centers).unsqueeze(-1)

    def evaluate_actions(self, obs, actions, action_masks=None, **kwargs):
        """Return scalar values, log_prob, entropy. Store logits for train()."""
        features = self.extract_features(obs, self.features_extractor)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features = self.extract_features(obs, self.pi_features_extractor)
            vf_features = self.extract_features(obs, self.vf_features_extractor)
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)

        # Value: compute logits and decode to scalar
        value_logits = self.value_net(latent_vf)
        values = decode_value(value_logits, self._bin_centers)

        # Store logits for cross-entropy loss in train()
        self._last_value_logits = value_logits

        # Action distribution
        distribution = self._get_action_dist_from_latent(latent_pi)
        if action_masks is not None:
            distribution.apply_masking(action_masks)
        log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()

        return values, log_prob, entropy


# ============================================================
# Categorical MaskablePPO
# ============================================================


class CategoricalMaskablePPO(MaskablePPO):
    """MaskablePPO with Two-Hot cross-entropy value loss instead of MSE."""

    def train(self) -> None:
        """Override train() to use categorical cross-entropy for value loss."""
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)
        clip_range_vf: Optional[float] = None
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        pg_losses, value_losses = [], []
        entropy_losses, clip_fractions = [], []

        # Get bin centers from policy
        bins = self.policy._bin_centers

        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    actions = rollout_data.actions.long().flatten()

                values, log_prob, entropy = self.policy.evaluate_actions(
                    rollout_data.observations,
                    actions,
                    action_masks=rollout_data.action_masks,
                )
                values = values.flatten()

                # Normalize advantage
                advantages = rollout_data.advantages
                if self.normalize_advantage:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # PPO clipped surrogate loss (unchanged)
                ratio = torch.exp(log_prob - rollout_data.old_log_prob)
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

                pg_losses.append(policy_loss.item())
                clip_fraction = torch.mean((torch.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                # === CATEGORICAL VALUE LOSS (replaces MSE) ===
                value_logits = self.policy._last_value_logits  # stored by evaluate_actions
                target_twohot = encode_twohot(rollout_data.returns, bins)
                value_loss = -(target_twohot * F.log_softmax(value_logits, dim=-1)).sum(dim=-1).mean()
                value_losses.append(value_loss.item())

                # Entropy loss (unchanged)
                if entropy is None:
                    entropy_loss = -torch.mean(-log_prob)
                else:
                    entropy_loss = -torch.mean(entropy)
                entropy_losses.append(entropy_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                # KL divergence early stopping
                with torch.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = torch.mean((torch.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            self._n_updates += self.n_epochs
            if self.target_kl is not None and np.mean(approx_kl_divs) > 1.5 * self.target_kl:
                break

        explained_var = self._compute_explained_variance()

        # Logging
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", np.mean(pg_losses) + np.mean(value_losses))
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", torch.exp(self.policy.log_std).mean().item())
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)

    def _compute_explained_variance(self) -> float:
        """Compute explained variance from rollout buffer."""
        values = self.rollout_buffer.values.flatten()
        returns = self.rollout_buffer.returns.flatten()
        var_returns = np.var(returns)
        if var_returns == 0:
            return float("nan")
        return float(1 - np.var(returns - values) / var_returns)
