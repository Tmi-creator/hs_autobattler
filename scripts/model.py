"""Standalone Actor-Critic model for HS:BG.

Contains ALL architecture components (no SB3 dependency):
- Building blocks: DecomposedEncoder, FiLM, MultiHeadAttention, GTrXL, PMA
- Two-hot categorical critic utilities (symlog, encode_twohot, decode_value)
- HSTransformerAgent: full actor-critic with transformer encoder

Usage:
    from scripts.model import HSTransformerAgent
    agent = HSTransformerAgent(num_card_ids=200)
    action_logits, value_logits = agent(obs_batch)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


# ============================================================
# Symlog Two-Hot Utilities (from DreamerV3)
# ============================================================

NUM_BINS = 255
SYMLOG_MIN = -20.0
SYMLOG_MAX = 20.0
BIN_CENTERS = torch.linspace(SYMLOG_MIN, SYMLOG_MAX, NUM_BINS)


def symlog(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.log1p(torch.abs(x))


def symexp(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1.0)


def encode_twohot(target: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
    """[B] scalar → [B, K] two-hot probability vector in symlog space."""
    z = symlog(target).clamp(bins[0], bins[-1])
    idx = torch.searchsorted(bins, z) - 1
    idx = idx.clamp(0, len(bins) - 2)
    lo, hi = bins[idx], bins[idx + 1]
    w_hi = (z - lo) / (hi - lo + 1e-8)
    twohot = torch.zeros(target.shape[0], len(bins), device=target.device)
    twohot.scatter_(1, idx.unsqueeze(1), (1.0 - w_hi).unsqueeze(1))
    twohot.scatter_(1, (idx + 1).unsqueeze(1), w_hi.unsqueeze(1))
    return twohot


def decode_value(logits: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
    """[B, K] logits → [B] scalar value via softmax + expectation + symexp."""
    probs = F.softmax(logits, dim=-1)
    return symexp((probs * bins.to(logits.device)).sum(dim=-1))


# ============================================================
# Building Blocks (from trans.py, zero SB3 deps)
# ============================================================

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * (self.weight / (x.pow(2).mean(-1, keepdim=True) + self.eps) ** 0.5)


class FFN_SwiGLU(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        size = (int(4 * d_model * 2 / 3) + 255) // 256 * 256
        self.w1 = nn.Linear(d_model, size, bias=False)
        self.v = nn.Linear(d_model, size, bias=False)
        self.w2 = nn.Linear(size, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.v(x))


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, q_x, k_x, v_x, mask=None):
        B, Lq, D = q_x.shape
        _, Lk, _ = k_x.shape
        q = self.q(q_x).view(B, Lq, self.n_heads, self.d_k).transpose(1, 2)
        k = self.k(k_x).view(B, Lk, self.n_heads, self.d_k).transpose(1, 2)
        v = self.v(v_x).view(B, Lk, self.n_heads, self.d_k).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / (self.d_k ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        return self.o((torch.softmax(scores, -1) @ v).transpose(1, 2).contiguous().view(B, Lq, D))


class GatedResidual(nn.Module):
    """GTrXL GRU gate. Identity-init: b_z=-3 → gate ≈ 0.05 at start."""
    def __init__(self, d_model: int, bias_init: float = -3.0):
        super().__init__()
        self.w_z = nn.Linear(2 * d_model, d_model)
        self.w_r = nn.Linear(2 * d_model, d_model)
        self.w_h = nn.Linear(2 * d_model, d_model)
        nn.init.constant_(self.w_z.bias, bias_init)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        xy = torch.cat([x, y], dim=-1)
        z = torch.sigmoid(self.w_z(xy))
        r = torch.sigmoid(self.w_r(xy))
        h = torch.tanh(self.w_h(torch.cat([r * x, y], dim=-1)))
        return (1 - z) * x + z * h


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, use_gating: bool = True):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.ffn = FFN_SwiGLU(d_model)
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        self.gate_attn = GatedResidual(d_model) if use_gating else None
        self.gate_ffn = GatedResidual(d_model) if use_gating else None

    def forward(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        h = self.norm1(x)
        attn_out = self.attn(h, h, h, mask=mask)
        x = self.gate_attn(x, attn_out) if self.gate_attn else x + attn_out
        ffn_out = self.ffn(self.norm2(x))
        x = self.gate_ffn(x, ffn_out) if self.gate_ffn else x + ffn_out
        return x


class PMA(nn.Module):
    """Pooling by Multihead Attention. K learnable seeds → [B, d_model]."""
    def __init__(self, d_model: int, n_heads: int, k_seeds: int = 4):
        super().__init__()
        self.seeds = nn.Parameter(torch.randn(1, k_seeds, d_model) * 0.01)
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.norm = RMSNorm(d_model)
        self.proj = nn.Linear(k_seeds * d_model, d_model)

    def forward(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        B = x.shape[0]
        seeds = self.seeds.expand(B, -1, -1)
        x_n = self.norm(x)
        out = seeds + self.attn(self.norm(seeds), x_n, x_n, mask=mask)
        return self.proj(out.reshape(B, -1))


class DecomposedEncoder(nn.Module):
    """Heterogeneous entity encoder: card_id + continuous + binary + types → d_model."""
    _CARD_ID_IDX = 2
    _CONTINUOUS_IDX = [3, 4, 6, 7]           # cost, tier, ATK, HP
    _BINARY_IDX = [0, 1, 5] + list(range(8, 26))
    _TYPE_IDX = list(range(26, 37))

    def __init__(self, d_model: int, max_teams: int = 5, num_card_ids: int = 300):
        super().__init__()
        d_card = d_model // 2
        self.emb_card = nn.Embedding(num_card_ids, d_card, padding_idx=0)
        self.proj_continuous = nn.Linear(len(self._CONTINUOUS_IDX), d_model)
        self.proj_binary = nn.Linear(len(self._BINARY_IDX), d_model)
        self.proj_types = nn.Linear(len(self._TYPE_IDX), d_model)
        self.proj_card = nn.Linear(d_card, d_model)
        self.emb_team = nn.Embedding(max_teams, d_model)

    def forward(self, val: torch.Tensor, team_id: torch.Tensor) -> torch.Tensor:
        card_ids = val[..., self._CARD_ID_IDX].long().clamp(0, self.emb_card.num_embeddings - 1)
        x = (
            self.proj_card(self.emb_card(card_ids))
            + self.proj_continuous(val[..., self._CONTINUOUS_IDX])
            + self.proj_binary(val[..., self._BINARY_IDX])
            + self.proj_types(val[..., self._TYPE_IDX])
            + self.emb_team(team_id)
        )
        return x


class FiLMGenerator(nn.Module):
    def __init__(self, d_context: int, d_model: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_context, d_model), nn.SiLU(),
            nn.Linear(d_model, 2 * d_model),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, c):
        gamma, beta = self.mlp(c).chunk(2, dim=-1)
        return gamma, beta


class FiLM(nn.Module):
    def __init__(self, d_context: int, d_model: int):
        super().__init__()
        self.gen = FiLMGenerator(d_context, d_model)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.gen(c)
        return x * (gamma.unsqueeze(1) + 1.0) + beta.unsqueeze(1)


# ============================================================
# Full Agent
# ============================================================

class HSTransformerAgent(nn.Module):
    """Actor-Critic agent with transformer encoder + categorical critic.

    Input:  flat obs [B, 1036] from HearthstoneEnv
    Output: action_logits [B, 34], value_logits [B, 255]
    """

    _ZONES = [(7, 1), (10, 2), (7, 3), (3, 4)]  # (slots, team_id)
    _GLOBAL_SIZE = 7
    _ENEMY_SIZE = 3
    _EF = 38  # features per entity slot

    def __init__(
        self,
        n_actions: int = 34,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        d_context: int = 10,
        num_card_ids: int = 300,
        use_gating: bool = True,
        pma_seeds: int = 4,
        actor_hidden: int = 128,
        critic_hidden: int = 128,
    ):
        super().__init__()
        self.d_model = d_model

        # Encoder
        self.encoder = DecomposedEncoder(d_model, num_card_ids=num_card_ids)
        self.film = FiLM(d_context, d_model)
        self.global_ctx_proj = nn.Sequential(
            nn.Linear(d_context, d_model), nn.SiLU(), nn.Linear(d_model, d_model),
        )
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, use_gating) for _ in range(n_layers)
        ])
        self.ln_f = RMSNorm(d_model)
        self.pma = PMA(d_model, n_heads, k_seeds=pma_seeds)

        # Actor head
        self.actor = nn.Sequential(
            nn.Linear(d_model, actor_hidden), nn.ReLU(),
            nn.Linear(actor_hidden, n_actions),
        )
        # Critic head (categorical two-hot, 255 bins)
        self.critic = nn.Sequential(
            nn.Linear(d_model, critic_hidden), nn.ReLU(),
            nn.Linear(critic_hidden, NUM_BINS),
        )
        # Zero-init output layers (DreamerV3 trick)
        nn.init.zeros_(self.actor[-1].weight)
        nn.init.zeros_(self.actor[-1].bias)
        nn.init.zeros_(self.critic[-1].weight)
        nn.init.zeros_(self.critic[-1].bias)

        # Register bin centers as buffer (auto device placement)
        self.register_buffer("bins", BIN_CENTERS.clone())

    def _parse_obs(self, flat: torch.Tensor):
        """[B, 1036] → val [B, 27, 38], team_id [B, 27], context [B, 10]"""
        B, device = flat.shape[0], flat.device
        global_vec = flat[:, :self._GLOBAL_SIZE]
        pos = self._GLOBAL_SIZE

        chunks, ids = [], []
        for n_slots, zone_id in self._ZONES:
            size = n_slots * self._EF
            chunks.append(flat[:, pos:pos + size].view(B, n_slots, self._EF))
            ids.append(torch.full((B, n_slots), zone_id, device=device, dtype=torch.long))
            pos += size

        enemy_vec = flat[:, pos:pos + self._ENEMY_SIZE]
        val = torch.cat(chunks, dim=1)
        team_id = torch.cat(ids, dim=1)
        team_id = team_id * (val[:, :, 0] > 0.5).long()
        context = torch.cat([global_vec, enemy_vec], dim=-1)
        return val, team_id, context

    def _encode(self, flat: torch.Tensor) -> torch.Tensor:
        """Flat obs → pooled features [B, d_model]."""
        val, team_id, context = self._parse_obs(flat)

        # Symlog continuous features, preserve card_id
        card_ids = val[..., 2].clone()
        val = symlog(val)
        val[..., 2] = card_ids

        x = self.encoder(val, team_id)
        x = self.film(x, context)

        # Prepend [GLOBAL_CTX] token
        global_token = self.global_ctx_proj(context).unsqueeze(1)
        x = torch.cat([global_token, x], dim=1)

        # Padding mask
        entity_mask = (team_id != 0)
        global_mask = torch.ones(x.shape[0], 1, device=x.device, dtype=torch.bool)
        pad_mask = torch.cat([global_mask, entity_mask], dim=1).unsqueeze(1).unsqueeze(2)

        for block in self.blocks:
            x = block(x, mask=pad_mask)
        x = self.ln_f(x)

        return self.pma(x, mask=pad_mask)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (action_logits [B, 34], value_logits [B, 255]).

        Critic is detached from the encoder: value-loss gradients update the
        critic head only, not shared representations. Prevents value-target
        noise from corrupting the actor's view of the board.
        """
        features = self._encode(obs)
        return self.actor(features), self.critic(features.detach())

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        action_mask: torch.Tensor,
        action: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """PPO interface: returns (action, log_prob, entropy, value, value_logits).

        Args:
            obs: [B, 1036] flat observations
            action_mask: [B, 34] bool mask (True = legal)
            action: [B] actions to evaluate (None = sample new)
        """
        action_logits, value_logits = self.forward(obs)

        # Mask illegal actions
        action_logits = action_logits.masked_fill(~action_mask, -1e8)
        dist = Categorical(logits=action_logits)

        if action is None:
            action = dist.sample()

        value = decode_value(value_logits, self.bins)

        return action, dist.log_prob(action), dist.entropy(), value, value_logits

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """Returns scalar value [B] for GAE bootstrap."""
        features = self._encode(obs)
        return decode_value(self.critic(features), self.bins)
