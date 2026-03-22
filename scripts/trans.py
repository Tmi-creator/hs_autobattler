from __future__ import annotations

import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class FFN_SwiGLU(nn.Module):
    """
    Feed-Forward сеть с функцией активации Swish-Gated Linear Unit (SiLU).
    Использует коэффициент 2/3 для управления скрытой размерностью (эвристика из LLaMA)
    и округляет до кратного 256 для аппаратной оптимизации на GPU.
    """

    def __init__(self, d_model):
        super().__init__()
        coef = 2 / 3
        new_size = int(4 * d_model * coef)
        new_size = (new_size + 255) // 256 * 256
        self.w1 = nn.Linear(d_model, new_size, bias=False)
        self.v = nn.Linear(d_model, new_size, bias=False)
        self.w2 = nn.Linear(new_size, d_model, bias=False)

    def forward(self, x):
        w = self.w1(x)
        w = F.silu(w)
        v = self.v(x)
        return self.w2(w * v)


class RMSNorm(nn.Module):
    """
    Root Mean Square Normalization.
    Более быстрая альтернатива классическому LayerNorm: масштабирует только по дисперсии,
    не центрируя по среднему (экономит вычисления, работает так же хорошо).
    """

    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        x = x * (self.weight / (x.pow(2).mean(-1, keepdim=True) + self.eps) ** 0.5)
        return x


class MultiHeadAttention(nn.Module):
    """
    Классический механизм Multi-Head Attention.
    В текущей архитектуре он принимает на вход плотные представления карт
    без позиционных эмбеддингов.
    За счет этого он работает как чистый Set Transformer (строго инвариантен
    к перестановкам токенов).
    """

    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_k = d_model // n_heads
        self.d_model = d_model
        self.n_heads = n_heads
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, q_x, k_x, v_x, mask=None):
        B, Lq, D = q_x.shape
        _, Lk, _ = k_x.shape
        q, k, v = self.q(q_x), self.k(k_x), self.v(v_x)
        q = q.view(B, Lq, self.n_heads, self.d_k).transpose(1, 2)
        k = k.view(B, Lk, self.n_heads, self.d_k).transpose(1, 2)
        v = v.view(B, Lk, self.n_heads, self.d_k).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / (self.d_k**0.5)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        attn = torch.softmax(scores, dim=-1) @ v
        attn = attn.transpose(1, 2).contiguous().view(B, Lq, D)
        return self.o(attn)


class GatedResidual(nn.Module):
    """
    GRU-подобный шлюз для Residual-связей (GTrXL).
    Заменяет тупое x + f(x) на:
      z = sigmoid(W_z·[x,y] + b_z)   — gate
      r = sigmoid(W_r·[x,y] + b_r)   — reset
      h = tanh(W_h·[r⊙x, y])         — candidate
      out = (1-z)⊙x + z⊙h

    Identity Init: b_z инициализируется как -3 → z ≈ 0.05 на старте.
    Трансформер фактически выключен, агент учит базу через линейные слои.
    Градиенты постепенно открывают шлюз по мере обучения.
    """

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


class PMA(nn.Module):
    """
    Pooling by Multihead Attention (Set Transformer).
    K обучаемых Seed-векторов выступают как Query, токены стола — как Key/Value.
    Каждый seed извлекает отдельный аспект оценки доски (сила, экономика, гибкость, синергии).
    Мультиаспектная агрегация динамически выделяет информацию, критичную для оценки состояния,
    вместо тупого mean() который уничтожает найденные синергии.
    """

    def __init__(self, d_model: int, n_heads: int, k_seeds: int = 4):
        super().__init__()
        self.k_seeds = k_seeds
        self.seeds = nn.Parameter(torch.randn(1, k_seeds, d_model) * 0.01)
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.norm = RMSNorm(d_model)
        # rFF: row-wise Feed-Forward после агрегации всех seed-векторов → единый вектор
        self.proj = nn.Linear(k_seeds * d_model, d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """x: [B, N, D], mask: [B, 1, 1, N] → [B, D]"""
        B = x.shape[0]
        seeds = self.seeds.expand(B, -1, -1)  # [B, k, D]
        x_norm = self.norm(x)
        out = self.attn(self.norm(seeds), x_norm, x_norm, mask=mask)
        out = seeds + out  # [B, k, D]
        # Flatten все seed-ы и проецировать в единый вектор
        return self.proj(out.reshape(B, -1))  # [B, D]


class TransformerBlock(nn.Module):
    """
    Pre-LN блок Трансформера с опциональным GRU-шлюзом (GTrXL).
    Без шлюза: классический x + f(x). С шлюзом: GatedResidual.
    Поддерживает padding mask для исключения пустых слотов из attention.
    """

    def __init__(
        self,
        attention_cls,
        attention_args,
        ffn_cls,
        ffn_args,
        norm_cls,
        norm_args,
        gate_cls=None,
        gate_args=None,
    ):
        super().__init__()
        self.attention = attention_cls(**attention_args)
        self.ffn = ffn_cls(**ffn_args)
        self.norm1 = norm_cls(**norm_args)
        self.norm2 = norm_cls(**norm_args)
        self.gate_attn = gate_cls(**gate_args) if gate_cls else None
        self.gate_ffn = gate_cls(**gate_args) if gate_cls else None

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        h = self.norm1(x)
        attn_out = self.attention(h, h, h, mask=mask)
        x = self.gate_attn(x, attn_out) if self.gate_attn else x + attn_out
        ffn_out = self.ffn(self.norm2(x))
        x = self.gate_ffn(x, ffn_out) if self.gate_ffn else x + ffn_out
        return x


class DecomposedEncoder(nn.Module):
    """
    Гетерогенный энкодер сущностей с раздельной обработкой разнотипных признаков.

    Вместо наивного Linear(37, d_model) для всех признаков сразу, энкодер разделяет
    входной вектор на три семантические группы и обрабатывает их независимо:

      1. Continuous (5):  card_id_norm, cost, tier, ATK, HP
         → symlog-совместимые скаляры → Linear → d_model
      2. Binary (20):     is_present, is_spell, is_frozen, 17 keyword/effect flags
         → бинарные индикаторы → Linear → d_model
      3. Types (11):      one-hot вектор расовых типов (Beast, Mech, Demon...)
         → категориальный вход → Linear → d_model

    Результаты суммируются (аддитивное смешивание, аналогично позиционным эмбеддингам
    в классическом Transformer) и дополняются Zone Embedding (emb_team).

    Раскладка вектора сущности из hs_env.py (37 float):
      [0] is_present  [1] is_spell  [2] card_id_norm  [3] cost  [4] tier
      [5] is_frozen  [6] ATK  [7] HP  [8..24] 17 flags  [25] is_selected
      [26..36] 11 type one-hots
    """

    # Индексы признаков в entity-векторе из hs_env.py
    _CONTINUOUS_IDX = [2, 3, 4, 6, 7]       # card_id_norm, cost, tier, ATK, HP
    _BINARY_IDX = [0, 1, 5] + list(range(8, 26))  # is_present, is_spell, is_frozen, 17 flags, is_selected
    _TYPE_IDX = list(range(26, 37))          # 11 type one-hots

    def __init__(self, d_model: int, max_teams: int = 5):
        super().__init__()
        self.proj_continuous = nn.Linear(len(self._CONTINUOUS_IDX), d_model)
        self.proj_binary = nn.Linear(len(self._BINARY_IDX), d_model)
        self.proj_types = nn.Linear(len(self._TYPE_IDX), d_model)

        # Zone Embedding: идентификатор зоны (доска, таверна, рука, дискавер)
        self.emb_team = nn.Embedding(max_teams, d_model)

    def forward(self, val: torch.Tensor, team_id: torch.Tensor) -> torch.Tensor:
        """
        val: [B, N, 37] — сырые entity-вектора
        team_id: [B, N] — ID зоны каждого слота
        → [B, N, d_model]
        """
        x_cont = self.proj_continuous(val[..., self._CONTINUOUS_IDX])
        x_bin = self.proj_binary(val[..., self._BINARY_IDX])
        x_type = self.proj_types(val[..., self._TYPE_IDX])

        # Аддитивное смешивание (как token + position в BERT)
        x = x_cont + x_bin + x_type + self.emb_team(team_id)
        return x


class FiLMGenerator(nn.Module):
    """
    Генератор гиперпараметров для Feature-wise Linear Modulation (FiLM).
    Принимает глобальный контекст игры (ХП героя, Голда, Тир таверны) и выдает коэффициенты
    мультипликативного (gamma) и аддитивного (beta) искажения локальных токенов карт.
    """

    def __init__(self, d_context, d_model):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_context, d_model),
            nn.SiLU(),  # Гладкая активация лучше подходит для генерации параметров масштабирования
            nn.Linear(d_model, 2 * d_model),
        )

        # Zero-Initialization Trick:
        # Инициализируем веса последнего слоя строго нулями.
        # В паре с формулой (gamma + 1.0) ниже, это гарантирует, что на шаге 0 инициализации сети
        # FiLM будет работать как Identity-функция (тождество) X' = X*1 + 0, не убивая градиенты.
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, c):
        out = self.mlp(c)
        gamma, beta = out.chunk(2, dim=-1)
        return gamma, beta


class FiLM(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM).
    Динамически искажает пространство локальных инкодингов X (карт)
    на основе глобального вектора C (стейт игры).
    Позволяет аппаратно "выключать" или "масштабировать" определенные
    признаки (например, обнулять интерес
    к покупке дорогих карт, если голды 0).
    Это эффективнее простой конкатенации X и C.
    """

    def __init__(self, d_context, d_model):
        super().__init__()
        self.generator = FiLMGenerator(d_context, d_model)

    def forward(self, x, c):
        gamma, beta = self.generator(c)
        gamma = gamma.unsqueeze(1)
        beta = beta.unsqueeze(1)
        # Прибавляем единицу к gamma для Identity Start (см. описание в FiLMGenerator)
        return x * (gamma + 1.0) + beta



# ==========================================
# Утилиты
# ==========================================


def symlog(x: torch.Tensor) -> torch.Tensor:
    """
    Симметричный логарифм из DreamerV3.
    Безопасно сжимает выбросы (50000 → ~10.8), около нуля ведет себя как y=x.
    """
    return torch.sign(x) * torch.log1p(torch.abs(x))


# ==========================================
# SB3-совместимый FeaturesExtractor
# ==========================================


class TransformerFeaturesExtractor(BaseFeaturesExtractor):
    """
    SB3-совместимая обёртка: принимает плоский Box(1009,) из hs_env.py,
    парсит → DecomposedEncoder → FiLM → [GLOBAL_CTX] → GatedTransformer → PMA → [B, d_model].

    Архитектурные особенности:
      • DecomposedEncoder: раздельные проекции для continuous/binary/categorical признаков
      • FiLM: pre-Attention модуляция токенов глобальным контекстом (тир, золото, ХП)
      • [GLOBAL_CTX]: обучаемый токен, аналог [CLS] из BERT / scalar features из AlphaStar,
        позволяет картам обращаться к глобальному состоянию внутри Self-Attention
      • Multi-Seed PMA (K=4): мультиаспектная агрегация через 4 обучаемых seed-вектора

    Раскладка obs: Global(7) | Board(7×37) | Hand(10×37) | Store(7×37) | Discover(3×37) | Enemy(3)
    """

    _ZONES = [(7, 1), (10, 2), (7, 3), (3, 4)]  # (slots, team_id)
    _GLOBAL_SIZE = 7
    _ENEMY_SIZE = 3
    _EF = 37

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        d_context: int = 10,
        max_teams: int = 5,
        use_symlog: bool = True,
        use_gating: bool = True,
        use_pma: bool = True,
        pma_seeds: int = 4,
    ):
        super().__init__(observation_space, features_dim=d_model)
        self.use_symlog = use_symlog
        self._total_slots = sum(s for s, _ in self._ZONES)

        # --- Encoder ---
        self.encoder = DecomposedEncoder(d_model, max_teams)

        # --- FiLM: pre-Attention модуляция ---
        self.film = FiLM(d_context, d_model)

        # --- [GLOBAL_CTX] token ---
        # Обучаемый токен глобального контекста: проецирует скаляры экономики (золото, тир, ХП)
        # в d_model-пространство и участвует в Self-Attention наравне с картами.
        # Аналог [CLS] из BERT: карты могут "спросить" у него "сколько у нас золота?"
        # через механизм внимания, а не только через FiLM (pre-Attention модуляцию).
        self.global_ctx_proj = nn.Sequential(
            nn.Linear(d_context, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        # --- Transformer Blocks ---
        self.blocks = nn.ModuleList([
            TransformerBlock(
                attention_cls=MultiHeadAttention,
                attention_args=dict(d_model=d_model, n_heads=n_heads),
                ffn_cls=FFN_SwiGLU, ffn_args=dict(d_model=d_model),
                norm_cls=RMSNorm, norm_args=dict(d_model=d_model),
                gate_cls=GatedResidual if use_gating else None,
                gate_args=dict(d_model=d_model) if use_gating else None,
            )
            for _ in range(n_layers)
        ])
        self.ln_f = RMSNorm(d_model)

        # --- Multi-Seed PMA ---
        self.pma = PMA(d_model, n_heads, k_seeds=pma_seeds) if use_pma else None

    def _parse_flat_obs(self, flat: torch.Tensor) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        """[B, 1009] → val [B, N, 37], team_id [B, N], context [B, 10]"""
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
        # Пустые слоты (is_present == 0) → team_id = 0 для корректного маскирования
        team_id = team_id * (val[:, :, 0] > 0.5).long()

        context = torch.cat([global_vec, enemy_vec], dim=-1)

        return val, team_id, context

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """[B, 1009] → [B, d_model]"""
        val, team_id, context = self._parse_flat_obs(observations)

        if self.use_symlog:
            val = symlog(val)

        # 1. DecomposedEncoder: val + zone → [B, N, D]
        x = self.encoder(val, team_id)

        # 2. FiLM: pre-Attention модуляция глобальным контекстом
        x = self.film(x, context)

        # 3. [GLOBAL_CTX]: конкатенируем обучаемый токен глобального контекста
        global_token = self.global_ctx_proj(context).unsqueeze(1)  # [B, 1, D]
        x = torch.cat([global_token, x], dim=1)  # [B, N+1, D]

        # Padding mask: [B, 1, 1, N+1] — GLOBAL_CTX всегда visible (True)
        entity_mask = (team_id != 0)  # [B, N]
        global_mask = torch.ones(x.shape[0], 1, device=x.device, dtype=torch.bool)
        pad_mask = torch.cat([global_mask, entity_mask], dim=1).unsqueeze(1).unsqueeze(2)

        # 4. Transformer blocks
        for block in self.blocks:
            x = block(x, mask=pad_mask)
        x = self.ln_f(x)

        # 5. Multi-Seed PMA (K=4) или masked mean pooling
        if self.pma is not None:
            x = self.pma(x, mask=pad_mask)  # [B, d_model]
        else:
            full_mask = torch.cat([global_mask, entity_mask], dim=1)
            mask_f = full_mask.float().unsqueeze(-1)
            x = (x * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1)

        return x

