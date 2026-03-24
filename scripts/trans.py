import torch
import torch.nn as nn
import torch.nn.functional as F


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


class TransformerBlock(nn.Module):
    """
    Классический блок Трансформера (Self-Attention + FFN с Residual-связями).
    Поскольку мы работаем с N ~ 30 (карты на столе/в таверне),
    сложность O(N^2) составляет всего ~900 операций,
    что делает этот блок быстрее и эффективнее, чем усложненные структуры типа ISAB.
    Из-за отсутствия внешних pos_encoding работает как чисто инвариантная к
    перестановкам архитектура (Pure Set Transformer).
    """

    def __init__(
        self,
        attention_cls,
        attention_args,
        ffn_cls,
        ffn_args,
        norm_cls,
        norm_args,
    ):
        super().__init__()
        self.attention = attention_cls(**attention_args)
        self.ffn = ffn_cls(**ffn_args)
        self.norm1 = norm_cls(**norm_args)
        self.norm2 = norm_cls(**norm_args)

    def forward(self, x):
        h = self.norm1(x)
        # Self-Attention: подаем один и тот же тензор как Query, Key, Value
        x = x + self.attention(h, h, h)
        x = x + self.ffn(self.norm2(x))
        return x


class ObservationEncoder(nn.Module):
    """
    Энкодер плотных наблюдений (Dense Entity Representation).
    Вместо EAV паттерна (Entity-Attribute-Value), каждая карта/существо
    подается как единый плотный вектор признаков d_features.
    Это аппаратно решает Binding Problem (трансформеру не нужно тратить
    слои на связывание ХП и Атаки одного существа).
    Отсутствуют координаты (pos_id) — мы относимся к борде как к
    неупорядоченному множеству.
    """

    def __init__(self, d_features, d_model, max_teams=5, max_time=10):
        super().__init__()
        # Проекция всех сырых признаков карты разом в скрытую размерность трансформера
        self.val_proj = nn.Linear(d_features, d_model)

        # Эмбеддинги для идентификации того, ГДЕ лежит карта (своя доска, таверна, рука и т.д.)
        self.emb_team = nn.Embedding(max_teams, d_model)

        # Эмбеддинг времени (история состояний)
        self.emb_time = nn.Embedding(max_time, d_model)

    def forward(self, val, team_id, time_id):
        x = self.val_proj(val)
        x = x + self.emb_team(team_id)
        x = x + self.emb_time(time_id)
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


class ActorHead(nn.Module):
    """
    Голова Actor в RL-архитектуре PPO/A2C.
    Проецирует скрытое представление всего множества карт в логиты действий агента
    (покупка, разыгрывание и т.д.)
    """

    def __init__(self, d_model, action_size):
        super().__init__()
        self.head = nn.Linear(d_model, action_size)

    def forward(self, h):
        return self.head(h)


class DiscreteCriticHead(nn.Module):
    """
    Голова Critic с категориальным распределением Q-value по корзинам (bins).
    Помогает лучше оценивать дисперсию и неопределенность ожидаемой пользы состояния,
    чем скалярный Critic.
    """

    def __init__(self, d_model, action_size, num_bins=20):
        super().__init__()
        self.action_size = action_size
        self.num_bins = num_bins
        self.hidden = nn.Linear(d_model, d_model)
        self.critic_head = nn.Linear(d_model, action_size * num_bins)

    def forward(self, h):
        c = F.relu(self.hidden(h))
        q_logits = self.critic_head(c)
        q_logits = q_logits.view(-1, self.action_size, self.num_bins)
        return q_logits


class MARLGPT(nn.Module):
    """
    Главная архитектура нейросети агента
    (Dense Entity Set Transformer с FiLM).
    Обладает максимальной гибкостью через инъекцию зависимостей
    (классы и их аргументы передаются в init).
    """

    def __init__(
        self,
        d_model,
        n_layers,
        transformer_block_cls,
        transformer_block_args,
        obs_encoder_cls,
        obs_encoder_args,
        film_cls,
        film_args,
        actor_head_cls,
        actor_head_args,
        critic_head_cls,
        critic_head_args,
        norm_cls,
        norm_args,
    ):
        super().__init__()
        self.d_model = d_model

        self.obs_encoder = obs_encoder_cls(**obs_encoder_args)

        if film_cls is not None and film_args is not None:
            self.film = film_cls(**film_args)
        else:
            self.film = None

        self.blocks = nn.ModuleList(
            [transformer_block_cls(**transformer_block_args) for _ in range(n_layers)]
        )
        self.ln_f = norm_cls(**norm_args)

        self.actor_head = actor_head_cls(**actor_head_args)
        self.critic_head = critic_head_cls(**critic_head_args)

    def forward(self, val, team_id, time_id, global_context=None, action_mask=None):
        # 1. Энкодинг локальных плотных наблюдений -> [B, N, D]
        x = self.obs_encoder(val, team_id, time_id)

        # 2. FiLM: Модуляция токенов карт на основе глобального стейта игры
        if self.film is not None and global_context is not None:
            x = self.film(x, global_context)

        # 3. Проход через Set Transformer
        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)

        # 4. Pooling над всем множеством карт (Mean Pooling) -> [B, D]
        h_t = x.mean(dim=1)

        # 5. Вычисление логитов Actor и аппаратное сокрытие нелегальных действий (-inf)
        actor_logits = self.actor_head(h_t)
        if action_mask is not None:
            actor_logits = actor_logits.masked_fill(action_mask == 0, float("-inf"))

        # 6. Вычисление логитов Critic для оценки состояний
        q_logits = self.critic_head(h_t)

        return actor_logits, q_logits


# ==========================================
# Пример конфигурации и инициализации
# ==========================================
if __name__ == "__main__":
    d_model = 128
    n_heads = 4
    d_context = 16

    transformer_args = dict(
        attention_cls=MultiHeadAttention,
        attention_args=dict(d_model=d_model, n_heads=n_heads),
        ffn_cls=FFN_SwiGLU,
        ffn_args=dict(d_model=d_model),
        norm_cls=RMSNorm,
        norm_args=dict(d_model=d_model),
    )

    args_marlgpt = dict(
        d_model=d_model,
        n_layers=4,
        transformer_block_cls=TransformerBlock,
        transformer_block_args=transformer_args,
        obs_encoder_cls=ObservationEncoder,
        obs_encoder_args=dict(d_features=35, d_model=d_model, max_teams=5, max_time=10),
        film_cls=FiLM,
        film_args=dict(d_context=d_context, d_model=d_model),
        actor_head_cls=ActorHead,
        actor_head_args=dict(d_model=d_model, action_size=32),
        critic_head_cls=DiscreteCriticHead,
        critic_head_args=dict(d_model=d_model, action_size=32, num_bins=20),
        norm_cls=RMSNorm,
        norm_args=dict(d_model=d_model),
    )

    model = MARLGPT(**args_marlgpt)
    print(
        "Архитектура Pure Set Transformer готова! Params:",
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    )
