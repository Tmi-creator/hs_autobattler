"""
Визуализация паттернов внимания трансформера.

Загружает обученную модель, прогоняет одно наблюдение через трансформер
и строит тепловые карты attention weights для каждого слоя и каждой головы.
Позволяет увидеть, какие сущности модель считает взаимосвязанными.

Использование:
    python scripts/visualize_attention.py --model models/quick_test/transformer_10k.zip
"""

import argparse
import os
import sys
from pathlib import Path
from typing import cast

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from sb3_contrib import MaskablePPO

root_path = Path(__file__).resolve().parent.parent
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))

from hearthstone.env.hs_env import HearthstoneEnv  # noqa: E402
from scripts.trans import MultiHeadAttention  # noqa: E402

OUTPUTS_DIR = Path(__file__).resolve().parent / "outputs"


# === Названия зон для оси X/Y на хитмапе ===
def _build_entity_labels() -> list[str]:
    """Генерирует метки для 28 токенов: GLOBAL_CTX + 27 entity slots."""
    labels = ["[CTX]"]
    for i in range(7):
        labels.append(f"B{i}")  # Board
    for i in range(10):
        labels.append(f"H{i}")  # Hand
    for i in range(7):
        labels.append(f"S{i}")  # Store
    for i in range(3):
        labels.append(f"D{i}")  # Discover
    return labels


class AttentionCapture:
    """
    Хук для перехвата attention weights из MultiHeadAttention.
    Регистрируется как forward_hook, сохраняет attention map каждого слоя.
    """

    def __init__(self):
        self.attention_maps: list[torch.Tensor] = []
        self._hooks: list[torch.utils.hooks.RemovableHook] = []

    def hook_fn(self, module: torch.nn.Module, input_args: tuple, output: torch.Tensor):
        """Перехватываем attention weights внутри MultiHeadAttention."""
        q_x, k_x = input_args[0], input_args[1]
        with torch.no_grad():
            mha = cast(MultiHeadAttention, module)
            B, Lq, D = q_x.shape
            _, Lk, _ = k_x.shape
            q = mha.q(q_x).view(B, Lq, mha.n_heads, mha.d_k).transpose(1, 2)
            k = mha.k(k_x).view(B, Lk, mha.n_heads, mha.d_k).transpose(1, 2)
            scores = (q @ k.transpose(-2, -1)) / (mha.d_k**0.5)
            attn_weights = torch.softmax(scores, dim=-1)  # [B, heads, Lq, Lk]
            self.attention_maps.append(attn_weights.cpu())

    def register(self, model: torch.nn.Module):
        """Регистрирует хуки на все MultiHeadAttention слои в TransformerBlock-ах."""
        extractor = model.policy.features_extractor
        for i, block in enumerate(extractor.blocks):
            hook = block.attention.register_forward_hook(self.hook_fn)
            self._hooks.append(hook)

    def remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def clear(self):
        self.attention_maps.clear()


def plot_attention(
    attention_maps: list[torch.Tensor],
    labels: list[str],
    is_present: list[bool],
    save_dir: str,
):
    """
    Строит тепловые карты внимания.
    attention_maps: список [1, n_heads, N+1, N+1] для каждого слоя
    """
    n_layers = len(attention_maps)
    n_heads = attention_maps[0].shape[1]
    N = attention_maps[0].shape[-1]

    # Ограничиваем метки размером attention
    labels = labels[:N]

    # Masked labels: помечаем пустые слоты
    display_labels = []
    for i, lbl in enumerate(labels):
        if i == 0:  # [CTX] всегда видимый
            display_labels.append(lbl)
        elif i - 1 < len(is_present) and not is_present[i - 1]:
            display_labels.append(f"({lbl})")
        else:
            display_labels.append(lbl)

    # === Средняя attention по головам для каждого слоя ===
    fig, axes = plt.subplots(1, n_layers, figsize=(6 * n_layers, 5))
    if n_layers == 1:
        axes = [axes]

    fig.suptitle("Attention Weights (averaged over heads)", fontsize=14, y=1.02)

    for layer_idx, attn in enumerate(attention_maps):
        ax = axes[layer_idx]
        avg_attn = attn[0].mean(dim=0).numpy()  # [N, N] средняя по головам

        im = ax.imshow(avg_attn, cmap="viridis", vmin=0, aspect="auto")
        ax.set_title(f"Layer {layer_idx}", fontsize=12)
        ax.set_xticks(range(N))
        ax.set_yticks(range(N))
        ax.set_xticklabels(display_labels, rotation=90, fontsize=7)
        ax.set_yticklabels(display_labels, fontsize=7)
        ax.set_xlabel("Key (attended to)")
        ax.set_ylabel("Query (attending)")

    plt.tight_layout()
    path = os.path.join(save_dir, "attention_avg_by_layer.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] {path}")

    # === Детальная: каждая голова первого слоя ===
    attn_layer0 = attention_maps[0][0]  # [n_heads, N, N]
    fig, axes = plt.subplots(1, n_heads, figsize=(5 * n_heads, 4))
    if n_heads == 1:
        axes = [axes]

    fig.suptitle("Layer 0 - Individual Attention Heads", fontsize=14, y=1.02)

    for head_idx in range(n_heads):
        ax = axes[head_idx]
        head_attn = attn_layer0[head_idx].numpy()

        im = ax.imshow(head_attn, cmap="magma", vmin=0, aspect="auto")
        ax.set_title(f"Head {head_idx}", fontsize=11)
        ax.set_xticks(range(N))
        ax.set_yticks(range(N))
        ax.set_xticklabels(display_labels, rotation=90, fontsize=6)
        ax.set_yticklabels(display_labels, fontsize=6)

    plt.tight_layout()
    path = os.path.join(save_dir, "attention_heads_layer0.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] {path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize Transformer attention")
    parser.add_argument(
        "--model",
        type=str,
        default=str(OUTPUTS_DIR / "models" / "transformer_final.zip"),
        help="Path to saved MaskablePPO model",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=str(OUTPUTS_DIR / "visualizations"),
        help="Directory to save attention heatmaps",
    )
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # Загрузка модели
    print(f"[LOAD] Loading model from {args.model}...")
    model = MaskablePPO.load(args.model, device="cpu")

    # Создаём среду и берём одно наблюдение
    env = HearthstoneEnv()
    obs, _ = env.reset(seed=123)

    # Собираем is_present для каждого слота
    EF = 37
    GLOBAL = 7
    is_present = []
    pos = GLOBAL
    for n_slots in [7, 10, 7, 3]:  # Board, Hand, Store, Discover
        for s in range(n_slots):
            is_present.append(obs[pos + s * EF] > 0.5)
        pos += n_slots * EF

    # Подключаем хуки
    capture = AttentionCapture()
    capture.register(model)

    # Forward pass
    obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        model.policy.features_extractor(obs_tensor)

    capture.remove()

    # Визуализация
    labels = _build_entity_labels()
    print(f"[INFO] Captured {len(capture.attention_maps)} layer attention maps")
    print(f"[INFO] Present entities: {sum(is_present)}/{len(is_present)}")

    plot_attention(capture.attention_maps, labels, is_present, args.save_dir)

    print(f"\n[DONE] Attention maps saved to {args.save_dir}/")


if __name__ == "__main__":
    main()
