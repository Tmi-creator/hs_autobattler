"""
Smoke test: проверяет что TransformerFeaturesExtractor корректно работает
с плоскими наблюдениями из HearthstoneEnv.

Тесты:
  1. Forward pass: shape [B, d_model]
  2. Gradient flow: no NaN
  3. Padding mask: пустые слоты маскируются
  4. Symlog sanity
  5. DecomposedEncoder: раздельная обработка фич
  6. GLOBAL_CTX: токен глобального контекста в attention
  7. PMA multi-seed: K=4 seeds → [B, d_model]
  8. Полный пайплайн: MaskablePPO + TransformerFeaturesExtractor + HearthstoneEnv
"""

import sys
from pathlib import Path

root_path = Path(__file__).resolve().parent.parent
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from gymnasium import spaces  # noqa: E402

from scripts.trans import (  # noqa: E402
    PMA,
    DecomposedEncoder,
    TransformerFeaturesExtractor,
    symlog,
)


def test_forward_pass() -> None:
    """Forward pass с рандомными данными → shape [B, d_model]"""
    print("Test 1: Forward pass shapes... ", end="")
    d_model = 128
    obs_space = spaces.Box(low=0, high=1, shape=(1009,), dtype=np.float32)

    extractor = TransformerFeaturesExtractor(obs_space, d_model=d_model, n_heads=4, n_layers=2)
    extractor.eval()

    batch = torch.rand(4, 1009)
    with torch.no_grad():
        out = extractor(batch)

    assert out.shape == (4, d_model), f"Expected (4, {d_model}), got {out.shape}"
    assert not torch.isnan(out).any(), "NaN in output!"
    print("PASSED")


def test_gradient_flow() -> None:
    """loss.backward() работает без NaN"""
    print("Test 2: Gradient flow... ", end="")
    obs_space = spaces.Box(low=0, high=1, shape=(1009,), dtype=np.float32)

    extractor = TransformerFeaturesExtractor(obs_space, d_model=64, n_heads=4, n_layers=2)
    extractor.train()

    batch = torch.rand(2, 1009)
    out = extractor(batch)
    loss = out.sum()
    loss.backward()

    has_grad = False
    for name, p in extractor.named_parameters():
        if p.grad is not None:
            assert not torch.isnan(p.grad).any(), f"NaN grad in {name}"
            if p.grad.abs().sum() > 0:
                has_grad = True

    assert has_grad, "No gradients flowed through the model!"
    print("PASSED")


def test_padding_mask() -> None:
    """Пустые слоты (is_present=0) корректно маскируются"""
    print("Test 3: Padding mask... ", end="")
    obs_space = spaces.Box(low=0, high=1, shape=(1009,), dtype=np.float32)

    extractor = TransformerFeaturesExtractor(obs_space, d_model=64, n_heads=4, n_layers=2)
    extractor.eval()

    # Создаём obs где все сущности = 0 (пустые), кроме первого на борде
    batch = torch.zeros(1, 1009)
    # Первая сущность на борде: is_present=1, остальные фичи рандомные
    batch[0, 7] = 1.0  # is_present
    batch[0, 7 + 6] = 0.5  # ATK
    batch[0, 7 + 7] = 0.3  # HP

    with torch.no_grad():
        out = extractor(batch)

    assert out.shape == (1, 64), f"Expected (1, 64), got {out.shape}"
    assert not torch.isnan(out).any(), "NaN with sparse obs!"
    print("PASSED")


def test_symlog() -> None:
    """symlog вычисляет корректно"""
    print("Test 4: Symlog sanity... ", end="")
    x = torch.tensor([0.0, 1.0, -1.0, 50000.0, -50000.0])
    y = symlog(x)

    assert abs(y[0].item()) < 1e-6, f"symlog(0) = {y[0].item()}"
    assert abs(y[1].item() - 0.6931) < 0.01, f"symlog(1) = {y[1].item()}"
    assert abs(y[2].item() + 0.6931) < 0.01, f"symlog(-1) = {y[2].item()}"
    assert abs(y[3].item() - 10.82) < 0.01, f"symlog(50000) = {y[3].item()}"
    assert abs(y[4].item() + 10.82) < 0.01, f"symlog(-50000) = {y[4].item()}"
    print("PASSED")


def test_decomposed_encoder() -> None:
    """DecomposedEncoder раздельно обрабатывает continuous/binary/types"""
    print("Test 5: DecomposedEncoder feature split... ", end="")
    d_model = 64
    enc = DecomposedEncoder(d_model, max_teams=5)

    B, N = 2, 27
    val = torch.rand(B, N, 37)
    team_id = torch.randint(0, 5, (B, N))

    out = enc(val, team_id)
    assert out.shape == (B, N, d_model), f"Expected ({B}, {N}, {d_model}), got {out.shape}"
    assert not torch.isnan(out).any(), "NaN in DecomposedEncoder output!"

    # Проверяем что разные ветки дают разные вклады
    val_zeros = torch.zeros(B, N, 37)
    val_zeros[..., 6] = 1.0  # только ATK (continuous)
    out_atk = enc(val_zeros, team_id)

    val_zeros2 = torch.zeros(B, N, 37)
    val_zeros2[..., 8] = 1.0  # только Taunt (binary)
    out_taunt = enc(val_zeros2, team_id)

    assert not torch.allclose(out_atk, out_taunt), "ATK and Taunt produce identical outputs!"
    print("PASSED")


def test_global_ctx_token() -> None:
    """[GLOBAL_CTX] токен добавляется в последовательность перед Transformer"""
    print("Test 6: GLOBAL_CTX token... ", end="")
    obs_space = spaces.Box(low=0, high=1, shape=(1009,), dtype=np.float32)

    extractor = TransformerFeaturesExtractor(obs_space, d_model=64, n_heads=4, n_layers=1)
    extractor.eval()

    batch = torch.rand(2, 1009)

    # Проверяем что global_ctx_proj существует и работает
    val, team_id, context = extractor._parse_flat_obs(batch)
    global_token = extractor.global_ctx_proj(context)
    assert global_token.shape == (2, 64), (
        f"Global token shape: expected (2, 64), got {global_token.shape}"
    )

    # Полный проход
    with torch.no_grad():
        out = extractor(batch)
    assert out.shape == (2, 64)
    assert not torch.isnan(out).any()
    print("PASSED")


def test_pma_multi_seed() -> None:
    """PMA с K=4 seeds: [B, N, D] → [B, D]"""
    print("Test 7: PMA multi-seed (K=4)... ", end="")
    d_model = 64
    pma = PMA(d_model, n_heads=4, k_seeds=4)

    x = torch.rand(3, 20, d_model)
    out = pma(x)
    assert out.shape == (3, d_model), f"Expected (3, {d_model}), got {out.shape}"
    assert not torch.isnan(out).any(), "NaN in PMA output!"
    print("PASSED")


def test_full_pipeline() -> None:
    """Полный пайплайн: MaskablePPO + TransformerFeaturesExtractor + HearthstoneEnv"""
    print("Test 8: Full SB3 pipeline... ", end="")
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
    from sb3_contrib.common.wrappers import ActionMasker

    from hearthstone.env.hs_env import HearthstoneEnv

    def mask_fn(env: object) -> np.ndarray:
        from typing import cast

        return np.asarray(cast(HearthstoneEnv, env).action_masks(), dtype=bool)

    env = HearthstoneEnv()
    env.reset(seed=42)
    env = ActionMasker(env, mask_fn)

    policy_kwargs = dict(
        features_extractor_class=TransformerFeaturesExtractor,
        features_extractor_kwargs=dict(
            d_model=64,
            n_heads=4,
            n_layers=2,
            d_context=10,
        ),
        net_arch=dict(pi=[64], vf=[64]),
    )

    model = MaskablePPO(
        MaskableActorCriticPolicy,
        env,
        policy_kwargs=policy_kwargs,
        n_steps=32,
        batch_size=16,
        verbose=0,
        device="cpu",
    )

    # Predict
    obs, _ = env.reset()
    masks = np.asarray(env.action_masks(), dtype=bool)
    action, _ = model.predict(obs, action_masks=masks, deterministic=True)

    assert 0 <= int(action) <= 31, f"Action {action} out of range!"

    # Короткий learn (64 шага)
    model.learn(total_timesteps=64)

    print("PASSED")


if __name__ == "__main__":
    print("=" * 50)
    print("[TEST] TransformerFeaturesExtractor Smoke Tests")
    print("=" * 50)

    test_forward_pass()
    test_gradient_flow()
    test_padding_mask()
    test_symlog()
    test_decomposed_encoder()
    test_global_ctx_token()
    test_pma_multi_seed()
    test_full_pipeline()

    print("\n" + "=" * 50)
    print("ALL TESTS PASSED!")
    print("=" * 50)
