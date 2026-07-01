import os
import sys
from pathlib import Path
from typing import Callable, Protocol, TypedDict, cast

import numpy as np
import torch
import wandb
from scripts.categorical_critic import CategoricalMaskablePPO, CategoricalValuePolicy
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv
from wandb.integration.sb3 import WandbCallback

from hearthstone.env.hs_env import HearthstoneEnv
from scripts.callbacks import BoardPowerCallback, GameLoggerCallback, SelfPlayCallback
from scripts.trans import TransformerFeaturesExtractor

# Добавляем корень проекта в sys.path
root_path = Path(__file__).resolve().parent.parent
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))

# === Все артефакты обучения складываются сюда ===
OUTPUTS_DIR = Path(__file__).resolve().parent / "outputs"


class TrainConfig(TypedDict):
    policy_type: str
    total_timesteps: int
    learning_rate: float
    gamma: float
    batch_size: int
    n_steps: int
    ent_coef: float
    n_envs: int
    seed: int
    # Transformer-specific
    d_model: int
    n_heads: int
    n_layers: int


class WandbApi(Protocol):
    def save(
        self,
        glob_str: str,
        base_path: str | None = None,
        policy: str = "live",
    ) -> bool | list[str]: ...


class SupportsLearn(Protocol):
    def learn(
        self,
        total_timesteps: int,
        callback: list[object],
    ) -> CategoricalMaskablePPO: ...


def _mask_fn(base_env: object) -> np.ndarray:
    return np.asarray(cast(HearthstoneEnv, base_env).action_masks(), dtype=bool)


# === MAIN DETERMINISM FUNCTION ===
def setup_determinism(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    set_random_seed(seed)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def make_env(rank: int, seed: int = 0) -> Callable[[], ActionMasker]:
    def _init() -> ActionMasker:
        env = HearthstoneEnv()
        env.reset(seed=seed + rank)
        return ActionMasker(env, _mask_fn)

    return _init


def main() -> None:
    SEED = 42
    setup_determinism(SEED)

    config: TrainConfig = {
        "policy_type": "TransformerPolicy",
        "total_timesteps": 1_500_000,
        "learning_rate": 0.0003,
        "gamma": 0.999,
        "batch_size": 256,
        "n_steps": 4096,
        "ent_coef": 0.04,
        "n_envs": 8,
        "seed": SEED,
        # Transformer hyperparams
        "d_model": 128,
        "n_heads": 4,
        "n_layers": 4,
    }

    run = wandb.init(
        project="hs_autobattler_transformer",
        config=dict(config),
        sync_tensorboard=True,
        monitor_gym=False,
        save_code=True,
    )

    # === Пути: всё в scripts/outputs/ ===
    models_dir = str(OUTPUTS_DIR / "models" / "transformer" / run.id)
    runs_dir = str(OUTPUTS_DIR / "runs" / "transformer" / run.id)
    logs_dir = str(OUTPUTS_DIR / "logs" / "transformer" / run.id)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    print("Checking single environment...")
    dummy_env = HearthstoneEnv()
    dummy_env.reset(seed=SEED)
    print("Environment check passed!")

    print(f"Initializing {config['n_envs']} parallel environments...")
    env = DummyVecEnv([make_env(i, SEED) for i in range(config["n_envs"])])

    # Get num_card_ids from environment
    _tmp_env = HearthstoneEnv()
    num_card_ids = _tmp_env.num_card_ids
    del _tmp_env
    print(f"Card vocabulary size: {num_card_ids}")

    # === TRANSFORMER POLICY ===
    policy_kwargs = dict(
        features_extractor_class=TransformerFeaturesExtractor,
        features_extractor_kwargs=dict(
            d_model=config["d_model"],
            n_heads=config["n_heads"],
            n_layers=config["n_layers"],
            d_context=10,  # global(7) + enemy(3)
            num_card_ids=num_card_ids,
        ),
        net_arch=dict(pi=[128], vf=[128]),
    )

    model = CategoricalMaskablePPO(
        CategoricalValuePolicy,
        env,
        verbose=1,
        learning_rate=config["learning_rate"],
        gamma=config["gamma"],
        batch_size=config["batch_size"],
        n_steps=config["n_steps"],
        ent_coef=config["ent_coef"],
        tensorboard_log=runs_dir,
        policy_kwargs=policy_kwargs,
        seed=SEED,
        device="cuda" if torch.cuda.is_available() else "auto",
    )

    # === ZERO-INIT OUTPUT WEIGHTS (DreamerV3) ===
    policy = model.policy
    for module in [policy.action_net, policy.value_net]:
        if hasattr(module, "weight"):
            torch.nn.init.zeros_(module.weight)
        if hasattr(module, "bias") and module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    print("[INIT] Zero-Init applied to Actor/Critic output layers")

    # Кол-во параметров
    extractor = model.policy.features_extractor
    n_params = sum(p.numel() for p in extractor.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.policy.parameters() if p.requires_grad)
    print(f"[MODEL] Transformer FeaturesExtractor: {n_params:,} params")
    print(f"[MODEL] Total trainable: {total_params:,} params")
    print(f"[TRAIN] Starting training for {config['total_timesteps']:,} timesteps...")

    # === CALLBACKS ===
    game_logger = GameLoggerCallback(check_freq=15000, log_dir=logs_dir)
    checkpoint_callback = CheckpointCallback(
        save_freq=100_000 // config["n_envs"],
        save_path=models_dir,
        name_prefix="hs_trans",
    )
    wandb_callback = WandbCallback(
        gradient_save_freq=500,
        model_save_path=models_dir,
        verbose=2,
    )
    self_play_callback = SelfPlayCallback(update_freq=50_000, model_save_path=models_dir)
    board_power_callback = BoardPowerCallback(log_freq=2000)

    model_learner = cast(SupportsLearn, model)
    model_learner.learn(
        total_timesteps=config["total_timesteps"],
        callback=[
            checkpoint_callback,
            wandb_callback,
            game_logger,
            self_play_callback,
            board_power_callback,
        ],
    )

    # === SAVE FINAL MODEL ===
    final_path = str(OUTPUTS_DIR / "models" / "transformer_final")
    model.save(final_path)
    wandb_api = cast(WandbApi, wandb)
    wandb_api.save(f"{final_path}.zip")
    print(f"[DONE] Model saved to {final_path}.zip")
    run.finish()
    env.close()


if __name__ == "__main__":
    main()
