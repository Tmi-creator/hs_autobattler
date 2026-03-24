import os
import sys
from pathlib import Path
from typing import Callable, Protocol, TypedDict, cast

# Добавляем корень проекта в sys.path, чтобы импорты из scripts и hearthstone работали везде
root_path = Path(__file__).resolve().parent.parent
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))

import numpy as np
import torch
import wandb
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv
from wandb.integration.sb3 import WandbCallback

from hearthstone.env.hs_env import HearthstoneEnv
from scripts.callbacks import GameLoggerCallback, SelfPlayCallback


class TrainConfig(TypedDict):
    policy_type: str
    total_timesteps: int
    learning_rate: float
    gamma: float
    batch_size: int
    n_steps: int
    ent_coef: float
    net_arch: list[int]
    n_envs: int
    seed: int


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
        callback: list[CheckpointCallback | WandbCallback | GameLoggerCallback | SelfPlayCallback],
    ) -> MaskablePPO: ...


def _mask_fn(base_env: object) -> np.ndarray:
    return np.asarray(cast(HearthstoneEnv, base_env).action_masks(), dtype=bool)


# === MAIN DETERMINISM FUNCTION ===
def setup_determinism(seed: int) -> None:
    # 1. Fix string hashing (important for dicts and sets)
    # should do it before all other processes, so change os.environ
    os.environ["PYTHONHASHSEED"] = str(seed)

    # 2. Make CUDA determine algorithm (Torch >= 1.7)
    # Without it torch.use_deterministic_algorithms(True) can throw error
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    # 3. standard seeds
    set_random_seed(seed)  # Fix random, numpy and torch cpu

    # 4. Settings PyTorch for GPU
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        # Жесткий режим (вызовет ошибку, если используется недетерминированная операция)
        # Раскомментируй, если хочешь быть уверен на 200%, но некоторые слои могут не работать
        # torch.use_deterministic_algorithms(True)


def make_env(rank: int, seed: int = 0) -> Callable[[], ActionMasker]:
    def _init() -> ActionMasker:
        env = HearthstoneEnv()
        env.reset(seed=seed + rank)
        wrapped_env = ActionMasker(
            env,
            _mask_fn,
        )
        return wrapped_env

    return _init


def main() -> None:
    SEED = 42
    setup_determinism(SEED)  # setup before all

    config: TrainConfig = {
        "policy_type": "MlpPolicy",
        "total_timesteps": 100_000,
        "learning_rate": 0.0003,
        "gamma": 0.99,
        "batch_size": 256,
        "n_steps": 2048,
        "ent_coef": 0.01,
        "net_arch": [512, 512],
        "n_envs": 8,
        "seed": SEED,
    }

    run = wandb.init(
        project="hs_autobattler",
        config=dict(config),
        sync_tensorboard=True,
        monitor_gym=False,
        save_code=True,
    )

    models_dir = f"models/{run.id}"
    os.makedirs(models_dir, exist_ok=True)

    print("Checking single environment...")
    # Важно: создаем dummy среду с фикс. сидом, чтобы проверки не сдвигали глобальный рандом непредсказуемо
    dummy_env = HearthstoneEnv()
    dummy_env.reset(seed=SEED)
    print("Environment check passed!")

    print(f"Initializing {config['n_envs']} parallel environments...")

    # make_vec_env сама передаст seed + rank в каждый подпроцесс
    env = SubprocVecEnv([make_env(i, SEED) for i in range(config["n_envs"])])

    # Оборачиваем векторизованную среду в нормализатор
    # norm_obs=True - нормализует входные данные (obs)
    # norm_reward=True - нормализует награду (reward)
    # clip_reward=10.0 - обрезает выбросы
    # add next
    # env = VecNormalize(env, norm_obs=False, norm_reward=True, clip_reward=10.0)

    policy_kwargs = dict(net_arch=config["net_arch"])

    model = MaskablePPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=config["learning_rate"],
        gamma=config["gamma"],
        batch_size=config["batch_size"],
        n_steps=config["n_steps"],
        ent_coef=config["ent_coef"],
        tensorboard_log=f"runs/{run.id}",
        policy_kwargs=policy_kwargs,
        seed=SEED,
        device="cuda" if torch.cuda.is_available() else "auto",
    )
    logs_dir = f"logs/{run.id}"
    os.makedirs(logs_dir, exist_ok=True)

    print(f"Starting training for {config['total_timesteps']} timesteps...")

    # === CALLBACKS ===
    game_logger = GameLoggerCallback(check_freq=15000, log_dir=logs_dir)

    checkpoint_callback = CheckpointCallback(
        save_freq=50000 // config["n_envs"], save_path=models_dir, name_prefix="hs_model"
    )

    wandb_callback = WandbCallback(
        gradient_save_freq=100,
        model_save_path=f"models/{run.id}",
        verbose=2,
    )

    self_play_callback = SelfPlayCallback(update_freq=15000, model_save_path=models_dir)

    model_learner = cast(SupportsLearn, model)
    model_learner.learn(
        total_timesteps=config["total_timesteps"],
        callback=[checkpoint_callback, wandb_callback, game_logger, self_play_callback],
    )

    model.save(f"{models_dir}/hs_final")
    wandb_api = cast(WandbApi, wandb)
    wandb_api.save(f"{models_dir}/hs_final.zip")
    print("Training finished.")
    run.finish()
    env.close()


if __name__ == "__main__":
    main()
