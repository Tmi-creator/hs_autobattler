import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback

from hs_env import HearthstoneEnv


def main():
    models_dir = "models/PPO"
    log_dir = "logs"
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    print("Initializing environment...")
    env = HearthstoneEnv()

    check_env(env)
    print("Environment check passed!")

    # 3. Настройка Нейросети (Policy)
    # MlpPolicy = MultiLayer Perceptron (обычная полносвязная сеть), так как у нас вектор чисел, а не картинка.
    # net_arch = архитектура сети. Вход (606) -> 256 -> 256 -> Выход (26 действий)
    policy_kwargs = dict(net_arch=[256, 256])

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=0.0003,  # Скорость обучения
        gamma=0.99,  # Важность будущих наград
        batch_size=64,  # Размер пакета данных для обновления
        ent_coef=0.01,  # Коэффициент энтропии (заставляет пробовать новое)
        tensorboard_log=log_dir,  # Логирование для графиков
        policy_kwargs=policy_kwargs,
        device="auto"  # Использует GPU (cuda/mps) если есть, иначе CPU
    )

    TIMESTEPS = 300_000
    print(f"Starting training for {TIMESTEPS} timesteps...")

    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path=models_dir,
        name_prefix="hs_model"
    )

    model.learn(total_timesteps=TIMESTEPS, callback=checkpoint_callback)

    model.save(f"{models_dir}/hs_final")
    print("Training finished. Model saved.")

    print("\n--- Testing the trained agent (1 game) ---")
    obs, _ = env.reset()
    done = False
    truncated = False
    total_reward = 0

    steps = 0
    while not done and not truncated:
        action, _ = model.predict(obs, deterministic=True)

        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        steps += 1

        act_str = str(action)
        if action == 0:
            act_str = "END TURN"
        elif action == 1:
            act_str = "ROLL"
        elif 2 <= action <= 8:
            act_str = f"BUY Slot {action - 2}"
        elif 9 <= action <= 15:
            act_str = f"SELL Slot {action - 9}"
        elif 16 <= action <= 25:
            act_str = f"PLAY Hand {action - 16}"

        print(f"Step {steps}: Action {act_str} | Reward: {reward:.2f}")

    print(f"Game finished. Total Reward: {total_reward}")


if __name__ == "__main__":
    main()
