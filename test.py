import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

from hs_env import HearthstoneEnv


def evaluate_model(model_path, num_games=100):
    print(f"Loading model from: {model_path}")

    env = HearthstoneEnv()

    try:
        model = PPO.load(model_path, env=env)
    except FileNotFoundError:
        print("Model file not found! Check the path.")
        return

    print(f"\n--- Starting Evaluation ({num_games} games) ---")

    wins = 0
    losses = 0
    draws = 0
    total_rewards = []
    turns_history = []

    for i in range(num_games):
        obs, _ = env.reset()
        done = False
        truncated = False
        episode_reward = 0

        while not done and not truncated:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward

        total_rewards.append(episode_reward)
        turns_history.append(env.game.turn_count)

        p0_hp = env.game.players[0].health
        p1_hp = env.game.players[1].health

        result_str = "DRAW"
        if p0_hp > 0 and p1_hp <= 0:
            wins += 1
            result_str = "WIN"
        elif p0_hp <= 0 and p1_hp > 0:
            losses += 1
            result_str = "LOSE"
            print(
                f"Game {i + 1}/{num_games} | Result: {result_str} | HP: {p0_hp} vs {p1_hp} | Reward: {episode_reward:.1f}")
            print([[u.max_atk, u.max_hp, u.card_id] for u in env.game.players[0].board])
            print([[u.max_atk, u.max_hp, u.card_id] for u in env.game.players[1].board])
        else:
            draws += 1

        # if (i + 1) % 10 == 0:
        #     print(
        #         f"Game {i + 1}/{num_games} | Result: {result_str} | HP: {p0_hp} vs {p1_hp} | Reward: {episode_reward:.1f}")

    win_rate = (wins / num_games) * 100
    avg_reward = np.mean(total_rewards)
    avg_turns = np.mean(turns_history)

    print("\n" + "=" * 30)
    print(f"EVALUATION RESULTS ({num_games} games)")
    print("=" * 30)
    print(f"Win Rate:      {win_rate:.1f}%")
    print(f"Wins:          {wins}")
    print(f"Losses:        {losses}")
    print(f"Draws:         {draws}")
    print(f"Avg Reward:    {avg_reward:.2f}")
    print(f"Avg Turns:     {avg_turns:.1f}")
    print("=" * 30)


if __name__ == "__main__":
    MODEL_PATH = "models/PPO/hs_final"

    evaluate_model(MODEL_PATH, num_games=200)
