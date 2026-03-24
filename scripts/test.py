from typing import Protocol, cast

import numpy as np
from sb3_contrib import MaskablePPO
from tqdm import tqdm

from hearthstone.env.hs_env import HearthstoneEnv


class SupportsPPOLoad(Protocol):
    @classmethod
    def load(cls, path: str) -> MaskablePPO: ...


def evaluate(model_path: str, num_games: int = 1000) -> None:
    print(f"Loading model from: {model_path}")

    env = HearthstoneEnv()

    # Грузим модель
    try:
        ppo_loader = cast(SupportsPPOLoad, MaskablePPO)
        model = ppo_loader.load(model_path)
    except FileNotFoundError:
        print("❌ Model not found! Check path.")
        return

    print(f"\n🥊 STARTING BATTLE ROYALE: {num_games} GAMES")

    wins = 0
    losses = 0
    draws = 0

    # Храним историю ходов, чтобы понять, насколько быстро он убивает
    turn_counts: list[int] = []

    for _ in tqdm(range(num_games)):
        obs, _ = env.reset()
        done = False
        truncated = False

        while not done and not truncated:
            action_masks = np.asarray(env.action_masks(), dtype=bool)

            # deterministic=True заставляет агента играть "лучший" ход, а не пробовать варианты
            action, _ = model.predict(obs, action_masks=action_masks, deterministic=True)

            obs, _, done, truncated, _ = env.step(int(action))

        p0_hp = env.game.players[0].health
        p1_hp = env.game.players[1].health

        turn_counts.append(env.game.turn_count)

        if p0_hp > 0 and p1_hp <= 0:
            wins += 1
        elif p0_hp <= 0 and p1_hp > 0:
            losses += 1
        else:
            draws += 1  # infinite swap / both died

    # --- STATS ---
    win_rate = (wins / num_games) * 100
    avg_turns = np.mean(turn_counts)

    print("\n" + "=" * 40)
    print(f"📊 FINAL RESULTS ({num_games} games)")
    print("=" * 40)
    print(f"🏆 Win Rate:   {win_rate:.2f}%")
    print(f"✅ Wins:       {wins}")
    print(f"❌ Losses:     {losses}")
    print(f"🤝 Draws:      {draws}")
    print(f"⏱️ Avg Turns:  {avg_turns:.1f}")
    print("=" * 40)

    if win_rate > 95.0:
        print("🤖 VERDICT: DOMINATOR (Ready for Tier 2)")
    elif win_rate > 80.0:
        print("🤔 VERDICT: GOOD, BUT NOT PERFECT")
    else:
        print("💩 VERDICT: NEEDS MORE TRAINING")


if __name__ == "__main__":
    # models/run_id/hs_final
    MODEL_PATH = "models/xctm4xer/hs_final"

    evaluate(MODEL_PATH)
