"""
PvP Evaluation: MLP vs Transformer.

Загружает обе модели и сталкивает их друг с другом.
Одна модель управляет агентом, другая — оппонентом (через set_opponent).
Проводит N игр в каждую сторону и выводит статистику.

Использование:
    python scripts/evaluate_pvp.py
    python scripts/evaluate_pvp.py --mlp path/to/mlp.zip --transformer path/to/trans.zip
    python scripts/evaluate_pvp.py --games 200
"""

import argparse
import sys
from pathlib import Path
from typing import cast

import numpy as np
from sb3_contrib import MaskablePPO

from hearthstone.env.hs_env import HearthstoneEnv

root_path = Path(__file__).resolve().parent.parent
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))


# === CONSTANTS ===
OUTPUTS_DIR = Path(__file__).resolve().parent / "outputs"
DEFAULT_MLP_PATH = str(OUTPUTS_DIR / "models" / "mlp_final.zip")
DEFAULT_TRANS_PATH = str(OUTPUTS_DIR / "models" / "transformer_final.zip")
PVP_GAMES = 200  # кол-во игр в каждую сторону


def run_match(
    agent: MaskablePPO,
    opponent: MaskablePPO,
    n_games: int,
    seed: int = 42,
) -> dict[str, float]:
    """
    Прогоняет n_games игр: agent управляет игроком, opponent — оппонентом.
    Возвращает статистику.
    """
    env = HearthstoneEnv()
    wins = 0
    losses = 0
    draws = 0
    total_hp_diff: list[float] = []
    total_agent_power: list[float] = []

    for game_idx in range(n_games):
        obs, _ = env.reset(seed=seed + game_idx)
        env.set_opponent(opponent)
        done = False
        truncated = False

        while not done and not truncated:
            masks = np.asarray(env.action_masks(), dtype=bool)
            action, _ = agent.predict(obs, action_masks=masks, deterministic=True)
            obs, _, done, truncated, _ = env.step(int(action))

        # Результат
        p0 = env.game.players[env.my_player_id]
        p1 = env.game.players[env.enemy_id]
        hp_diff = p0.health - p1.health
        total_hp_diff.append(hp_diff)
        total_agent_power.append(env.get_board_power())

        if p0.health > 0 and p1.health <= 0:
            wins += 1
        elif p0.health <= 0 and p1.health > 0:
            losses += 1
        else:
            draws += 1

    return {
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": wins / n_games * 100,
        "avg_hp_diff": float(np.mean(total_hp_diff)),
        "avg_board_power": float(np.mean(total_agent_power)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="PvP: MLP vs Transformer")
    parser.add_argument(
        "--mlp",
        type=str,
        default=DEFAULT_MLP_PATH,
        help="Path to MLP model",
    )
    parser.add_argument(
        "--transformer",
        type=str,
        default=DEFAULT_TRANS_PATH,
        help="Path to Transformer model",
    )
    parser.add_argument(
        "--games",
        type=int,
        default=PVP_GAMES,
        help="Number of games per matchup",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("PvP Evaluation: MLP vs Transformer")
    print("=" * 60)

    # Загрузка моделей
    print(f"\n[LOAD] MLP: {args.mlp}")
    mlp_model = MaskablePPO.load(args.mlp, device="cpu")
    print(f"[LOAD] Transformer: {args.transformer}")
    trans_model = MaskablePPO.load(args.transformer, device="cpu")

    # === Раунд 1: Transformer (agent) vs MLP (opponent) ===
    print(f"\n[MATCH 1] Transformer vs MLP ({args.games} games)...")
    stats_trans = run_match(trans_model, mlp_model, args.games, seed=100)

    # === Раунд 2: MLP (agent) vs Transformer (opponent) ===
    print(f"[MATCH 2] MLP vs Transformer ({args.games} games)...")
    stats_mlp = run_match(mlp_model, trans_model, args.games, seed=200)

    # === Результаты ===
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"{'Metric':<25} {'Transformer':>15} {'MLP':>15}")
    print("-" * 55)
    print(
        f"{'Win Rate (as agent)':<25} "
        f"{stats_trans['win_rate']:>14.1f}% "
        f"{stats_mlp['win_rate']:>14.1f}%"
    )
    print(
        f"{'Wins / Losses / Draws':<25} "
        f"{stats_trans['wins']:>4}/{stats_trans['losses']:>4}/"
        f"{stats_trans['draws']:>4}  "
        f"{stats_mlp['wins']:>4}/{stats_mlp['losses']:>4}/"
        f"{stats_mlp['draws']:>4}"
    )
    print(
        f"{'Avg HP Diff':<25} "
        f"{stats_trans['avg_hp_diff']:>+14.1f} "
        f"{stats_mlp['avg_hp_diff']:>+14.1f}"
    )
    print(
        f"{'Avg Board Power':<25} "
        f"{stats_trans['avg_board_power']:>14.1f} "
        f"{stats_mlp['avg_board_power']:>14.1f}"
    )
    print("=" * 60)

    # Verdict
    total_trans_wr = (stats_trans["win_rate"] + (100 - stats_mlp["win_rate"])) / 2
    total_mlp_wr = 100 - total_trans_wr
    print(f"\n[OVERALL] Transformer: {total_trans_wr:.1f}% | MLP: {total_mlp_wr:.1f}%")

    if total_trans_wr > 55:
        print("[VERDICT] Transformer is significantly better!")
    elif total_trans_wr > 50:
        print("[VERDICT] Transformer has a slight edge.")
    elif total_trans_wr > 45:
        print("[VERDICT] Results are roughly equal.")
    else:
        print("[VERDICT] MLP performs better (more training needed?).")


if __name__ == "__main__":
    main()
