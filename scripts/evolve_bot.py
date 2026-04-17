"""(mu+lambda)-ES trainer for the parametric ES bot v2.

Evolves a weight vector (~22 floats) for ``es_bot.es_bot_turn`` by playing
tournaments between candidate individuals.

Usage:
    python scripts/evolve_bot.py --generations 50
    python scripts/evolve_bot.py --generations 100 --mu 25 --lambda 25
    python scripts/evolve_bot.py --generations 5 --mu 4 --lambda 4 --quick
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

# Make the C++ engine importable in both the main process and spawned workers.
# On Windows we also need to register the MinGW DLL directory before import.
_CPP_BUILD = ROOT / "cpp" / "build"
if _CPP_BUILD.exists() and str(_CPP_BUILD) not in sys.path:
    sys.path.insert(0, str(_CPP_BUILD))
if os.name == "nt":
    _MINGW_BIN = r"C:\msys64\mingw64\bin"
    if os.path.isdir(_MINGW_BIN):
        try:
            os.add_dll_directory(_MINGW_BIN)  # type: ignore[attr-defined]
        except (OSError, AttributeError):
            pass

from hearthstone.engine.game import Game
from hearthstone.env.es_bot import (
    N_WEIGHTS,
    es_bot_turn,
    save_weights,
)
from hearthstone.env.smart_bot import smart_bot_turn


# ============================================================
# Headless match driver — no Gym wrapper, no obs encoding
# ============================================================

MAX_TURNS = 40


def _play_one_side(game: Game, p_idx: int, bot_fn) -> None:
    """bot_fn(game, p_idx) drives one side's tavern turn."""
    try:
        bot_fn(game, p_idx)
    except Exception:
        # Fatal bot error → forfeit: mark player ready and end turn.
        if not game.players_ready.get(p_idx, False):
            try:
                game.step(p_idx, "END_TURN")
            except Exception:
                pass


def play_match(
    bot_a_fn,
    bot_b_fn,
    seed: int,
    max_tier: int = 6,
) -> int:
    """Return 1 if A wins, -1 if B wins, 0 on draw/timeout."""
    random.seed(seed)
    np.random.seed(seed & 0xFFFFFFFF)
    game = Game(max_tier=max_tier)

    turns = 0
    while not game.game_over and turns < MAX_TURNS:
        _play_one_side(game, 0, bot_a_fn)
        _play_one_side(game, 1, bot_b_fn)
        turns += 1

    if game.winner_id == 0:
        return 1
    if game.winner_id == 1:
        return -1
    # Timeout / no winner → compare remaining HP
    h0 = game.players[0].health
    h1 = game.players[1].health
    if h0 > h1:
        return 1
    if h1 > h0:
        return -1
    return 0


def make_es_bot_fn(weights: np.ndarray):
    def fn(game: Game, p_idx: int) -> None:
        es_bot_turn(game, p_idx, weights)
    return fn


def smart_bot_fn(game: Game, p_idx: int) -> None:
    smart_bot_turn(game, p_idx)


# ============================================================
# Multiprocessing workers (module-level for picklability)
# ============================================================
#
# Windows defaults to spawn-start workers, which re-import this module. Worker
# functions must be at module level (no closures) because the pool serializes
# them by qualified name. All state goes through arguments.

def _worker_match_es_es(args: tuple) -> Tuple[int, int, int]:
    """Play one ES-vs-ES match.

    args = (idx_i, idx_j, w_a, w_b, seed, max_tier).
    Returns (idx_i, idx_j, result_for_i).
    """
    idx_i, idx_j, w_a, w_b, seed, max_tier = args
    fn_a = make_es_bot_fn(w_a)
    fn_b = make_es_bot_fn(w_b)
    result = play_match(fn_a, fn_b, seed=seed, max_tier=max_tier)
    return (idx_i, idx_j, result)


def _worker_match_es_smart(args: tuple) -> Tuple[int, int]:
    """Play one ES-vs-Smart anchor match.

    args = (idx, w, seed, max_tier).
    Returns (idx, result_for_es).
    """
    idx, w, seed, max_tier = args
    fn = make_es_bot_fn(w)
    result = play_match(fn, smart_bot_fn, seed=seed, max_tier=max_tier)
    return (idx, result)


def _worker_match_es_hof(args: tuple) -> Tuple[int, int]:
    """Play one ES-vs-HallOfFame match.

    args = (idx, w_es, w_hof, seed, max_tier).
    Returns (idx, result_for_es).
    """
    idx, w_es, w_hof, seed, max_tier = args
    fn_es = make_es_bot_fn(w_es)
    fn_hof = make_es_bot_fn(w_hof)
    result = play_match(fn_es, fn_hof, seed=seed, max_tier=max_tier)
    return (idx, result)


# ============================================================
# (μ+λ)-ES individual
# ============================================================

@dataclass
class Individual:
    weights: np.ndarray  # shape (N,)
    sigmas: np.ndarray   # shape (N,)
    fitness: float = 0.0
    played: int = 0
    wins: float = 0.0

    def reset_score(self) -> None:
        self.fitness = 0.0
        self.played = 0
        self.wins = 0.0


# ============================================================
# Hall of Fame — archive of past champions
# ============================================================

class HallOfFame:
    """Stores weight snapshots of past best individuals.

    Opponents sampled from HoF force the population to stay robust against
    diverse strategies, not overfit to the current generation's meta.
    """

    def __init__(self, max_size: int = 50) -> None:
        self.entries: List[np.ndarray] = []
        self.max_size = max_size

    def add(self, weights: np.ndarray) -> None:
        self.entries.append(weights.copy())
        if len(self.entries) > self.max_size:
            # Keep every other entry (thin out oldest)
            self.entries = self.entries[::2] + self.entries[-1:]

    def sample(self, rng: random.Random) -> np.ndarray:
        return rng.choice(self.entries)

    @property
    def size(self) -> int:
        return len(self.entries)


def random_individual(n: int, rng: random.Random) -> Individual:
    w = np.array([rng.gauss(0.0, 1.0) for _ in range(n)], dtype=np.float32)
    s = np.full(n, 0.3, dtype=np.float32)
    return Individual(weights=w, sigmas=s)


def mutate(parent: Individual, rng: random.Random) -> Individual:
    n = parent.weights.shape[0]
    tau = 1.0 / math.sqrt(2.0 * math.sqrt(n))
    tau_prime = 1.0 / math.sqrt(2.0 * n)

    global_noise = rng.gauss(0.0, 1.0)
    eps_sigma = np.array([rng.gauss(0.0, 1.0) for _ in range(n)], dtype=np.float32)
    new_sigmas = parent.sigmas * np.exp(
        tau_prime * global_noise + tau * eps_sigma
    ).astype(np.float32)
    new_sigmas = np.clip(new_sigmas, 1e-3, 3.0)

    eps_w = np.array([rng.gauss(0.0, 1.0) for _ in range(n)], dtype=np.float32)
    new_weights = parent.weights + new_sigmas * eps_w

    return Individual(weights=new_weights.astype(np.float32), sigmas=new_sigmas)


def crossover(
    parent_a: Individual, parent_b: Individual, rng: random.Random
) -> Individual:
    """Uniform crossover: each weight independently from parent A or B."""
    n = parent_a.weights.shape[0]
    mask = np.array([rng.random() < 0.5 for _ in range(n)], dtype=bool)
    child_w = np.where(mask, parent_a.weights, parent_b.weights)
    child_s = np.where(mask, parent_a.sigmas, parent_b.sigmas)
    return Individual(weights=child_w.astype(np.float32), sigmas=child_s.astype(np.float32))


# ============================================================
# Tournament
# ============================================================

def sample_pairings(
    pop_size: int, n_per_agent: int, rng: random.Random
) -> List[Tuple[int, int]]:
    """Sample unique (i, j) pairs so each agent plays ≈ n_per_agent games."""
    target_total = pop_size * n_per_agent // 2
    seen: set[tuple[int, int]] = set()
    pairs: List[Tuple[int, int]] = []
    attempts = 0
    while len(pairs) < target_total and attempts < target_total * 20:
        i = rng.randrange(pop_size)
        j = rng.randrange(pop_size)
        if i == j:
            attempts += 1
            continue
        key = (min(i, j), max(i, j))
        if key in seen:
            attempts += 1
            continue
        seen.add(key)
        pairs.append((i, j))
        attempts += 1
    return pairs


def _tally_result(ind_i: Individual, ind_j: Individual, result: int) -> None:
    if result == 1:
        ind_i.wins += 1
    elif result == -1:
        ind_j.wins += 1
    else:
        ind_i.wins += 0.5
        ind_j.wins += 0.5
    ind_i.played += 1
    ind_j.played += 1


def evaluate_population(
    population: List[Individual],
    n_match: int,
    n_anchor: int,
    generation: int,
    rng: random.Random,
    executor: ProcessPoolExecutor | None = None,
    max_tier: int = 6,
    hof: HallOfFame | None = None,
    n_hof: int = 4,
) -> None:
    """Play random pairings + anchor (Smart Bot) + Hall of Fame games.

    If ``executor`` is provided, matches run in parallel across workers. With
    ``executor=None`` this degrades to the sequential path (used in tests).
    """
    for ind in population:
        ind.reset_score()

    # --- 1. Build match job lists ---
    pairs = sample_pairings(len(population), n_match, rng)
    es_jobs = []
    for (i, j) in pairs:
        seed = abs(hash((generation, i, j))) & 0xFFFFFFFF
        es_jobs.append((
            i, j,
            population[i].weights, population[j].weights,
            seed, max_tier,
        ))

    anchor_jobs = []
    for i, ind in enumerate(population):
        for k in range(n_anchor):
            seed = abs(hash((generation, "anchor", i, k))) & 0xFFFFFFFF
            anchor_jobs.append((i, ind.weights, seed, max_tier))

    # Hall of Fame games — each individual plays vs random HoF members
    hof_jobs = []
    if hof is not None and hof.size > 0:
        for i, ind in enumerate(population):
            for k in range(n_hof):
                hof_w = hof.sample(rng)
                seed = abs(hash((generation, "hof", i, k))) & 0xFFFFFFFF
                hof_jobs.append((i, ind.weights, hof_w, seed, max_tier))

    # --- 2. Helper to tally anchor/hof result for individual i ---
    def _tally_solo(i: int, result: int) -> None:
        if result == 1:
            population[i].wins += 1
        elif result == 0:
            population[i].wins += 0.5
        population[i].played += 1

    # --- 3. Dispatch ---
    if executor is not None:
        for (i, j, result) in executor.map(
            _worker_match_es_es, es_jobs, chunksize=4
        ):
            _tally_result(population[i], population[j], result)

        for (i, result) in executor.map(
            _worker_match_es_smart, anchor_jobs, chunksize=4
        ):
            _tally_solo(i, result)

        if hof_jobs:
            for (i, result) in executor.map(
                _worker_match_es_hof, hof_jobs, chunksize=4
            ):
                _tally_solo(i, result)
    else:
        bot_fns = [make_es_bot_fn(ind.weights) for ind in population]
        for (i, j, _wa, _wb, seed, _mt) in es_jobs:
            result = play_match(
                bot_fns[i], bot_fns[j], seed=seed, max_tier=max_tier
            )
            _tally_result(population[i], population[j], result)
        for (i, _w, seed, _mt) in anchor_jobs:
            result = play_match(
                bot_fns[i], smart_bot_fn, seed=seed, max_tier=max_tier
            )
            _tally_solo(i, result)
        for (i, _w_es, w_hof, seed, _mt) in hof_jobs:
            fn_hof = make_es_bot_fn(w_hof)
            result = play_match(bot_fns[i], fn_hof, seed=seed, max_tier=max_tier)
            _tally_solo(i, result)

    for ind in population:
        ind.fitness = ind.wins / max(1, ind.played)


def _board_power(game: Game, p_idx: int) -> float:
    """Compute board power using the same formula as HearthstoneEnv."""
    import math
    power = 0.0
    for unit in game.players[p_idx].board:
        u_score = unit.cur_atk * 1.0 + unit.cur_hp * 0.8
        if unit.has_divine_shield:
            u_score += unit.cur_atk * 1.0 + 5.0
        if unit.has_poisonous or unit.has_venomous:
            poison_value = 30.0
            if unit.cur_atk < poison_value:
                u_score += poison_value - unit.cur_atk
        if unit.has_windfury:
            u_score += unit.cur_atk * 0.7
        if unit.has_reborn:
            u_score += unit.base_atk * 0.8 + 1.0
        if unit.has_cleave:
            u_score += unit.cur_atk * 1.0
        power += math.sqrt(u_score)
    return power


def anchor_winrate(
    ind: Individual,
    n_games: int,
    generation: int,
) -> float:
    bot = make_es_bot_fn(ind.weights)
    wins = 0.0
    for k in range(n_games):
        seed = abs(hash((generation, "ar_anchor", k))) & 0xFFFFFFFF
        result = play_match(bot, smart_bot_fn, seed=seed)
        if result == 1:
            wins += 1
        elif result == 0:
            wins += 0.5
    return wins / n_games


def detailed_eval(
    ind: Individual,
    n_games: int,
    generation: int,
    max_tier: int = 6,
) -> dict:
    """Detailed evaluation of best individual vs Smart Bot.

    Returns dict with winrate, avg/max board_power, avg turns, avg tier.
    """
    bot = make_es_bot_fn(ind.weights)
    wins = 0.0
    board_powers: list[float] = []
    max_tiers: list[int] = []
    game_lengths: list[int] = []

    for k in range(n_games):
        seed = abs(hash((generation, "detailed", k))) & 0xFFFFFFFF
        random.seed(seed)
        np.random.seed(seed & 0xFFFFFFFF)
        game = Game(max_tier=max_tier)

        turns = 0
        max_bp = 0.0
        while not game.game_over and turns < MAX_TURNS:
            bot(game, 0)
            smart_bot_fn(game, 1)
            bp = _board_power(game, 0)
            if bp > max_bp:
                max_bp = bp
            turns += 1

        board_powers.append(max_bp)
        max_tiers.append(game.players[0].tavern_tier)
        game_lengths.append(turns)

        if game.winner_id == 0:
            wins += 1
        elif game.winner_id is None:
            h0 = game.players[0].health
            h1 = game.players[1].health
            if h0 > h1:
                wins += 1
            elif h0 == h1:
                wins += 0.5

    return {
        "winrate": wins / n_games,
        "avg_board_power": float(np.mean(board_powers)),
        "max_board_power": float(np.max(board_powers)),
        "avg_max_tier": float(np.mean(max_tiers)),
        "avg_game_length": float(np.mean(game_lengths)),
        "n_games": n_games,
    }


# ============================================================
# Main loop
# ============================================================

def _maybe_init_wandb(args: argparse.Namespace, n_weights: int):
    """Initialize a wandb run if enabled. Returns the run or None."""
    if not getattr(args, "wandb", False):
        return None
    try:
        import wandb
    except ImportError:
        print("[wandb] package not installed; running without logging")
        return None
    run = wandb.init(
        project=getattr(args, "wandb_project", "hs_autobattler_es"),
        name=getattr(args, "wandb_name", None) or f"es_{args.seed}",
        config={
            "n_weights": n_weights,
            "mu": args.mu,
            "lam": args.lam,
            "generations": args.generations,
            "n_match": args.n_match,
            "n_anchor": args.n_anchor,
            "max_tier": args.max_tier,
            "seed": args.seed,
            "workers": args.workers,
        },
    )
    return run


def run_evolution(args: argparse.Namespace) -> None:
    n_weights = N_WEIGHTS

    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    wandb_run = _maybe_init_wandb(args, n_weights)

    # Resolve worker count. 0 or negative → sequential (for tests/debug).
    n_workers = args.workers
    if n_workers is None:
        n_workers = max(1, (os.cpu_count() or 2) - 1)

    hof_interval = getattr(args, "hof_interval", 5)
    n_hof = getattr(args, "n_hof", 4)

    print(f"[evolve] n_weights={n_weights} "
          f"mu={args.mu} lam={args.lam} gens={args.generations}")
    print(f"[evolve] n_match={args.n_match} n_anchor={args.n_anchor} "
          f"n_hof={n_hof} hof_interval={hof_interval} "
          f"workers={n_workers} out_dir={out_dir}")

    # --- Init population ---
    parents = [random_individual(n_weights, rng) for _ in range(args.mu)]
    hof = HallOfFame(max_size=50)

    best_fitness_history: List[float] = []
    all_time_best: Individual | None = None

    executor: ProcessPoolExecutor | None = None
    if n_workers > 1:
        executor = ProcessPoolExecutor(max_workers=n_workers)

    try:
        for gen in range(args.generations):
            gen_start = time.time()

            # --- Generate offspring ---
            offspring: List[Individual] = []
            for _ in range(args.lam):
                parent = rng.choice(parents)
                offspring.append(mutate(parent, rng))

            population = parents + offspring

            # --- Evaluate ---
            evaluate_population(
                population,
                args.n_match,
                args.n_anchor,
                gen,
                rng,
                executor=executor,
                max_tier=args.max_tier,
                hof=hof,
                n_hof=n_hof,
            )

            # --- Select top-μ ---
            population.sort(key=lambda ind: ind.fitness, reverse=True)
            parents = population[: args.mu]

            best = parents[0]
            fit_best = best.fitness

            # --- Elitism: guarantee all-time best survives ---
            if all_time_best is None or fit_best > all_time_best.fitness:
                all_time_best = Individual(
                    weights=best.weights.copy(),
                    sigmas=best.sigmas.copy(),
                    fitness=fit_best,
                )
            elif all_time_best is not None:
                # Inject all-time best as last parent (displaces weakest)
                parents[-1] = Individual(
                    weights=all_time_best.weights.copy(),
                    sigmas=all_time_best.sigmas.copy(),
                )

            # --- Hall of Fame: add best every K generations ---
            if gen % hof_interval == 0:
                hof.add(best.weights)
            fit_mean = float(np.mean([ind.fitness for ind in population]))
            sigma_mean = float(
                np.mean([float(np.mean(ind.sigmas)) for ind in parents])
            )
            best_fitness_history.append(fit_best)

            dt = time.time() - gen_start
            # Win-rate of parents (human-readable metric alongside ELO)
            parents_wr = float(np.mean([
                ind.wins / max(1, ind.played) for ind in parents
            ]))

            print(
                f"[gen {gen:3d}] best={fit_best:.3f} mean={fit_mean:.3f} "
                f"parents_wr={parents_wr:.3f} "
                f"sigma={sigma_mean:.3f} hof={hof.size} time={dt:.1f}s"
            )

            # --- Periodic benchmark vs SmartBot (the REAL metric) ---
            bench_interval = getattr(args, "bench_interval", 20)
            log_data = {
                "gen": gen,
                "fitness/best": fit_best,
                "fitness/population_mean": fit_mean,
                "fitness/parents_mean": parents_wr,
                "sigma_mean": sigma_mean,
                "time/gen_seconds": dt,
            }
            if gen % bench_interval == 0 or gen == args.generations - 1:
                bench = detailed_eval(
                    best, n_games=50, generation=gen, max_tier=args.max_tier,
                )
                log_data["benchmark/vs_smart_winrate"] = bench["winrate"]
                log_data["benchmark/avg_board_power"] = bench["avg_board_power"]
                log_data["benchmark/max_board_power"] = bench["max_board_power"]
                log_data["benchmark/avg_max_tier"] = bench["avg_max_tier"]
                print(
                    f"  [bench] vs SmartBot: wr={bench['winrate']:.3f} "
                    f"avg_bp={bench['avg_board_power']:.1f} "
                    f"max_bp={bench['max_board_power']:.1f} "
                    f"tier={bench['avg_max_tier']:.1f}"
                )

            if wandb_run is not None:
                wandb_run.log(log_data, step=gen)

            # --- Checkpoint ---
            save_weights(
                str(out_dir / f"gen_{gen:03d}.npz"),
                best.weights,
                sigmas=best.sigmas,
                fitness=fit_best,
                generation=gen,
            )

        # --- Final best ---
        save_weights(
            str(out_dir / "best.npz"),
            parents[0].weights,
            sigmas=parents[0].sigmas,
            fitness=parents[0].fitness,
            generation=args.generations - 1,
        )
        np.save(
            out_dir / "fitness_history.npy",
            np.array(best_fitness_history, dtype=np.float32),
        )

        # --- Detailed eval vs Smart Bot ---
        if args.final_eval > 0:
            stats = detailed_eval(
                parents[0], args.final_eval, args.generations,
                max_tier=args.max_tier,
            )
            print(
                f"[final] vs Smart Bot ({stats['n_games']} games):\n"
                f"  winrate:         {stats['winrate']:.3f}\n"
                f"  avg_board_power: {stats['avg_board_power']:.1f}\n"
                f"  max_board_power: {stats['max_board_power']:.1f}\n"
                f"  avg_max_tier:    {stats['avg_max_tier']:.1f}\n"
                f"  avg_game_length: {stats['avg_game_length']:.1f} turns"
            )
            if wandb_run is not None:
                for k, v in stats.items():
                    wandb_run.summary[f"final/{k}"] = v

        if wandb_run is not None:
            # Upload the best weights as a wandb artifact so the run is
            # self-contained (weights + fitness curve visible in the UI).
            import wandb  # local import to keep the optional dep optional
            artifact = wandb.Artifact(
                name="es_best", type="model",
                metadata={
                    "generations": args.generations,
                    "fitness": float(parents[0].fitness),
                },
            )
            artifact.add_file(str(out_dir / "best.npz"))
            artifact.add_file(str(out_dir / "fitness_history.npy"))
            wandb_run.log_artifact(artifact)
            wandb_run.finish()

        print(f"[evolve] done. Best weights -> {out_dir / 'best.npz'}")
    finally:
        if executor is not None:
            executor.shutdown(wait=True)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--generations", type=int, default=50)
    p.add_argument("--mu", type=int, default=25)
    p.add_argument("--lam", "--lambda", dest="lam", type=int, default=25)
    p.add_argument("--n-match", type=int, default=20,
                   help="games per agent inside intra-pop pairings")
    p.add_argument("--n-anchor", type=int, default=4,
                   help="games per agent vs Smart Bot anchor")
    p.add_argument("--final-eval", type=int, default=50,
                   help="extra vs-SmartBot games for best individual at end (0=skip)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default="artifacts/es_bot")
    p.add_argument("--workers", type=int, default=None,
                   help="parallel match workers (default: cpu_count-1, 0/1 = sequential)")
    p.add_argument("--max-tier", type=int, default=6,
                   help="tavern max tier for simulated games")
    p.add_argument("--wandb", action="store_true",
                   help="log per-generation metrics to Weights & Biases")
    p.add_argument("--wandb-project", default="hs_autobattler_es")
    p.add_argument("--wandb-name", default=None,
                   help="wandb run name (default: es_{preset}_{seed})")
    p.add_argument("--n-hof", type=int, default=4,
                   help="games per agent vs Hall of Fame members")
    p.add_argument("--hof-interval", type=int, default=5,
                   help="add best to Hall of Fame every N generations")
    p.add_argument("--bench-interval", type=int, default=20,
                   help="run detailed eval vs SmartBot every N generations")
    p.add_argument("--quick", action="store_true",
                   help="shortcut: mu=4 lam=4 n_match=6 n_anchor=2 final-eval=20")
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.quick:
        args.mu = 4
        args.lam = 4
        args.n_match = 6
        args.n_anchor = 2
        args.final_eval = 20

    run_evolution(args)


if __name__ == "__main__":
    main()
