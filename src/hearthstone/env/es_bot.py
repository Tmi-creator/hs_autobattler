"""ES Bot v2 — rule-based priority loop with evolved weights.

Same priority chain as smart_bot (discovery -> triplets -> spells -> upgrade ->
play hand -> buy -> sell+buy -> roll -> end turn) but every numeric threshold
and unit-scoring coefficient comes from a weight vector that evolution tunes.

No deepcopy, no pickle, no lookahead. One game ~ 1-5 ms.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Set

import numpy as np

from hearthstone.engine.configs import CARD_DB
from hearthstone.engine.enums import SpellIDs, UnitType
from hearthstone.engine.spells import SPELLS_REQUIRE_TARGET

if TYPE_CHECKING:
    from hearthstone.engine.entities import Player, Unit
    from hearthstone.engine.game import Game


# ============================================================
# Weight layout (index == position in weight vector)
# ============================================================

WEIGHT_NAMES: List[str] = [
    # --- Unit scoring (0..3) ---
    "w_atk",              # 0
    "w_hp",               # 1
    "w_tier",             # 2
    "w_golden",           # 3
    # --- Keyword values (4..11) ---
    "w_taunt",            # 4
    "w_ds",               # 5
    "w_poison",           # 6
    "w_wf",               # 7
    "w_reborn",           # 8
    "w_dr",               # 9
    "w_cleave",           # 10
    "w_magnetize",        # 11
    # --- Synergy (12..14) ---
    "w_type_match",       # 12  bonus per matching type on board
    "w_pair",             # 13  bonus for 2nd copy (pair)
    "w_triple",           # 14  bonus for 3rd copy (triple-ready)
    # --- Economy (15..18) ---
    "w_upgrade_aggro",    # 15  multiplier: upgrade when gold >= up_cost * this
    "w_roll_threshold",   # 16  roll if best_shop_score < this
    "w_sell_delta",       # 17  sell weakest + buy if shop_score - weak_score > this
    "w_buy_threshold",    # 18  buy if score > this (prevents buying trash)
    # --- Turn scaling (19..20) ---
    "w_early_atk",        # 19  extra atk weight on turns 1-5
    "w_late_hp",          # 20  extra hp weight on turns 6+
    # --- Misc (21..22) ---
    "w_board_full_roll",  # 21  roll aggressiveness when board is full
    "w_buy_empty",        # 22  lower buy threshold when board < 4 (typically negative)
]

N_WEIGHTS: int = len(WEIGHT_NAMES)  # 23

# Weight indices
W_ATK = 0
W_HP = 1
W_TIER = 2
W_GOLDEN = 3
W_TAUNT = 4
W_DS = 5
W_POISON = 6
W_WF = 7
W_REBORN = 8
W_DR = 9
W_CLEAVE = 10
W_MAGNETIZE = 11
W_TYPE_MATCH = 12
W_PAIR = 13
W_TRIPLE = 14
W_UPGRADE_AGGRO = 15
W_ROLL_THRESH = 16
W_SELL_DELTA = 17
W_BUY_THRESH = 18
W_EARLY_ATK = 19
W_LATE_HP = 20
W_BOARD_FULL_ROLL = 21
W_BUY_EMPTY = 22


# ============================================================
# Helpers
# ============================================================

def _card_is_deathrattle(card_id: str) -> bool:
    data = CARD_DB.get(card_id)
    return bool(data and data.get("deathrattle", False))


def _get_board_types(player: "Player") -> Set[UnitType]:
    types: Set[UnitType] = set()
    for u in player.board:
        for t in u.types:
            types.add(t)
    return types


def _get_card_counts(player: "Player") -> dict[str, int]:
    counts: dict[str, int] = {}
    for u in player.board:
        counts[u.card_id] = counts.get(u.card_id, 0) + 1
    for c in player.hand:
        if c.unit:
            counts[c.unit.card_id] = counts.get(c.unit.card_id, 0) + 1
    return counts


def _weakest_board_unit(player: "Player", w: np.ndarray, turn: int) -> tuple[int, float]:
    if not player.board:
        return -1, 0.0
    board_types = _get_board_types(player)
    worst_idx = 0
    worst_score = float("inf")
    for i, u in enumerate(player.board):
        sc = _score_unit_on_board(u, board_types, turn, w)
        if sc < worst_score:
            worst_score = sc
            worst_idx = i
    return worst_idx, worst_score


def _best_board_unit_idx(player: "Player", w: np.ndarray, turn: int) -> int:
    if not player.board:
        return -1
    board_types = _get_board_types(player)
    best_idx = 0
    best_score = -float("inf")
    for i, u in enumerate(player.board):
        sc = _score_unit_on_board(u, board_types, turn, w)
        if sc > best_score:
            best_score = sc
            best_idx = i
    return best_idx


# ============================================================
# Unit scoring
# ============================================================

def score_unit_es(
    card_id: str,
    board_types: Set[UnitType],
    card_counts: dict[str, int],
    turn: int,
    w: np.ndarray,
) -> float:
    """Score a unit from the shop (or discovery) using evolved weights."""
    data = CARD_DB.get(card_id)
    if not data:
        return 0.0

    atk: float = data["atk"]
    hp: float = data["hp"]
    tier: float = data["tier"]

    atk_weight = w[W_ATK] + (w[W_EARLY_ATK] if turn <= 5 else 0.0)
    hp_weight = w[W_HP] + (w[W_LATE_HP] if turn > 5 else 0.0)

    s: float = atk_weight * atk + hp_weight * hp + w[W_TIER] * tier

    # Keywords from card DB tags
    tags = data.get("tags", set())
    from hearthstone.engine.enums import Tags
    if Tags.TAUNT in tags:
        s += w[W_TAUNT]
    if Tags.DIVINE_SHIELD in tags:
        s += w[W_DS]
    if Tags.POISONOUS in tags or Tags.VENOMOUS in tags:
        s += w[W_POISON]
    if Tags.WINDFURY in tags:
        s += w[W_WF]
    if Tags.REBORN in tags:
        s += w[W_REBORN]
    if Tags.CLEAVE in tags:
        s += w[W_CLEAVE]
    if Tags.MAGNETIC in tags:
        s += w[W_MAGNETIZE]

    if data.get("deathrattle", False):
        s += w[W_DR]

    s += w[W_GOLDEN]

    # Tribal synergy
    unit_types = data.get("type", [])
    for t in unit_types:
        if t in board_types:
            s += w[W_TYPE_MATCH]

    # Triple hunting
    copies = card_counts.get(card_id, 0)
    if copies >= 2:
        s += w[W_TRIPLE]
    elif copies == 1:
        s += w[W_PAIR]

    return s


def _score_unit_on_board(
    unit: "Unit",
    board_types: Set[UnitType],
    turn: int,
    w: np.ndarray,
) -> float:
    """Score an already-placed unit (for sell/buff target decisions)."""
    atk_weight = w[W_ATK] + (w[W_EARLY_ATK] if turn <= 5 else 0.0)
    hp_weight = w[W_HP] + (w[W_LATE_HP] if turn > 5 else 0.0)

    s: float = atk_weight * unit.cur_atk + hp_weight * unit.cur_hp + w[W_TIER] * unit.tier

    if unit.is_golden:
        s += w[W_GOLDEN]
    if unit.has_taunt:
        s += w[W_TAUNT]
    if unit.has_divine_shield:
        s += w[W_DS]
    if unit.has_poisonous or unit.has_venomous:
        s += w[W_POISON]
    if unit.has_windfury:
        s += w[W_WF]
    if unit.has_reborn:
        s += w[W_REBORN]
    if unit.has_cleave:
        s += w[W_CLEAVE]
    if unit.has_magnetic:
        s += w[W_MAGNETIZE]

    for t in unit.types:
        if t in board_types:
            s += w[W_TYPE_MATCH]

    return s


# ============================================================
# Bot turn — priority loop (mirrors smart_bot structure)
# ============================================================

def es_bot_turn(
    game: "Game",
    p_idx: int,
    weights: np.ndarray,
    max_actions: int = 40,
) -> None:
    """Execute one full tavern turn using evolved heuristics."""
    player = game.players[p_idx]
    turn = game.turn_count
    w = weights

    for action_n in range(max_actions):
        if game.game_over or game.players_ready.get(p_idx, False):
            return

        # --- Discovery ---
        if player.is_discovering and player.discovery.options:
            board_types = _get_board_types(player)
            card_counts = _get_card_counts(player)
            best_idx = 0
            best_score = -1e9
            for i, opt in enumerate(player.discovery.options):
                if opt.unit:
                    sc = score_unit_es(opt.unit.card_id, board_types, card_counts, turn, w)
                else:
                    sc = 0.0
                if sc > best_score:
                    best_score = sc
                    best_idx = i
            game.step(p_idx, "DISCOVER_CHOICE", index=best_idx)
            continue

        # --- Play Triplet Reward ---
        played_something = False
        for i, card in enumerate(player.hand):
            if card.spell and card.spell.card_id == SpellIDs.TRIPLET_REWARD:
                game.step(p_idx, "PLAY", hand_index=i, insert_index=-1)
                played_something = True
                break
        if played_something:
            continue

        # --- Play buff spells on best unit ---
        if player.board:
            best_target = _best_board_unit_idx(player, w, turn)
            for i, card in enumerate(player.hand):
                if card.spell and card.spell.card_id != SpellIDs.TAVERN_COIN:
                    spell_id = card.spell.card_id
                    if spell_id in SPELLS_REQUIRE_TARGET:
                        if best_target >= 0:
                            game.step(p_idx, "PLAY", hand_index=i, target_index=best_target)
                            played_something = True
                            break
                    elif spell_id != SpellIDs.TRIPLET_REWARD:
                        game.step(p_idx, "PLAY", hand_index=i, insert_index=-1)
                        played_something = True
                        break
        if played_something:
            continue

        # --- Play Coins ---
        for i, card in enumerate(player.hand):
            if card.spell and card.spell.card_id == SpellIDs.TAVERN_COIN:
                game.step(p_idx, "PLAY", hand_index=i, insert_index=-1)
                played_something = True
                break
        if played_something:
            continue

        # --- Upgrade tavern ---
        if player.tavern_tier < 6:
            upgrade_threshold = player.up_cost * max(0.5, w[W_UPGRADE_AGGRO])
            if player.gold >= upgrade_threshold:
                success, _, _ = game.step(p_idx, "UPGRADE")
                if success:
                    continue

        # --- Play all minions from hand ---
        for i, card in enumerate(player.hand):
            if card.unit and len(player.board) < 7:
                game.step(p_idx, "PLAY", hand_index=i, insert_index=-1)
                played_something = True
                break
        if played_something:
            continue

        # --- Score shop and buy best ---
        if player.gold >= 3 and player.store:
            board_types = _get_board_types(player)
            card_counts = _get_card_counts(player)

            best_shop_idx = -1
            best_shop_score = -1e9
            for i, item in enumerate(player.store):
                if item.unit:
                    sc = score_unit_es(item.unit.card_id, board_types, card_counts, turn, w)
                    if sc > best_shop_score:
                        best_shop_score = sc
                        best_shop_idx = i

            # Board-size-aware buy threshold
            buy_thresh = w[W_BUY_THRESH]
            if len(player.board) < 4:
                buy_thresh += w[W_BUY_EMPTY]

            if best_shop_idx >= 0 and best_shop_score > buy_thresh:
                if len(player.board) >= 7:
                    weak_idx, weak_score = _weakest_board_unit(player, w, turn)
                    if best_shop_score > weak_score + w[W_SELL_DELTA]:
                        game.step(p_idx, "SELL", index=weak_idx)
                        game.step(p_idx, "BUY", index=best_shop_idx)
                        continue
                elif len(player.hand) < 10:
                    game.step(p_idx, "BUY", index=best_shop_idx)
                    continue

        # --- Roll if shop is mediocre ---
        if player.gold >= 2 and player.store:
            board_types = _get_board_types(player)
            card_counts = _get_card_counts(player)
            max_shop_score = max(
                (
                    score_unit_es(item.unit.card_id, board_types, card_counts, turn, w)
                    for item in player.store
                    if item.unit
                ),
                default=-1e9,
            )
            roll_thresh = w[W_ROLL_THRESH]
            if len(player.board) >= 7:
                roll_thresh += w[W_BOARD_FULL_ROLL]
            if max_shop_score < roll_thresh and player.gold >= 3:
                game.step(p_idx, "ROLL")
                continue

        break

    if not game.game_over and not game.players_ready.get(p_idx, False):
        game.step(p_idx, "END_TURN")


# ============================================================
# Checkpoint I/O
# ============================================================

def save_weights(
    path: str,
    weights: np.ndarray,
    sigmas: np.ndarray | None = None,
    fitness: float | None = None,
    generation: int | None = None,
) -> None:
    payload: dict = {"weights": weights.astype(np.float32)}
    if sigmas is not None:
        payload["sigmas"] = sigmas.astype(np.float32)
    if fitness is not None:
        payload["fitness"] = np.float32(fitness)
    if generation is not None:
        payload["generation"] = np.int32(generation)
    np.savez(path, **payload)


def load_weights(path: str) -> np.ndarray:
    data = np.load(path, allow_pickle=False)
    return data["weights"].astype(np.float32)
