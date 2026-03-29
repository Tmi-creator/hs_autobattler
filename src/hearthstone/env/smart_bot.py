"""Smart Bot: Score-based heuristic opponent for fast training.

Zero neural inference. Uses a greedy scoring function to make
buy/sell/play decisions based on tribal synergies and triple hunting.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, List, Set

from hearthstone.engine.configs import CARD_DB, TIER_UPGRADE_COSTS
from hearthstone.engine.entities import Player, Unit
from hearthstone.engine.enums import CardIDs, SpellIDs, UnitType
from hearthstone.engine.spells import SPELLS_REQUIRE_TARGET

if TYPE_CHECKING:
    from hearthstone.engine.game import Game

# Standard upgrade curve: (turn, max_gold_to_spend_on_upgrade)
# Turn 2: upgrade if gold >= 5 (costs 4 after discount)
# Turn 5: upgrade if gold >= 7
UPGRADE_TURNS = {3: 5, 5: 7, 7: 9}


def score_unit(
    card_id: str,
    board_types: Set[UnitType],
    board_card_counts: dict[str, int],
    turn: int,
) -> float:
    """Score a unit from the shop for the smart bot."""
    data = CARD_DB.get(card_id)
    if not data:
        return 0.0

    atk = data["atk"]
    hp = data["hp"]
    tier = data["tier"]
    s: float = (atk + hp) + tier * 2.0

    # Triplet hunting
    copies = board_card_counts.get(card_id, 0)
    if copies >= 2:
        s += 100.0  # MUST buy for triple
    elif copies == 1:
        s += 10.0  # pair potential

    # Tribal synergy: +6 per matching tribe on board
    unit_types = data.get("type", [])
    for t in unit_types:
        if t in board_types:
            s += 6.0

    # Specific card bonuses (Tier 1 pool)
    if card_id == CardIDs.WRATH_WEAVER:
        if turn <= 4:
            s += 15.0
    elif card_id == CardIDs.CRACKLING_CYCLONE:
        s += 8.0  # DS + Windfury is strong early
    elif card_id == CardIDs.ANNOY_O_TRON:
        s += 5.0  # DS + Taunt is solid
    elif card_id == CardIDs.ROT_HIDE_GNOLL:
        s += 6.0  # Scales in combat
    elif card_id == CardIDs.MISFIT_DRAGONLING:
        s += 4.0  # SoC buff scales with tier

    return s


def _get_board_types(player: Player) -> Set[UnitType]:
    types: Set[UnitType] = set()
    for u in player.board:
        for t in u.types:
            types.add(t)
    return types


def _get_card_counts(player: Player) -> dict[str, int]:
    """Count copies of each card_id on board + hand."""
    counts: dict[str, int] = {}
    for u in player.board:
        counts[u.card_id] = counts.get(u.card_id, 0) + 1
    for c in player.hand:
        if c.unit:
            cid = c.unit.card_id
            counts[cid] = counts.get(cid, 0) + 1
    return counts


def _weakest_board_unit(player: Player) -> tuple[int, float]:
    """Return (index, power) of weakest board unit."""
    if not player.board:
        return -1, 0.0
    worst_idx = 0
    worst_power = float("inf")
    for i, u in enumerate(player.board):
        power = u.cur_atk + u.cur_hp
        if power < worst_power:
            worst_power = power
            worst_idx = i
    return worst_idx, worst_power


def _best_board_unit_idx(player: Player) -> int:
    """Return index of strongest board unit (for buff spells)."""
    if not player.board:
        return -1
    best_idx = 0
    best_power = 0.0
    for i, u in enumerate(player.board):
        power = u.cur_atk + u.cur_hp
        if u.has_divine_shield:
            power += 10
        if power > best_power:
            best_power = power
            best_idx = i
    return best_idx


def smart_bot_turn(game: Game, p_idx: int) -> None:
    """Execute a full tavern turn for player p_idx using heuristics.

    Priority order:
    1. Handle discovery (pick highest tier/power)
    2. Play Triplet Reward immediately
    3. Play buff spells on strongest unit
    4. Play Coins if needed
    5. Upgrade tavern on curve
    6. Play all minions from hand
    7. Buy best-scoring shop unit
    8. Sell weakest if board full and shop has much better
    9. Roll if gold >= 2 and shop is bad
    """
    player = game.players[p_idx]
    turn = game.turn_count
    max_actions = 40
    actions_taken = 0

    for _ in range(max_actions):
        actions_taken += 1
        if actions_taken > max_actions:
            break

        # --- Discovery ---
        if player.is_discovering and player.discovery.options:
            # Pick highest tier, then highest stats
            best_idx = 0
            best_score = -1.0
            for i, opt in enumerate(player.discovery.options):
                if opt.unit:
                    sc = opt.unit.tier * 10 + opt.unit.cur_atk + opt.unit.cur_hp
                    if sc > best_score:
                        best_score = sc
                        best_idx = i
            game.step(p_idx, "DISCOVER_CHOICE", index=best_idx)
            continue

        # --- Play Triplet Reward from hand ---
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
            best_target = _best_board_unit_idx(player)
            for i, card in enumerate(player.hand):
                if card.spell and card.spell.card_id != SpellIDs.TAVERN_COIN:
                    spell_id = card.spell.card_id
                    if spell_id in SPELLS_REQUIRE_TARGET:
                        if best_target >= 0:
                            game.step(
                                p_idx,
                                "PLAY",
                                hand_index=i,
                                target_index=best_target,
                            )
                            played_something = True
                            break
                    elif spell_id != SpellIDs.TRIPLET_REWARD:
                        game.step(
                            p_idx,
                            "PLAY",
                            hand_index=i,
                            insert_index=-1,
                        )
                        played_something = True
                        break
        if played_something:
            continue

        # --- Play Coins if we need gold ---
        for i, card in enumerate(player.hand):
            if card.spell and card.spell.card_id == SpellIDs.TAVERN_COIN:
                game.step(p_idx, "PLAY", hand_index=i, insert_index=-1)
                played_something = True
                break
        if played_something:
            continue

        # --- Upgrade tavern on curve ---
        if player.tavern_tier < 6:
            should_upgrade = False
            # Standard curve
            if turn in UPGRADE_TURNS and player.gold >= UPGRADE_TURNS[turn]:
                should_upgrade = True
            # Also upgrade if board is full and we have gold
            if len(player.board) >= 6 and player.gold >= player.up_cost:
                should_upgrade = True
            if should_upgrade:
                success, _, _ = game.step(p_idx, "UPGRADE")
                if success:
                    continue

        # --- Play all minions from hand ---
        for i, card in enumerate(player.hand):
            if card.unit and len(player.board) < 7:
                game.step(
                    p_idx, "PLAY", hand_index=i, insert_index=-1
                )
                played_something = True
                break
        if played_something:
            continue

        # --- Score shop and buy best ---
        if player.gold >= 3 and player.store:
            board_types = _get_board_types(player)
            card_counts = _get_card_counts(player)

            best_shop_idx = -1
            best_shop_score = -1.0
            for i, item in enumerate(player.store):
                if item.unit:
                    sc = score_unit(
                        item.unit.card_id, board_types, card_counts, turn
                    )
                    if sc > best_shop_score:
                        best_shop_score = sc
                        best_shop_idx = i

            if best_shop_idx >= 0:
                # If board full, consider selling weakest
                if len(player.board) >= 7:
                    weak_idx, weak_power = _weakest_board_unit(player)
                    # Only sell+buy if shop unit is much better
                    if best_shop_score > weak_power + 15:
                        game.step(p_idx, "SELL", index=weak_idx)
                        game.step(p_idx, "BUY", index=best_shop_idx)
                        continue
                elif len(player.hand) < 10:
                    game.step(p_idx, "BUY", index=best_shop_idx)
                    continue

        # --- Roll if gold >= 2 and shop is mediocre ---
        if player.gold >= 2 and player.store:
            board_types = _get_board_types(player)
            card_counts = _get_card_counts(player)
            max_score = max(
                (
                    score_unit(
                        item.unit.card_id, board_types, card_counts, turn
                    )
                    for item in player.store
                    if item.unit
                ),
                default=0,
            )
            if max_score < 8 and player.gold >= 3:
                game.step(p_idx, "ROLL")
                continue

        # Nothing useful to do
        break

    game.step(p_idx, "END_TURN")
