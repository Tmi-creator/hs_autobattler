import random
from engine.game import Game


def print_player_state(player, name):
    print(f"--- {name} (HP: {player.health}, Gold: {player.gold}, Tier: {player.tavern_tier}) ---")

    board_str = " | ".join([f"{u.card_id} ({u.cur_atk}/{u.cur_hp})" for u in player.board])
    print(f"Board: [{board_str}]")

    hand_str = ", ".join([c.unit.card_id for c in player.hand if c.unit])
    print(f"Hand:  [{hand_str}]")

    store_str = ", ".join([f"{u.card_id}" for u in player.store])
    print(f"Store: [{store_str}]")
    print("-" * 40)


def simple_bot_turn(game, player_idx):
    """
    Простой бот с адаптированной логикой под RL-интерфейс (reward, done, info).
    """
    player = game.players[player_idx]

    while len(player.board) < 7 and len(player.hand) > 0:
        reward, done, info = game.step(player_idx, "PLAY", hand_index=0, insert_index=-1)

        if reward >= 0:
            print(f"[P{player_idx}] PLAY success: {info}")
        else:
            print(f"[P{player_idx}] PLAY failed: {info}")
            break

    while player.gold >= 3:
        if not player.store or (player.gold >= 4 and random.random() < 0.1):
            reward, done, info = game.step(player_idx, "ROLL")
            if reward >= 0:
                print(f"[P{player_idx}] ROLLED tavern")
            continue

        if player.store:
            store_idx = random.randint(0, len(player.store) - 1)
            card_id = player.store[store_idx].card_id

            reward, done, info = game.step(player_idx, "BUY", index=store_idx)

            if reward >= 0:
                print(f"[P{player_idx}] BOUGHT {card_id}")

                if len(player.hand) > 0:
                    r_play, _, i_play = game.step(player_idx, "PLAY", hand_index=len(player.hand) - 1, insert_index=-1)
                    if r_play >= 0:
                        print(f"[P{player_idx}] ...and PLAYED it immediately")
            else:
                break
        else:
            break

    game.step(player_idx, "END_TURN")


def run_simulation():
    print("=== STARTING HS BATTLEGROUNDS SIMULATION ===")
    game = Game()
    max_turns = 30

    while not game.game_over and game.turn_count <= max_turns:
        current_turn = game.turn_count
        print(f"\n>>> TURN {current_turn} <<<")

        hp_p0_start = game.players[0].health
        hp_p1_start = game.players[1].health

        print_player_state(game.players[0], "Player 0")
        simple_bot_turn(game, 0)

        print_player_state(game.players[1], "Player 1")
        simple_bot_turn(game, 1)

        if game.turn_count > current_turn or game.game_over:
            print(f"\n*** COMBAT RESOLVED ***")

            dmg_p0 = hp_p0_start - game.players[0].health
            dmg_p1 = hp_p1_start - game.players[1].health

            if dmg_p0 > 0: print(f"Player 0 took {dmg_p0} damage!")
            if dmg_p1 > 0: print(f"Player 1 took {dmg_p1} damage!")
            if dmg_p0 == 0 and dmg_p1 == 0: print("It's a DRAW!")

        else:
            print("ERROR: Turn did not advance! Check game logic.")
            break

    print("\n=== GAME OVER ===")
    p0 = game.players[0]
    p1 = game.players[1]

    if p0.health <= 0 and p1.health <= 0:
        print("Result: DRAW (Both died)")
    elif p0.health <= 0:
        print(f"Result: Player 1 WINS! (HP: {p1.health})")
    elif p1.health <= 0:
        print(f"Result: Player 0 WINS! (HP: {p0.health})")
    else:
        print("Result: Turn limit reached")


if __name__ == "__main__":
    run_simulation()
