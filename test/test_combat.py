from engine.combat import Combat_Manager
from engine.entities import Unit, Player

board1 = [Unit.create_from_db("103", 1, 1)]
board2 = [Unit.create_from_db("101", 2, 2)]

Player1 = Player(uid=1, board=board1, hand=[], tavern_tier=1)
Player2 = Player(uid=2, board=board2, hand=[], tavern_tier=1)
combat_manager = Combat_Manager()
print(combat_manager.resolve_combat(Player1, Player2))
