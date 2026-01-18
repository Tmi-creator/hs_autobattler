from engine.combat import CombatManager
from engine.entities import Unit, Player
from engine.configs import CARD_DB, CardIDs
board1 = [Unit.create_from_db(CardIDs.ANNOY_O_TRON, 1, 1)]
board2 = [Unit.create_from_db(CardIDs.SCALLYWAG, 2, 2)]

Player1 = Player(uid=1, board=board1, hand=[])
Player2 = Player(uid=2, board=board2, hand=[])
combat_manager = CombatManager()
print(combat_manager.resolve_combat(Player1, Player2))
print(combat_manager.resolve_combat(Player1, Player2))