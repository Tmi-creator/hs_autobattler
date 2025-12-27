from engine.entities import Player
from engine.pool import CardPool
from engine.tavern import TavernManager

global_pool = CardPool()
print(f"Cards in Tier 1 pool: {len(global_pool.tiers[1])}")

tavern = TavernManager(global_pool)
p1 = Player(uid=1, board=[], hand=[], tavern_tier=1, gold=10)

print(f"\n--- Turn 1 Start (Gold: {p1.gold}) ---")

tavern.roll_tavern(p1)

print("Store after roll:")
for i, u in enumerate(p1.store):
    print(f"[{i}] {u.card_id} (Stats: {u.max_atk}/{u.max_hp})")

success, msg = tavern.buy_unit(p1, 0)
if success:
    print(f"\nBought unit! Gold left: {p1.gold}")
    print(f"Hand: {[c.unit.card_id for c in p1.hand]}")
    print(f"Pool count for Tier 1: {len(global_pool.tiers[1])} (Should decrease)")
else:
    print(f"Error: {msg}")

print("\n--- Rerolling ---")
tavern.roll_tavern(p1)
print("Store after reroll:")
for i, u in enumerate(p1.store):
    print(f"[{i}] {u.card_id}")
print(f"Gold left: {p1.gold}")
