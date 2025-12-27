from engine.entities import Player
from engine.pool import CardPool, SpellPool
from engine.tavern import TavernManager

global_pool = CardPool()
spell_pool = SpellPool()
print(f"Cards in Tier 1 pool: {len(global_pool.tiers[1])}")

tavern = TavernManager(global_pool, spell_pool)
p1 = Player(uid=1, board=[], hand=[], tavern_tier=1, gold=10)

print(f"\n--- Turn 1 Start (Gold: {p1.gold}) ---")

tavern.roll_tavern(p1)

print("Store after roll:")
for i, item in enumerate(p1.store):
    if item.unit:
        print(f"[{i}] {item.card_id} (Stats: {item.unit.max_atk}/{item.unit.max_hp})")
    else:
        print(f"[{i}] {item.card_id} (Spell)")

success, msg = tavern.buy_unit(p1, 0)
if success:
    print(f"\nBought unit! Gold left: {p1.gold}")
    print(f"Hand: {[c.unit.card_id if c.unit else c.spell.card_id for c in p1.hand]}")
    print(f"Pool count for Tier 1: {len(global_pool.tiers[1])} (Should decrease)")
else:
    print(f"Error: {msg}")

print("\n--- Rerolling ---")
tavern.roll_tavern(p1)
print("Store after reroll:")
for i, item in enumerate(p1.store):
    print(f"[{i}] {item.card_id}")
print(f"Gold left: {p1.gold}")
