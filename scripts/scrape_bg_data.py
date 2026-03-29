"""Scrape Hearthstone Battlegrounds card data from HearthstoneJSON API.

Downloads the full card database for a specific game build,
filters ALL BG-relevant cards and saves them as frozen JSON files.

Categories scraped:
  - Pool minions (isBattlegroundsPoolMinion)
  - Heroes (battlegroundsHero)
  - Hero powers (linked via heroPowerDbfId)
  - Tavern spells (isBattlegroundsPoolSpell + BATTLEGROUND_SPELL type)
  - Trinkets (BATTLEGROUND_TRINKET type)
  - Anomalies (BATTLEGROUND_ANOMALY type)
  - Quest rewards (BATTLEGROUND_QUEST_REWARD type)
  - Tokens (referenced by pool minions via battlegroundsRelatedCard)
  - Buddies (isBattlegroundsBuddy)

Usage:
    python scripts/scrape_bg_data.py                    # default build
    python scripts/scrape_bg_data.py --build 235290     # specific build
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

# --- Config ---
DEFAULT_BUILD = 234747  # patch before 34.6 (no anomalies)
API_URL = "https://api.hearthstonejson.com/v1/{build}/enUS/cards.json"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data"

# BG minion race mapping (hearthstonejson uses these strings)
RACE_MAP = {
    "BEAST": "Beast",
    "DEMON": "Demon",
    "DRAGON": "Dragon",
    "ELEMENTAL": "Elemental",
    "MECH": "Mech",
    "MECHANICAL": "Mech",
    "MURLOC": "Murloc",
    "PIRATE": "Pirate",
    "QUILBOAR": "Quilboar",
    "UNDEAD": "Undead",
    "NAGA": "Naga",
    "ALL": "All",
}


# ============================================================
# Fetching
# ============================================================

def fetch_cards(build: int) -> list[dict]:
    """Download cards.json for a specific build."""
    url = API_URL.format(build=build)
    print(f"[*] Fetching {url} ...")
    req = Request(url, headers={"User-Agent": "hs_autobattler/1.0"})
    try:
        with urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except URLError as e:
        print(f"[!] Failed to fetch: {e}")
        sys.exit(1)
    print(f"[+] Downloaded {len(data)} cards total")
    return data


# ============================================================
# Helpers
# ============================================================

def extract_mechanics(card: dict) -> list[str]:
    """Extract mechanic tags from a card."""
    return sorted(card.get("mechanics", []))


def extract_race(card: dict) -> str | list[str] | None:
    """Extract minion race/type. Prefers `races` list (handles dual-type)."""
    races = card.get("races")
    if races:
        mapped = [RACE_MAP.get(r, r) for r in races]
        return mapped if len(mapped) > 1 else mapped[0]
    race = card.get("race")
    if race:
        return RACE_MAP.get(race, race)
    return None


def clean_text(text: str | None) -> str | None:
    """Remove HTML tags from card text."""
    if not text:
        return None
    cleaned = re.sub(r"<[^>]+>", "", text)
    cleaned = cleaned.replace("\n", " ").strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned if cleaned else None


# ============================================================
# Formatters
# ============================================================

def format_minion(card: dict) -> dict:
    """Extract relevant fields from a BG minion card."""
    result = {
        "id": card["id"],
        "dbfId": card["dbfId"],
        "name": card["name"],
        "tier": card.get("techLevel", 0),
        "attack": card.get("attack", 0),
        "health": card.get("health", 0),
        "race": extract_race(card),
        "mechanics": extract_mechanics(card),
        "text": clean_text(card.get("text")),
    }
    if "battlegroundsPremiumDbfId" in card:
        result["goldenDbfId"] = card["battlegroundsPremiumDbfId"]
    return result


def format_hero(card: dict) -> dict:
    """Extract relevant fields from a BG hero card."""
    return {
        "id": card["id"],
        "dbfId": card["dbfId"],
        "name": card["name"],
        "health": card.get("health", 0),
        "armor": card.get("armor", 0),
        "heroPowerDbfId": card.get("heroPowerDbfId"),
        "buddyDbfId": card.get("battlegroundsBuddyDbfId"),
        "text": clean_text(card.get("text")),
    }


def format_hero_power(card: dict) -> dict:
    """Extract relevant fields from a BG hero power."""
    return {
        "id": card["id"],
        "dbfId": card["dbfId"],
        "name": card["name"],
        "cost": card.get("cost", 0),
        "text": clean_text(card.get("text")),
    }


def format_spell(card: dict) -> dict:
    """Extract relevant fields from a BG spell."""
    return {
        "id": card["id"],
        "dbfId": card["dbfId"],
        "name": card["name"],
        "tier": card.get("techLevel", 0),
        "cost": card.get("cost", 0),
        "mechanics": extract_mechanics(card),
        "text": clean_text(card.get("text")),
    }


TRINKET_SLOT_MAP = {
    "LESSER_TRINKET": "lesser",
    "GREATER_TRINKET": "greater",
}


def format_trinket(card: dict) -> dict:
    """Extract relevant fields from a BG trinket."""
    slot_raw = card.get("spellSchool", "")
    slot = TRINKET_SLOT_MAP.get(slot_raw, slot_raw.lower() if slot_raw else "unknown")
    races = card.get("battlegroundsAssociatedRaces", [])
    return {
        "id": card["id"],
        "dbfId": card["dbfId"],
        "name": card["name"],
        "slot": slot,  # "lesser" or "greater"
        "cost": card.get("cost", 0),
        "associatedRaces": [RACE_MAP.get(r, r) for r in races] if races else [],
        "mechanics": extract_mechanics(card),
        "text": clean_text(card.get("text")),
    }


def format_anomaly(card: dict) -> dict:
    """Extract relevant fields from a BG anomaly."""
    return {
        "id": card["id"],
        "dbfId": card["dbfId"],
        "name": card["name"],
        "text": clean_text(card.get("text")),
    }


def format_quest_reward(card: dict) -> dict:
    """Extract relevant fields from a BG quest reward."""
    return {
        "id": card["id"],
        "dbfId": card["dbfId"],
        "name": card["name"],
        "text": clean_text(card.get("text")),
    }


def format_token(card: dict) -> dict:
    """Extract relevant fields from a BG token/related card."""
    result = {
        "id": card["id"],
        "dbfId": card["dbfId"],
        "name": card["name"],
        "type": card.get("type", ""),
        "attack": card.get("attack", 0),
        "health": card.get("health", 0),
        "race": extract_race(card),
        "mechanics": extract_mechanics(card),
        "text": clean_text(card.get("text")),
    }
    if card.get("cost") is not None:
        result["cost"] = card["cost"]
    return result


def format_buddy(card: dict) -> dict:
    """Extract relevant fields from a BG buddy."""
    return {
        "id": card["id"],
        "dbfId": card["dbfId"],
        "name": card["name"],
        "tier": card.get("techLevel", 0),
        "attack": card.get("attack", 0),
        "health": card.get("health", 0),
        "race": extract_race(card),
        "mechanics": extract_mechanics(card),
        "text": clean_text(card.get("text")),
    }


# ============================================================
# Save / Print
# ============================================================

def save_json(data: list | dict, path: Path) -> None:
    """Save data as formatted JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    count = len(data) if isinstance(data, list) else "meta"
    print(f"  -> Saved {path.name} ({count})")


def print_summary(results: dict[str, list]) -> None:
    """Print a summary of all scraped data."""
    print("\n" + "=" * 60)
    print("SCRAPE SUMMARY")
    print("=" * 60)

    minions = results["minions"]
    tier_counts: dict[int, int] = defaultdict(int)
    for m in minions:
        tier_counts[m["tier"]] += 1

    print(f"\nPool Minions: {len(minions)}")
    for tier in sorted(tier_counts):
        print(f"  Tier {tier}: {tier_counts[tier]}")

    race_counts: dict[str, int] = defaultdict(int)
    for m in minions:
        race = m.get("race")
        if race is None:
            race_counts["Neutral"] += 1
        elif isinstance(race, list):
            for r in race:
                race_counts[r] += 1
        else:
            race_counts[race] += 1
    print(f"\nRace distribution:")
    for race in sorted(race_counts):
        print(f"  {race}: {race_counts[race]}")

    for key in ["heroes", "hero_powers", "spells", "trinkets",
                 "anomalies", "quest_rewards", "tokens", "buddies"]:
        label = key.replace("_", " ").title()
        print(f"{label}: {len(results[key])}")

    print("=" * 60)


# ============================================================
# Main
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scrape ALL BG card data from HearthstoneJSON",
    )
    parser.add_argument(
        "--build", type=int, default=DEFAULT_BUILD,
        help=f"Game build number (default: {DEFAULT_BUILD})",
    )
    args = parser.parse_args()
    build = args.build

    print(f"[*] Scraping Battlegrounds data for build {build}")

    # 1. Download
    all_cards = fetch_cards(build)
    dbf_map = {c["dbfId"]: c for c in all_cards}

    # 2. Pool minions (isBattlegroundsPoolMinion)
    bg_minions = sorted(
        [format_minion(c) for c in all_cards
         if c.get("isBattlegroundsPoolMinion") is True],
        key=lambda m: (m["tier"], m["name"]),
    )

    # 3. Heroes (battlegroundsHero, excluding skins)
    bg_heroes = sorted(
        [format_hero(c) for c in all_cards
         if c.get("battlegroundsHero") is True
         and not c.get("battlegroundsSkinParentId")],
        key=lambda h: h["name"],
    )

    # 4. Hero powers (linked from heroes)
    hero_power_dbf_ids = {
        h["heroPowerDbfId"]
        for h in bg_heroes
        if h.get("heroPowerDbfId")
    }
    bg_hero_powers = sorted(
        [format_hero_power(c) for c in all_cards
         if c.get("type") == "HERO_POWER"
         and c.get("dbfId") in hero_power_dbf_ids],
        key=lambda hp: hp["name"],
    )

    # 5. Tavern spells (isBattlegroundsPoolSpell — the 60 real pool spells)
    bg_spells = sorted(
        [format_spell(c) for c in all_cards
         if c.get("isBattlegroundsPoolSpell") is True],
        key=lambda s: (s["tier"], s["name"]),
    )

    # 6. Trinkets (type=BATTLEGROUND_TRINKET, exclude meta/shop-button cards)
    bg_trinkets = sorted(
        [format_trinket(c) for c in all_cards
         if c.get("type") == "BATTLEGROUND_TRINKET"
         and "battlegroundsNormalDbfId" not in c
         and c.get("text")
         and "shop opens" not in c.get("text", "").lower()],
        key=lambda t: (t["slot"], t["name"]),
    )

    # 7. Anomalies (type=BATTLEGROUND_ANOMALY, exclude golden)
    bg_anomalies = sorted(
        [format_anomaly(c) for c in all_cards
         if c.get("type") == "BATTLEGROUND_ANOMALY"
         and "battlegroundsNormalDbfId" not in c],
        key=lambda a: a["name"],
    )

    # 8. Quest rewards (type=BATTLEGROUND_QUEST_REWARD, exclude golden)
    bg_quest_rewards = sorted(
        [format_quest_reward(c) for c in all_cards
         if c.get("type") == "BATTLEGROUND_QUEST_REWARD"
         and "battlegroundsNormalDbfId" not in c],
        key=lambda q: q["name"],
    )

    # 9. Tokens (referenced by pool minions via battlegroundsRelatedCard)
    token_dbf_ids: set[int] = set()
    for c in all_cards:
        if c.get("isBattlegroundsPoolMinion") is True:
            rel = c.get("battlegroundsRelatedCard")
            if rel:
                token_dbf_ids.add(rel)
    bg_tokens = sorted(
        [format_token(dbf_map[d]) for d in token_dbf_ids if d in dbf_map],
        key=lambda t: t["name"],
    )

    # 10. Buddies (isBattlegroundsBuddy, exclude golden)
    bg_buddies = sorted(
        [format_buddy(c) for c in all_cards
         if c.get("isBattlegroundsBuddy") is True
         and "battlegroundsNormalDbfId" not in c],
        key=lambda b: b["name"],
    )

    # ---- Save ----
    out_dir = OUTPUT_DIR / f"patch_{build}"
    print(f"\n[*] Saving to {out_dir}")

    save_json(bg_minions, out_dir / "bg_minions.json")
    save_json(bg_heroes, out_dir / "bg_heroes.json")
    save_json(bg_hero_powers, out_dir / "bg_hero_powers.json")
    save_json(bg_spells, out_dir / "bg_spells.json")
    save_json(bg_trinkets, out_dir / "bg_trinkets.json")
    save_json(bg_anomalies, out_dir / "bg_anomalies.json")
    save_json(bg_quest_rewards, out_dir / "bg_quest_rewards.json")
    save_json(bg_tokens, out_dir / "bg_tokens.json")
    save_json(bg_buddies, out_dir / "bg_buddies.json")

    results = {
        "minions": bg_minions,
        "heroes": bg_heroes,
        "hero_powers": bg_hero_powers,
        "spells": bg_spells,
        "trinkets": bg_trinkets,
        "anomalies": bg_anomalies,
        "quest_rewards": bg_quest_rewards,
        "tokens": bg_tokens,
        "buddies": bg_buddies,
    }

    # Count lesser/greater trinkets
    lesser_count = sum(1 for t in bg_trinkets if t["slot"] == "lesser")
    greater_count = sum(1 for t in bg_trinkets if t["slot"] == "greater")

    metadata = {
        "build": build,
        "scraped_at": datetime.now(timezone.utc).isoformat(),
        "source": API_URL.format(build=build),
        "counts": {k: len(v) for k, v in results.items()},
        "trinket_breakdown": {
            "lesser": lesser_count,
            "greater": greater_count,
        },
        "notes": {
            "minions": "Active pool (isBattlegroundsPoolMinion). Authoritative.",
            "heroes": "All BG heroes (battlegroundsHero). No per-patch pool flag exists.",
            "spells": "Active pool (isBattlegroundsPoolSpell). Authoritative.",
            "trinkets": "All trinkets in game files. No pool flag — assumed all active.",
            "anomalies": "All anomalies in game files. No pool flag — assumed all active.",
            "quest_rewards": "All quest rewards. Quests may not be active every season.",
            "buddies": "All buddy cards. Buddy system may not be active every season.",
            "tokens": "Minions/spells referenced by pool minions (deathrattle summons etc).",
        },
    }
    save_json(metadata, out_dir / "metadata.json")

    # ---- Summary ----
    print_summary(results)
    print(f"\n[+] Done! Data frozen in {out_dir}")


if __name__ == "__main__":
    main()
