#!/usr/bin/env python3
"""
Analyze what features from schema_143.json can be extracted from Tennis Abstract
"""
import json
from pathlib import Path

# Load schema
schema_path = Path(__file__).parent.parent / "features" / "schema_143.json"
with open(schema_path) as f:
    schema = json.load(f)

# TA matchhead columns (from JavaScript var)
ta_columns = [
    "date", "tourn", "surf", "level", "wl", "rank", "seed", "entry", "round",
    "score", "max", "opp", "orank", "oseed", "oentry", "ohand", "obday",
    "oht", "ocountry", "oactive", "time", "aces", "dfs", "pts", "firsts", "fwon",
    "swon", "games", "saved", "chances", "oaces", "odfs", "opts", "ofirsts",
    "ofwon", "oswon", "ogames", "osaved", "ochances", "obackhand", "chartlink",
    "pslink", "whserver", "matchid", "wh", "roundnum", "matchnum"
]

# TA profile variables
ta_profile_vars = [
    "fullname", "lastname", "currentrank", "peakrank", "peakfirst", "peaklast",
    "dob", "ht", "hand", "backhand", "country", "active"
]

print("=" * 80)
print("TENNIS ABSTRACT DATA AVAILABILITY ANALYSIS")
print("=" * 80)

print("\n📊 Per-Match Columns Available (matchmx array):")
print(f"   Total: {len(ta_columns)} columns")
for i, col in enumerate(ta_columns, 1):
    print(f"   {i:2d}. {col}")

print("\n👤 Profile Variables Available:")
for var in ta_profile_vars:
    print(f"   - {var}")

print("\n" + "=" * 80)
print("FEATURE MAPPING TO SCHEMA_143")
print("=" * 80)

# Group features by category
features_by_category = {}
for feat in schema["features"]:
    name = feat["name"]

    # Categorize by prefix
    if name.startswith("P2_"):
        category = "Player Stats"
    elif name.startswith("P1_"):
        category = "Opponent Stats"
    elif name.startswith("H2H_"):
        category = "Head-to-Head"
    elif name.startswith("Surface_"):
        category = "Surface (One-Hot)"
    elif name.startswith("Level_"):
        category = "Level (One-Hot)"
    elif name.startswith("Round_"):
        category = "Round (One-Hot)"
    elif name.startswith("P2_Hand_"):
        category = "Player Hand (One-Hot)"
    elif name.startswith("P1_Hand_"):
        category = "Opponent Hand (One-Hot)"
    elif name.startswith("P2_Country_"):
        category = "Player Country (One-Hot)"
    elif name.startswith("P1_Country_"):
        category = "Opponent Country (One-Hot)"
    else:
        category = "Other"

    if category not in features_by_category:
        features_by_category[category] = []
    features_by_category[category].append(name)

# Print summary
for category, feats in sorted(features_by_category.items()):
    print(f"\n📁 {category}: {len(feats)} features")
    for f in feats[:3]:  # Show first 3
        print(f"   - {f}")
    if len(feats) > 3:
        print(f"   ... and {len(feats) - 3} more")

print("\n" + "=" * 80)
print("DATA SOURCE MAPPING")
print("=" * 80)

print("""
✅ AVAILABLE DIRECTLY FROM TA MATCH DATA:
   - Surface, Level, Round (from 'surf', 'level', 'round' columns)
   - Player/Opponent Hand (from profile 'hand' variable + 'ohand' column)
   - Player/Opponent Country (from profile 'country' + 'ocountry' column)
   - Player/Opponent Rank (from 'rank', 'orank' columns per match)
   - Match Result (from 'wl' column)
   - Match Stats: aces, double faults, points, first serves, etc.

✅ CALCULABLE FROM TA MATCH HISTORY:
   - Win Streaks (count consecutive W in 'wl' column)
   - Win Rates by Surface/Level/Round (filter matches + calculate)
   - Rank Changes (compare 'rank' across time windows)
   - Form Trends (recent win%, games won%, etc.)
   - Recent Match Counts (by surface, level, opponent hand)

✅ AVAILABLE FROM TA H2H PAGE:
   - H2H_TotalMatches
   - H2H_P2Wins, H2H_P1Wins
   - H2H_RecentAdvantage (recent form in H2H)

❓ NEED ALTERNATIVE SOURCE:
   - Rank Points (ATP website or Sackmann GitHub repo)

📝 FEATURE CALCULATION STRATEGY:
   1. Profile Variables → One-hot encodings (hand, country)
   2. Match History (matchmx) → Time-series features (streaks, form, rates)
   3. H2H Page → H2H stats
   4. ATP Website → Rank points (fallback)
""")

print("\n" + "=" * 80)
print("✨ NEXT STEPS:")
print("=" * 80)
print("""
1. Enhance ta_scraper.py to expose matchmx as DataFrame
2. Create ta_feature_calculator.py with methods:
   - _calculate_streaks(matches_df, player_id)
   - _calculate_win_rates(matches_df, surface, level, etc.)
   - _calculate_rank_changes(matches_df, windows)
   - _calculate_form_features(matches_df, windows)
   - _get_h2h_stats(player_slug, opponent_slug)
   - _onehot_encode(surface, level, round, hand, country)
   - build_143_features(player1_slug, player2_slug, match_context)
3. Build player name→slug mapping CSV
4. Add routing in main.py (ATP/Challenger→TA, ITF→UTR)
""")
