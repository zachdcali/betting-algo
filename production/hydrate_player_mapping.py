#!/usr/bin/env python3
"""
Safely hydrate player_mapping.csv from existing data sources.

This script MERGES (not replaces) player_mapping.csv by:
1. Reading existing player_mapping.csv to preserve all current data
2. Reading the UTR rankings file to get top players
3. Scanning all existing match files to find additional player IDs
4. Creating mapping entries for newly discovered players
5. Writing to player_mapping.NEW.csv for safety (you swap it manually)

Run this once to backfill the mapping with players that were scraped
but never had get_player_profile() called (which triggers update_player_mapping).
"""

import csv
import json
from pathlib import Path
import pandas as pd
from datetime import datetime

def is_valid_id(s: str) -> bool:
    """Check if a string is a valid UTR ID (numeric only)"""
    return isinstance(s, str) and s.strip().isdigit()

def normalize_name(name: str) -> str:
    """Normalize player name for consistent matching"""
    return name.strip().lower()

def generate_name_variants(full_name: str) -> str:
    """Generate name variants like 'Djokovic N.', 'N. Djokovic', 'DJOKOVIC N.'"""
    if not full_name:
        return ""

    parts = full_name.split()
    if len(parts) < 2:
        return full_name

    first, last = parts[0], parts[-1]
    variants = [
        full_name,  # Full name
        f"{last} {first[0]}." if first else last,  # "Djokovic N."
        f"{first[0]}. {last}" if first else last,  # "N. Djokovic"
        f"{last.upper()} {first[0]}." if first else last.upper(),  # "DJOKOVIC N."
    ]
    return "|".join(variants)

def merge_name_variants(existing: str, new: str) -> str:
    """Merge two pipe-separated variant lists, removing duplicates"""
    if not existing and not new:
        return ""
    if not existing:
        return new
    if not new:
        return existing

    # Split, normalize, dedupe, rejoin
    variants = set()
    for v in existing.split("|") + new.split("|"):
        v = v.strip()
        if v:
            variants.add(v)
    return "|".join(sorted(variants))

def extract_name_from_match_file(match_file: Path) -> str:
    """Try to extract player name from match CSV"""
    try:
        df = pd.read_csv(match_file, nrows=1)
        # Common columns that might have player name
        for col in ['player_name', 'name', 'player']:
            if col in df.columns:
                name = str(df[col].iloc[0]).strip()
                if name and name != 'nan':
                    return name
    except Exception:
        pass
    return ""

def hydrate_mapping():
    """Main function to hydrate player_mapping.csv"""
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / "data"

    mapping_file = data_dir / "player_mapping.csv"
    new_mapping_file = data_dir / "player_mapping.NEW.csv"

    timestamp = datetime.now().strftime("%Y-%m-%d")

    # Step 0: Read existing mapping to preserve data
    print("📂 Step 0: Reading existing player_mapping.csv...")
    existing = {}  # utr_id -> row dict
    skipped_invalid = 0
    if mapping_file.exists():
        try:
            with open(mapping_file, 'r', encoding='utf-8', newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    utr_id = (row.get('utr_id') or '').strip()
                    if not is_valid_id(utr_id):
                        skipped_invalid += 1
                        continue
                    existing[utr_id] = row
            print(f"   ✅ Loaded {len(existing)} existing entries")
            if skipped_invalid > 0:
                print(f"   ℹ️  Skipped {skipped_invalid} rows with invalid/empty UTR IDs")
        except Exception as e:
            print(f"   ⚠️  Error reading existing mapping: {e}")
            existing = {}
    else:
        print("   ℹ️  No existing mapping file found")

    # Track newly discovered players
    discovered = {}  # utr_id -> {name, utr, country, source}

    # Step 1: Read rankings
    print("\n🔍 Step 1: Reading UTR rankings file...")
    ranking_files = list(data_dir.glob("*_utr_rankings_*.csv"))
    if ranking_files:
        ranking_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        latest_ranking = ranking_files[0]
        print(f"   Found: {latest_ranking.name}")

        rankings_df = pd.read_csv(latest_ranking)
        for _, row in rankings_df.iterrows():
            player_id = str(row['id']).strip()
            if not is_valid_id(player_id):
                continue
            player_name = str(row['name']).strip()
            utr_val = row.get('utr', '')

            discovered[player_id] = {
                'name': player_name,
                'utr': utr_val if pd.notna(utr_val) else '',
                'country': '',
                'source': 'rankings'
            }

        print(f"   ✅ Found {len(discovered)} players from rankings")
    else:
        print("   ⚠️  No rankings file found")

    # Step 2: Scan match files
    print("\n🔍 Step 2: Scanning match files for additional player IDs...")
    matches_dir = data_dir / "matches"
    players_dir = data_dir / "players"

    if matches_dir.exists():
        match_files = list(matches_dir.glob("player_*_matches.csv"))
        print(f"   Found {len(match_files)} match files")

        for match_file in match_files:
            # Extract player ID from filename: player_3568338_matches.csv -> 3568338
            filename = match_file.stem
            if "_matches" in filename:
                player_id = filename.replace("player_", "").replace("_matches", "")
                if not is_valid_id(player_id):
                    continue

                if player_id not in discovered:
                    # Try to get name from profile JSON first
                    profile_file = players_dir / f"player_{player_id}.json"
                    player_name = ""
                    country = ""
                    utr_val = ""

                    if profile_file.exists():
                        try:
                            with open(profile_file, 'r', encoding='utf-8') as f:
                                profile = json.load(f)
                                player_name = profile.get('name', '')
                                country = profile.get('country', '')
                                utr_val = profile.get('current_utr', '')
                        except Exception:
                            pass

                    # If no profile, try to extract name from match file
                    if not player_name:
                        player_name = extract_name_from_match_file(match_file)

                    # Last resort: placeholder
                    if not player_name:
                        player_name = f'Player_{player_id}'

                    discovered[player_id] = {
                        'name': player_name,
                        'utr': utr_val,
                        'country': country,
                        'source': 'matches'
                    }

        print(f"   ✅ Total unique players discovered: {len(discovered)}")
    else:
        print("   ⚠️  Matches directory not found")

    # Step 3: Merge existing + discovered
    print("\n🔀 Step 3: Merging existing and discovered data...")
    merged = {}

    # Start with existing (to preserve all current data)
    for utr_id, row in existing.items():
        merged[utr_id] = row.copy()

    # Merge in discovered
    new_count = 0
    updated_count = 0

    for utr_id, info in discovered.items():
        if utr_id in merged:
            # Update existing entry (preserve non-empty fields)
            row = merged[utr_id]

            # Update primary_name if current is empty or placeholder
            if not row.get('primary_name') or row['primary_name'].startswith('Player_'):
                if info['name'] and not info['name'].startswith('Player_'):
                    row['primary_name'] = info['name']
                    updated_count += 1

            # Update/merge name_variants
            new_variants = generate_name_variants(info['name'])
            row['name_variants'] = merge_name_variants(row.get('name_variants', ''), new_variants)

            # Update current_utr if we have a newer value
            if info['utr'] and not row.get('current_utr'):
                row['current_utr'] = info['utr']

            # Update country if empty
            if info['country'] and not row.get('country'):
                row['country'] = info['country']

            # Update bovada_name if empty
            if not row.get('bovada_name'):
                row['bovada_name'] = info['name']

            row['last_updated'] = timestamp

        else:
            # New entry
            merged[utr_id] = {
                'utr_id': utr_id,
                'primary_name': info['name'],
                'name_variants': generate_name_variants(info['name']),
                'bovada_name': info['name'],
                'current_utr': info['utr'],
                'country': info['country'],
                'last_updated': timestamp
            }
            new_count += 1

    print(f"   ✅ Total entries: {len(merged)} ({new_count} new, {updated_count} updated)")

    # Step 4: Write to NEW file
    print(f"\n📝 Step 4: Writing to {new_mapping_file.name}...")

    written_count = 0
    with open(new_mapping_file, 'w', encoding='utf-8', newline='') as f:
        fieldnames = ['utr_id', 'primary_name', 'name_variants', 'bovada_name',
                      'current_utr', 'country', 'last_updated']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for utr_id in sorted(merged.keys()):
            # Final safety check: only write valid IDs
            if not is_valid_id(utr_id):
                continue
            row = merged[utr_id]
            # Ensure all fields are present
            output_row = {field: row.get(field, '') for field in fieldnames}
            writer.writerow(output_row)
            written_count += 1

    print(f"   ✅ Wrote {written_count} entries to {new_mapping_file.name}")

    # Step 5: Verification
    print("\n🔍 Step 5: Verification checks...")

    # Check Djokovic
    djokovic_found = False
    with open(new_mapping_file, 'r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if 'djokovic' in row.get('primary_name', '').lower():
                print(f"   ✅ Found: {row['primary_name']} (ID: {row['utr_id']})")
                djokovic_found = True
                break

    if not djokovic_found:
        print("   ⚠️  Djokovic not found in new mapping")

    # Line count comparison
    old_lines = len(existing)
    new_lines = len(merged)
    print(f"\n📊 Line count: {old_lines} (old) → {new_lines} (new) [+{new_lines - old_lines}]")

    print(f"\n✅ Success! New mapping written to: {new_mapping_file}")
    print(f"\n📋 Next steps:")
    print(f"   1. Review: cat {new_mapping_file}")
    print(f"   2. Backup: cp {mapping_file} {mapping_file}.backup")
    print(f"   3. Swap: mv {new_mapping_file} {mapping_file}")

    return len(merged), new_count, updated_count

if __name__ == "__main__":
    print("🚀 Safely hydrating player_mapping.csv from existing data...\n")
    total, new, updated = hydrate_mapping()
    print(f"\n📊 Summary: {total} total, {new} new, {updated} updated")
