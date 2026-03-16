#!/usr/bin/env python3
"""
Audit script to validate player_mapping.NEW.csv before swapping into production.

Verifies:
1. Row count doesn't decrease
2. Required columns present
3. Key players present (Djokovic, Medvedev, FAA, de Minaur)
4. Data quality metrics (placeholders, coverage)
5. Bovada slate coverage
6. Generates diff files for manual review

Usage:
    python audit_mapping.py

Outputs:
    - player_mapping.AUDIT.txt (detailed report)
    - DIFF_added.csv (newly added players)
    - DIFF_updated.csv (modified existing players)
    - SLATE_MISSING.csv (players in today's slate not in mapping)
"""

import csv
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd

def normalize_name(name: str) -> str:
    """Normalize player name for matching"""
    return name.strip().lower()

def audit_mapping():
    """Main audit function"""
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / "data"
    logs_dir = base_dir / "production" / "logs"

    old_file = data_dir / "player_mapping.csv"
    new_file = data_dir / "player_mapping.NEW.csv"

    # Find latest odds file
    odds_dir = logs_dir / "odds"
    odds_file = None
    if odds_dir.exists():
        odds_files = list(odds_dir.glob("bovada_tennis_*.csv"))
        if odds_files:
            odds_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            odds_file = odds_files[0]

    report_lines = []
    issues = []

    def log(msg):
        report_lines.append(msg)
        print(msg)

    log("=" * 80)
    log("PLAYER MAPPING AUDIT REPORT")
    log(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log("=" * 80)
    log("")

    # Check files exist
    if not old_file.exists():
        log(f"❌ ERROR: Old mapping not found: {old_file}")
        issues.append("Old mapping file missing")
        return False

    if not new_file.exists():
        log(f"❌ ERROR: New mapping not found: {new_file}")
        issues.append("New mapping file missing")
        return False

    log(f"✅ Old mapping: {old_file}")
    log(f"✅ New mapping: {new_file}")
    log("")

    # Load mappings
    log("📂 Loading mapping files...")
    try:
        old_df = pd.read_csv(old_file, encoding='utf-8')
        new_df = pd.read_csv(new_file, encoding='utf-8')
    except Exception as e:
        log(f"❌ ERROR: Failed to load mappings: {e}")
        issues.append(f"Load error: {e}")
        return False

    log(f"   Old: {len(old_df)} rows")
    log(f"   New: {len(new_df)} rows")
    log("")

    # 1. ROW COUNT CHECK
    log("1️⃣  ROW COUNT CHECK")
    log("-" * 80)

    if len(new_df) < len(old_df):
        log(f"❌ FAIL: New mapping has FEWER rows ({len(new_df)}) than old ({len(old_df)})")
        issues.append(f"Row count decreased by {len(old_df) - len(new_df)}")
    else:
        diff = len(new_df) - len(old_df)
        log(f"✅ PASS: New mapping has {diff} additional rows")
    log("")

    # 2. COLUMN CHECK
    log("2️⃣  COLUMN CHECK")
    log("-" * 80)

    required_cols = ['utr_id', 'primary_name', 'name_variants', 'bovada_name', 'current_utr', 'country', 'last_updated']
    missing_cols = [col for col in required_cols if col not in new_df.columns]

    if missing_cols:
        log(f"❌ FAIL: Missing required columns: {missing_cols}")
        issues.append(f"Missing columns: {missing_cols}")
    else:
        log("✅ PASS: All required columns present")

    log(f"   Columns: {list(new_df.columns)}")
    log("")

    # 3. UNIQUENESS CHECK
    log("3️⃣  UNIQUENESS CHECK")
    log("-" * 80)

    if 'utr_id' in new_df.columns:
        duplicate_ids = new_df[new_df['utr_id'].duplicated()]
        if not duplicate_ids.empty:
            log(f"❌ FAIL: Found {len(duplicate_ids)} duplicate UTR IDs")
            issues.append(f"{len(duplicate_ids)} duplicate UTR IDs")
            for _, row in duplicate_ids.head(5).iterrows():
                log(f"   Duplicate: {row['utr_id']} - {row.get('primary_name', 'N/A')}")
        else:
            log("✅ PASS: All UTR IDs are unique")
    log("")

    # 4. KEY PLAYERS CHECK
    log("4️⃣  KEY PLAYERS CHECK")
    log("-" * 80)

    key_players = [
        ('djokovic', 'Novak Djokovic'),
        ('medvedev', 'Daniil Medvedev'),
        ('auger', 'Felix Auger-Aliassime'),
        ('de minaur', 'Alex de Minaur'),
        ('sinner', 'Jannik Sinner'),
        ('alcaraz', 'Carlos Alcaraz')
    ]

    for search_term, full_name in key_players:
        matches = new_df[new_df['primary_name'].str.lower().str.contains(search_term, na=False)]
        if not matches.empty:
            player = matches.iloc[0]
            log(f"✅ Found: {player['primary_name']} (ID: {player['utr_id']}, UTR: {player.get('current_utr', 'N/A')})")
        else:
            log(f"❌ Missing: {full_name}")
            issues.append(f"Key player missing: {full_name}")
    log("")

    # 5. DATA QUALITY METRICS
    log("5️⃣  DATA QUALITY METRICS")
    log("-" * 80)

    if 'primary_name' in new_df.columns:
        placeholders = new_df[new_df['primary_name'].str.startswith('Player_', na=False)]
        placeholder_pct = len(placeholders) / len(new_df) * 100
        log(f"   Placeholder names: {len(placeholders)} ({placeholder_pct:.1f}%)")

        if placeholder_pct > 10:
            log(f"   ⚠️  Warning: >10% placeholder names")
            issues.append(f"High placeholder percentage: {placeholder_pct:.1f}%")

    if 'current_utr' in new_df.columns:
        has_utr = new_df[new_df['current_utr'].notna() & (new_df['current_utr'] != '')]
        utr_pct = len(has_utr) / len(new_df) * 100
        log(f"   Players with UTR: {len(has_utr)} ({utr_pct:.1f}%)")

    if 'country' in new_df.columns:
        has_country = new_df[new_df['country'].notna() & (new_df['country'] != '')]
        country_pct = len(has_country) / len(new_df) * 100
        log(f"   Players with country: {len(has_country)} ({country_pct:.1f}%)")
    log("")

    # 6. DIFF ANALYSIS
    log("6️⃣  DIFF ANALYSIS")
    log("-" * 80)

    if 'utr_id' in old_df.columns and 'utr_id' in new_df.columns:
        old_ids = set(old_df['utr_id'].astype(str))
        new_ids = set(new_df['utr_id'].astype(str))

        added_ids = new_ids - old_ids
        removed_ids = old_ids - new_ids
        common_ids = old_ids & new_ids

        log(f"   Added: {len(added_ids)} players")
        log(f"   Removed: {len(removed_ids)} players")
        log(f"   Unchanged IDs: {len(common_ids)} players")

        if removed_ids:
            log(f"   ⚠️  Warning: {len(removed_ids)} players were removed")
            issues.append(f"{len(removed_ids)} players removed")

        # Save added players
        if added_ids:
            added_df = new_df[new_df['utr_id'].astype(str).isin(added_ids)]
            added_file = data_dir / "DIFF_added.csv"
            added_df.to_csv(added_file, index=False, encoding='utf-8')
            log(f"   📄 Saved added players to: {added_file.name}")

        # Check for updated players (same ID, different data)
        updated_players = []
        for uid in common_ids:
            old_row = old_df[old_df['utr_id'].astype(str) == uid].iloc[0]
            new_row = new_df[new_df['utr_id'].astype(str) == uid].iloc[0]

            # Check if any field changed
            changed = False
            for col in ['primary_name', 'name_variants', 'bovada_name', 'current_utr', 'country']:
                if col in old_row and col in new_row:
                    old_val = str(old_row[col]) if pd.notna(old_row[col]) else ''
                    new_val = str(new_row[col]) if pd.notna(new_row[col]) else ''
                    if old_val != new_val:
                        changed = True
                        break

            if changed:
                updated_players.append(uid)

        if updated_players:
            log(f"   Updated: {len(updated_players)} players had data changes")
            updated_df = new_df[new_df['utr_id'].astype(str).isin(updated_players)]
            updated_file = data_dir / "DIFF_updated.csv"
            updated_df.to_csv(updated_file, index=False, encoding='utf-8')
            log(f"   📄 Saved updated players to: {updated_file.name}")
    log("")

    # 7. SLATE COVERAGE CHECK
    log("7️⃣  SLATE COVERAGE CHECK")
    log("-" * 80)

    if odds_file and odds_file.exists():
        log(f"   Latest odds: {odds_file.name}")
        try:
            odds_df = pd.read_csv(odds_file)

            # Extract unique player names
            slate_players = set()
            for col in ['player1_normalized', 'player2_normalized']:
                if col in odds_df.columns:
                    slate_players.update(odds_df[col].dropna().unique())

            log(f"   Slate size: {len(slate_players)} unique players")

            # Check coverage
            missing_players = []
            for player in slate_players:
                player_norm = normalize_name(player)

                # Check if player in mapping (by primary_name or name_variants)
                found = False
                for _, row in new_df.iterrows():
                    primary = normalize_name(str(row.get('primary_name', '')))
                    variants = str(row.get('name_variants', ''))

                    if player_norm == primary or player_norm in normalize_name(variants):
                        found = True
                        break

                if not found:
                    missing_players.append(player)

            if missing_players:
                log(f"   ❌ Missing from mapping: {len(missing_players)} players")
                issues.append(f"{len(missing_players)} slate players missing from mapping")

                # Save missing slate players
                missing_file = data_dir / "SLATE_MISSING.csv"
                missing_df = pd.DataFrame({'player_name': missing_players})
                missing_df.to_csv(missing_file, index=False)
                log(f"   📄 Saved missing players to: {missing_file.name}")

                # Show first few
                for player in missing_players[:10]:
                    log(f"      - {player}")
                if len(missing_players) > 10:
                    log(f"      ... and {len(missing_players) - 10} more")
            else:
                log("   ✅ All slate players found in mapping")
        except Exception as e:
            log(f"   ⚠️  Could not check slate coverage: {e}")
    else:
        log("   ℹ️  No odds file found, skipping slate coverage check")
    log("")

    # SUMMARY
    log("=" * 80)
    log("SUMMARY")
    log("=" * 80)

    if issues:
        log(f"❌ AUDIT FAILED: {len(issues)} issue(s) found")
        log("")
        log("Issues:")
        for i, issue in enumerate(issues, 1):
            log(f"   {i}. {issue}")
        log("")
        log("⚠️  DO NOT swap player_mapping.NEW.csv until issues are resolved")
    else:
        log("✅ AUDIT PASSED: No issues found")
        log("")
        log("Next steps:")
        log("   1. Review DIFF_added.csv and DIFF_updated.csv")
        log("   2. Backup: cp data/player_mapping.csv data/player_mapping.csv.backup")
        log("   3. Swap: mv data/player_mapping.NEW.csv data/player_mapping.csv")
        log("   4. Re-run dry-run: cd production && python main.py")

    log("")
    log("=" * 80)

    # Write report to file
    report_file = data_dir / "player_mapping.AUDIT.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))

    print(f"\n📄 Full audit report saved to: {report_file}")

    return len(issues) == 0

if __name__ == "__main__":
    print("🔍 Starting player mapping audit...\n")
    success = audit_mapping()
    sys.exit(0 if success else 1)
