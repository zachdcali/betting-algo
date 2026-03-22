#!/usr/bin/env python3
"""
Auto-settle pending predictions by checking Tennis Abstract match history.

For each unsettled row in prediction_log.csv:
  1. Fetch p1's recent matches from TA
  2. Look for a completed match against p2 on or after the logged match_date
  3. If found, record the winner, score, and compute model_correct / market_correct

Usage:
    python auto_settle.py           # check and settle all pending
    python auto_settle.py --dry-run # show what would be settled without writing
    python auto_settle.py --stats   # show accuracy stats only (no settling)
"""

import argparse
import sys
import os
import re
import time
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "scraping"))

from ta_scraper import TennisAbstractScraper
from features.ta_feature_calculator import TAFeatureCalculator

LOG_PATH = Path(__file__).parent / "prediction_log.csv"
SCRAPER = TennisAbstractScraper(rate_limit_delay=3.0)


# ---------------------------------------------------------------------------
# Name matching
# ---------------------------------------------------------------------------

def _last_name(name: str) -> str:
    """Extract last name, lowercased."""
    parts = name.strip().split()
    return parts[-1].lower() if parts else ""


def _names_match(opp_name: str, candidate: str) -> bool:
    """
    Check if TA opponent name matches a logged player name.
    TA opp_name is often 'FirstLast' or 'F. Last' or just 'Last'.
    Candidate is the full name from Bovada e.g. 'Luca Van Assche'.
    """
    opp = opp_name.lower().strip()
    cand = candidate.lower().strip()

    # Exact
    if opp == cand:
        return True

    # Last name match (both)
    if _last_name(opp) == _last_name(cand) and _last_name(cand):
        return True

    # TA sometimes stores 'Last' only — if opp is a single token, check last name
    if " " not in opp and opp == _last_name(cand):
        return True

    # TA 'F. Last' format
    m = re.match(r"^([a-z])\.?\s+(.+)$", opp)
    if m:
        first_initial = m.group(1)
        last = m.group(2).strip()
        cand_parts = cand.split()
        if cand_parts and cand_parts[0][0] == first_initial and last == _last_name(cand):
            return True

    return False


_NAME_SUFFIXES = {'jr', 'sr', 'ii', 'iii', 'iv'}

def _strip_suffixes(name: str) -> str:
    """Strip name suffixes (Jr, Sr, II, etc.) before slug derivation."""
    parts = name.strip().split()
    clean = [p for p in parts if p.lower().rstrip('.') not in _NAME_SUFFIXES]
    return ' '.join(clean) if len(clean) >= 2 else name


def _resolve_slug(player_name: str, calc: TAFeatureCalculator) -> str:
    """Get TA slug for a player name via mapping or derivation."""
    name_lower = player_name.lower().strip()
    # Check player mapping
    if name_lower in calc.player_slug_map:
        return calc.player_slug_map[name_lower]
    # Strip Jr/Sr/II/III suffixes and try again
    clean = _strip_suffixes(player_name)
    if clean != player_name:
        clean_lower = clean.lower().strip()
        if clean_lower in calc.player_slug_map:
            return calc.player_slug_map[clean_lower]
        return TennisAbstractScraper.name_to_slug(clean)
    return TennisAbstractScraper.name_to_slug(player_name)


# ---------------------------------------------------------------------------
# Core settle logic
# ---------------------------------------------------------------------------

def try_settle_from_ta(p1: str, p2: str, match_date_str: str,
                        calc: TAFeatureCalculator,
                        dry_run: bool = False) -> dict | None:
    """
    Fetch p1's recent matches and look for a result vs p2 on/after match_date.
    Returns a dict with keys: actual_winner (1|2), score, settled_at
    or None if result not yet found.
    """
    match_date = pd.to_datetime(match_date_str, errors='coerce')
    if pd.isna(match_date):
        print(f"  ⚠️  Could not parse match_date '{match_date_str}' — skipping")
        return None

    slug1 = _resolve_slug(p1, calc)
    current_year = datetime.now().year
    years = [current_year]
    # Include previous year if match_date was in prior year
    if match_date.year < current_year:
        years.append(match_date.year)

    print(f"  Checking TA for {p1} ({slug1}) vs {p2}...")
    matches = SCRAPER.get_player_matches(slug1, years=years, force_refresh=True)

    if matches.empty:
        print(f"    No match data found for {slug1}")
        return None

    # TA stores the tournament START DATE for all rounds (same as Sackmann CSV),
    # not the actual match date. So we can't use a tight date filter.
    # Instead: look back up to 21 days before the logged match_date (covers any
    # tournament start), and forward 3 days (grace for in-progress matches).
    # Rely on opponent name matching to identify the specific match.
    matches['date'] = pd.to_datetime(matches['date'], errors='coerce')
    window_start = match_date - timedelta(days=21)
    window_end   = match_date + timedelta(days=3)
    recent = matches[
        (matches['date'] >= window_start) &
        (matches['date'] <= window_end)
    ].copy()

    if recent.empty:
        print(f"    No matches found within window of {match_date_str}")
        return None

    # Look for a match vs p2
    found = recent[recent['opp_name'].apply(lambda n: _names_match(str(n), p2))]

    if found.empty:
        print(f"    No result found yet vs '{p2}'")
        return None

    row = found.iloc[0]
    result = str(row.get('result', '')).upper()
    score = str(row.get('score', ''))
    match_date_found = row['date'].strftime('%Y-%m-%d') if pd.notna(row['date']) else '?'

    if result == 'W':
        actual_winner = 1
        winner_name = p1
    elif result == 'L':
        actual_winner = 2
        winner_name = p2
    else:
        print(f"    Unexpected result value '{result}' — skipping")
        return None

    print(f"    Found result ({match_date_found}): {winner_name} won  |  score: {score}")
    return {
        'actual_winner': actual_winner,
        'score': score,
        'settled_at': datetime.now().isoformat(),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def show_stats(df: pd.DataFrame):
    settled = df[df['actual_winner'].notna()].copy()
    if settled.empty:
        print("No settled predictions yet.")
        return

    # Full stats (all settled, for reference)
    n_all = len(settled)
    print(f"\n=== LIVE ACCURACY ===")
    print(f"  Total settled: {n_all}  (includes ITF/no-model entries)")

    # Clean stats: only rows with complete features and a real model prediction
    clean = settled[
        settled['model_correct'].notna() &
        settled['features_complete'].fillna(True).astype(bool)
    ].copy()

    if clean.empty:
        print("  No complete-feature predictions settled yet.")
        return

    n = len(clean)
    model_acc  = clean['model_correct'].mean()
    market_acc = clean['market_correct'].mean()
    print(f"\n  [Complete features only — {n} matches]")
    print(f"  Model:  {model_acc:.1%}  ({int(clean['model_correct'].sum())}/{n})")
    print(f"  Market: {market_acc:.1%}  ({int(clean['market_correct'].sum())}/{n})")
    print(f"  Edge:   {model_acc - market_acc:+.1%}")

    # By model version
    if 'model_version' in clean.columns:
        print("\n  By model version:")
        for ver, grp in clean.groupby('model_version'):
            if grp.empty:
                continue
            ma = grp['model_correct'].mean()
            mk = grp['market_correct'].mean()
            print(f"    {ver}: model {ma:.1%}  market {mk:.1%}  ({len(grp)} matches)")

    # By surface
    if 'surface' in clean.columns and clean['surface'].notna().any():
        print("\n  By surface:")
        for surf, grp in clean.groupby('surface'):
            if grp.empty:
                continue
            print(f"    {surf}: model {grp['model_correct'].mean():.0%}  market {grp['market_correct'].mean():.0%}  ({len(grp)} matches)")


def run(dry_run: bool = False, stats_only: bool = False):
    if not LOG_PATH.exists():
        print("No prediction_log.csv found.")
        return

    df = pd.read_csv(LOG_PATH)

    if stats_only:
        show_stats(df)
        return

    pending = df[df['actual_winner'].isna()].copy()
    if pending.empty:
        print("No unsettled predictions.")
        show_stats(df)
        return

    print(f"Found {len(pending)} unsettled predictions\n")

    # Load player mapping once
    calc = TAFeatureCalculator.__new__(TAFeatureCalculator)
    calc.player_slug_map = TAFeatureCalculator._load_player_mapping(calc)

    newly_settled = 0

    for idx, row in pending.iterrows():
        p1 = str(row['p1'])
        p2 = str(row['p2'])
        match_date = str(row.get('match_date', ''))
        print(f"\n[{idx}] {p1} vs {p2}  ({row.get('tournament', '')}  {match_date})")
        print(f"     Model: {float(row['model_p1_prob']):.0%} P1 | Market: {float(row['market_p1_prob']):.0%} P1")

        result = try_settle_from_ta(p1, p2, match_date, calc, dry_run=dry_run)

        if result is None:
            print(f"  → Not settled yet")
            continue

        if dry_run:
            winner_label = p1 if result['actual_winner'] == 1 else p2
            print(f"  [DRY RUN] Would settle: winner={winner_label}, score={result['score']}")
            continue

        # Write result back
        df.at[idx, 'actual_winner'] = result['actual_winner']
        df.at[idx, 'score'] = result['score']
        df.at[idx, 'settled_at'] = result['settled_at']

        model_p1 = float(row['model_p1_prob'])
        market_p1 = float(row['market_p1_prob'])
        w = result['actual_winner']
        model_correct = int((w == 1 and model_p1 > 0.5) or (w == 2 and model_p1 < 0.5))
        market_correct = int((w == 1 and market_p1 > 0.5) or (w == 2 and market_p1 < 0.5))
        df.at[idx, 'model_correct'] = model_correct
        df.at[idx, 'market_correct'] = market_correct

        winner_name = p1 if w == 1 else p2
        print(f"  ✓ Settled: {winner_name} won | Model {'✓' if model_correct else '✗'} | Market {'✓' if market_correct else '✗'}")
        newly_settled += 1

        # Rate limit between players
        time.sleep(3.0)

    if not dry_run:
        df.to_csv(LOG_PATH, index=False)
        print(f"\nSaved {newly_settled} newly settled prediction(s) to prediction_log.csv")

    show_stats(df)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Auto-settle predictions from Tennis Abstract results')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be settled without writing')
    parser.add_argument('--stats', action='store_true', help='Show accuracy stats only')
    args = parser.parse_args()

    run(dry_run=args.dry_run, stats_only=args.stats)
