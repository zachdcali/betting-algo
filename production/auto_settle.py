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
from collections import Counter
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "scraping"))

from ta_scraper import TennisAbstractScraper
from features.ta_feature_calculator import TAFeatureCalculator
from utils.bet_tracker import BetTracker
from audit_logger import log_settlement_event, upsert_run_history
from logging_utils import make_run_id, utc_now
from prediction_logger import upgrade_prediction_log

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
                        session_cache: dict | None = None,
                        dry_run: bool = False) -> dict:
    """
    Fetch p1's recent matches and look for a result vs p2 on/after match_date.
    Returns a dict with a status code plus settlement metadata when available.
    """
    match_date = pd.to_datetime(match_date_str, errors='coerce')
    if pd.isna(match_date):
        print(f"  ⚠️  Could not parse match_date '{match_date_str}' — skipping")
        return {
            'status': 'parse_error',
            'outcome_detail': f"Could not parse match_date '{match_date_str}'",
            'ta_player_slug': '',
        }

    slug1 = _resolve_slug(p1, calc)
    current_year = datetime.now().year
    years = [current_year]
    # Include previous year if match_date was in prior year
    if match_date.year < current_year:
        years.append(match_date.year)

    print(f"  Checking TA for {p1} ({slug1}) vs {p2}...")
    matches = SCRAPER.get_player_matches(
        slug1,
        years=years,
        force_refresh=True,
        session_cache=session_cache,
    )

    if matches.empty:
        print(f"    No match data found for {slug1}")
        return {
            'status': 'ta_empty',
            'outcome_detail': f"No match data found for {slug1}",
            'ta_player_slug': slug1,
        }

    # TA stores the tournament START DATE for all rounds (same as Sackmann CSV),
    # not the actual match date. So we can't use a tight date filter.
    # Use a wide window: 21 days before (covers tournament start offset)
    # and 14 days after (covers delayed runs, rain delays, rescheduled matches).
    # Rely on opponent name matching to identify the specific match.
    matches['date'] = pd.to_datetime(matches['date'], errors='coerce')
    window_start = match_date - timedelta(days=21)
    window_end   = match_date + timedelta(days=14)
    recent = matches[
        (matches['date'] >= window_start) &
        (matches['date'] <= window_end)
    ].copy()

    if recent.empty:
        print(f"    No matches found within window of {match_date_str}")
        return {
            'status': 'outside_window',
            'outcome_detail': f"No matches found within window of {match_date_str}",
            'ta_player_slug': slug1,
        }

    # Look for a match vs p2
    found = recent[recent['opp_name'].apply(lambda n: _names_match(str(n), p2))]

    if found.empty:
        print(f"    No result found yet vs '{p2}'")
        return {
            'status': 'opponent_not_found',
            'outcome_detail': f"No result found yet vs '{p2}'",
            'ta_player_slug': slug1,
        }

    # If multiple matches vs same opponent in window, pick closest to logged date
    if len(found) > 1:
        found = found.copy()
        found['_date_diff'] = (found['date'] - match_date).abs()
        found = found.sort_values('_date_diff')

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
        return {
            'status': 'unexpected_result',
            'outcome_detail': f"Unexpected result value '{result}'",
            'ta_player_slug': slug1,
            'ta_match_date_found': match_date_found,
            'ta_event_found': str(row.get('event', '')),
            'ta_round_found': str(row.get('round', '')),
        }

    print(f"    Found result ({match_date_found}): {winner_name} won  |  score: {score}")
    return {
        'status': 'matched_and_settled',
        'actual_winner': actual_winner,
        'score': score,
        'settled_at': datetime.now().isoformat(),
        'ta_player_slug': slug1,
        'ta_match_date_found': match_date_found,
        'ta_event_found': str(row.get('event', '')),
        'ta_round_found': str(row.get('round', '')),
        'outcome_detail': f"{winner_name} won",
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def show_stats(df: pd.DataFrame):
    settled = df[df['actual_winner'].notna()].copy()
    if settled.empty:
        print("No settled predictions yet.")
        return

    n_all = len(settled)
    print(f"\n{'='*70}")
    print(f"  LIVE ACCURACY REPORT")
    print(f"{'='*70}")
    print(f"  Total settled: {n_all}  (includes ITF/no-model entries)")

    # Clean stats: only rows with complete features and a real model prediction
    clean = settled[
        settled['model_correct'].notna() &
        settled['features_complete'].fillna(True).astype(bool)
    ].copy()

    if clean.empty:
        print("  No complete-feature predictions settled yet.")
        return

    # Exclude 50/50 market (market not making a pick) for fair comparison
    has_market_pick = clean['market_p1_prob'] != 0.50
    clean_no5050 = clean[has_market_pick]
    n_5050 = len(clean) - len(clean_no5050)

    n = len(clean_no5050)
    model_acc  = clean_no5050['model_correct'].mean()
    market_acc = clean_no5050['market_correct'].mean()
    print(f"\n  [Complete features, market has pick — {n} matches]")
    print(f"  (Excluded {n_5050} matches where market was 50/50)")
    print(f"  Model:  {model_acc:.1%}  ({int(clean_no5050['model_correct'].sum())}/{n})")
    print(f"  Market: {market_acc:.1%}  ({int(clean_no5050['market_correct'].sum())}/{n})")
    print(f"  Edge:   {model_acc - market_acc:+.1%}")

    # By model version
    if 'model_version' in clean_no5050.columns:
        print(f"\n  {'─'*66}")
        print(f"  By model version:")
        for ver, grp in clean_no5050.groupby('model_version'):
            if grp.empty:
                continue
            ma = grp['model_correct'].mean()
            mk = grp['market_correct'].mean()
            edge = ma - mk
            print(f"    {ver}: model {ma:.1%}  market {mk:.1%}  edge {edge:+.1%}  ({len(grp)} matches)")

    # By surface
    if 'surface' in clean_no5050.columns and clean_no5050['surface'].notna().any():
        print(f"\n  {'─'*66}")
        print(f"  By surface:")
        for surf, grp in clean_no5050.groupby('surface'):
            if grp.empty:
                continue
            ma = grp['model_correct'].mean()
            mk = grp['market_correct'].mean()
            print(f"    {surf}: model {ma:.1%}  market {mk:.1%}  edge {ma-mk:+.1%}  ({len(grp)} matches)")

    # Feature completeness summary
    all_settled = df[df['actual_winner'].notna()]
    incomplete = all_settled[all_settled['features_complete'].fillna(True).astype(bool) == False]
    if len(incomplete) > 0:
        print(f"\n  {'─'*66}")
        print(f"  Feature completeness:")
        print(f"    Complete: {len(all_settled) - len(incomplete)}  |  Incomplete (excluded): {len(incomplete)}")
        # Show which features defaulted most
        all_defaults = []
        for _, r in incomplete.iterrows():
            if pd.notna(r.get('defaulted_features', '')) and str(r.get('defaulted_features', '')).strip():
                all_defaults.extend(str(r['defaulted_features']).split(','))
        if all_defaults:
            from collections import Counter
            top = Counter(all_defaults).most_common(5)
            print(f"    Top defaulted features:")
            for feat, cnt in top:
                print(f"      {feat.strip()}: {cnt}x")

    print(f"{'='*70}")


def run(
    dry_run: bool = False,
    stats_only: bool = False,
    stale_days: int = 7,
    include_market_only: bool = False,
    run_id: str | None = None,
    record_run_history: bool | None = None,
):
    started = utc_now()
    owns_run_id = run_id is None
    if run_id is None:
        run_id = make_run_id(started, prefix='settle')
    if record_run_history is None:
        record_run_history = owns_run_id

    summary = {
        'run_id': run_id,
        'run_kind': 'auto_settle',
        'started_at': started.replace(microsecond=0).isoformat(),
        'completed_at': '',
        'status': 'running',
        'auto_settle_enabled': True,
        'rankings_refresh_enabled': '',
        'odds_rows_fetched': 0,
        'odds_rows_candidate': 0,
        'feature_rows_total': 0,
        'feature_rows_ok': 0,
        'feature_rows_skipped': 0,
        'feature_skip_reason_summary': {},
        'prediction_rows_total': 0,
        'prediction_rows_success': 0,
        'prediction_rows_error': 0,
        'prediction_log_attempts': 0,
        'prediction_log_created': 0,
        'prediction_log_updated': 0,
        'prediction_log_skipped_incomplete': 0,
        'bet_opportunities': 0,
        'bets_logged': 0,
        'settlement_candidates': 0,
        'settlement_newly_settled': 0,
        'settlement_auto_settled_bets': 0,
        'settlement_reason_summary': {},
        'error_message': '',
    }
    if record_run_history:
        upsert_run_history(summary)

    if not LOG_PATH.exists():
        print("No prediction_log.csv found.")
        summary['status'] = 'missing_prediction_log'
        summary['completed_at'] = utc_now().replace(microsecond=0).isoformat()
        if record_run_history:
            upsert_run_history(summary)
        return summary

    df = upgrade_prediction_log(LOG_PATH, stale_days=stale_days, write=not dry_run)

    if stats_only:
        show_stats(df)
        summary['status'] = 'stats_only'
        summary['completed_at'] = utc_now().replace(microsecond=0).isoformat()
        if record_run_history:
            upsert_run_history(summary)
        return summary

    pending = df[df['actual_winner'].isna()].copy()
    if 'record_status' in pending.columns:
        pending = pending[~pending['record_status'].isin(['stale_no_model'])]
    if not include_market_only:
        pending = pending[pending['model_p1_prob'].notna()]
    summary['settlement_candidates'] = len(pending)
    if pending.empty:
        print("No unsettled predictions.")
        show_stats(df)
        summary['status'] = 'success'
        summary['completed_at'] = utc_now().replace(microsecond=0).isoformat()
        if record_run_history:
            upsert_run_history(summary)
        return summary

    print(f"Found {len(pending)} unsettled prediction(s) to check\n")

    # Load player mapping once
    calc = TAFeatureCalculator.__new__(TAFeatureCalculator)
    calc.player_slug_map = TAFeatureCalculator._load_player_mapping(calc)
    tracker = BetTracker(str(Path(__file__).parent / "logs"))
    session_cache = {}

    newly_settled = 0
    newly_settled_bets = 0
    reason_counts = Counter()

    for idx, row in pending.iterrows():
        p1 = str(row['p1'])
        p2 = str(row['p2'])
        match_date = str(row.get('match_date', ''))
        print(f"\n[{idx}] {p1} vs {p2}  ({row.get('tournament', '')}  {match_date})")
        print(f"     Model: {float(row['model_p1_prob']):.0%} P1 | Market: {float(row['market_p1_prob']):.0%} P1")

        result = try_settle_from_ta(
            p1,
            p2,
            match_date,
            calc,
            session_cache=session_cache,
            dry_run=dry_run,
        )
        outcome_code = result.get('status', 'unknown')
        reason_counts[outcome_code] += 1
        log_settlement_event(
            run_id=run_id,
            dry_run=dry_run,
            row_index=idx,
            record_status_before=str(row.get('record_status', '')),
            match_uid=row.get('match_uid', ''),
            prediction_uid=row.get('prediction_uid', ''),
            match_date=match_date,
            match_start_time=str(row.get('match_start_time', '')),
            tournament=str(row.get('tournament', '')),
            round_code=str(row.get('round', '')),
            surface=str(row.get('surface', '')),
            p1=p1,
            p2=p2,
            model_version=str(row.get('model_version', '')),
            ta_player_slug=result.get('ta_player_slug', ''),
            outcome_code=outcome_code,
            outcome_detail=result.get('outcome_detail', ''),
            ta_match_date_found=result.get('ta_match_date_found', ''),
            ta_event_found=result.get('ta_event_found', ''),
            ta_round_found=result.get('ta_round_found', ''),
            actual_winner=result.get('actual_winner'),
            score=result.get('score', ''),
        )

        if outcome_code != 'matched_and_settled':
            print(f"  → Not settled yet ({outcome_code})")
            continue

        if dry_run:
            winner_label = p1 if result['actual_winner'] == 1 else p2
            print(f"  [DRY RUN] Would settle: winner={winner_label}, score={result['score']}")
            continue

        # Write result back
        df.at[idx, 'actual_winner'] = result['actual_winner']
        df.at[idx, 'score'] = result['score']
        df.at[idx, 'settled_at'] = result['settled_at']

        import math
        model_p1_raw = row['model_p1_prob']
        market_p1 = float(row['market_p1_prob'])
        w = result['actual_winner']
        # Only score model_correct if there was actually a model prediction
        if pd.isna(model_p1_raw) or (isinstance(model_p1_raw, float) and math.isnan(model_p1_raw)):
            model_correct = float('nan')
        else:
            model_p1 = float(model_p1_raw)
            model_correct = int((w == 1 and model_p1 > 0.5) or (w == 2 and model_p1 < 0.5))
        market_correct = int((w == 1 and market_p1 > 0.5) or (w == 2 and market_p1 < 0.5))
        if not pd.isna(model_correct):
            df.at[idx, 'model_correct'] = model_correct
        df.at[idx, 'market_correct'] = market_correct

        # Score XGBoost if prediction exists
        xgb_p1_raw = row.get('xgb_p1_prob')
        if 'xgb_correct' not in df.columns:
            df['xgb_correct'] = None
        if pd.notna(xgb_p1_raw) and not (isinstance(xgb_p1_raw, float) and math.isnan(xgb_p1_raw)):
            xgb_p1 = float(xgb_p1_raw)
            xgb_correct = int((w == 1 and xgb_p1 > 0.5) or (w == 2 and xgb_p1 < 0.5))
            df.at[idx, 'xgb_correct'] = xgb_correct

        # Score Random Forest if prediction exists
        rf_p1_raw = row.get('rf_p1_prob')
        if 'rf_correct' not in df.columns:
            df['rf_correct'] = None
        if pd.notna(rf_p1_raw) and not (isinstance(rf_p1_raw, float) and math.isnan(rf_p1_raw)):
            rf_p1 = float(rf_p1_raw)
            rf_correct = int((w == 1 and rf_p1 > 0.5) or (w == 2 and rf_p1 < 0.5))
            df.at[idx, 'rf_correct'] = rf_correct

        winner_name = p1 if w == 1 else p2
        model_str = ('✓' if model_correct else '✗') if not (isinstance(model_correct, float) and math.isnan(model_correct)) else 'N/A'
        print(f"  ✓ Settled: {winner_name} won | Model {model_str} | Market {'✓' if market_correct else '✗'}")
        newly_settled += 1

        settled_bets = tracker.settle_pending_bets_for_match(
            match_uid=row.get('match_uid'),
            p1=p1,
            p2=p2,
            actual_winner=w,
            notes=f"Auto-settled from Tennis Abstract | score={result['score']}",
        )
        if settled_bets:
            print(f"  💰 Auto-settled {settled_bets} pending bet(s)")
            newly_settled_bets += settled_bets

        # Rate limit between players
        time.sleep(3.0)

    if not dry_run:
        df.to_csv(LOG_PATH, index=False)
        df = upgrade_prediction_log(LOG_PATH, stale_days=stale_days, write=True)
        print(f"\nSaved {newly_settled} newly settled prediction(s) to prediction_log.csv")
        if newly_settled_bets:
            print(f"Auto-settled {newly_settled_bets} pending tracked bet(s)")

    summary['settlement_newly_settled'] = newly_settled
    summary['settlement_auto_settled_bets'] = newly_settled_bets
    summary['settlement_reason_summary'] = dict(reason_counts)
    summary['status'] = 'success'
    summary['completed_at'] = utc_now().replace(microsecond=0).isoformat()
    if record_run_history:
        upsert_run_history(summary)

    show_stats(df)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Auto-settle predictions from Tennis Abstract results')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be settled without writing')
    parser.add_argument('--stats', action='store_true', help='Show accuracy stats only')
    parser.add_argument('--stale-days', type=int, default=7, help='Mark legacy no-model rows older than this as stale and skip them')
    parser.add_argument('--include-market-only', action='store_true', help='Also check old market-only rows with no model prediction')
    args = parser.parse_args()

    run(
        dry_run=args.dry_run,
        stats_only=args.stats,
        stale_days=args.stale_days,
        include_market_only=args.include_market_only,
    )
