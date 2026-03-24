#!/usr/bin/env python3
"""Append or update a prediction in prediction_log.csv."""
import pandas as pd
import os
from datetime import datetime, date

LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'prediction_log.csv')

COLUMNS = [
    'logged_at', 'match_date', 'tournament', 'surface', 'level', 'round',
    'p1', 'p2',
    'model_p1_prob', 'model_p2_prob',
    'market_p1_prob', 'market_p2_prob',
    'p1_odds_american', 'p2_odds_american',
    'edge_p1', 'model_version', 'features_complete', 'defaulted_features',
    'actual_winner', 'score', 'settled_at', 'model_correct', 'market_correct',
]


def _parse_match_date(match_date) -> str:
    """
    Convert match_date to YYYY-MM-DD string.
    If it looks like a time string (e.g. "8:00 AM", "Today 7:30 PM"), use today's date.
    """
    s = str(match_date).strip()
    # Already looks like a date
    if len(s) >= 10 and s[4] == '-':
        return s[:10]
    # Looks like a time / relative label → use today
    return date.today().isoformat()


def log_prediction(
    p1: str, p2: str,
    tournament: str, surface: str, level: str, round_code: str,
    match_date,
    model_p1_prob: float, model_p2_prob: float,
    market_p1_prob: float, market_p2_prob: float,
    p1_odds_american: float = None, p2_odds_american: float = None,
    model_version: str = None,
    actual_winner: int = None, score: str = None,
    features_complete: bool = True,
    defaulted_features: str = '',
    allow_update: bool = True,
):
    """
    Append or update a prediction row.

    If allow_update=True (default) and a row with the same p1+p2+match_date
    already exists without a settled result, overwrite it with fresh probabilities
    and odds rather than adding a duplicate.

    features_complete=False marks predictions that used defaulted feature values
    and should be excluded from accuracy analysis.
    """
    # Default model_version from registry if not provided
    if model_version is None:
        try:
            import json
            registry_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'model_registry.json')
            with open(registry_path) as _f:
                model_version = json.load(_f).get('current_version', 'unknown')
        except Exception:
            model_version = 'unknown'

    match_date_str = _parse_match_date(match_date)

    settled_at = datetime.now().isoformat() if actual_winner is not None else None
    model_correct = None
    market_correct = None
    if actual_winner is not None and model_p1_prob is not None:
        model_correct = int((actual_winner == 1) == (model_p1_prob > 0.5))
        market_correct = int((actual_winner == 1) == (market_p1_prob > 0.5))

    row = {
        'logged_at': datetime.now().isoformat(),
        'match_date': match_date_str,
        'tournament': tournament,
        'surface': surface,
        'level': level,
        'round': round_code,
        'p1': p1, 'p2': p2,
        'model_p1_prob': round(model_p1_prob, 4) if model_p1_prob is not None else None,
        'model_p2_prob': round(model_p2_prob, 4) if model_p2_prob is not None else None,
        'market_p1_prob': round(market_p1_prob, 4) if market_p1_prob is not None else None,
        'market_p2_prob': round(market_p2_prob, 4) if market_p2_prob is not None else None,
        'p1_odds_american': p1_odds_american,
        'p2_odds_american': p2_odds_american,
        'edge_p1': round(model_p1_prob - market_p1_prob, 4) if (model_p1_prob is not None and market_p1_prob is not None) else None,
        'model_version': model_version,
        'features_complete': features_complete,
        'defaulted_features': defaulted_features or '',
        'actual_winner': actual_winner,
        'score': score,
        'settled_at': settled_at,
        'model_correct': model_correct,
        'market_correct': market_correct,
    }

    if os.path.exists(LOG_PATH):
        df = pd.read_csv(LOG_PATH)

        # Ensure new columns exist in older logs
        if 'features_complete' not in df.columns:
            df.insert(df.columns.get_loc('model_version') + 1, 'features_complete', True)
        if 'defaulted_features' not in df.columns:
            df.insert(df.columns.get_loc('features_complete') + 1, 'defaulted_features', '')

        # Dedup: find unsettled row for same matchup on same date
        if allow_update:
            p1_lower = str(p1).lower().strip()
            p2_lower = str(p2).lower().strip()
            mask = (
                (df['match_date'] == match_date_str) &
                (df['actual_winner'].isna()) &
                (
                    (df['p1'].str.lower().str.strip() == p1_lower) &
                    (df['p2'].str.lower().str.strip() == p2_lower)
                    |
                    (df['p1'].str.lower().str.strip() == p2_lower) &
                    (df['p2'].str.lower().str.strip() == p1_lower)
                )
            )
            if mask.any():
                idx = df[mask].index[0]
                # Preserve original market odds — opening lines are less efficient
                # and more valuable for edge analysis than lines closer to match time.
                PRESERVE_IF_SET = {'market_p1_prob', 'market_p2_prob',
                                   'p1_odds_american', 'p2_odds_american'}
                for col, val in row.items():
                    if col not in df.columns:
                        continue
                    if col in PRESERVE_IF_SET and pd.notna(df.at[idx, col]):
                        continue  # keep original market odds
                    df.at[idx, col] = val
                df.to_csv(LOG_PATH, index=False)
                return

        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row], columns=COLUMNS)

    df.to_csv(LOG_PATH, index=False)
