#!/usr/bin/env python3
"""Append or update a prediction in prediction_log.csv."""
import pandas as pd
import os
from datetime import datetime, date
from pathlib import Path

from logging_utils import (
    append_unique_row,
    build_match_uid,
    build_odds_snapshot_uid,
    build_prediction_uid,
    ensure_csv_columns,
)

LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'prediction_log.csv')
BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
SNAPSHOT_LOG_PATH = BASE_DIR / 'prediction_snapshots.csv'
ODDS_HISTORY_LOG_PATH = BASE_DIR / 'odds_history.csv'
SCHEMA_VERSION = 'prediction_log_v2'

COLUMNS = [
    'logged_at', 'match_date', 'tournament', 'surface', 'level', 'round',
    'p1', 'p2',
    'p1_rank', 'p2_rank',
    'model_p1_prob', 'model_p2_prob',
    'xgb_p1_prob', 'xgb_p2_prob',
    'rf_p1_prob', 'rf_p2_prob',
    'market_p1_prob', 'market_p2_prob',
    'p1_odds_american', 'p2_odds_american',
    'p1_odds_decimal', 'p2_odds_decimal',
    'spread_handicap', 'spread_odds_p1', 'spread_odds_p2',
    'total_games', 'total_odds_over', 'total_odds_under',
    'edge_p1', 'primary_model_family', 'model_version', 'nn_model_version',
    'xgb_model_version', 'rf_model_version',
    'logging_schema_version', 'logging_quality', 'rescore_quality',
    'record_status', 'record_note',
    'features_complete', 'defaulted_features',
    'odds_scraped_at', 'match_start_time',
    'actual_winner', 'score', 'settled_at', 'model_correct', 'market_correct',
    'xgb_correct',
    'rf_correct',
    'run_id', 'latest_run_id',
    'match_uid',
    'feature_snapshot_id', 'latest_feature_snapshot_id',
    'prediction_uid', 'latest_prediction_uid',
    'latest_logged_at', 'latest_model_version_seen',
    'latest_nn_model_version_seen', 'latest_xgb_model_version_seen', 'latest_rf_model_version_seen',
    'latest_odds_scraped_at', 'latest_match_start_time', 'latest_match_date',
]

SNAPSHOT_COLUMNS = [
    'prediction_uid',
    'logged_at', 'match_date', 'tournament', 'surface', 'level', 'round',
    'p1', 'p2',
    'run_id', 'match_uid', 'feature_snapshot_id',
    'p1_rank', 'p2_rank',
    'model_p1_prob', 'model_p2_prob',
    'xgb_p1_prob', 'xgb_p2_prob',
    'rf_p1_prob', 'rf_p2_prob',
    'market_p1_prob', 'market_p2_prob',
    'p1_odds_american', 'p2_odds_american',
    'p1_odds_decimal', 'p2_odds_decimal',
    'spread_handicap', 'spread_odds_p1', 'spread_odds_p2',
    'total_games', 'total_odds_over', 'total_odds_under',
    'edge_p1', 'primary_model_family', 'model_version', 'nn_model_version',
    'xgb_model_version', 'rf_model_version',
    'logging_schema_version', 'logging_quality', 'rescore_quality',
    'record_status', 'record_note',
    'features_complete', 'defaulted_features',
    'odds_scraped_at', 'match_start_time',
    'actual_winner', 'score', 'settled_at', 'model_correct', 'market_correct',
    'xgb_correct', 'rf_correct',
    'snapshot_role',
]

ODDS_HISTORY_COLUMNS = [
    'odds_snapshot_uid',
    'logged_at',
    'run_id',
    'match_uid',
    'match_date',
    'tournament',
    'surface',
    'level',
    'round',
    'p1',
    'p2',
    'odds_scraped_at',
    'match_start_time',
    'market_p1_prob',
    'market_p2_prob',
    'p1_odds_american',
    'p2_odds_american',
    'p1_odds_decimal',
    'p2_odds_decimal',
    'spread_handicap',
    'spread_odds_p1',
    'spread_odds_p2',
    'total_games',
    'total_odds_over',
    'total_odds_under',
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


def _load_registry_versions() -> tuple[str, str, str]:
    """Return the current NN/XGB/RF versions from the model registry."""
    try:
        import json
        registry_path = BASE_DIR / 'models' / 'model_registry.json'
        with open(registry_path) as f:
            registry = json.load(f)
        return (
            registry.get('current_version', 'unknown'),
            registry.get('xgboost', {}).get('current_version', 'unknown'),
            registry.get('random_forest', {}).get('current_version', 'unknown'),
        )
    except Exception:
        return ('unknown', 'unknown', 'unknown')


def _coerce_bool_series(values: pd.Series, default: bool = False) -> pd.Series:
    """Convert common CSV-ish truthy values into a boolean Series."""
    normalized = values.fillna(default).astype(str).str.strip().str.lower()
    truthy = {'1', 'true', 't', 'yes', 'y'}
    falsy = {'0', 'false', 'f', 'no', 'n', ''}
    return normalized.map(lambda v: True if v in truthy else False if v in falsy else default)


def _clean_text(value) -> str:
    """Return a safe stripped string for scalars that may be pd.NA/None."""
    if pd.isna(value):
        return ''
    return str(value).strip()


def upgrade_prediction_log(path: Path | None = None, stale_days: int = 7, write: bool = True) -> pd.DataFrame:
    """
    Backfill logging metadata on the prediction log.

    Legacy rows get synthetic match/prediction ids where possible plus explicit
    quality/status markers so downstream tooling can distinguish reliable
    snapshot-based rows from older fallback-only history.
    """
    target = path or Path(LOG_PATH)
    df = ensure_csv_columns(target, COLUMNS)

    nn_current, xgb_current, rf_current = _load_registry_versions()

    for idx, row in df.iterrows():
        match_date_str = _parse_match_date(row.get('match_date', ''))
        df.at[idx, 'match_date'] = match_date_str

        match_uid = _clean_text(row.get('match_uid', ''))
        if not match_uid:
            match_uid = build_match_uid(
                row.get('p1', ''),
                row.get('p2', ''),
                match_date_str,
                row.get('tournament', ''),
                row.get('round', ''),
                row.get('surface', ''),
            )
            df.at[idx, 'match_uid'] = match_uid

        prediction_uid = _clean_text(row.get('prediction_uid', ''))
        model_version = _clean_text(row.get('model_version', '')) or nn_current
        if not prediction_uid:
            prediction_uid = build_prediction_uid(
                match_uid,
                model_version,
                _clean_text(row.get('logged_at', '')),
                row.get('p1', ''),
                row.get('p2', ''),
            )
            df.at[idx, 'prediction_uid'] = prediction_uid

        if not _clean_text(row.get('primary_model_family', '')):
            df.at[idx, 'primary_model_family'] = 'nn'
        df.at[idx, 'model_version'] = model_version

        nn_version = _clean_text(row.get('nn_model_version', ''))
        xgb_version = _clean_text(row.get('xgb_model_version', ''))
        rf_version = _clean_text(row.get('rf_model_version', ''))

        df.at[idx, 'nn_model_version'] = nn_version or model_version
        if not xgb_version and pd.notna(row.get('xgb_p1_prob')):
            df.at[idx, 'xgb_model_version'] = 'unknown_legacy'
        elif not xgb_version:
            df.at[idx, 'xgb_model_version'] = ''
        if not rf_version and pd.notna(row.get('rf_p1_prob')):
            df.at[idx, 'rf_model_version'] = 'unknown_legacy'
        elif not rf_version:
            df.at[idx, 'rf_model_version'] = ''

        has_run_id = bool(_clean_text(row.get('run_id', '')))
        has_feature_snapshot = bool(_clean_text(row.get('feature_snapshot_id', '')))
        has_lineage = has_run_id and has_feature_snapshot

        df.at[idx, 'logging_schema_version'] = SCHEMA_VERSION if has_lineage else 'legacy_v1'
        df.at[idx, 'logging_quality'] = 'snapshot_v2' if has_lineage else 'legacy_backfilled'
        df.at[idx, 'rescore_quality'] = 'exact_feature_snapshot' if has_feature_snapshot else 'legacy_fallback_match'

        row_logged_at = pd.to_datetime(row.get('logged_at'), errors='coerce')
        row_match_date = pd.to_datetime(match_date_str, errors='coerce')
        effective_dt = row_logged_at if pd.notna(row_logged_at) else row_match_date
        age_days = None
        if pd.notna(effective_dt):
            age_days = (pd.Timestamp.now() - effective_dt).days

        has_model_prediction = pd.notna(row.get('model_p1_prob'))
        is_settled = pd.notna(row.get('actual_winner'))
        if is_settled:
            status = 'settled'
            note = ''
        elif not has_model_prediction and age_days is not None and age_days >= stale_days:
            status = 'stale_no_model'
            note = f"Legacy row without model output older than {stale_days} days"
        elif not has_model_prediction:
            status = 'pending_no_model'
            note = 'Pending market-only row'
        elif has_lineage:
            status = 'pending'
            note = ''
        else:
            status = 'pending_legacy'
            note = 'Legacy row without exact feature snapshot lineage'

        df.at[idx, 'record_status'] = status
        df.at[idx, 'record_note'] = note

        if not _clean_text(row.get('latest_logged_at', '')):
            df.at[idx, 'latest_logged_at'] = row.get('logged_at', '')
        if not _clean_text(row.get('latest_model_version_seen', '')):
            df.at[idx, 'latest_model_version_seen'] = model_version
        if not _clean_text(row.get('latest_nn_model_version_seen', '')):
            df.at[idx, 'latest_nn_model_version_seen'] = df.at[idx, 'nn_model_version']
        if not _clean_text(row.get('latest_xgb_model_version_seen', '')):
            df.at[idx, 'latest_xgb_model_version_seen'] = df.at[idx, 'xgb_model_version']
        if not _clean_text(row.get('latest_rf_model_version_seen', '')):
            df.at[idx, 'latest_rf_model_version_seen'] = df.at[idx, 'rf_model_version']
        if not _clean_text(row.get('latest_match_date', '')):
            df.at[idx, 'latest_match_date'] = match_date_str
        if not _clean_text(row.get('latest_prediction_uid', '')):
            df.at[idx, 'latest_prediction_uid'] = prediction_uid
        if not _clean_text(row.get('latest_run_id', '')):
            df.at[idx, 'latest_run_id'] = row.get('run_id', '')
        if not _clean_text(row.get('latest_feature_snapshot_id', '')):
            df.at[idx, 'latest_feature_snapshot_id'] = row.get('feature_snapshot_id', '')
        if not _clean_text(row.get('latest_odds_scraped_at', '')):
            df.at[idx, 'latest_odds_scraped_at'] = row.get('odds_scraped_at', '')
        if not _clean_text(row.get('latest_match_start_time', '')):
            df.at[idx, 'latest_match_start_time'] = row.get('match_start_time', '')

    if 'features_complete' in df.columns:
        df['features_complete'] = _coerce_bool_series(df['features_complete'], default=True)

    if write:
        df.to_csv(target, index=False)
    return df


def log_prediction(
    p1: str, p2: str,
    tournament: str, surface: str, level: str, round_code: str,
    match_date,
    model_p1_prob: float, model_p2_prob: float,
    market_p1_prob: float, market_p2_prob: float,
    run_id: str = None,
    match_uid: str = None,
    feature_snapshot_id: str = None,
    p1_rank: float = None, p2_rank: float = None,
    p1_odds_american: float = None, p2_odds_american: float = None,
    p1_odds_decimal: float = None, p2_odds_decimal: float = None,
    spread_handicap: float = None, spread_odds_p1: float = None, spread_odds_p2: float = None,
    total_games: float = None, total_odds_over: float = None, total_odds_under: float = None,
    xgb_p1_prob: float = None, xgb_p2_prob: float = None,
    rf_p1_prob: float = None, rf_p2_prob: float = None,
    model_version: str = None,
    nn_model_version: str = None,
    xgb_model_version: str = None,
    rf_model_version: str = None,
    odds_scraped_at: str = None,
    match_start_time: str = None,
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
    current_nn, current_xgb, current_rf = _load_registry_versions()
    if model_version is None:
        model_version = current_nn
    if nn_model_version is None:
        nn_model_version = model_version
    if xgb_model_version is None and xgb_p1_prob is not None:
        xgb_model_version = current_xgb
    if rf_model_version is None and rf_p1_prob is not None:
        rf_model_version = current_rf

    match_date_str = _parse_match_date(match_date)
    logged_at = datetime.now().isoformat()
    prediction_uid = build_prediction_uid(match_uid or '', model_version, logged_at, p1, p2)

    settled_at = datetime.now().isoformat() if actual_winner is not None else None
    model_correct = None
    market_correct = None
    xgb_correct = None
    rf_correct = None
    if actual_winner is not None and model_p1_prob is not None:
        model_correct = int((actual_winner == 1) == (model_p1_prob > 0.5))
        market_correct = int((actual_winner == 1) == (market_p1_prob > 0.5))
    if actual_winner is not None and xgb_p1_prob is not None:
        xgb_correct = int((actual_winner == 1) == (xgb_p1_prob > 0.5))
    if actual_winner is not None and rf_p1_prob is not None:
        rf_correct = int((actual_winner == 1) == (rf_p1_prob > 0.5))

    row = {
        'logged_at': logged_at,
        'match_date': match_date_str,
        'tournament': tournament,
        'surface': surface,
        'level': level,
        'round': round_code,
        'p1': p1, 'p2': p2,
        'p1_rank': p1_rank,
        'p2_rank': p2_rank,
        'model_p1_prob': round(model_p1_prob, 4) if model_p1_prob is not None else None,
        'model_p2_prob': round(model_p2_prob, 4) if model_p2_prob is not None else None,
        'xgb_p1_prob': round(xgb_p1_prob, 4) if xgb_p1_prob is not None else None,
        'xgb_p2_prob': round(xgb_p2_prob, 4) if xgb_p2_prob is not None else None,
        'rf_p1_prob': round(rf_p1_prob, 4) if rf_p1_prob is not None else None,
        'rf_p2_prob': round(rf_p2_prob, 4) if rf_p2_prob is not None else None,
        'market_p1_prob': round(market_p1_prob, 4) if market_p1_prob is not None else None,
        'market_p2_prob': round(market_p2_prob, 4) if market_p2_prob is not None else None,
        'p1_odds_american': p1_odds_american,
        'p2_odds_american': p2_odds_american,
        'p1_odds_decimal': p1_odds_decimal,
        'p2_odds_decimal': p2_odds_decimal,
        'spread_handicap': spread_handicap,
        'spread_odds_p1': spread_odds_p1,
        'spread_odds_p2': spread_odds_p2,
        'total_games': total_games,
        'total_odds_over': total_odds_over,
        'total_odds_under': total_odds_under,
        'edge_p1': round(model_p1_prob - market_p1_prob, 4) if (model_p1_prob is not None and market_p1_prob is not None) else None,
        'primary_model_family': 'nn',
        'model_version': model_version,
        'nn_model_version': nn_model_version,
        'xgb_model_version': xgb_model_version or '',
        'rf_model_version': rf_model_version or '',
        'logging_schema_version': SCHEMA_VERSION if (run_id and feature_snapshot_id) else 'legacy_v1',
        'logging_quality': 'snapshot_v2' if (run_id and feature_snapshot_id) else 'legacy_backfilled',
        'rescore_quality': 'exact_feature_snapshot' if feature_snapshot_id else 'legacy_fallback_match',
        'record_status': 'settled' if actual_winner is not None else 'pending',
        'record_note': '',
        'features_complete': features_complete,
        'defaulted_features': defaulted_features or '',
        'odds_scraped_at': odds_scraped_at,
        'match_start_time': match_start_time,
        'actual_winner': actual_winner,
        'score': score,
        'settled_at': settled_at,
        'model_correct': model_correct,
        'market_correct': market_correct,
        'xgb_correct': xgb_correct,
        'rf_correct': rf_correct,
        'run_id': run_id,
        'latest_run_id': run_id,
        'match_uid': match_uid,
        'feature_snapshot_id': feature_snapshot_id,
        'latest_feature_snapshot_id': feature_snapshot_id,
        'prediction_uid': prediction_uid,
        'latest_prediction_uid': prediction_uid,
        'latest_logged_at': logged_at,
        'latest_model_version_seen': model_version,
        'latest_nn_model_version_seen': nn_model_version,
        'latest_xgb_model_version_seen': xgb_model_version or '',
        'latest_rf_model_version_seen': rf_model_version or '',
        'latest_odds_scraped_at': odds_scraped_at,
        'latest_match_start_time': match_start_time,
        'latest_match_date': match_date_str,
    }

    snapshot_row = {
        key: row.get(key)
        for key in SNAPSHOT_COLUMNS
        if key != 'snapshot_role'
    }
    snapshot_row['snapshot_role'] = 'live'
    append_unique_row(SNAPSHOT_LOG_PATH, snapshot_row, SNAPSHOT_COLUMNS, unique_key='prediction_uid')

    odds_snapshot_uid = build_odds_snapshot_uid(
        match_uid or '',
        odds_scraped_at or logged_at,
        match_start_time or '',
        p1_odds_decimal,
        p2_odds_decimal,
    )
    has_market_snapshot = any(
        value is not None and str(value) != ''
        for value in [
            market_p1_prob, market_p2_prob,
            p1_odds_american, p2_odds_american,
            p1_odds_decimal, p2_odds_decimal,
            spread_handicap, total_games,
        ]
    )
    if has_market_snapshot:
        odds_row = {
            'odds_snapshot_uid': odds_snapshot_uid,
            'logged_at': logged_at,
            'run_id': run_id,
            'match_uid': match_uid,
            'match_date': match_date_str,
            'tournament': tournament,
            'surface': surface,
            'level': level,
            'round': round_code,
            'p1': p1,
            'p2': p2,
            'odds_scraped_at': odds_scraped_at,
            'match_start_time': match_start_time,
            'market_p1_prob': round(market_p1_prob, 4) if market_p1_prob is not None else None,
            'market_p2_prob': round(market_p2_prob, 4) if market_p2_prob is not None else None,
            'p1_odds_american': p1_odds_american,
            'p2_odds_american': p2_odds_american,
            'p1_odds_decimal': p1_odds_decimal,
            'p2_odds_decimal': p2_odds_decimal,
            'spread_handicap': spread_handicap,
            'spread_odds_p1': spread_odds_p1,
            'spread_odds_p2': spread_odds_p2,
            'total_games': total_games,
            'total_odds_over': total_odds_over,
            'total_odds_under': total_odds_under,
        }
        append_unique_row(ODDS_HISTORY_LOG_PATH, odds_row, ODDS_HISTORY_COLUMNS, unique_key='odds_snapshot_uid')

    if os.path.exists(LOG_PATH):
        df = upgrade_prediction_log(Path(LOG_PATH), write=False)

        # Ensure new columns exist in older logs
        # Dedup: find unsettled row for same matchup within ±3 days
        # (Bovada sometimes shifts match dates between pipeline runs)
        if allow_update:
            p1_lower = str(p1).lower().strip()
            p2_lower = str(p2).lower().strip()
            players_match = (
                (df['actual_winner'].isna()) &
                (
                    (df['p1'].str.lower().str.strip() == p1_lower) &
                    (df['p2'].str.lower().str.strip() == p2_lower)
                    |
                    (df['p1'].str.lower().str.strip() == p2_lower) &
                    (df['p2'].str.lower().str.strip() == p1_lower)
                )
            )
            # Date window: same players within ±3 days
            if players_match.any():
                from datetime import timedelta
                try:
                    new_date = datetime.strptime(match_date_str, '%Y-%m-%d').date()
                    existing_dates = pd.to_datetime(df.loc[players_match, 'match_date'], errors='coerce').dt.date
                    date_close = existing_dates.apply(
                        lambda d: abs((d - new_date).days) <= 3 if pd.notna(d) else False
                    )
                    players_match.loc[players_match] = date_close.values
                except Exception:
                    pass  # fall back to players-only match
            mask = players_match
            if mask.any():
                idx = df[mask].index[0]
                # Preserve original odds + model probs — opening lines are less efficient
                # and more valuable for edge analysis than lines closer to match time.
                PRESERVE_IF_SET = {'model_p1_prob', 'model_p2_prob',
                                   'xgb_p1_prob', 'xgb_p2_prob',
                                   'rf_p1_prob', 'rf_p2_prob',
                                   'market_p1_prob', 'market_p2_prob',
                                   'p1_odds_american', 'p2_odds_american',
                                   'p1_odds_decimal', 'p2_odds_decimal',
                                   'spread_handicap', 'spread_odds_p1', 'spread_odds_p2',
                                   'total_games', 'total_odds_over', 'total_odds_under',
                                   'edge_p1', 'primary_model_family', 'model_version', 'nn_model_version',
                                   'xgb_model_version', 'rf_model_version',
                                   'p1_rank', 'p2_rank',
                                   'odds_scraped_at', 'match_start_time',
                                   'run_id', 'feature_snapshot_id', 'prediction_uid'}
                for col, val in row.items():
                    if col not in df.columns:
                        continue
                    if col in PRESERVE_IF_SET and pd.notna(df.at[idx, col]):
                        continue  # keep original market odds
                    df.at[idx, col] = val
                df.at[idx, 'latest_logged_at'] = logged_at
                df.at[idx, 'latest_model_version_seen'] = model_version
                df.at[idx, 'latest_nn_model_version_seen'] = nn_model_version
                df.at[idx, 'latest_xgb_model_version_seen'] = xgb_model_version or ''
                df.at[idx, 'latest_rf_model_version_seen'] = rf_model_version or ''
                df.at[idx, 'latest_run_id'] = run_id
                df.at[idx, 'latest_feature_snapshot_id'] = feature_snapshot_id
                df.at[idx, 'latest_prediction_uid'] = prediction_uid
                df.at[idx, 'latest_odds_scraped_at'] = odds_scraped_at
                df.at[idx, 'latest_match_start_time'] = match_start_time
                df.at[idx, 'latest_match_date'] = match_date_str
                df.at[idx, 'record_status'] = 'settled' if actual_winner is not None else 'pending'
                if feature_snapshot_id:
                    df.at[idx, 'logging_schema_version'] = SCHEMA_VERSION
                    df.at[idx, 'logging_quality'] = 'snapshot_v2'
                    df.at[idx, 'rescore_quality'] = 'exact_feature_snapshot'
                    df.at[idx, 'record_note'] = ''
                df.to_csv(LOG_PATH, index=False)
                return

        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row], columns=COLUMNS)

    df.to_csv(LOG_PATH, index=False)
