#!/usr/bin/env python3
"""Append or update a prediction in prediction_log.csv."""
import pandas as pd
import os
from datetime import datetime, date
from pathlib import Path

try:
    from logging_utils import (
        append_unique_row,
        build_feature_snapshot_id,
        build_match_uid,
        build_odds_snapshot_uid,
        build_prediction_uid,
        canonicalize_live_event_key,
        ensure_csv_columns,
        normalize_name,
    )
except ModuleNotFoundError:  # pragma: no cover - package import path
    from .logging_utils import (
        append_unique_row,
        build_feature_snapshot_id,
        build_match_uid,
        build_odds_snapshot_uid,
        build_prediction_uid,
        canonicalize_live_event_key,
        ensure_csv_columns,
        normalize_name,
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
    'xgb_model_version', 'rf_model_version', 'nn_probability_source',
    'logging_schema_version', 'logging_quality', 'rescore_quality',
    'record_status', 'record_note',
    'identity_status', 'identity_event_key',
    'identity_related_match_uid', 'identity_conflict_fields',
    'features_complete', 'defaulted_features',
    'feature_schema_sha256', 'feature_vector_sha256',
    'odds_scraped_at', 'match_start_time', 'match_start_at_utc',
    'p1_hand', 'p2_hand',
    'actual_winner', 'score', 'settled_at', 'model_correct', 'market_correct',
    'xgb_correct',
    'rf_correct',
    'run_id', 'latest_run_id',
    'match_uid',
    'feature_snapshot_id', 'latest_feature_snapshot_id',
    'prediction_uid', 'latest_prediction_uid',
    'latest_logged_at', 'latest_model_version_seen',
    'latest_nn_model_version_seen', 'latest_xgb_model_version_seen', 'latest_rf_model_version_seen',
    'latest_nn_probability_source_seen',
    'latest_odds_scraped_at', 'latest_match_start_time', 'latest_match_start_at_utc', 'latest_match_date',
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
    'xgb_model_version', 'rf_model_version', 'nn_probability_source',
    'logging_schema_version', 'logging_quality', 'rescore_quality',
    'record_status', 'record_note',
    'identity_status', 'identity_event_key',
    'identity_related_match_uid', 'identity_conflict_fields',
    'features_complete', 'defaulted_features',
    'feature_schema_sha256', 'feature_vector_sha256',
    'odds_scraped_at', 'match_start_time', 'match_start_at_utc',
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
    'match_start_at_utc',
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


class LiveMatchIdentityError(ValueError):
    """Raised when prediction lineage violates the live identity contract."""


_PRESERVED_IDENTITY_STATUSES = {'identity_conflict', 'superseded_identity'}


def _validate_feature_snapshot_identity(
    *,
    match_uid: str | None,
    feature_snapshot_id: str | None,
    run_id: str | None,
    p1: str,
    p2: str,
) -> None:
    """Require snapshot-backed predictions to name the exact producing match.

    ``feature_snapshot_id`` is deterministic over match, run, and oriented
    players.  Recomputing it here catches cross-match pointers before any
    prediction, odds, or operational row is written.
    """
    snapshot = _clean_text(feature_snapshot_id)
    if not snapshot:
        return
    match = _clean_text(match_uid)
    run = _clean_text(run_id)
    if not match or not run:
        raise LiveMatchIdentityError(
            'feature_snapshot_id requires nonblank match_uid and run_id'
        )
    expected = build_feature_snapshot_id(match, run, p1, p2)
    if snapshot != expected:
        raise LiveMatchIdentityError(
            'feature_snapshot_id does not belong to prediction match_uid/run/player orientation: '
            f'expected {expected}, received {snapshot}'
        )


def _validate_live_match_uid(
    *,
    match_uid: str | None,
    p1: str,
    p2: str,
    match_date: str,
    tournament: str,
    identity_event_key: str,
    round_code: str,
    surface: str,
) -> None:
    """Require a supplied live UID to match every canonical identity input."""
    incoming = _clean_text(match_uid)
    if not incoming:
        return
    event_identity = _clean_text(identity_event_key) or tournament
    expected = build_match_uid(
        p1,
        p2,
        match_date,
        event_identity,
        round_code,
        surface,
    )
    if incoming != expected:
        raise LiveMatchIdentityError(
            'match_uid does not match canonical player/date/event/round/surface identity: '
            f'expected {expected}, received {incoming}'
        )


def _unsettled_pair_mask(
    df: pd.DataFrame,
    *,
    p1: str,
    p2: str,
) -> pd.Series:
    """Return same-player unsettled rows, independent of player order."""
    p1_key = normalize_name(p1)
    p2_key = normalize_name(p2)
    left = df.get('p1', pd.Series('', index=df.index)).fillna('').map(normalize_name)
    right = df.get('p2', pd.Series('', index=df.index)).fillna('').map(normalize_name)
    unsettled = df.get('actual_winner', pd.Series(pd.NA, index=df.index)).isna()
    status = df.get(
        'record_status', pd.Series('', index=df.index)
    ).fillna('').astype(str).str.strip().str.lower()
    # A superseded alias is immutable historical context, never an active row
    # that can be selected again by the operational refresh path. Identity
    # conflicts remain selectable so a repeat run stays fail-closed.
    active = status.ne('superseded_identity')
    players = ((left == p1_key) & (right == p2_key)) | (
        (left == p2_key) & (right == p1_key)
    )
    return unsettled & active & players


def _nearby_unsettled_pair_mask(
    df: pd.DataFrame,
    *,
    p1: str,
    p2: str,
    match_date: str,
) -> pd.Series:
    """Return same-player unsettled rows within the explicit three-day window."""
    mask = _unsettled_pair_mask(df, p1=p1, p2=p2)
    if not mask.any():
        return mask

    new_date = pd.to_datetime(match_date, errors='coerce')
    if pd.isna(new_date):
        return pd.Series(False, index=df.index)
    existing = pd.to_datetime(
        df.get('match_date', pd.Series('', index=df.index)), errors='coerce'
    )
    day_delta = (existing - new_date).abs().dt.days
    return mask & day_delta.le(3).fillna(False)


def _identity_metadata_differences(existing: pd.Series, incoming: dict) -> tuple[str, ...]:
    """Name mutable metadata differences that require an explicit decision."""
    def date_key(value) -> str:
        parsed = pd.to_datetime(value, errors='coerce')
        return parsed.date().isoformat() if pd.notna(parsed) else _clean_text(value)

    comparisons = {
        'match_date': (
            date_key(existing.get('match_date')),
            date_key(incoming.get('match_date')),
        ),
        'round': (
            _clean_text(existing.get('round')).upper(),
            _clean_text(incoming.get('round')).upper(),
        ),
        'surface': (
            _clean_text(existing.get('surface')).lower(),
            _clean_text(incoming.get('surface')).lower(),
        ),
    }
    stored_event_key = _clean_text(existing.get('identity_event_key'))
    incoming_event_key = _clean_text(incoming.get('identity_event_key'))
    if stored_event_key and incoming_event_key:
        # The canonical source event key owns identity. Display tournament text
        # may legitimately enrich from blank/resolver-only to a human label.
        comparisons['event_key'] = (stored_event_key, incoming_event_key)
    else:
        comparisons['tournament'] = (
            canonicalize_live_event_key(existing.get('tournament')),
            canonicalize_live_event_key(incoming.get('tournament')),
        )
    return tuple(
        field for field, (stored, current) in comparisons.items()
        if stored != current
    )


def _is_safe_round_enrichment(existing: pd.Series, incoming: dict) -> bool:
    """Allow one-way blank-round incomplete -> explicit-round complete alias."""
    differences = set(_identity_metadata_differences(existing, incoming))
    existing_complete = _coerce_bool_series(
        pd.Series([existing.get('features_complete', False)]), default=False
    ).iloc[0]
    incoming_complete = bool(incoming.get('features_complete'))
    existing_status = _clean_text(existing.get('record_status')).lower()
    return bool(
        differences == {'round'}
        and not _clean_text(existing.get('round'))
        and bool(_clean_text(incoming.get('round')))
        and not existing_complete
        and incoming_complete
        and existing_status not in _PRESERVED_IDENTITY_STATUSES
        and normalize_name(existing.get('p1')) == normalize_name(incoming.get('p1'))
        and normalize_name(existing.get('p2')) == normalize_name(incoming.get('p2'))
    )


def _canonical_exact_row_index(df: pd.DataFrame, indices: list[int]) -> int:
    """Choose the first complete observation, otherwise the earliest row."""
    candidates = df.loc[indices].copy()
    candidates['_complete_rank'] = _coerce_bool_series(
        candidates.get(
            'features_complete', pd.Series(False, index=candidates.index)
        ),
        default=False,
    ).astype(int)
    candidates['_logged_rank'] = pd.to_datetime(
        candidates.get('logged_at', pd.Series('', index=candidates.index)),
        errors='coerce',
        utc=True,
        format='mixed',
    )
    # Complete rows outrank incomplete rows; within that tier the earliest
    # observation preserves the opening evidence contract.
    candidates = candidates.sort_values(
        ['_complete_rank', '_logged_rank'],
        ascending=[False, True],
        kind='stable',
        na_position='last',
    )
    return int(candidates.index[0])


def upgrade_prediction_log(path: Path | None = None, stale_days: int = 7, write: bool = True) -> pd.DataFrame:
    """
    Backfill logging metadata on the prediction log.

    Legacy rows get synthetic match/prediction ids where possible plus explicit
    quality/status markers so downstream tooling can distinguish reliable
    snapshot-based rows from older fallback-only history.
    """
    target = path or Path(LOG_PATH)
    # The log mixes numeric and textual columns. Use object storage while
    # upgrading so filling newly-added identifiers/status strings cannot rely
    # on pandas' deprecated implicit dtype widening.
    df = ensure_csv_columns(target, COLUMNS).astype(object)

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
        nn_probability_source = _clean_text(row.get('nn_probability_source', ''))

        df.at[idx, 'nn_model_version'] = nn_version or model_version
        if not xgb_version and pd.notna(row.get('xgb_p1_prob')):
            df.at[idx, 'xgb_model_version'] = 'unknown_legacy'
        elif not xgb_version:
            df.at[idx, 'xgb_model_version'] = ''
        if not rf_version and pd.notna(row.get('rf_p1_prob')):
            df.at[idx, 'rf_model_version'] = 'unknown_legacy'
        elif not rf_version:
            df.at[idx, 'rf_model_version'] = ''
        if not nn_probability_source:
            if '+calibrated' in model_version or '+calibrated' in df.at[idx, 'nn_model_version']:
                nn_probability_source = 'calibrated'
            elif pd.notna(row.get('model_p1_prob')):
                nn_probability_source = 'raw'
            else:
                nn_probability_source = ''
        df.at[idx, 'nn_probability_source'] = nn_probability_source

        has_run_id = bool(_clean_text(row.get('run_id', '')))
        has_feature_snapshot = bool(_clean_text(row.get('feature_snapshot_id', '')))
        has_lineage = has_run_id and has_feature_snapshot

        df.at[idx, 'logging_schema_version'] = SCHEMA_VERSION if has_lineage else 'legacy_v1'
        df.at[idx, 'logging_quality'] = 'snapshot_v2' if has_lineage else 'legacy_backfilled'
        df.at[idx, 'rescore_quality'] = 'exact_feature_snapshot' if has_feature_snapshot else 'legacy_fallback_match'

        row_logged_at = pd.to_datetime(
            row.get('logged_at'), errors='coerce', utc=True
        )
        row_match_date = pd.to_datetime(
            match_date_str, errors='coerce', utc=True
        )
        effective_dt = row_logged_at if pd.notna(row_logged_at) else row_match_date
        age_days = None
        if pd.notna(effective_dt):
            age_days = (pd.Timestamp.now(tz='UTC') - effective_dt).days

        has_model_prediction = pd.notna(row.get('model_p1_prob'))
        winner = pd.to_numeric(row.get('actual_winner'), errors='coerce')
        is_settled = winner in (1, 2)
        preserved_status = _clean_text(row.get('record_status')).lower()
        if preserved_status in _PRESERVED_IDENTITY_STATUSES:
            status = preserved_status
            note = _clean_text(row.get('record_note'))
        elif is_settled:
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
        if not _clean_text(row.get('identity_status')):
            df.at[idx, 'identity_status'] = 'legacy_unclassified'
        if not _clean_text(row.get('identity_event_key')):
            df.at[idx, 'identity_event_key'] = ''
        if not _clean_text(row.get('identity_related_match_uid')):
            df.at[idx, 'identity_related_match_uid'] = ''
        if not _clean_text(row.get('identity_conflict_fields')):
            df.at[idx, 'identity_conflict_fields'] = ''

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
        if not _clean_text(row.get('latest_nn_probability_source_seen', '')):
            df.at[idx, 'latest_nn_probability_source_seen'] = df.at[idx, 'nn_probability_source']
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
        if not _clean_text(row.get('latest_match_start_at_utc', '')):
            df.at[idx, 'latest_match_start_at_utc'] = row.get('match_start_at_utc', '')

    if 'features_complete' in df.columns:
        df['features_complete'] = _coerce_bool_series(df['features_complete'], default=False)

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
    identity_event_key: str = '',
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
    nn_probability_source: str = None,
    odds_scraped_at: str = None,
    match_start_time: str = None,
    match_start_at_utc: str = None,
    actual_winner: int = None, score: str = None,
    features_complete: bool = False,
    p1_hand: str = '',
    p2_hand: str = '',
    defaulted_features: str = '',
    feature_schema_sha256: str = '',
    feature_vector_sha256: str = '',
    allow_update: bool = True,
    identity_conflict_uids_out: set[str] | None = None,
):
    """
    Append or update a prediction row.

    Snapshot-backed rows refresh only an unsettled row with the exact same
    ``match_uid``. A nearby same-player row with a different UID is recorded as
    an explicit canonical alias (metadata identical) or a non-decision identity
    conflict (date/round/surface/tournament changed). Legacy rows without exact
    IDs retain the old player/date fallback.

    features_complete=False marks predictions that used defaulted feature values
    and should be excluded from accuracy analysis.
    Returns:
        "created" if a new operational row was appended,
        "updated" if an existing unsettled row was refreshed,
        "created_alias" for a canonical replacement of a legacy UID,
        "identity_conflict" for a fail-closed metadata shift.
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
    if nn_probability_source is None and model_p1_prob is not None:
        nn_probability_source = 'raw'

    match_date_str = _parse_match_date(match_date)
    identity_event_key = (
        canonicalize_live_event_key(identity_event_key)
        if _clean_text(identity_event_key) else ''
    )
    _validate_live_match_uid(
        match_uid=match_uid,
        p1=p1,
        p2=p2,
        match_date=match_date_str,
        tournament=tournament,
        identity_event_key=identity_event_key,
        round_code=round_code,
        surface=surface,
    )
    _validate_feature_snapshot_identity(
        match_uid=match_uid,
        feature_snapshot_id=feature_snapshot_id,
        run_id=run_id,
        p1=p1,
        p2=p2,
    )
    logged_at = datetime.now().isoformat()
    prediction_uid = build_prediction_uid(match_uid or '', model_version, logged_at, p1, p2)

    valid_winner = actual_winner in (1, 2)
    settled_at = datetime.now().isoformat() if valid_winner else None
    model_correct = None
    market_correct = None
    xgb_correct = None
    rf_correct = None
    if valid_winner and model_p1_prob is not None:
        model_correct = int((actual_winner == 1) == (model_p1_prob > 0.5))
        if market_p1_prob is not None:
            market_correct = int((actual_winner == 1) == (market_p1_prob > 0.5))
    if valid_winner and xgb_p1_prob is not None:
        xgb_correct = int((actual_winner == 1) == (xgb_p1_prob > 0.5))
    if valid_winner and rf_p1_prob is not None:
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
        'nn_probability_source': nn_probability_source or '',
        'logging_schema_version': SCHEMA_VERSION if (run_id and feature_snapshot_id) else 'legacy_v1',
        'logging_quality': 'snapshot_v2' if (run_id and feature_snapshot_id) else 'legacy_backfilled',
        'rescore_quality': 'exact_feature_snapshot' if feature_snapshot_id else 'legacy_fallback_match',
        'record_status': 'settled' if valid_winner else 'pending',
        'record_note': '',
        'identity_status': 'canonical' if match_uid else 'legacy_unclassified',
        'identity_event_key': _clean_text(identity_event_key),
        'identity_related_match_uid': '',
        'identity_conflict_fields': '',
        'features_complete': features_complete,
        'p1_hand': p1_hand,
        'p2_hand': p2_hand,
        'defaulted_features': defaulted_features or '',
        'feature_schema_sha256': feature_schema_sha256 or '',
        'feature_vector_sha256': feature_vector_sha256 or '',
        'odds_scraped_at': odds_scraped_at,
        'match_start_time': match_start_time,
        'match_start_at_utc': match_start_at_utc,
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
        'latest_nn_probability_source_seen': nn_probability_source or '',
        'latest_odds_scraped_at': odds_scraped_at,
        'latest_match_start_time': match_start_time,
        'latest_match_start_at_utc': match_start_at_utc,
        'latest_match_date': match_date_str,
    }

    if os.path.exists(LOG_PATH):
        df = upgrade_prediction_log(Path(LOG_PATH), write=False)
    else:
        df = pd.DataFrame(columns=COLUMNS).astype(object)

    action = 'created'
    update_idx = None
    reconciled_duplicate_count = 0
    incoming_uid = _clean_text(match_uid)
    if allow_update and not df.empty:
        nearby = _nearby_unsettled_pair_mask(
            df, p1=p1, p2=p2, match_date=match_date_str,
        )
        if incoming_uid:
            stored_uids = df.get(
                'match_uid', pd.Series('', index=df.index)
            ).fillna('').astype(str).str.strip()
            exact_mask = stored_uids.eq(incoming_uid)
            if exact_mask.any():
                exact_indices = exact_mask[exact_mask].index.tolist()
                exact_winners = pd.to_numeric(
                    df.loc[exact_indices, 'actual_winner'], errors='coerce'
                )
                if exact_winners.isin([1, 2]).any():
                    raise LiveMatchIdentityError(
                        f'match_uid {incoming_uid} is already settled; refusing a new live row'
                    )
                exact_statuses = df.loc[
                    exact_indices, 'record_status'
                ].fillna('').astype(str).str.strip().str.lower()
                if exact_statuses.eq('superseded_identity').any():
                    raise LiveMatchIdentityError(
                        f'match_uid {incoming_uid} is superseded; refusing identity reuse'
                    )
                incompatible_fields: set[str] = set()
                for candidate_idx in exact_indices:
                    candidate = df.loc[candidate_idx]
                    if not (
                        normalize_name(candidate.get('p1')) == normalize_name(p1)
                        and normalize_name(candidate.get('p2')) == normalize_name(p2)
                    ):
                        incompatible_fields.add('player_orientation')
                    incompatible_fields.update(
                        _identity_metadata_differences(candidate, row)
                    )
                if incompatible_fields:
                    raise LiveMatchIdentityError(
                        f'match_uid {incoming_uid} has incompatible exact-row metadata: '
                        f'{",".join(sorted(incompatible_fields))}'
                    )
                preserved_indices = [
                    candidate_idx for candidate_idx in exact_indices
                    if _clean_text(
                        df.at[candidate_idx, 'record_status']
                    ).lower() in _PRESERVED_IDENTITY_STATUSES
                ]
                # Safety tombstones dominate ordinary complete/pending rows.
                # Otherwise duplicate repair could select the higher-quality
                # inference row, delete the conflict row, and silently restore
                # decision eligibility for a disputed identity.
                update_idx = _canonical_exact_row_index(
                    df, preserved_indices or exact_indices
                )
                duplicate_indices = [
                    candidate_idx for candidate_idx in exact_indices
                    if candidate_idx != update_idx
                ]
                if duplicate_indices:
                    # prediction_log.csv is one operational row per match UID.
                    # Immutable prediction_snapshots.csv retains every original
                    # observation, so removing redundant operational copies does
                    # not discard point-in-time evidence.
                    reconciled_duplicate_count = len(duplicate_indices)
                    df = df.drop(index=duplicate_indices)
                existing_status = _clean_text(
                    df.at[update_idx, 'record_status']
                ).lower()
                if existing_status in _PRESERVED_IDENTITY_STATUSES:
                    action = 'identity_conflict'
                    row['record_status'] = existing_status
                    row['record_note'] = _clean_text(
                        df.at[update_idx, 'record_note']
                    )
                    row['identity_status'] = _clean_text(
                        df.at[update_idx, 'identity_status']
                    ) or 'conflict'
                    row['identity_related_match_uid'] = _clean_text(
                        df.at[update_idx, 'identity_related_match_uid']
                    )
                    row['identity_conflict_fields'] = _clean_text(
                        df.at[update_idx, 'identity_conflict_fields']
                    )
                    row['features_complete'] = False
            elif nearby.any():
                candidates = df.loc[nearby].copy()
                differences: set[str] = set()
                orientations_match = True
                safe_round_enrichment = False
                alias_compatible = True
                for _, candidate in candidates.iterrows():
                    candidate_differences = set(
                        _identity_metadata_differences(candidate, row)
                    )
                    differences.update(candidate_differences)
                    candidate_round_enrichment = _is_safe_round_enrichment(
                        candidate, row
                    )
                    safe_round_enrichment = (
                        safe_round_enrichment or candidate_round_enrichment
                    )
                    alias_compatible = alias_compatible and (
                        not candidate_differences or candidate_round_enrichment
                    )
                    orientations_match = orientations_match and (
                        normalize_name(candidate.get('p1')) == normalize_name(p1)
                        and normalize_name(candidate.get('p2')) == normalize_name(p2)
                    )
                if not orientations_match:
                    differences.add('player_orientation')

                related = sorted({
                    _clean_text(value)
                    for value in candidates.get('match_uid', pd.Series(dtype=str))
                    if _clean_text(value) and _clean_text(value) != incoming_uid
                })
                related_text = '|'.join(related)
                if alias_compatible and orientations_match and related:
                    # One-time bridge for pre-contract rows whose UID included
                    # volatile source text. Immutable snapshots stay untouched;
                    # the old operational rows are retired and the new canonical
                    # row starts with its own exact snapshot.
                    action = 'created_alias'
                    row['identity_status'] = 'canonical_alias'
                    row['identity_related_match_uid'] = related_text
                    row['record_note'] = (
                        f'canonical_round_enrichment_from:{related_text}'
                        if safe_round_enrichment
                        else f'canonical_identity_alias_from:{related_text}'
                    )
                    for candidate_idx in candidates.index:
                        df.at[candidate_idx, 'record_status'] = 'superseded_identity'
                        df.at[candidate_idx, 'record_note'] = (
                            f'superseded_by_canonical_match_uid:{incoming_uid}'
                        )
                        df.at[candidate_idx, 'identity_status'] = 'superseded_alias'
                        df.at[candidate_idx, 'identity_related_match_uid'] = incoming_uid
                        df.at[candidate_idx, 'identity_conflict_fields'] = ''
                        df.at[candidate_idx, 'features_complete'] = False
                else:
                    action = 'identity_conflict'
                    fields = ','.join(sorted(differences or {'ambiguous_identity'}))
                    row['record_status'] = 'identity_conflict'
                    row['record_note'] = (
                        f'match_identity_conflict:{fields};related:{related_text}'
                    )
                    row['identity_status'] = 'conflict'
                    row['identity_related_match_uid'] = related_text
                    row['identity_conflict_fields'] = fields
                    row['features_complete'] = False
                    defaults = _clean_text(row.get('defaulted_features'))
                    marker = 'match_identity_conflict'
                    row['defaulted_features'] = (
                        f'{defaults},{marker}' if defaults else marker
                    )
                    # Both sides of an unresolved identity shift are unsafe.
                    # Blocking only the incoming UID would leave the older row
                    # eligible for settlement/shadow scoring and could attach
                    # the disputed result to the wrong round or event.
                    for candidate_idx in candidates.index:
                        prior_related = _clean_text(
                            df.at[candidate_idx, 'identity_related_match_uid']
                        )
                        related_uids = {
                            value for value in prior_related.split('|') if value
                        }
                        related_uids.add(incoming_uid)
                        df.at[candidate_idx, 'record_status'] = 'identity_conflict'
                        df.at[candidate_idx, 'record_note'] = (
                            f'match_identity_conflict:{fields};related:{incoming_uid}'
                        )
                        df.at[candidate_idx, 'identity_status'] = 'conflict_related'
                        df.at[candidate_idx, 'identity_related_match_uid'] = '|'.join(
                            sorted(related_uids)
                        )
                        df.at[candidate_idx, 'identity_conflict_fields'] = fields
                        df.at[candidate_idx, 'features_complete'] = False
                        prior_defaults = _clean_text(
                            df.at[candidate_idx, 'defaulted_features']
                        )
                        if marker not in {
                            value.strip() for value in prior_defaults.split(',')
                            if value.strip()
                        }:
                            df.at[candidate_idx, 'defaulted_features'] = (
                                f'{prior_defaults},{marker}'
                                if prior_defaults else marker
                            )
        elif nearby.any():
            # Compatibility only: rows without immutable live IDs retain the
            # historical player/date update path.
            update_idx = nearby[nearby].index[0]

    if action == 'identity_conflict' and identity_conflict_uids_out is not None:
        if incoming_uid:
            identity_conflict_uids_out.add(incoming_uid)
        identity_conflict_uids_out.update(
            value for value in _clean_text(
                row.get('identity_related_match_uid')
            ).split('|')
            if value
        )

    snapshot_row = {
        key: row.get(key)
        for key in SNAPSHOT_COLUMNS
        if key != 'snapshot_role'
    }
    snapshot_row['snapshot_role'] = 'live'
    append_unique_row(
        SNAPSHOT_LOG_PATH,
        snapshot_row,
        SNAPSHOT_COLUMNS,
        unique_key='prediction_uid',
    )

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
            'match_start_at_utc': match_start_at_utc,
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

    if update_idx is not None:
        idx = update_idx
        stored_uid = _clean_text(df.at[idx, 'match_uid'])
        if incoming_uid and stored_uid != incoming_uid:
            raise LiveMatchIdentityError(
                'refusing to attach feature snapshot to a different match_uid: '
                f'stored={stored_uid}, incoming={incoming_uid}'
            )
        # Preserve original odds + model probs — opening lines are less efficient
        # and more valuable for edge analysis than lines closer to match time.
        # EXCEPTION: if the stored row was computed on INCOMPLETE features and
        # this prediction is complete, upgrade everything (probs+odds+edge).
        existing_complete = str(df.at[idx, 'features_complete']) in (
            'True', '1', '1.0'
        )
        upgrading_to_complete = (
            action != 'identity_conflict'
            and bool(features_complete)
            and not existing_complete
        )
        if upgrading_to_complete:
            print(
                f"  ⬆️  Upgrading incomplete prediction to first complete one: "
                f"{p1} vs {p2}"
            )
        # A model-regime bump unfreezes complete rows. The refreshing prediction
        # must itself be complete; identity conflicts never enter this branch.
        stored_version = _clean_text(df.at[idx, 'nn_model_version'])
        new_version = _clean_text(nn_model_version)
        real_version = lambda value: value.lower() not in ('', 'nan', 'none')
        regime_refresh = (
            action != 'identity_conflict'
            and bool(features_complete)
            and existing_complete
            and real_version(stored_version)
            and real_version(new_version)
            and stored_version != new_version
        )
        if regime_refresh:
            print(
                f"  🔁 Regime refresh {stored_version} → {new_version}: "
                f"re-pricing open match {p1} vs {p2}"
            )
        preserve_if_set = set() if (
            upgrading_to_complete or regime_refresh
        ) else {
            'model_p1_prob', 'model_p2_prob',
            'xgb_p1_prob', 'xgb_p2_prob',
            'rf_p1_prob', 'rf_p2_prob',
            'market_p1_prob', 'market_p2_prob',
            'p1_odds_american', 'p2_odds_american',
            'p1_odds_decimal', 'p2_odds_decimal',
            'spread_handicap', 'spread_odds_p1', 'spread_odds_p2',
            'total_games', 'total_odds_over', 'total_odds_under',
            'edge_p1', 'primary_model_family', 'model_version',
            'nn_model_version', 'xgb_model_version', 'rf_model_version',
            'nn_probability_source', 'p1_rank', 'p2_rank',
            'odds_scraped_at', 'match_start_time', 'match_start_at_utc',
            'run_id', 'feature_snapshot_id', 'prediction_uid',
        }
        if not (
            existing_complete and not upgrading_to_complete and not regime_refresh
        ):
            for col, val in row.items():
                if col not in df.columns:
                    continue
                if col in preserve_if_set and pd.notna(df.at[idx, col]):
                    continue
                df.at[idx, col] = val

        df.at[idx, 'latest_logged_at'] = logged_at
        df.at[idx, 'latest_model_version_seen'] = model_version
        df.at[idx, 'latest_nn_model_version_seen'] = nn_model_version
        df.at[idx, 'latest_xgb_model_version_seen'] = xgb_model_version or ''
        df.at[idx, 'latest_rf_model_version_seen'] = rf_model_version or ''
        df.at[idx, 'latest_nn_probability_source_seen'] = nn_probability_source or ''
        df.at[idx, 'latest_run_id'] = run_id
        df.at[idx, 'latest_feature_snapshot_id'] = feature_snapshot_id
        df.at[idx, 'latest_prediction_uid'] = prediction_uid
        df.at[idx, 'latest_odds_scraped_at'] = odds_scraped_at
        df.at[idx, 'latest_match_start_time'] = match_start_time
        df.at[idx, 'latest_match_start_at_utc'] = match_start_at_utc
        df.at[idx, 'latest_match_date'] = match_date_str
        if action == 'identity_conflict':
            df.at[idx, 'record_status'] = row['record_status']
            df.at[idx, 'record_note'] = row['record_note']
            df.at[idx, 'identity_status'] = row['identity_status']
            df.at[idx, 'identity_related_match_uid'] = row[
                'identity_related_match_uid'
            ]
            df.at[idx, 'identity_conflict_fields'] = row[
                'identity_conflict_fields'
            ]
            df.at[idx, 'features_complete'] = False
        else:
            df.at[idx, 'record_status'] = 'settled' if valid_winner else 'pending'
            existing_identity_status = _clean_text(
                df.at[idx, 'identity_status']
            ).lower()
            if existing_identity_status in {'', 'legacy_unclassified'}:
                df.at[idx, 'identity_status'] = (
                    'canonical_reconciled_duplicate'
                    if reconciled_duplicate_count
                    else 'canonical'
                )
            if not _clean_text(df.at[idx, 'identity_event_key']):
                df.at[idx, 'identity_event_key'] = row['identity_event_key']
        if feature_snapshot_id:
            df.at[idx, 'logging_schema_version'] = SCHEMA_VERSION
            df.at[idx, 'logging_quality'] = 'snapshot_v2'
            df.at[idx, 'rescore_quality'] = 'exact_feature_snapshot'
            if action != 'identity_conflict':
                df.at[idx, 'record_note'] = ''
        if reconciled_duplicate_count and action != 'identity_conflict':
            df.at[idx, 'record_note'] = (
                f'reconciled_{reconciled_duplicate_count}_duplicate_operational_rows'
            )
        df.to_csv(LOG_PATH, index=False)
        return action if action == 'identity_conflict' else 'updated'

    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

    df.to_csv(LOG_PATH, index=False)
    return action
