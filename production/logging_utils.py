#!/usr/bin/env python3
"""Utilities for stable logging IDs and append-only CSV logs."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd


def utc_now() -> datetime:
    """Return the current UTC time."""
    return datetime.now(timezone.utc)


def utc_now_iso() -> str:
    """Return the current UTC timestamp in ISO-8601 format."""
    return utc_now().replace(microsecond=0).isoformat()


def make_run_id(started_at: datetime | None = None, prefix: str = "run") -> str:
    """Create a deterministic-looking run identifier from a timestamp."""
    dt = started_at or utc_now()
    stamp = dt.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}_{stamp}"


def normalize_text(value) -> str:
    """Normalize free text to a compact ASCII-ish form for hashing."""
    if value is None:
        return ""
    text = str(value).strip().lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^a-z0-9\s:/._-]+", "", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_name(name: str) -> str:
    """Normalize a player name for matching across logs."""
    return normalize_text(name).replace("-", " ")


def stable_hash(*parts, length: int = 20) -> str:
    """Hash a set of normalized parts."""
    joined = "||".join(normalize_text(part) for part in parts)
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()[:length]


def build_match_uid(
    p1: str,
    p2: str,
    match_date: str,
    tournament: str = "",
    round_code: str = "",
    surface: str = "",
) -> str:
    """
    Build a stable, order-insensitive match identifier.

    The UID is intentionally based on the canonical player pair plus coarse match
    metadata so that live odds, feature rows, and later rescoring can refer to
    the same match without relying on a per-run row index.
    """
    players = sorted([normalize_name(p1), normalize_name(p2)])
    return "match_" + stable_hash(
        players[0],
        players[1],
        match_date,
        tournament,
        round_code,
        surface,
    )


def build_feature_snapshot_id(match_uid: str, run_id: str, p1: str, p2: str) -> str:
    """Build a unique feature-snapshot identifier for one run of one match."""
    return "feat_" + stable_hash(match_uid, run_id, normalize_name(p1), normalize_name(p2))


def build_prediction_uid(match_uid: str, model_version: str, logged_at: str, p1: str, p2: str) -> str:
    """Build an immutable prediction-snapshot identifier."""
    return "pred_" + stable_hash(match_uid, model_version, logged_at, normalize_name(p1), normalize_name(p2))


def build_odds_snapshot_uid(
    match_uid: str,
    odds_scraped_at: str,
    match_start_time: str = "",
    p1_odds_decimal=None,
    p2_odds_decimal=None,
) -> str:
    """Build an immutable odds-snapshot identifier."""
    return "odds_" + stable_hash(
        match_uid,
        odds_scraped_at,
        match_start_time,
        p1_odds_decimal,
        p2_odds_decimal,
    )


def ensure_csv_columns(path: Path, columns: Iterable[str]) -> pd.DataFrame:
    """Load a CSV and backfill any missing columns."""
    if path.exists():
        df = pd.read_csv(path)
    else:
        df = pd.DataFrame(columns=list(columns))

    for col in columns:
        if col not in df.columns:
            df[col] = pd.NA

    ordered = [col for col in columns] + [col for col in df.columns if col not in columns]
    return df[ordered]


def append_unique_row(path: Path, row: dict, columns: Iterable[str], unique_key: str | None = None) -> bool:
    """Append a row to a CSV, optionally deduplicating on one key column."""
    df = ensure_csv_columns(path, columns)

    if unique_key and unique_key in row and unique_key in df.columns:
        existing = set(df[unique_key].dropna().astype(str))
        candidate = str(row[unique_key])
        if candidate in existing:
            return False

    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(path, index=False)
    return True
