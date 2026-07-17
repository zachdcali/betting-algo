#!/usr/bin/env python3
"""Unauthenticated, read-only Kalshi tennis market logger.

Kalshi prices are contract probabilities.  This module preserves the suffixed
``*_dollars`` / ``*_fp`` fields exactly as returned and never de-vigs them.
Market-to-board binding is exact on player pair, Eastern match date, and tour
series after applying only the reviewed aliases in
``data/kalshi_player_aliases.json``.  Ambiguity remains explicitly unmatched.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import re
import sys
import unicodedata
from urllib.parse import urlencode

import pandas as pd
import requests

_PRODUCTION_DIR = str(Path(__file__).resolve().parent.parent)
if _PRODUCTION_DIR not in sys.path:
    sys.path.insert(0, _PRODUCTION_DIR)

from logging_utils import atomic_write_csv, make_run_id, utc_now_iso
from operations.operational_lock import operational_csv_lock


BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"
TENNIS_SERIES = (
    "KXATPMATCH",
    "KXATPCHALLENGERMATCH",
    "KXITFMATCH",
    "KXWTAMATCH",
)
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ALIAS_PATH = REPO_ROOT / "data" / "kalshi_player_aliases.json"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "production" / "kalshi_odds_history.csv"
OBSERVATION_SCHEMA_VERSION = "kalshi_odds_observation@1.0.0"

KALSHI_OBSERVATION_COLUMNS = [
    "kalshi_observation_uid",
    "observation_schema_version",
    "polled_at",
    "run_id",
    "source",
    "series_ticker",
    "event_ticker",
    "market_ticker",
    "market_status",
    "match_date",
    "yes_player_raw",
    "yes_player_key",
    "opponent_player_raw",
    "opponent_player_key",
    "yes_bid_dollars",
    "yes_ask_dollars",
    "last_price_dollars",
    "yes_bid_size_fp",
    "yes_ask_size_fp",
    "volume_fp",
    "open_interest_fp",
    "market_open_time",
    "market_close_time",
    "expected_expiration_time",
    "source_uri",
    "source_payload_sha256",
    "match_uid",
    "board_side",
    "board_p1",
    "board_p2",
    "board_p1_identity_key",
    "board_p2_identity_key",
    "board_event",
    "match_method",
    "match_status",
    "alias_schema_version",
    "alias_applied",
]


@dataclass(frozen=True)
class AliasRegistry:
    schema_version: str
    aliases: dict[str, str]


def normalize_player_name(value) -> str:
    """ASCII, punctuation-free, case-folded player name used only for joins."""
    if value is None or pd.isna(value):
        return ""
    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(char for char in text if not unicodedata.combining(char))
    return " ".join(re.findall(r"[a-z0-9]+", text.casefold()))


def load_alias_registry(path: Path | str = DEFAULT_ALIAS_PATH) -> AliasRegistry:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    version = str(payload.get("schema_version", "")).strip()
    if not version:
        raise ValueError("Kalshi alias registry requires schema_version")
    aliases: dict[str, str] = {}
    for entry in payload.get("aliases", []):
        source = normalize_player_name(entry.get("kalshi_name"))
        target = normalize_player_name(entry.get("board_name"))
        if not source or not target:
            raise ValueError("Kalshi alias registry contains a blank name")
        previous = aliases.get(source)
        if previous is not None and previous != target:
            raise ValueError(f"conflicting Kalshi alias for {source}")
        aliases[source] = target
    return AliasRegistry(version, aliases)


def _payload_sha256(market: dict) -> str:
    raw = {key: value for key, value in market.items() if not key.startswith("_")}
    encoded = json.dumps(
        raw, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def fetch_open_tennis_markets(
    *,
    session=None,
    series_tickers: tuple[str, ...] = TENNIS_SERIES,
    limit: int = 200,
    timeout: float = 20.0,
    max_pages_per_series: int = 20,
    polled_at: str | None = None,
) -> tuple[list[dict], str]:
    """Fetch every open market for the configured tennis series.

    No authentication header is created or accepted by this function. Cursor
    cycles, duplicate tickers with changed payloads, and malformed response
    bodies fail loudly rather than silently dropping market evidence.
    """
    if not 1 <= int(limit) <= 1000:
        raise ValueError("Kalshi page limit must be in [1, 1000]")
    poll_time = polled_at or utc_now_iso()
    client = session or requests.Session()
    owns_client = session is None
    markets_by_ticker: dict[str, tuple[str, dict]] = {}
    try:
        for series in series_tickers:
            cursor = ""
            seen_cursors: set[str] = set()
            for _page in range(max_pages_per_series):
                params = {
                    "series_ticker": series,
                    "status": "open",
                    "limit": int(limit),
                }
                if cursor:
                    params["cursor"] = cursor
                response = client.get(
                    f"{BASE_URL}/markets", params=params, timeout=timeout,
                )
                response.raise_for_status()
                payload = response.json()
                page_markets = payload.get("markets") if isinstance(payload, dict) else None
                if not isinstance(page_markets, list):
                    raise RuntimeError(f"Kalshi {series} response has no markets list")
                source_uri = getattr(response, "url", "") or (
                    f"{BASE_URL}/markets?{urlencode(params)}"
                )
                for market in page_markets:
                    if not isinstance(market, dict):
                        raise RuntimeError(f"Kalshi {series} returned a non-object market")
                    ticker = str(market.get("ticker", "")).strip()
                    if not ticker:
                        raise RuntimeError(f"Kalshi {series} returned a market without ticker")
                    digest = _payload_sha256(market)
                    previous = markets_by_ticker.get(ticker)
                    if previous is not None:
                        if previous[0] != digest:
                            raise RuntimeError(
                                f"Kalshi market {ticker} changed within one poll"
                            )
                        continue
                    annotated = dict(market)
                    annotated["_requested_series"] = series
                    annotated["_source_uri"] = source_uri
                    markets_by_ticker[ticker] = (digest, annotated)

                next_cursor = str(payload.get("cursor") or "").strip()
                if not next_cursor:
                    break
                if next_cursor in seen_cursors:
                    raise RuntimeError(f"Kalshi {series} repeated cursor {next_cursor}")
                seen_cursors.add(next_cursor)
                cursor = next_cursor
            else:
                raise RuntimeError(
                    f"Kalshi {series} exceeded {max_pages_per_series} pages"
                )
    finally:
        if owns_client:
            client.close()
    return [value[1] for value in markets_by_ticker.values()], poll_time


def _market_match_date(event_ticker: str) -> str:
    match = re.search(r"-(\d{2})([A-Z]{3})(\d{2})", str(event_ticker).upper())
    if not match:
        return ""
    try:
        return datetime.strptime("".join(match.groups()), "%y%b%d").date().isoformat()
    except ValueError:
        return ""


def _first_value(row: pd.Series, *columns: str):
    for column in columns:
        value = row.get(column, "")
        if value is not None and not pd.isna(value) and str(value).strip():
            return value
    return ""


def _board_match_date(row: pd.Series) -> str:
    exact = _first_value(row, "meta_match_date", "match_date", "latest_match_date")
    if exact:
        parsed = pd.to_datetime(exact, errors="coerce")
        if not pd.isna(parsed):
            return parsed.date().isoformat()
    match_time = str(_first_value(row, "match_time", "match_start_time"))
    absolute = re.search(r"\b(\d{1,2}/\d{1,2}/\d{2,4})\b", match_time)
    if absolute:
        parsed = pd.to_datetime(absolute.group(1), errors="coerce")
        if not pd.isna(parsed):
            return parsed.date().isoformat()
    return ""


def _series_matches_board(series: str, event: str) -> bool:
    event_key = str(event or "").casefold()
    if series == "KXATPCHALLENGERMATCH":
        return "challenger" in event_key
    if series == "KXITFMATCH":
        return "itf" in event_key
    if series == "KXWTAMATCH":
        return "wta" in event_key or "women" in event_key
    if series == "KXATPMATCH":
        return not any(
            marker in event_key for marker in ("challenger", "itf", "wta", "women")
        )
    return False


def _board_candidates(board: pd.DataFrame) -> list[dict]:
    candidates: list[dict] = []
    if board is None or board.empty:
        return candidates
    for index, row in board.iterrows():
        p1 = str(_first_value(row, "player1_raw", "p1"))
        p2 = str(_first_value(row, "player2_raw", "p2"))
        p1_key = normalize_player_name(p1)
        p2_key = normalize_player_name(p2)
        match_date = _board_match_date(row)
        if not p1_key or not p2_key or p1_key == p2_key or not match_date:
            continue
        candidates.append({
            "index": str(index),
            "p1": p1,
            "p2": p2,
            "p1_key": p1_key,
            "p2_key": p2_key,
            "pair": tuple(sorted((p1_key, p2_key))),
            "match_date": match_date,
            "match_uid": str(_first_value(row, "match_uid")),
            "event": str(_first_value(row, "event", "tournament")),
        })
    return candidates


def _observation_uid(run_id: str, polled_at: str, ticker: str, digest: str) -> str:
    encoded = "\x1f".join((run_id, polled_at, ticker, digest)).encode("utf-8")
    return "kalshi_" + sha256(encoded).hexdigest()[:24]


def _raw(value) -> str:
    return "" if value is None else str(value)


def build_kalshi_observations(
    markets: list[dict],
    board: pd.DataFrame,
    *,
    run_id: str,
    polled_at: str,
    alias_registry: AliasRegistry | None = None,
) -> pd.DataFrame:
    """Bind one raw Kalshi poll to a board without fuzzy identity inference."""
    aliases = alias_registry or load_alias_registry()
    board_rows = _board_candidates(board)
    grouped: dict[str, list[dict]] = {}
    for market in markets:
        event_ticker = str(market.get("event_ticker", "")).strip()
        grouped.setdefault(event_ticker, []).append(market)

    rows: list[dict] = []
    for event_ticker, event_markets in grouped.items():
        series = str(event_markets[0].get("_requested_series") or "").strip()
        market_date = _market_match_date(event_ticker)
        raw_names = [str(market.get("yes_sub_title") or "").strip() for market in event_markets]
        raw_keys = [normalize_player_name(name) for name in raw_names]
        canonical_keys = [aliases.aliases.get(key, key) for key in raw_keys]
        unique_canonical = {key for key in canonical_keys if key}

        status = "unmatched_pair_date"
        method = ""
        selected = None
        if len(event_markets) != 2 or len(unique_canonical) != 2:
            status = "invalid_two_sided_event"
        elif not market_date:
            status = "invalid_event_date"
        else:
            pair = tuple(sorted(unique_canonical))
            matches = [
                candidate for candidate in board_rows
                if candidate["pair"] == pair
                and candidate["match_date"] == market_date
                and _series_matches_board(series, candidate["event"])
            ]
            unique_matches = {
                (
                    candidate["match_uid"], candidate["p1_key"],
                    candidate["p2_key"], candidate["event"],
                ): candidate
                for candidate in matches
            }
            matches = list(unique_matches.values())
            if len(matches) > 1:
                status = "ambiguous_board_match"
            elif len(matches) == 1:
                selected = matches[0]
                if not selected["match_uid"]:
                    status = "board_match_missing_uid"
                else:
                    status = "matched"
                    method = (
                        "reviewed_alias_pair_date"
                        if any(source != target for source, target in zip(raw_keys, canonical_keys))
                        else "exact_pair_date"
                    )

        for position, market in enumerate(event_markets):
            yes_raw = raw_names[position]
            yes_raw_key = raw_keys[position]
            yes_key = canonical_keys[position]
            opponent_raw = ""
            opponent_key = ""
            if len(event_markets) == 2:
                opponent_position = 1 - position
                opponent_raw = raw_names[opponent_position]
                opponent_key = canonical_keys[opponent_position]
            digest = _payload_sha256(market)
            board_side = ""
            match_uid = ""
            if status == "matched" and selected is not None:
                match_uid = selected["match_uid"]
                if yes_key == selected["p1_key"]:
                    board_side = "p1"
                elif yes_key == selected["p2_key"]:
                    board_side = "p2"
                else:  # Defensive: the selected pair must orient every market.
                    match_uid = ""
                    status = "orientation_conflict"
                    method = ""
            alias_applied = (
                f"{yes_raw_key}=>{yes_key}" if yes_raw_key and yes_raw_key != yes_key else ""
            )
            rows.append({
                "kalshi_observation_uid": _observation_uid(
                    run_id, polled_at, str(market.get("ticker", "")), digest,
                ),
                "observation_schema_version": OBSERVATION_SCHEMA_VERSION,
                "polled_at": polled_at,
                "run_id": run_id,
                "source": "kalshi",
                "series_ticker": series,
                "event_ticker": event_ticker,
                "market_ticker": _raw(market.get("ticker")),
                "market_status": _raw(market.get("status")),
                "match_date": market_date,
                "yes_player_raw": yes_raw,
                "yes_player_key": yes_key,
                "opponent_player_raw": opponent_raw,
                "opponent_player_key": opponent_key,
                "yes_bid_dollars": _raw(market.get("yes_bid_dollars")),
                "yes_ask_dollars": _raw(market.get("yes_ask_dollars")),
                "last_price_dollars": _raw(market.get("last_price_dollars")),
                "yes_bid_size_fp": _raw(market.get("yes_bid_size_fp")),
                "yes_ask_size_fp": _raw(market.get("yes_ask_size_fp")),
                "volume_fp": _raw(market.get("volume_fp")),
                "open_interest_fp": _raw(market.get("open_interest_fp")),
                "market_open_time": _raw(market.get("open_time")),
                "market_close_time": _raw(market.get("close_time")),
                "expected_expiration_time": _raw(market.get("expected_expiration_time")),
                "source_uri": _raw(market.get("_source_uri")),
                "source_payload_sha256": digest,
                "match_uid": match_uid,
                "board_side": board_side,
                "board_p1": selected["p1"] if selected is not None else "",
                "board_p2": selected["p2"] if selected is not None else "",
                "board_p1_identity_key": selected["p1_key"] if selected is not None else "",
                "board_p2_identity_key": selected["p2_key"] if selected is not None else "",
                "board_event": selected["event"] if selected is not None else "",
                "match_method": method,
                "match_status": status,
                "alias_schema_version": aliases.schema_version,
                "alias_applied": alias_applied,
            })
    return pd.DataFrame(rows, columns=KALSHI_OBSERVATION_COLUMNS)


def _string_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.reindex(columns=KALSHI_OBSERVATION_COLUMNS, fill_value="").copy()
    for column in result.columns:
        result[column] = result[column].map(
            lambda value: "" if value is None or pd.isna(value) else str(value)
        )
    return result


def append_kalshi_observations(
    observations: pd.DataFrame,
    path: Path | str = DEFAULT_OUTPUT_PATH,
) -> int:
    """Atomically append immutable observations and return newly created rows."""
    incoming = _string_frame(observations)
    if incoming.empty:
        return 0
    if incoming["kalshi_observation_uid"].eq("").any():
        raise ValueError("Kalshi observations require kalshi_observation_uid")
    if incoming["kalshi_observation_uid"].duplicated().any():
        raise ValueError("duplicate Kalshi observation UID within one poll")
    output = Path(path)
    lock_dir = output.parent / "logs" if output.parent.name == "production" else output.parent
    with operational_csv_lock(lock_dir):
        if output.exists() and output.stat().st_size:
            existing = pd.read_csv(
                output, dtype=str, keep_default_na=False, low_memory=False,
            )
            existing = _string_frame(existing)
        else:
            existing = pd.DataFrame(columns=KALSHI_OBSERVATION_COLUMNS)
        existing_by_id = existing.set_index("kalshi_observation_uid", drop=False)
        for _, row in incoming.iterrows():
            observation_id = row["kalshi_observation_uid"]
            if observation_id not in existing_by_id.index:
                continue
            previous = existing_by_id.loc[observation_id]
            if isinstance(previous, pd.DataFrame):
                raise RuntimeError(f"duplicate durable Kalshi observation {observation_id}")
            differing = [
                column for column in KALSHI_OBSERVATION_COLUMNS
                if str(previous.get(column, "")) and str(row.get(column, ""))
                and str(previous.get(column, "")) != str(row.get(column, ""))
            ]
            if differing:
                raise RuntimeError(
                    f"conflicting durable Kalshi observation {observation_id}: {differing}"
                )
        new_rows = incoming[
            ~incoming["kalshi_observation_uid"].isin(set(existing_by_id.index))
        ]
        if new_rows.empty:
            return 0
        combined = pd.concat([existing, new_rows], ignore_index=True)
        atomic_write_csv(combined, output)
        return len(new_rows)


def latest_feature_board() -> Path | None:
    candidates = sorted((REPO_ROOT / "production" / "logs").glob("features_*.csv"))
    return candidates[-1] if candidates else None


def poll_summary(observations: pd.DataFrame, created: int) -> dict[str, int | str]:
    matched = observations[observations.get("match_status", "").eq("matched")]
    two_sided_events = 0
    if not observations.empty and "event_ticker" in observations.columns:
        for _, group in observations.groupby("event_ticker", sort=False):
            if (
                len(group) == 2
                and group.get("market_ticker", pd.Series(dtype=str)).nunique() == 2
                and group.get("yes_player_key", pd.Series(dtype=str)).nunique() == 2
            ):
                two_sided_events += 1
    return {
        "market_rows": int(len(observations)),
        "two_sided_events": int(two_sided_events),
        "matched_market_rows": int(len(matched)),
        "matched_matches": int(matched.get("match_uid", pd.Series(dtype=str)).nunique()),
        "rows_created": int(created),
        "logging_start": str(observations.get("polled_at", pd.Series(dtype=str)).min() or ""),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Log unauthenticated Kalshi tennis market observations",
    )
    parser.add_argument("--board", help="Feature/board CSV carrying exact match_uid")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--aliases", default=str(DEFAULT_ALIAS_PATH))
    parser.add_argument("--run-id", default="")
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args(argv)

    board_path = Path(args.board) if args.board else latest_feature_board()
    board = (
        pd.read_csv(board_path, low_memory=False)
        if board_path is not None and board_path.exists()
        else pd.DataFrame()
    )
    run_id = args.run_id or make_run_id(prefix="kalshi_poll")
    markets, polled_at = fetch_open_tennis_markets()
    observations = build_kalshi_observations(
        markets,
        board,
        run_id=run_id,
        polled_at=polled_at,
        alias_registry=load_alias_registry(args.aliases),
    )
    created = 0 if args.no_write else append_kalshi_observations(
        observations, args.output,
    )
    print(json.dumps(poll_summary(observations, created), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
