"""Player match-history frames served from the canonical store (Supabase).

Replaces Tennis Abstract as the live pipeline's source of career histories:
``get_history_frame`` returns the same player-perspective, most-recent-first
schema that ``ta_scraper.get_player_matches`` emits, so every rolling feature
computes unchanged. Recent-gap rows (since the store's last ingest) still come
from ATP stitching on top; TA remains an optional cross-check.

Why this matters: TA 403-blocks datacenter IPs, the store does not — once
features read from here, the pipeline can run hourly in the cloud.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

try:
    import canonical_store as cs
except ImportError:  # imported from features/ dir
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import canonical_store as cs

# chronological order within a tournament, for the most-recent-first sort
_ROUND_ORDER_SQL = """
CASE m.round
    WHEN 'Q1' THEN 1 WHEN 'Q2' THEN 2 WHEN 'Q3' THEN 3 WHEN 'Q4' THEN 4
    WHEN 'R128' THEN 5 WHEN 'R64' THEN 6 WHEN 'R32' THEN 7 WHEN 'R16' THEN 8
    WHEN 'RR' THEN 8 WHEN 'QF' THEN 9 WHEN 'SF' THEN 10 WHEN 'BR' THEN 11 WHEN 'F' THEN 12
    ELSE 0 END
"""

_HISTORY_SQL = f"""
WITH sides AS (
    SELECT m.match_id, m.match_date, m.event, m.surface, m.level, m.round, m.source AS match_source,
           {_ROUND_ORDER_SQL} AS round_ord,
           'W' AS result, m.score,
           m.winner_rank AS rank, m.winner_rank_points AS rank_points,
           m.loser_id AS opp_id, m.loser_rank AS opp_rank,
           st.minutes,
           st.w_ace AS aces, st.w_df AS double_faults, st.w_svpt AS serve_points,
           st.w_1stin AS first_serves_in, st.w_1stwon AS first_serve_won,
           st.w_2ndwon AS second_serve_won, st.w_svgms AS service_games,
           st.w_bpsaved AS bp_saved, st.w_bpfaced AS bp_faced,
           st.l_ace AS opp_aces, st.l_df AS opp_double_faults, st.l_svpt AS opp_serve_points,
           st.l_1stin AS opp_first_serves_in, st.l_1stwon AS opp_first_serve_won,
           st.l_2ndwon AS opp_second_serve_won, st.l_svgms AS opp_service_games,
           st.l_bpsaved AS opp_bp_saved, st.l_bpfaced AS opp_bp_faced
    FROM matches m LEFT JOIN match_stats st USING (match_id)
    WHERE m.winner_id = %(pid)s
    UNION ALL
    SELECT m.match_id, m.match_date, m.event, m.surface, m.level, m.round, m.source AS match_source,
           {_ROUND_ORDER_SQL},
           'L', m.score,
           m.loser_rank, m.loser_rank_points,
           m.winner_id, m.winner_rank,
           st.minutes,
           st.l_ace, st.l_df, st.l_svpt, st.l_1stin, st.l_1stwon,
           st.l_2ndwon, st.l_svgms, st.l_bpsaved, st.l_bpfaced,
           st.w_ace, st.w_df, st.w_svpt, st.w_1stin, st.w_1stwon,
           st.w_2ndwon, st.w_svgms, st.w_bpsaved, st.w_bpfaced
    FROM matches m LEFT JOIN match_stats st USING (match_id)
    WHERE m.loser_id = %(pid)s
)
SELECT s.*, p.name AS opp_name, p.hand AS opp_hand, p.country AS opp_country
FROM sides s LEFT JOIN players p ON p.player_id = s.opp_id
ORDER BY s.match_date DESC, s.round_ord DESC
"""


def quarantine_shifted_duplicates(df: pd.DataFrame, max_shift_days: int = 14) -> pd.DataFrame:
    """Drop later copies of an exact result that was shifted into a newer week.

    ATP event discovery has occasionally re-ingested a finished tournament with
    the scrape week's Monday as ``match_date``.  The runtime identity uses the
    source, the event label before its first comma, and the player-perspective
    equivalent of winner/loser/round/score: opponent, result, round, and score.
    Keeping the earliest row is correct for this failure mode and prevents the
    later copy from inflating recent-form windows.

    The 14-day bound covers the observed +7 and +14 day shifts while avoiding
    broad, unbounded deduplication of genuine rematches.
    """
    if df is None or df.empty:
        return df

    required = {"date", "event", "match_source", "opp_id", "result", "round", "score"}
    if not required.issubset(df.columns):
        return df

    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    # The observed aliases differ only after the first comma (for example
    # ``Braunschweig`` vs ``Braunschweig, Germany BRAWO OPEN``).  Every one of
    # the 595 audited shifted pairs shares this canonical event key and source;
    # requiring both avoids collapsing a rare identical-score rematch at a
    # different tournament.
    out["_event_key"] = out["event"].fillna("").astype(str).map(
        lambda value: " ".join(value.split(",", 1)[0].casefold().split())
    )
    eligible = (
        out["date"].notna()
        & out["_event_key"].ne("")
        & out["match_source"].fillna("").astype(str).str.strip().ne("")
        & out["opp_id"].notna()
        & out["result"].astype(str).str.upper().isin(["W", "L"])
        & out["round"].fillna("").astype(str).str.strip().ne("")
        & out["score"].fillna("").astype(str).str.strip().ne("")
    )

    drop_indices: list[int] = []
    signature = ["match_source", "_event_key", "opp_id", "result", "round", "score"]
    candidates = out.loc[eligible].sort_values(signature + ["date"], kind="stable")
    for _, group in candidates.groupby(signature, sort=False, dropna=False):
        last_kept_date = None
        for idx, row in group.iterrows():
            row_date = row["date"]
            if last_kept_date is not None:
                shift_days = int((row_date - last_kept_date).days)
                if 0 < shift_days <= int(max_shift_days):
                    drop_indices.append(idx)
                    continue
            last_kept_date = row_date

    if drop_indices:
        out = out.drop(index=drop_indices)
    out = out.drop(columns=["_event_key"])
    out.attrs["shifted_duplicate_rows_quarantined"] = len(drop_indices)
    return out


def find_player_id(conn, name: str) -> Optional[int]:
    """Unique player id by exact, then unaccent-normalized, name match."""
    with conn.cursor() as cur:
        return cs._resolve_player_id(cur, name)


def get_history_frame(conn, player_id: int) -> pd.DataFrame:
    """Full career history, player-perspective, most-recent-first (TA schema)."""
    df = pd.read_sql(_HISTORY_SQL, conn, params={"pid": int(player_id)})
    if df.empty:
        return df
    df = df.rename(columns={"match_date": "date"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["surface"] = df["surface"].fillna("").astype(str).str.title()
    for col in ("event", "round", "level", "opp_name", "opp_hand", "opp_country", "score"):
        df[col] = df[col].fillna("")
    df["source"] = "store"
    # Quarantine the known wrong-week ingestion failure before any rolling
    # feature sees it.  Exact-date label duplicates are handled separately
    # below; this pass catches the +7/+14 day copies that otherwise survive.
    df = quarantine_shifted_duplicates(df)
    # same real match can exist under two event labels (hub vs calendar
    # discovery) — one row per (date, opponent, round) or form windows double-count
    df = df.drop_duplicates(subset=["date", "opp_id", "round"], keep="first")
    # Keep ``opp_id``: shared candidate H2H selection must prefer stable
    # identity over display-name matching.  The additive column is ignored by
    # legacy calculators.
    return df.drop(columns=["round_ord", "match_id", "match_source"])


def get_profile(conn, name: str) -> Optional[dict]:
    """Static profile from the players table (TA-profile-shaped, sans current rank —
    callers supply rank/points from the live rankings scrape as they already do)."""
    with conn.cursor() as cur:
        pid = cs._resolve_player_id(cur, name)
        if pid is None:
            return None
        cur.execute(
            "SELECT player_id, name, hand, height_cm, country, birthdate FROM players WHERE player_id=%s",
            (pid,),
        )
        r = cur.fetchone()
    if r is None:
        return None
    birthdate = r[5].isoformat() if r[5] is not None else None
    age = None
    if r[5] is not None:
        age = (pd.Timestamp.now() - pd.Timestamp(r[5])).days / 365.25
    return {
        "player_id": r[0],
        "name": r[1],
        "hand": r[2] or "U",
        "height_cm": float(r[3]) if r[3] is not None else None,
        "country": r[4] or "",
        "birthdate": birthdate,
        "age": round(age, 1) if age is not None else None,
    }


def latest_recorded_rank(conn, player_id: int, max_age_days: int = 120) -> Optional[int]:
    """Most recent rank recorded on the player's matches (training lineage).

    Covers players ranked deeper than the live rankings file reaches
    (~1500): training data carried their real ranks via Sackmann's columns,
    so serving 999 for a #1800 player is train/serve skew. Staleness-capped —
    beyond max_age_days the snapshot is too old to trust and callers fall
    through to the unranked convention.
    """
    with conn.cursor() as cur:
        cur.execute(
            """SELECT CASE WHEN winner_id = %s THEN winner_rank ELSE loser_rank END AS r
               FROM matches
               WHERE (winner_id = %s OR loser_id = %s)
                 AND match_date >= CURRENT_DATE - %s
                 AND CASE WHEN winner_id = %s THEN winner_rank ELSE loser_rank END IS NOT NULL
               ORDER BY match_date DESC LIMIT 1""",
            (player_id, player_id, player_id, max_age_days, player_id),
        )
        row = cur.fetchone()
        return int(row[0]) if row and row[0] else None
