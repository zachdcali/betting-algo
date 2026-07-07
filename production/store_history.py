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
    SELECT m.match_id, m.match_date, m.event, m.surface, m.level, m.round,
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
    SELECT m.match_id, m.match_date, m.event, m.surface, m.level, m.round,
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
    return df.drop(columns=["round_ord", "opp_id", "match_id"])


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
