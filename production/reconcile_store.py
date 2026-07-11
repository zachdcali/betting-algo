"""Cross-source reconciliation over the canonical store (build 2).

Three passes, all loud, none silent:

1. detect_conflicts — the same real match recorded by two sources (same
   winner+loser within a 6-day window) with disagreeing fields writes a
   match_conflicts row per disagreement (level/surface/round/score), plus
   'winner' when the two sources flip the result. Already-recorded unresolved
   conflicts are not duplicated.

2. repair_from_curated — the one class safe to auto-repair: Sackmann is the
   curated record for LEVEL (the live tour hub cannot distinguish Masters 'M'
   from 'A'; Davis Cup weeks etc.), so a live-source row's level is updated to
   Sackmann's and the conflict marked resolved. NULL level/surface on either
   twin is filled from the other (fill, not overwrite). Score/round/winner
   disagreements are never auto-picked — they stay open for eyes.

3. fill_missing_stats — store matches we can see on an ATP results page whose
   match_stats row is missing get their serve stats fetched (shared browser)
   and inserted with source='atp_stats'. Covers the Slam/tour granular gap
   (Wimbledon 2026 sat at 0/350 before this).

CLI:  python reconcile_store.py [--since-days 75] [--detect-only]
      [--stats-url URL ...] [--stats-cap 60]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "scraping"))
sys.path.insert(0, str(Path(__file__).parent / "features"))

import canonical_store as cs

COMPARE_FIELDS = ("level", "surface", "round", "score")


def _pairs_sql(since_days: int) -> str:
    # one sackmann row 'a' joined to its live-source twin 'b'
    return f"""
        SELECT a.match_id, b.match_id, a.source, b.source,
               a.level, b.level, a.surface, b.surface,
               a.round, b.round, a.score, b.score
        FROM matches a
        JOIN matches b
          ON a.winner_id = b.winner_id AND a.loser_id = b.loser_id
         AND abs(a.match_date - b.match_date) <= 6
         AND a.source = 'sackmann' AND b.source <> 'sackmann'
        WHERE a.match_date >= now()::date - {int(since_days)}
    """


def detect_conflicts(conn, since_days: int = 75) -> dict:
    """Write one match_conflicts row per cross-source field disagreement."""
    found: dict[str, int] = {}
    with conn.cursor() as cur:
        cur.execute(_pairs_sql(since_days))
        pairs = cur.fetchall()
        for (a_id, b_id, a_src, b_src,
             a_lvl, b_lvl, a_surf, b_surf, a_rnd, b_rnd, a_sc, b_sc) in pairs:
            vals = {"level": (b_lvl, a_lvl), "surface": (b_surf, a_surf),
                    "round": (b_rnd, a_rnd), "score": (b_sc, a_sc)}
            for field, (existing, incoming) in vals.items():
                if not existing or not incoming:
                    continue  # NULLs are gaps (filled by repair), not conflicts
                if str(existing).strip().lower() == str(incoming).strip().lower():
                    continue
                cur.execute(
                    """INSERT INTO match_conflicts (match_id, field, existing_value,
                                                    incoming_value, incoming_source)
                       SELECT %s, %s, %s, %s, %s
                       WHERE NOT EXISTS (
                           SELECT 1 FROM match_conflicts
                           WHERE match_id = %s AND field = %s AND NOT resolved)""",
                    (b_id, field, str(existing), str(incoming), a_src, b_id, field),
                )
                if cur.rowcount:
                    found[field] = found.get(field, 0) + 1
        # winner flips: same pair, opposite orientation, same window
        cur.execute(f"""
            SELECT b.match_id, a.source
            FROM matches a
            JOIN matches b
              ON a.winner_id = b.loser_id AND a.loser_id = b.winner_id
             AND abs(a.match_date - b.match_date) <= 6
             AND a.source = 'sackmann' AND b.source <> 'sackmann'
            WHERE a.match_date >= now()::date - {int(since_days)}
        """)
        for b_id, a_src in cur.fetchall():
            cur.execute(
                """INSERT INTO match_conflicts (match_id, field, existing_value,
                                                incoming_value, incoming_source)
                   SELECT %s, 'winner', 'as-recorded', 'FLIPPED vs sackmann', %s
                   WHERE NOT EXISTS (
                       SELECT 1 FROM match_conflicts
                       WHERE match_id = %s AND field = 'winner' AND NOT resolved)""",
                (b_id, a_src, b_id),
            )
            if cur.rowcount:
                found["winner"] = found.get("winner", 0) + 1
    return found


def repair_from_curated(conn, since_days: int = 75) -> dict:
    """Sackmann wins on level; NULL level/surface filled from the twin."""
    out = {"level_corrected": 0, "level_filled": 0, "surface_filled": 0}
    with conn.cursor() as cur:
        cur.execute(f"""
            UPDATE matches b SET level = a.level
            FROM matches a
            WHERE a.winner_id = b.winner_id AND a.loser_id = b.loser_id
              AND abs(a.match_date - b.match_date) <= 6
              AND a.source = 'sackmann' AND b.source <> 'sackmann'
              AND a.level IS NOT NULL AND b.level IS NOT NULL
              AND a.level <> b.level
              AND a.match_date >= now()::date - {int(since_days)}
        """)
        out["level_corrected"] = cur.rowcount
        cur.execute(
            """UPDATE match_conflicts c SET resolved = TRUE
               WHERE c.field = 'level' AND NOT c.resolved""")
        for field in ("level", "surface"):
            cur.execute(f"""
                UPDATE matches b SET {field} = a.{field}
                FROM matches a
                WHERE a.winner_id = b.winner_id AND a.loser_id = b.loser_id
                  AND abs(a.match_date - b.match_date) <= 6
                  AND a.source <> b.source
                  AND b.{field} IS NULL AND a.{field} IS NOT NULL
                  AND b.match_date >= now()::date - {int(since_days)}
            """)
            out[f"{field}_filled"] = cur.rowcount
    return out


_STATS_COLMAP = {  # parser key -> (winner col, loser col) stem
    "aces": "ace", "double_faults": "df", "serve_points": "svpt",
    "first_serves_in": "1stin", "first_serve_won": "1stwon",
    "second_serve_won": "2ndwon", "bp_saved": "bpsaved", "bp_faced": "bpfaced",
}


def fill_missing_stats(conn, results_url: str, cap: int = 60) -> dict:
    """Fetch serve stats for store matches visible on one results page."""
    from atp_results_scraper import fetch_tournament_results, fetch_match_stats
    from history_stitch import _names_loosely_match

    out = {"cards": 0, "already": 0, "no_store_row": 0, "fetched": 0, "no_stats_page": 0}
    cards = fetch_tournament_results(results_url)
    if cards is None or cards.empty:
        print(f"  ⚠️ no cards parsed from {results_url}")
        return out
    out["cards"] = len(cards)
    with conn.cursor() as cur:
        for _, card in cards.iterrows():
            if out["fetched"] >= cap:
                print(f"  stats cap {cap} reached — remaining cards left for next run")
                break
            stats_url = card.get("stats_url")
            if not stats_url or card.get("winner") not in (1, 2):
                continue
            w_name = card["p1"] if card["winner"] == 1 else card["p2"]
            l_name = card["p2"] if card["winner"] == 1 else card["p1"]
            wid = cs._resolve_player_id(cur, w_name)
            lid = cs._resolve_player_id(cur, l_name)
            if wid is None or lid is None:
                out["no_store_row"] += 1
                continue
            rnd = str(card.get("round") or "") or None
            cur.execute(
                """SELECT m.match_id, s.match_id FROM matches m
                   LEFT JOIN match_stats s USING (match_id)
                   WHERE m.winner_id=%s AND m.loser_id=%s
                     AND m.round IS NOT DISTINCT FROM %s
                     AND m.match_date >= now()::date - 40
                   ORDER BY m.match_date DESC LIMIT 1""",
                (wid, lid, rnd),
            )
            row = cur.fetchone()
            if row is None:
                out["no_store_row"] += 1
                continue
            match_id, has_stats = row
            if has_stats is not None:
                out["already"] += 1
                continue
            stats = fetch_match_stats(str(stats_url))
            if not stats:
                out["no_stats_page"] += 1
                continue
            # orient parser sides to store winner/loser by name
            if _names_loosely_match(stats["p1_name"], str(w_name)):
                w_side, l_side = stats["p1"], stats["p2"]
            elif _names_loosely_match(stats["p2_name"], str(w_name)):
                w_side, l_side = stats["p2"], stats["p1"]
            else:
                print(f"  ⚠️ stats page names {stats['p1_name']}/{stats['p2_name']} "
                      f"don't match {w_name} — skipped")
                continue
            cols, vals = ["match_id", "source"], [match_id, "atp_stats"]
            for key, stem in _STATS_COLMAP.items():
                cols += [f"w_{stem}", f"l_{stem}"]
                vals += [w_side.get(key), l_side.get(key)]
            placeholders = ",".join(["%s"] * len(vals))
            cur.execute(
                f"""INSERT INTO match_stats ({",".join(cols)})
                    VALUES ({placeholders}) ON CONFLICT DO NOTHING""",
                vals,
            )
            out["fetched"] += cur.rowcount
    return out


def run(conn=None, since_days: int = 75, stats_urls: list[str] | None = None,
        stats_cap: int = 60, detect_only: bool = False) -> dict:
    own = conn is None
    if own:
        conn = cs.connect()
    summary: dict = {}
    try:
        with conn.transaction():
            summary["conflicts"] = detect_conflicts(conn, since_days)
        if not detect_only:
            with conn.transaction():
                summary["repairs"] = repair_from_curated(conn, since_days)
        for url in (stats_urls or []):
            with conn.transaction():
                summary.setdefault("stats", []).append(
                    {url: fill_missing_stats(conn, url, cap=stats_cap)})
    finally:
        if own:
            conn.close()
    print(f"  🔎 reconcile: {summary}")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--since-days", type=int, default=75)
    ap.add_argument("--stats-url", action="append", default=[])
    ap.add_argument("--stats-cap", type=int, default=60)
    ap.add_argument("--detect-only", action="store_true")
    args = ap.parse_args()
    run(since_days=args.since_days, stats_urls=args.stats_url,
        stats_cap=args.stats_cap, detect_only=args.detect_only)
