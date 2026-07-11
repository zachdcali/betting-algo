"""Canonical match/player store (Supabase Postgres).

The owned database Zach asked for: full player histories + granular stats with
source provenance, so live feature computation reads OUR data instead of
re-scraping, and TA/ATP lag becomes an ingestion detail. Schema:

- players        Sackmann player_id as canonical identity (+ ta_slug/atp_url mapping)
- matches        one row per match, winner/loser ids, tournament-start date
                 (training convention), UNIQUE NULLS NOT DISTINCT dedupe key,
                 `source` provenance (sackmann | ta | atp_results | atp_activity)
- match_stats    Sackmann-shaped granular stats (w_/l_ mirrors), only when real
- match_conflicts cross-source disagreements (reconciliation writes here —
                 conflicts are logged, never silently overwritten)
- ingest_runs    provenance of every ingestion pass

Bulk backfill uses COPY into an all-TEXT staging table, then set-based casts
server-side (fast over the pooler; no per-row round trips).

CLI:
    tennis_env/bin/python production/canonical_store.py --create
    tennis_env/bin/python production/canonical_store.py --seed-players
    tennis_env/bin/python production/canonical_store.py --backfill
    tennis_env/bin/python production/canonical_store.py --summary
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import psycopg

REPO_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = REPO_ROOT / ".env.supabase"
PLAYERS_CSV = REPO_ROOT / "data" / "JeffSackmann" / "ATP Players" / "atp_players.csv"
MASTER_CSV = REPO_ROOT / "data" / "JeffSackmann" / "jeffsackmann_master_combined.csv"

SCHEMA_SQL = """
CREATE SEQUENCE IF NOT EXISTS players_new_id_seq START 9000000;
CREATE TABLE IF NOT EXISTS players (
    player_id  BIGINT PRIMARY KEY DEFAULT nextval('players_new_id_seq'),
    name       TEXT NOT NULL,
    hand       TEXT,
    height_cm  REAL,
    country    TEXT,
    birthdate  DATE,
    ta_slug    TEXT,
    atp_url    TEXT,
    updated_at TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX IF NOT EXISTS players_name_idx ON players (lower(name));

CREATE TABLE IF NOT EXISTS matches (
    match_id   BIGSERIAL PRIMARY KEY,
    tourney_id TEXT,
    match_date DATE NOT NULL,
    event      TEXT,
    surface    TEXT,
    level      TEXT,
    round      TEXT,
    draw_size  INT,
    best_of    INT,
    winner_id  BIGINT REFERENCES players(player_id),
    loser_id   BIGINT REFERENCES players(player_id),
    score      TEXT,
    source     TEXT NOT NULL,
    scraped_at TIMESTAMPTZ DEFAULT now(),
    UNIQUE NULLS NOT DISTINCT (match_date, event, round, winner_id, loser_id)
);
CREATE INDEX IF NOT EXISTS matches_winner_date_idx ON matches (winner_id, match_date DESC);
CREATE INDEX IF NOT EXISTS matches_loser_date_idx  ON matches (loser_id, match_date DESC);

CREATE TABLE IF NOT EXISTS match_stats (
    match_id  BIGINT PRIMARY KEY REFERENCES matches(match_id) ON DELETE CASCADE,
    minutes   REAL,
    w_ace REAL, w_df REAL, w_svpt REAL, w_1stin REAL, w_1stwon REAL,
    w_2ndwon REAL, w_svgms REAL, w_bpsaved REAL, w_bpfaced REAL,
    l_ace REAL, l_df REAL, l_svpt REAL, l_1stin REAL, l_1stwon REAL,
    l_2ndwon REAL, l_svgms REAL, l_bpsaved REAL, l_bpfaced REAL,
    source    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS match_conflicts (
    conflict_id     BIGSERIAL PRIMARY KEY,
    match_id        BIGINT REFERENCES matches(match_id),
    field           TEXT NOT NULL,
    existing_value  TEXT,
    incoming_value  TEXT,
    incoming_source TEXT,
    detected_at     TIMESTAMPTZ DEFAULT now(),
    resolved        BOOLEAN DEFAULT FALSE
);

CREATE TABLE IF NOT EXISTS ingest_runs (
    run_id        BIGSERIAL PRIMARY KEY,
    source        TEXT NOT NULL,
    started_at    TIMESTAMPTZ,
    finished_at   TIMESTAMPTZ,
    rows_inserted INT,
    rows_skipped  INT,
    notes         TEXT
);
"""


def load_database_url(env_path: Path = ENV_PATH) -> str:
    for line in env_path.read_text().splitlines():
        if line.startswith("DATABASE_URL="):
            return line.split("=", 1)[1].strip()
    raise RuntimeError(f"DATABASE_URL not found in {env_path}")


def connect() -> psycopg.Connection:
    return psycopg.connect(load_database_url(), connect_timeout=20)


def create_schema(conn) -> None:
    with conn.cursor() as cur:
        cur.execute(SCHEMA_SQL)
    conn.commit()
    print("schema created / verified")


def _copy_csv(cur, table: str, csv_path: Path, lines_per_batch: int = 150_000) -> None:
    """Stream a local CSV into an existing all-TEXT staging table.

    Chunked into multiple COPY statements so no single statement outlives
    Supabase's statement timeout (which killed a one-shot 916k-row COPY).
    """
    cur.execute("SET statement_timeout = 0")
    with open(csv_path, "r") as fh:
        fh.readline()  # skip header; batches use HEADER false
        batch: list[str] = []
        batch_no = 0
        while True:
            line = fh.readline()
            if line:
                batch.append(line)
            if batch and (len(batch) >= lines_per_batch or not line):
                batch_no += 1
                with cur.copy(f"COPY {table} FROM STDIN WITH (FORMAT csv, HEADER false)") as copy:
                    copy.write("".join(batch))
                print(f"    copied batch {batch_no} ({len(batch)} rows)")
                batch = []
            if not line:
                break


def _staging_from_header(cur, table: str, csv_path: Path) -> list[str]:
    header = csv_path.open().readline().strip().split(",")
    cols = ", ".join(f'"{c}" TEXT' for c in header)
    cur.execute(f"DROP TABLE IF EXISTS {table}")
    cur.execute(f"CREATE UNLOGGED TABLE {table} ({cols})")
    return header


def seed_players(conn, csv_path: Path = PLAYERS_CSV) -> None:
    t0 = time.time()
    with conn.cursor() as cur:
        _staging_from_header(cur, "stage_players", csv_path)
        _copy_csv(cur, "stage_players", csv_path)
        cur.execute("""
            INSERT INTO players (player_id, name, hand, height_cm, country, birthdate)
            SELECT DISTINCT ON (player_id::bigint)
                player_id::bigint,
                NULLIF(trim(coalesce(name_first,'') || ' ' || coalesce(name_last,'')), ''),
                NULLIF(hand, ''),
                NULLIF(height, '')::real,
                NULLIF(ioc, ''),
                CASE WHEN dob ~ '^\\d{8}$' AND substring(dob,5,2) BETWEEN '01' AND '12'
                     THEN to_date(dob, 'YYYYMMDD') END
            FROM stage_players
            WHERE player_id ~ '^\\d+$'
              AND NULLIF(trim(coalesce(name_first,'') || ' ' || coalesce(name_last,'')), '') IS NOT NULL
            ON CONFLICT (player_id) DO NOTHING
        """)
        inserted = cur.rowcount
        cur.execute("DROP TABLE stage_players")
    conn.commit()
    print(f"players seeded: {inserted} rows in {time.time()-t0:.0f}s")


def backfill_sackmann(conn, csv_path: Path = MASTER_CSV) -> None:
    """Full historical backfill from the local Sackmann master CSV (one-time)."""
    t0 = time.time()
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO ingest_runs (source, started_at, notes) VALUES ('sackmann', now(), %s) RETURNING run_id",
            (str(csv_path.name),),
        )
        run_id = cur.fetchone()[0]

        _staging_from_header(cur, "stage_matches", csv_path)
        print("  uploading master CSV via COPY (may take a few minutes)...")
        _copy_csv(cur, "stage_matches", csv_path)
        cur.execute("SELECT count(*) FROM stage_matches")
        staged = cur.fetchone()[0]
        print(f"  staged rows: {staged}")

        # players referenced by matches but absent from atp_players.csv
        cur.execute("""
            INSERT INTO players (player_id, name, hand, height_cm, country)
            SELECT DISTINCT ON (pid) pid, pname, NULLIF(phand,''), NULLIF(pht,'')::real, NULLIF(pioc,'')
            FROM (
                SELECT winner_id::bigint pid, winner_name pname, winner_hand phand,
                       winner_ht pht, winner_ioc pioc FROM stage_matches WHERE winner_id ~ '^\\d+$'
                UNION ALL
                SELECT loser_id::bigint, loser_name, loser_hand, loser_ht, loser_ioc
                FROM stage_matches WHERE loser_id ~ '^\\d+$'
            ) u
            WHERE NULLIF(trim(pname), '') IS NOT NULL
            ON CONFLICT (player_id) DO NOTHING
        """)
        print(f"  players added from match data: {cur.rowcount}")

        cur.execute("""
            INSERT INTO matches (tourney_id, match_date, event, surface, level, round,
                                 draw_size, best_of, winner_id, loser_id, score, source)
            SELECT tourney_id,
                   to_date(tourney_date, 'YYYYMMDD'),
                   tourney_name, NULLIF(surface,''), NULLIF(tourney_level,''), NULLIF(round,''),
                   NULLIF(regexp_replace(draw_size, '\\D', '', 'g'), '')::int,
                   NULLIF(best_of,'')::numeric::int,
                   winner_id::bigint, loser_id::bigint, NULLIF(score,''), 'sackmann'
            FROM stage_matches
            WHERE tourney_date ~ '^\\d{8}$' AND winner_id ~ '^\\d+$' AND loser_id ~ '^\\d+$'
            ON CONFLICT DO NOTHING
        """)
        n_matches = cur.rowcount
        print(f"  matches inserted: {n_matches}")

        cur.execute("""
            INSERT INTO match_stats (match_id, minutes,
                w_ace, w_df, w_svpt, w_1stin, w_1stwon, w_2ndwon, w_svgms, w_bpsaved, w_bpfaced,
                l_ace, l_df, l_svpt, l_1stin, l_1stwon, l_2ndwon, l_svgms, l_bpsaved, l_bpfaced,
                source)
            SELECT m.match_id, NULLIF(s.minutes,'')::real,
                NULLIF(s."w_ace",'')::real, NULLIF(s."w_df",'')::real, NULLIF(s."w_svpt",'')::real,
                NULLIF(s."w_1stIn",'')::real, NULLIF(s."w_1stWon",'')::real, NULLIF(s."w_2ndWon",'')::real,
                NULLIF(s."w_SvGms",'')::real, NULLIF(s."w_bpSaved",'')::real, NULLIF(s."w_bpFaced",'')::real,
                NULLIF(s."l_ace",'')::real, NULLIF(s."l_df",'')::real, NULLIF(s."l_svpt",'')::real,
                NULLIF(s."l_1stIn",'')::real, NULLIF(s."l_1stWon",'')::real, NULLIF(s."l_2ndWon",'')::real,
                NULLIF(s."l_SvGms",'')::real, NULLIF(s."l_bpSaved",'')::real, NULLIF(s."l_bpFaced",'')::real,
                'sackmann'
            FROM stage_matches s
            JOIN matches m ON m.match_date = to_date(s.tourney_date,'YYYYMMDD')
                          AND m.event = s.tourney_name
                          AND (m.round = NULLIF(s.round,'') OR (m.round IS NULL AND NULLIF(s.round,'') IS NULL))
                          AND m.winner_id = s.winner_id::bigint AND m.loser_id = s.loser_id::bigint
            WHERE NULLIF(s."w_svpt",'') IS NOT NULL AND s.tourney_date ~ '^\\d{8}$'
                  AND s.winner_id ~ '^\\d+$' AND s.loser_id ~ '^\\d+$'
            ON CONFLICT (match_id) DO NOTHING
        """)
        n_stats = cur.rowcount
        print(f"  match_stats inserted: {n_stats}")

        cur.execute("DROP TABLE stage_matches")
        cur.execute(
            "UPDATE ingest_runs SET finished_at=now(), rows_inserted=%s, rows_skipped=%s WHERE run_id=%s",
            (n_matches, staged - n_matches, run_id),
        )
    conn.commit()
    print(f"backfill complete in {time.time()-t0:.0f}s")


def backfill_ranks(conn, csv_path: Path) -> None:
    """One-time migration: add rank-at-match-time columns (needed by the
    rank-momentum features) and fill them from a Sackmann CSV via update-join."""
    t0 = time.time()
    with conn.cursor() as cur:
        cur.execute("""
            ALTER TABLE matches
                ADD COLUMN IF NOT EXISTS winner_rank REAL,
                ADD COLUMN IF NOT EXISTS winner_rank_points REAL,
                ADD COLUMN IF NOT EXISTS loser_rank REAL,
                ADD COLUMN IF NOT EXISTS loser_rank_points REAL
        """)
        _staging_from_header(cur, "stage_ranks", csv_path)
        _copy_csv(cur, "stage_ranks", csv_path)
        cur.execute("""
            UPDATE matches m SET
                winner_rank        = NULLIF(s.winner_rank,'')::real,
                winner_rank_points = NULLIF(s.winner_rank_points,'')::real,
                loser_rank         = NULLIF(s.loser_rank,'')::real,
                loser_rank_points  = NULLIF(s.loser_rank_points,'')::real
            FROM stage_ranks s
            WHERE s.tourney_date ~ '^\\d{8}$' AND s.winner_id ~ '^\\d+$' AND s.loser_id ~ '^\\d+$'
              AND m.match_date = to_date(s.tourney_date,'YYYYMMDD')
              AND m.event = s.tourney_name
              AND m.round IS NOT DISTINCT FROM NULLIF(s.round,'')
              AND m.winner_id = s.winner_id::bigint AND m.loser_id = s.loser_id::bigint
              AND m.source = 'sackmann'
        """)
        print(f"  ranks filled on {cur.rowcount} matches from {csv_path.name}")
        cur.execute("DROP TABLE stage_ranks")
    conn.commit()
    print(f"  done in {time.time()-t0:.0f}s")


def _resolve_player_id(cur, name: str) -> int | None:
    """Unique name match against players; exact first, then normalized
    (case/diacritics/hyphens/apostrophes stripped). None if ambiguous/missing —
    never guessed."""
    cur.execute("SELECT player_id FROM players WHERE lower(name) = lower(%s)", (name,))
    rows = cur.fetchall()
    if len(rows) == 1:
        return rows[0][0]
    try:  # explicit alias map (cross-source spellings)
        cur.execute(
            """SELECT player_id FROM player_aliases
               WHERE alias_norm = regexp_replace(lower(unaccent(%s)), '[^a-z]', '', 'g')""",
            (name,),
        )
        rows = cur.fetchall()
        if len(rows) == 1:
            return rows[0][0]
    except Exception:
        conn_ = cur.connection
        conn_.rollback()  # alias table absent — continue with name matching
    cur.execute(
        """SELECT player_id FROM players
           WHERE regexp_replace(lower(unaccent(name)), '[^a-z]', '', 'g')
               = regexp_replace(lower(unaccent(%s)), '[^a-z]', '', 'g')""",
        (name,),
    )
    rows = cur.fetchall()
    if len(rows) == 1:
        return rows[0][0]
    # prefix: ATP shortens some names ("Daniel Merida" -> "Daniel Merida Aguilar")
    cur.execute(
        """SELECT player_id FROM players
           WHERE regexp_replace(lower(unaccent(name)), '[^a-z]', '', 'g')
             LIKE regexp_replace(lower(unaccent(%s)), '[^a-z]', '', 'g') || '%%'""",
        (name,),
    )
    rows = cur.fetchall()
    if len(rows) == 1:
        return rows[0][0]
    # reversed order: "Yunchaokete Bu" (ATP) vs "Bu Yunchaokete" (Sackmann)
    parts = str(name).strip().split()
    if len(parts) >= 2:
        reversed_name = " ".join(parts[::-1])
        cur.execute(
            """SELECT player_id FROM players
               WHERE regexp_replace(lower(unaccent(name)), '[^a-z]', '', 'g')
                   = regexp_replace(lower(unaccent(%s)), '[^a-z]', '', 'g')""",
            (reversed_name,),
        )
        rows = cur.fetchall()
        if len(rows) == 1:
            return rows[0][0]
    return None


ALIAS_SQL = """
CREATE TABLE IF NOT EXISTS player_aliases (
    alias_norm TEXT PRIMARY KEY,
    player_id  BIGINT NOT NULL REFERENCES players(player_id)
);
"""

# Known cross-source spellings (ATP site vs Sackmann). Extend as found.
KNOWN_ALIASES = {
    "Aleksandr Shevchenko": "Alexander Shevchenko",
    "Abdullah Shelbayh": "Abedallah Shelbayh",
}


def seed_aliases(conn) -> None:
    with conn.cursor() as cur:
        cur.execute(ALIAS_SQL)
        for alias, canonical in KNOWN_ALIASES.items():
            cur.execute(
                """INSERT INTO player_aliases (alias_norm, player_id)
                   SELECT regexp_replace(lower(unaccent(%s)), '[^a-z]', '', 'g'), player_id
                   FROM players WHERE lower(name) = lower(%s)
                   ON CONFLICT (alias_norm) DO NOTHING""",
                (alias, canonical),
            )
    conn.commit()
    print("aliases seeded")


def merge_duplicate_players(conn) -> int:
    """Merge Sackmann's duplicate player entries (same normalized name AND same
    birthdate → same human). Matches repoint to the id with the larger match
    count; duplicate match rows (same match under both ids) are deleted; the
    minor player row is removed. Conservative: birthdate must be non-null."""
    merged = 0
    with conn.cursor() as cur:
        cur.execute(ALIAS_SQL)  # idempotent; merge cleans alias rows of merged ids
        cur.execute("""
            SELECT array_agg(player_id ORDER BY player_id), count(*)
            FROM players
            WHERE birthdate IS NOT NULL
            GROUP BY regexp_replace(lower(unaccent(name)), '[^a-z]', '', 'g'), birthdate
            HAVING count(*) > 1
        """)
        groups = [r[0] for r in cur.fetchall()]
        for ids in groups:
            counts = []
            for pid in ids:
                cur.execute("SELECT count(*) FROM matches WHERE winner_id=%s OR loser_id=%s", (pid, pid))
                counts.append((cur.fetchone()[0], pid))
            counts.sort(reverse=True)
            major = counts[0][1]
            for _, minor in counts[1:]:
                for col in ("winner_id", "loser_id"):
                    cur.execute(f"SELECT match_id FROM matches WHERE {col}=%s", (minor,))
                    for (mid,) in cur.fetchall():
                        try:
                            with conn.transaction():
                                cur.execute(f"UPDATE matches SET {col}=%s WHERE match_id=%s", (major, mid))
                        except Exception:  # unique dedupe key hit -> same match already under major
                            cur.execute("DELETE FROM matches WHERE match_id=%s", (mid,))
                cur.execute("DELETE FROM player_aliases WHERE player_id=%s", (minor,))
                cur.execute("DELETE FROM players WHERE player_id=%s", (minor,))
                merged += 1
    conn.commit()
    print(f"duplicate players merged: {merged}")
    return merged


def _current_rank_lookup():
    """(rank, points) by player name from the weekly rankings CSV.

    Used for scraped event rows, which carry no rank: for matches within the
    last few weeks the current official ranking IS the near-match-time rank
    (rankings are weekly). Provenance: data/atp_rankings.csv scraped_at.
    """
    try:
        import sys as _sys
        _sys.path.insert(0, str(Path(__file__).parent / "scraping"))
        from atp_rankings_scraper import load_rankings, get_player_rank, get_player_points
        df = load_rankings()
        if df is None or df.empty:
            return lambda nm: (None, None)
        return lambda nm: (get_player_rank(nm, df), get_player_points(nm, df))
    except Exception:
        return lambda nm: (None, None)


def fill_missing_ranks(conn, source: str = "atp_results") -> int:
    """Backfill NULL ranks on scraped rows by BACKWARD carry-forward: each row
    gets that player's most recent real rank observation strictly BEFORE the
    row's date. Never uses later observations — a future rank stamped onto a
    dated row is temporal leakage (it fabricated volatility in testing)."""
    updated = 0
    with conn.cursor() as cur:
        cur.execute("SET statement_timeout = 0")
        for side, rank_col, pts_col in (("winner", "winner_rank", "winner_rank_points"),
                                        ("loser", "loser_rank", "loser_rank_points")):
            cur.execute(f"""
                UPDATE matches m SET {rank_col} = sub.rank, {pts_col} = sub.pts
                FROM (
                    SELECT t.match_id,
                        (SELECT CASE WHEN p.winner_id = t.pid THEN p.winner_rank ELSE p.loser_rank END
                         FROM matches p
                         WHERE (p.winner_id = t.pid OR p.loser_id = t.pid)
                           AND p.match_date < t.match_date
                           AND (CASE WHEN p.winner_id = t.pid THEN p.winner_rank ELSE p.loser_rank END) IS NOT NULL
                         ORDER BY p.match_date DESC LIMIT 1) AS rank,
                        (SELECT CASE WHEN p.winner_id = t.pid THEN p.winner_rank_points ELSE p.loser_rank_points END
                         FROM matches p
                         WHERE (p.winner_id = t.pid OR p.loser_id = t.pid)
                           AND p.match_date < t.match_date
                           AND (CASE WHEN p.winner_id = t.pid THEN p.winner_rank_points ELSE p.loser_rank_points END) IS NOT NULL
                         ORDER BY p.match_date DESC LIMIT 1) AS pts
                    FROM (SELECT match_id, {side}_id AS pid, match_date
                          FROM matches WHERE source = %s AND {rank_col} IS NULL) t
                ) sub
                WHERE m.match_id = sub.match_id AND sub.rank IS NOT NULL
            """, (source,))
            updated += cur.rowcount
    conn.commit()
    print(f"  ranks carry-forward-filled ({updated} column-updates)")
    return updated


def ingest_event_results(conn, results_df, event: str, start_date: str,
                         surface: str, level: str, source: str = "atp_results") -> dict:
    """Upsert one tournament's scraped results (atp_results_scraper frame) into the store.

    Rows whose players can't be uniquely resolved are skipped and counted —
    never guessed. Existing rows (e.g. Sackmann later covering the same match)
    are left untouched by the dedupe key; reconciliation compares them separately.
    """
    inserted = skipped = 0
    unresolved: list[str] = []
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO ingest_runs (source, started_at, notes) VALUES (%s, now(), %s) RETURNING run_id",
            (source, f"{event} {start_date}"),
        )
        run_id = cur.fetchone()[0]
        for _, card in results_df.iterrows():
            if card.get("winner") not in (1, 2):
                skipped += 1
                continue
            w_name = card["p1"] if card["winner"] == 1 else card["p2"]
            l_name = card["p2"] if card["winner"] == 1 else card["p1"]
            wid, lid = _resolve_player_id(cur, w_name), _resolve_player_id(cur, l_name)
            if wid is None or lid is None:
                skipped += 1
                unresolved.append(w_name if wid is None else l_name)
                continue
            w_sets = card["p1_sets"] if card["winner"] == 1 else card["p2_sets"]
            l_sets = card["p2_sets"] if card["winner"] == 1 else card["p1_sets"]
            score = " ".join(f"{a}-{b}" for a, b in zip(str(w_sets).split(), str(l_sets).split()))
            # ranks left NULL here; fill_missing_ranks() carry-forward-fills them
            # from each player's own prior observations (leak-free by construction)
            rnd = str(card.get("round") or "") or None
            # label-blind guard: the same tournament can be discovered under two
            # names (live hub vs calendar) across runs — same players+date+round
            # is the same match regardless of the event string
            cur.execute(
                """INSERT INTO matches (match_date, event, surface, level, round,
                                        winner_id, loser_id, score, source)
                   SELECT %s,%s,%s,%s,%s,%s,%s,%s,%s
                   WHERE NOT EXISTS (
                       SELECT 1 FROM matches x
                       WHERE x.match_date=%s AND x.winner_id=%s AND x.loser_id=%s
                         AND x.round IS NOT DISTINCT FROM %s)
                   ON CONFLICT DO NOTHING""",
                (start_date, event, surface, level, rnd, wid, lid, score or None, source,
                 start_date, wid, lid, rnd),
            )
            inserted += cur.rowcount
        cur.execute(
            "UPDATE ingest_runs SET finished_at=now(), rows_inserted=%s, rows_skipped=%s WHERE run_id=%s",
            (inserted, skipped, run_id),
        )
    if unresolved:
        uniq = sorted(set(unresolved))
        print(f"  unresolved players ({len(uniq)}): {', '.join(uniq[:8])}{'...' if len(uniq) > 8 else ''}")
    print(f"  event ingest [{event}]: inserted={inserted} skipped={skipped}")
    return {"inserted": inserted, "skipped": skipped, "unresolved": len(set(unresolved))}


def summary(conn) -> None:
    with conn.cursor() as cur:
        for tbl in ("players", "matches", "match_stats", "match_conflicts", "ingest_runs"):
            cur.execute(f"SELECT count(*) FROM {tbl}")
            print(f"  {tbl:16} {cur.fetchone()[0]:>9,}")
        cur.execute("SELECT source, count(*), max(match_date) FROM matches GROUP BY source")
        for src, n, latest in cur.fetchall():
            print(f"  matches[{src}]: {n:,} rows, latest {latest}")
        cur.execute("SELECT pg_size_pretty(pg_database_size(current_database()))")
        print(f"  db size: {cur.fetchone()[0]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Canonical match/player store")
    ap.add_argument("--create", action="store_true")
    ap.add_argument("--seed-players", action="store_true")
    ap.add_argument("--backfill", action="store_true")
    ap.add_argument("--summary", action="store_true")
    args = ap.parse_args()
    with connect() as conn:
        if args.create:
            create_schema(conn)
        if args.seed_players:
            seed_players(conn)
        if args.backfill:
            backfill_sackmann(conn)
        if args.summary or not any(vars(args).values()):
            summary(conn)


def surface_from_store(event_name: str, cache: dict | None = None) -> str | None:
    """Resolve a tournament's surface from our own match history.

    The store holds years of ground truth (Braunschweig=Clay, Newport=Grass);
    a new season's event just needs a name match. Self-extends as ingest
    appends new events. Returns None when no confident match exists.
    """
    key = str(event_name).strip().lower()
    if cache is not None and ("surf::" + key) in cache:
        return cache["surf::" + key]
    needle = key.split(",")[0].split("(")[0].strip()
    needle = " ".join(w for w in needle.split()
                      if w.isalpha() and w not in ("atp", "challenger", "itf", "men", "mens", "wta"))
    result = None
    if len(needle) >= 4:
        try:
            with connect() as conn, conn.cursor() as cur:
                # month window disambiguates multi-venue cities: July 'Newport'
                # = grass Rhode Island; the hard Newport Beach event is January
                from datetime import date as _date
                _m = _date.today().month
                cur.execute(
                    """SELECT surface, count(*) AS n FROM matches
                       WHERE surface IS NOT NULL AND match_date >= '2018-01-01'
                         AND event ILIKE %s
                         AND EXTRACT(MONTH FROM match_date) BETWEEN %s AND %s
                       GROUP BY surface ORDER BY n DESC LIMIT 2""",
                    (f"%{needle}%", max(1, _m - 1), min(12, _m + 1)),
                )
                rows = cur.fetchall()
                # confident only when one surface clearly dominates
                if rows and (len(rows) == 1 or rows[0][1] >= 3 * max(1, rows[1][1])):
                    result = rows[0][0]
        except Exception as exc:
            print(f"      ⚠️ store surface lookup failed ({exc})")
    if cache is not None:
        cache["surf::" + key] = result
    return result


def ingest_itf_results(conn, em_df, event: str, start_date: str,
                       surface: str | None, level: str) -> dict:
    """Upsert one ITF event's completed order-of-play matches into the store.

    ITF frames carry (p1, p2, winner, score winner-first, round, completed).
    Unlike ATP ingest, unknown players are CREATED (name + nothing else):
    deep-ITF juniors predate any Sackmann coverage, and their histories can
    only accumulate if their first professional rows are storable.
    """
    inserted = skipped = created = 0
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO ingest_runs (source, started_at, notes) VALUES ('itf_oop', now(), %s) RETURNING run_id",
            (f"{event} {start_date}",),
        )
        run_id = cur.fetchone()[0]

        def resolve_or_create(name: str):
            nonlocal created
            pid = _resolve_player_id(cur, name)
            if pid is None and len(str(name).strip()) >= 5:
                cur.execute(
                    "INSERT INTO players (name, updated_at) VALUES (%s, now()) RETURNING player_id",
                    (str(name).strip(),),
                )
                pid = cur.fetchone()[0]
                created += 1
            return pid

        for _, card in em_df.iterrows():
            if not bool(card.get("completed")) or card.get("winner") not in (1, 2):
                skipped += 1
                continue
            w_name = card["p1"] if int(card["winner"]) == 1 else card["p2"]
            l_name = card["p2"] if int(card["winner"]) == 1 else card["p1"]
            wid, lid = resolve_or_create(w_name), resolve_or_create(l_name)
            if wid is None or lid is None or wid == lid:
                skipped += 1
                continue
            rnd = str(card.get("round") or "") or None
            cur.execute(
                """INSERT INTO matches (match_date, event, surface, level, round,
                                        winner_id, loser_id, score, source)
                   SELECT %s,%s,%s,%s,%s,%s,%s,%s,'itf_oop'
                   WHERE NOT EXISTS (
                       SELECT 1 FROM matches x
                       WHERE x.match_date=%s AND x.winner_id=%s AND x.loser_id=%s
                         AND x.round IS NOT DISTINCT FROM %s)
                   ON CONFLICT DO NOTHING""",
                (start_date, event, surface, level, rnd, wid, lid,
                 str(card.get("score") or "") or None,
                 start_date, wid, lid, rnd),
            )
            inserted += cur.rowcount
        cur.execute(
            "UPDATE ingest_runs SET finished_at=now(), rows_inserted=%s, rows_skipped=%s WHERE run_id=%s",
            (inserted, skipped, run_id),
        )
    print(f"  itf ingest [{event}]: inserted={inserted} skipped={skipped} players_created={created}")
    return {"inserted": inserted, "skipped": skipped, "created": created}
