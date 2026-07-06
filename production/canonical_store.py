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
CREATE TABLE IF NOT EXISTS players (
    player_id  BIGINT PRIMARY KEY,
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
