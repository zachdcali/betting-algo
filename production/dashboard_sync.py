"""Sync pipeline outputs to Supabase dash_* tables for the live dashboard.

Runs after every pipeline run (cloud + local). The dashboard is a static page
reading these tables via Supabase's REST API (anon key, RLS read-only), so it
auto-updates hourly regardless of where the pipeline ran or whether the git
repo is public or private.

v1 strategy: truncate + COPY the full CSV each sync. Idempotent, no drift, and
cheap at current sizes (~2.5k prediction rows). This also doubles as an
off-machine backup of the log files — the 2026-07-08 git-clobber incident is
the origin story here.
"""
from __future__ import annotations

import io
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from canonical_store import connect  # noqa: E402

BASE = os.path.dirname(__file__)

# (table, csv path, columns to sync — None = all)
SYNC_SPECS = [
    ("dash_predictions", os.path.join(BASE, "prediction_log.csv"), None),
    ("dash_odds_history", os.path.join(BASE, "odds_history.csv"), None),
    ("dash_shadow", os.path.join(BASE, "logs", "performance_v1_shadow_predictions.csv"), None),
    ("dash_runs", os.path.join(BASE, "logs", "audit", "run_history.csv"), None),
    ("dash_bets", os.path.join(BASE, "logs", "all_bets.csv"), None),
    ("dash_snapshots", os.path.join(BASE, "prediction_snapshots.csv"), None),
    ("dash_settlement_audit", os.path.join(BASE, "logs", "audit", "settlement_audit.csv"), None),
]

# Refuse to replace a table with a dramatically smaller one: a shrunken local
# CSV (corruption, another clobber) must never propagate into the backup.
# Override with ALLOW_DASH_SHRINK=1 only for a deliberate, understood shrink.
SHRINK_GUARD_RATIO = 0.9


def _pg_type(dtype) -> str:
    # display mirror: everything is text. Inferring numeric types from a CSV
    # snapshot breaks the day a column's first real value is a string (the
    # score column was created double-precision while scores were all empty,
    # then "4-6 4-6 2-6" arrived and the whole sync failed). The dashboard
    # parses numbers client-side.
    return "text"


def _sync_table(cur, table: str, df: pd.DataFrame) -> None:
    cur.execute("SELECT to_regclass(%s)", (f'public."{table}"',))
    if cur.fetchone()[0] is not None:
        cur.execute(f'SELECT count(*) FROM "{table}"')
        existing_n = cur.fetchone()[0]
        if (len(df) < existing_n * SHRINK_GUARD_RATIO
                and os.environ.get("ALLOW_DASH_SHRINK") != "1"):
            print(f"   🚨 SHRINK GUARD: refusing to sync {table} "
                  f"({existing_n} rows in Supabase, only {len(df)} locally) — "
                  f"local file may be damaged; backup preserved. "
                  f"Set ALLOW_DASH_SHRINK=1 to override deliberately.")
            return
    cols = [c.strip().lower().replace(" ", "_") for c in df.columns]
    col_defs = ", ".join(
        f'"{c}" {_pg_type(df[orig].dtype)}' for c, orig in zip(cols, df.columns)
    )
    cur.execute(f'CREATE TABLE IF NOT EXISTS "{table}" ({col_defs})')
    # schema drift (new CSV columns) → add as text; never silently drop data
    cur.execute(
        "SELECT column_name FROM information_schema.columns WHERE table_name = %s",
        (table,),
    )
    existing = {r[0] for r in cur.fetchall()}
    for c, orig in zip(cols, df.columns):
        if c not in existing:
            cur.execute(f'ALTER TABLE "{table}" ADD COLUMN "{c}" {_pg_type(df[orig].dtype)}')
    cur.execute(f'TRUNCATE "{table}"')
    buf = io.StringIO()
    df.to_csv(buf, index=False, header=False)
    buf.seek(0)
    col_list = ", ".join(f'"{c}"' for c in cols)
    with cur.copy(
        f'COPY "{table}" ({col_list}) FROM STDIN WITH (FORMAT csv, NULL \'\')'
    ) as copy:
        while chunk := buf.read(65536):
            copy.write(chunk)


def _enable_public_read(cur, table: str) -> None:
    cur.execute(f'ALTER TABLE "{table}" ENABLE ROW LEVEL SECURITY')
    cur.execute(
        "SELECT 1 FROM pg_policies WHERE tablename = %s AND policyname = 'dash_read'",
        (table,),
    )
    if not cur.fetchone():
        cur.execute(f'CREATE POLICY dash_read ON "{table}" FOR SELECT USING (true)')
    cur.execute(f'GRANT SELECT ON "{table}" TO anon')


def sync_dashboard_tables(verbose: bool = True) -> dict:
    """Push all dashboard tables. Returns {table: row_count}."""
    counts = {}
    with connect() as conn:
        with conn.cursor() as cur:
            for table, path, cols in SYNC_SPECS:
                if not os.path.exists(path):
                    if verbose:
                        print(f"   ⚠️ dashboard sync: {path} missing — skipped")
                    continue
                df = pd.read_csv(path, low_memory=False)
                if cols:
                    df = df[[c for c in cols if c in df.columns]]
                _sync_table(cur, table, df)
                _enable_public_read(cur, table)
                counts[table] = len(df)
        conn.commit()
    if verbose:
        summary = ", ".join(f"{t}={n}" for t, n in counts.items())
        print(f"   📊 Dashboard sync → Supabase: {summary}")
    return counts


if __name__ == "__main__":
    sync_dashboard_tables()
