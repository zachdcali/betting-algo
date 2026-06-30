"""SQLite store for the production logs.

ADDITIVE / non-destructive: this builds a SQLite database *from* the existing
CSV logs. The live pipeline still writes the CSVs; this importer mirrors them
into a queryable DB so the ledger/dashboard can run off SQL and so a later step
can switch the write path over without surprises.

Each table is rebuilt from its source CSV with the natural key as PRIMARY KEY,
loaded via INSERT OR REPLACE so duplicate operational rows collapse to the last
write (matching the "operational latest" semantics of prediction_log.csv).
"""
from __future__ import annotations

import os
import sqlite3

import pandas as pd

# table name -> (csv path relative to prod_dir, primary-key column)
TABLE_REGISTRY = {
    "predictions": ("prediction_log.csv", "match_uid"),
    "prediction_snapshots": ("prediction_snapshots.csv", "prediction_uid"),
    "odds_snapshots": ("odds_history.csv", "odds_snapshot_uid"),
    "shadow_predictions": ("logs/performance_v1_shadow_predictions.csv", "shadow_prediction_uid"),
    "runs": ("logs/audit/run_history.csv", "run_id"),
    "settlement_events": ("logs/audit/settlement_audit.csv", "settlement_event_id"),
    "skip_events": ("logs/audit/skipped_live_matches.csv", "skip_event_id"),
}


def connect(db_path: str) -> sqlite3.Connection:
    return sqlite3.connect(db_path)


def _sqlite_type(dtype) -> str:
    kind = getattr(dtype, "kind", "O")
    if kind in ("i", "u", "b"):  # bool stored as 0/1 INTEGER, not TEXT "True"/"False"
        return "INTEGER"
    if kind == "f":
        return "REAL"
    return "TEXT"


def _load_table(conn: sqlite3.Connection, table: str, df: pd.DataFrame, key: str) -> None:
    cols = list(df.columns)
    has_key = key in cols

    def coldef(c: str) -> str:
        t = _sqlite_type(df[c].dtype)
        if has_key and c == key:
            return f'"{c}" {t} PRIMARY KEY'
        return f'"{c}" {t}'

    conn.execute(f'DROP TABLE IF EXISTS "{table}"')
    conn.execute(f'CREATE TABLE "{table}" ({", ".join(coldef(c) for c in cols)})')

    collist = ", ".join(f'"{c}"' for c in cols)
    placeholders = ", ".join("?" for _ in cols)
    verb = "INSERT OR REPLACE" if has_key else "INSERT"
    rows = [
        tuple(None if pd.isna(v) else v for v in row)
        for row in df.itertuples(index=False, name=None)
    ]
    conn.executemany(f'{verb} INTO "{table}" ({collist}) VALUES ({placeholders})', rows)
    conn.commit()


def build_database(prod_dir: str, db_path: str) -> dict:
    """Rebuild every registered table from its source CSV. Returns {table: row_count}."""
    conn = connect(db_path)
    summary = {}
    try:
        for table, (rel_path, key) in TABLE_REGISTRY.items():
            path = os.path.join(prod_dir, rel_path)
            if not os.path.exists(path):
                summary[table] = 0
                continue
            df = pd.read_csv(path, low_memory=False)
            _load_table(conn, table, df, key)
            summary[table] = int(conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0])
    finally:
        conn.close()
    return summary


def read_table(db_path: str, table: str) -> pd.DataFrame:
    conn = connect(db_path)
    try:
        return pd.read_sql(f'SELECT * FROM "{table}"', conn)
    finally:
        conn.close()


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Build the SQLite logs DB from production CSVs.")
    ap.add_argument("--prod-dir", default=".")
    ap.add_argument("--db", default="logs/betting.db")
    args = ap.parse_args()
    result = build_database(args.prod_dir, args.db)
    for t, n in result.items():
        print(f"  {t:22s} {n:6d} rows")
    print(f"Wrote {args.db}")
