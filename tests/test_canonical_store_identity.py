import sys
from pathlib import Path

import pandas as pd


PRODUCTION_DIR = Path(__file__).resolve().parents[1] / "production"
if str(PRODUCTION_DIR) not in sys.path:
    sys.path.insert(0, str(PRODUCTION_DIR))

import canonical_store  # noqa: E402


class _Cursor:
    def __init__(self):
        self.rowcount = 0
        self._fetchone = None
        self.player_inserts = 0
        self.match_inserts = 0
        self.run_note = ""

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def execute(self, sql, params=()):
        compact = " ".join(str(sql).split())
        self.rowcount = 0
        if compact.startswith("INSERT INTO ingest_runs"):
            self._fetchone = (77,)
        elif compact.startswith("INSERT INTO players"):
            self.player_inserts += 1
            self._fetchone = (9_100_000 + self.player_inserts,)
        elif compact.startswith("INSERT INTO matches"):
            self.match_inserts += 1
            self.rowcount = 1
        elif compact.startswith("UPDATE ingest_runs"):
            self.run_note = params[2]
            self.rowcount = 1
        else:  # all identity lookups are monkeypatched in these focused tests
            raise AssertionError(f"unexpected SQL: {compact}")

    def fetchone(self):
        return self._fetchone


class _Connection:
    def __init__(self):
        self.cur = _Cursor()

    def cursor(self):
        return self.cur


def _one_match(p1: str) -> pd.DataFrame:
    return pd.DataFrame([{
        "p1": p1,
        "p2": "Known Opponent",
        "winner": 1,
        "score": "6-4 6-3",
        "round": "R32",
        "completed": True,
    }])


def test_itf_ambiguous_identity_is_skipped_without_creating(monkeypatch):
    conn = _Connection()
    monkeypatch.setattr(
        canonical_store,
        "_resolve_player_id",
        lambda _cur, name: 42 if name == "Known Opponent" else None,
    )
    monkeypatch.setattr(
        canonical_store,
        "_exact_or_normalized_player_candidates",
        lambda _cur, name: [101, 202] if name == "Ambiguous Junior" else [],
    )

    result = canonical_store.ingest_itf_results(
        conn, _one_match("Ambiguous Junior"), "M15 Test", "2026-07-13", "Clay", "15"
    )

    assert result == {"inserted": 0, "skipped": 1, "created": 0, "ambiguous": 1}
    assert conn.cur.player_inserts == 0
    assert conn.cur.match_inserts == 0
    assert "Ambiguous Junior[101|202]" in conn.cur.run_note


def test_itf_zero_candidate_identity_can_be_created(monkeypatch):
    conn = _Connection()
    monkeypatch.setattr(
        canonical_store,
        "_resolve_player_id",
        lambda _cur, name: 42 if name == "Known Opponent" else None,
    )
    monkeypatch.setattr(
        canonical_store,
        "_exact_or_normalized_player_candidates",
        lambda _cur, _name: [],
    )

    result = canonical_store.ingest_itf_results(
        conn, _one_match("Brand New Junior"), "M15 Test", "2026-07-13", "Clay", "15"
    )

    assert result == {"inserted": 1, "skipped": 0, "created": 1, "ambiguous": 0}
    assert conn.cur.player_inserts == 1
    assert conn.cur.match_inserts == 1
    assert conn.cur.run_note == ""
