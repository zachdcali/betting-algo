from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
import sys

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_DIR = REPO_ROOT / "production"
if str(PRODUCTION_DIR) not in sys.path:
    sys.path.insert(0, str(PRODUCTION_DIR))


def test_interrupted_finalizer_closes_only_latest_running_pipeline(monkeypatch, tmp_path):
    import audit_logger

    run_path = tmp_path / "run_history.csv"
    monkeypatch.setattr(audit_logger, "RUN_HISTORY_LOG_PATH", run_path)
    pd.DataFrame([
        {
            "run_id": "run_20260721T100000Z",
            "run_kind": "prediction_pipeline",
            "started_at": "2026-07-21T10:00:00+00:00",
            "status": "running",
        },
        {
            "run_id": "run_20260721T110000Z",
            "run_kind": "prediction_pipeline",
            "started_at": "2026-07-21T11:00:00+00:00",
            "status": "running",
            "feature_rows_ok": 19,
        },
        {
            "run_id": "settle_20260721T120000Z",
            "run_kind": "auto_settle",
            "started_at": "2026-07-21T12:00:00+00:00",
            "status": "running",
        },
    ]).to_csv(run_path, index=False)

    finalized = audit_logger.finalize_latest_running_run(
        status="cancelled", error_message="external deadline"
    )

    assert finalized == "run_20260721T110000Z"
    rows = pd.read_csv(run_path, keep_default_na=False).set_index("run_id")
    assert rows.loc[finalized, "status"] == "cancelled"
    assert rows.loc[finalized, "completed_at"]
    assert rows.loc[finalized, "error_message"] == "external deadline"
    assert str(rows.loc[finalized, "feature_rows_ok"]) in {"19", "19.0"}
    assert rows.loc["run_20260721T100000Z", "status"] == "running"
    assert rows.loc["settle_20260721T120000Z", "status"] == "running"


def test_interrupted_finalizer_does_not_rewrite_terminal_run(monkeypatch, tmp_path):
    import audit_logger

    run_path = tmp_path / "run_history.csv"
    monkeypatch.setattr(audit_logger, "RUN_HISTORY_LOG_PATH", run_path)
    pd.DataFrame([{
        "run_id": "run_20260721T110000Z",
        "run_kind": "prediction_pipeline",
        "started_at": "2026-07-21T11:00:00+00:00",
        "completed_at": "2026-07-21T11:30:00+00:00",
        "status": "success",
    }]).to_csv(run_path, index=False)

    assert audit_logger.finalize_latest_running_run() == ""
    row = pd.read_csv(run_path, keep_default_na=False).iloc[0]
    assert row["status"] == "success"
    assert row["completed_at"] == "2026-07-21T11:30:00+00:00"


def test_live_canonical_reconcile_never_starts_optional_stats_scrape(monkeypatch):
    import main

    calls = []

    class FakeConnection:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def transaction(self):
            return nullcontext()

    fake_store = SimpleNamespace(
        connect=lambda: FakeConnection(),
        ingest_event_results=lambda *_args, **_kwargs: {"inserted": 2},
    )
    fake_reconcile = SimpleNamespace(
        run=lambda **kwargs: calls.append(kwargs) or {}
    )
    monkeypatch.setitem(sys.modules, "canonical_store", fake_store)
    monkeypatch.setitem(sys.modules, "reconcile_store", fake_reconcile)

    orchestrator = main.LiveBettingOrchestrator.__new__(
        main.LiveBettingOrchestrator
    )
    orchestrator.run_metrics = {
        "canonical_ingest_status": "not_started",
        "canonical_ingest_rows": 0,
        "canonical_ingest_error": "",
        "reconcile_status": "not_started",
        "reconcile_error": "",
    }
    url = "https://www.atptour.com/en/scores/current/example/1/results"
    orchestrator._session_cache = {
        "atp_event_results": {
            url: pd.DataFrame([{"p1": "A", "p2": "B", "winner": 1}])
        },
        "atp_event_meta": {
            url: {
                "event": "Example",
                "start_date": "2026-07-20",
                "surface": "Hard",
                "level": "C",
                "id": "1",
            }
        },
    }

    orchestrator._ingest_store_events()

    assert len(calls) == 1
    assert calls[0]["since_days"] == 75
    assert "stats_urls" not in calls[0]
    assert "stats_cap" not in calls[0]
    assert orchestrator.run_metrics["canonical_ingest_status"] == "success"
    assert orchestrator.run_metrics["canonical_ingest_rows"] == 2
    assert orchestrator.run_metrics["reconcile_status"] == "success"


def test_dashboard_sync_creates_generation_read_index():
    import dashboard_sync

    statements = []

    class Cursor:
        def execute(self, statement, *_args):
            statements.append(statement)

    frame = pd.DataFrame(columns=["sync_id", "logged_at", "other"])
    dashboard_sync._ensure_query_index(Cursor(), "dash_shadow", frame)

    assert statements == [
        'CREATE INDEX IF NOT EXISTS "dash_shadow_generation_read_v2_idx" '
        'ON "dash_shadow" ("sync_id", "logged_at" DESC NULLS LAST)'
    ]
