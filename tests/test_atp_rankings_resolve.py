import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_DIR = REPO_ROOT / "production"
if str(PRODUCTION_DIR) not in sys.path:
    sys.path.insert(0, str(PRODUCTION_DIR))

import pandas as pd
from scraping.atp_rankings_scraper import resolve_rankings


def _df():
    return pd.DataFrame([
        {"rank": 1, "player_name": "J. Sinner", "points": 13450,
         "scraped_at": "2026-06-29T10:00:00"},
    ])


def test_live_success_returns_fresh():
    df, src = resolve_rankings(_fetch=lambda headless=True: _df(), _load=lambda: pd.DataFrame())
    assert src == "fresh"
    assert len(df) == 1


def test_empty_live_falls_back_to_cached():
    df, src = resolve_rankings(_fetch=lambda headless=True: pd.DataFrame(), _load=_df)
    assert src.startswith("cached")
    assert "2026-06-29" in src       # cache date surfaced for an honest message
    assert len(df) == 1


def test_exception_live_falls_back_to_cached():
    def boom(headless=True):
        raise RuntimeError("playwright timeout")
    df, src = resolve_rankings(_fetch=boom, _load=_df)
    assert src.startswith("cached")
    assert len(df) == 1


def test_both_empty_returns_none():
    df, src = resolve_rankings(_fetch=lambda headless=True: pd.DataFrame(), _load=lambda: pd.DataFrame())
    assert src == "none"
    assert df.empty
