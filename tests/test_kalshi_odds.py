import json
from pathlib import Path
import sys

import pandas as pd
import pytest

PRODUCTION = Path(__file__).resolve().parents[1] / "production"
sys.path.insert(0, str(PRODUCTION))

from odds import fetch_kalshi as kalshi  # noqa: E402


class FakeResponse:
    def __init__(self, payload, url="https://example.test/markets"):
        self._payload = payload
        self.url = url

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def get(self, url, *, params, timeout):
        self.calls.append((url, dict(params), timeout))
        return self.responses.pop(0)


def market(ticker, event, player, ask, *, bid="0.40", series="KXATPMATCH"):
    return {
        "ticker": ticker,
        "event_ticker": event,
        "yes_sub_title": player,
        "status": "active",
        "yes_bid_dollars": bid,
        "yes_ask_dollars": ask,
        "last_price_dollars": "0.5000",
        "yes_bid_size_fp": "12.34",
        "yes_ask_size_fp": "56.78",
        "volume_fp": "90.12",
        "open_interest_fp": "34.56",
        "open_time": "2026-07-16T10:00:00Z",
        "close_time": "2026-07-17T20:00:00Z",
        "expected_expiration_time": "2026-07-17T18:00:00Z",
        "_requested_series": series,
        "_source_uri": "https://example.test/source",
    }


def test_fetch_paginates_and_preserves_suffixed_market_fields():
    first = market("MKT-A", "KXATPMATCH-26JUL17AB", "Player A", "0.6100")
    second = market("MKT-B", "KXATPMATCH-26JUL17AB", "Player B", "0.4100")
    session = FakeSession([
        FakeResponse({"markets": [first], "cursor": "next"}),
        FakeResponse({"markets": [second], "cursor": ""}),
    ])

    rows, polled_at = kalshi.fetch_open_tennis_markets(
        session=session,
        series_tickers=("KXATPMATCH",),
        polled_at="2026-07-17T10:00:00+00:00",
    )

    assert polled_at == "2026-07-17T10:00:00+00:00"
    assert [row["ticker"] for row in rows] == ["MKT-A", "MKT-B"]
    assert rows[0]["yes_ask_dollars"] == "0.6100"
    assert rows[0]["volume_fp"] == "90.12"
    assert "yes_ask" not in rows[0]
    assert session.calls[0][1] == {
        "series_ticker": "KXATPMATCH", "status": "open", "limit": 200,
    }
    assert session.calls[1][1]["cursor"] == "next"


def test_reviewed_alias_pair_date_binding_orients_both_raw_markets():
    registry = kalshi.AliasRegistry(
        "aliases@test", {"daniel merida": "daniel merida aguilar"},
    )
    markets = [
        market(
            "MKT-MER", "KXATPMATCH-26JUL17BURMER", "Daniel Merida", "0.4200",
        ),
        market(
            "MKT-BUR", "KXATPMATCH-26JUL17BURMER",
            "Roman Andres Burruchaga", "0.5900",
        ),
    ]
    board = pd.DataFrame([{
        "match_uid": "match_exact",
        "player1_raw": "Roman Andres Burruchaga",
        "player2_raw": "Daniel Merida Aguilar",
        "meta_match_date": "2026-07-17",
        "event": "ATP - Umag (2)",
    }])

    observations = kalshi.build_kalshi_observations(
        markets,
        board,
        run_id="run_1",
        polled_at="2026-07-17T10:00:00+00:00",
        alias_registry=registry,
    ).set_index("market_ticker")

    assert observations["match_status"].eq("matched").all()
    assert observations["match_method"].eq("reviewed_alias_pair_date").all()
    assert observations.loc["MKT-BUR", "board_side"] == "p1"
    assert observations.loc["MKT-MER", "board_side"] == "p2"
    assert observations.loc["MKT-MER", "yes_ask_dollars"] == "0.4200"
    assert observations.loc["MKT-MER", "alias_applied"] == (
        "daniel merida=>daniel merida aguilar"
    )
    assert observations["match_uid"].eq("match_exact").all()
    assert observations["source_payload_sha256"].str.fullmatch(r"[0-9a-f]{64}").all()


def test_same_pair_date_with_multiple_board_uids_fails_ambiguous():
    markets = [
        market("MKT-A", "KXATPMATCH-26JUL17AB", "Player A", "0.5000"),
        market("MKT-B", "KXATPMATCH-26JUL17AB", "Player B", "0.5100"),
    ]
    board = pd.DataFrame([
        {"match_uid": "m1", "player1_raw": "Player A", "player2_raw": "Player B",
         "meta_match_date": "2026-07-17", "event": "ATP - Event One"},
        {"match_uid": "m2", "player1_raw": "Player A", "player2_raw": "Player B",
         "meta_match_date": "2026-07-17", "event": "ATP - Event Two"},
    ])

    observations = kalshi.build_kalshi_observations(
        markets,
        board,
        run_id="run_1",
        polled_at="2026-07-17T10:00:00+00:00",
        alias_registry=kalshi.AliasRegistry("aliases@test", {}),
    )

    assert observations["match_status"].eq("ambiguous_board_match").all()
    assert observations["match_uid"].eq("").all()
    assert observations["board_side"].eq("").all()


def test_alias_registry_rejects_conflicting_normalized_sources(tmp_path):
    path = tmp_path / "aliases.json"
    path.write_text(json.dumps({
        "schema_version": "test",
        "aliases": [
            {"kalshi_name": "José Name", "board_name": "Target One"},
            {"kalshi_name": "Jose Name", "board_name": "Target Two"},
        ],
    }))
    with pytest.raises(ValueError, match="conflicting Kalshi alias"):
        kalshi.load_alias_registry(path)


def test_append_is_idempotent_and_rejects_changed_immutable_row(tmp_path):
    observations = pd.DataFrame([{
        column: "" for column in kalshi.KALSHI_OBSERVATION_COLUMNS
    }])
    observations.loc[0, "kalshi_observation_uid"] = "kalshi_1"
    observations.loc[0, "yes_ask_dollars"] = "0.6100"
    path = tmp_path / "kalshi.csv"

    assert kalshi.append_kalshi_observations(observations, path) == 1
    assert kalshi.append_kalshi_observations(observations, path) == 0
    changed = observations.copy()
    changed.loc[0, "yes_ask_dollars"] = "0.6200"
    with pytest.raises(RuntimeError, match="conflicting durable Kalshi observation"):
        kalshi.append_kalshi_observations(changed, path)
    stored = pd.read_csv(path, dtype=str, keep_default_na=False)
    assert stored.loc[0, "yes_ask_dollars"] == "0.6100"
