from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "production"))

from odds.fetch_bovada import american_to_decimal  # noqa: E402


def test_american_odds_conversion_does_not_invent_even_money():
    assert american_to_decimal("EVEN") == 2.0
    assert american_to_decimal("+150") == 2.5
    assert american_to_decimal("-200") == 1.5
    assert american_to_decimal(None) is None
    assert american_to_decimal("") is None
    assert american_to_decimal("not a price") is None
    assert american_to_decimal("0") is None
