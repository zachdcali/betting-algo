from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_dashboard_renders():
    app_path = Path(__file__).resolve().parents[1] / "dashboard" / "app.py"
    at = AppTest.from_file(str(app_path))
    at.run(timeout=60)
    assert not at.exception
    assert at.title[0].value == "Tennis Betting Operations Dashboard"
