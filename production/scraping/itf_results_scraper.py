"""ITF World Tennis Tour data source (itftennis.com JSON APIs).

Opens the ITF tier (M15/M25 futures) for the live pipeline: event discovery,
upcoming-match rounds, and completed results — the pieces that make ITF matches
bettable (rounds + current histories) and settleable without Tennis Abstract.

Access pattern: itftennis.com sits behind Imperva/Incapsula, which starves
plain headless requests. The working method is a persistent Playwright page —
load any itftennis page once (accept the Termly consent), then issue API calls
via in-page ``fetch`` so they carry the Incapsula session cookies.

Confirmed endpoints (2026-07-08):
- ``/tennis/api/TournamentApi/GetCalendar`` — events with tournamentKey,
  surface, dates, venue, category (discovery + Bovada label mapping)
- ``/tennis/api/TournamentApi/GetOrderOfPlayDays?tournamentKey=<key>``
- ``/tennis/api/TournamentApi/GetOrderOfPlay?orderOfPlayDayId=<id>`` — per-day
  matches with roundCode, Main/Qualifying classification, player names/ids,
  per-team ``isWinner`` and set scores, playStatus ("Played and completed")

Granular serve/BP stats are NOT in these payloads — ITF live rows are
score-only; the variants' coverage features (Stat_Matches_Last10) count that
honestly, and historical ITF stats (~97%) remain in the canonical store.
"""
from __future__ import annotations

import json
import time
from typing import Optional

import pandas as pd

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

BASE = "https://www.itftennis.com"


class ItfClient:
    """Persistent browser session that can call itftennis JSON APIs."""

    def __init__(self):
        self._pw = None
        self._browser = None
        self._page = None

    def _ensure(self):
        if self._page is not None:
            return
        from playwright.sync_api import sync_playwright
        self._pw = sync_playwright().start()
        self._browser = self._pw.chromium.launch(headless=True)
        self._page = self._browser.new_page()
        self._page.set_extra_http_headers({"User-Agent": USER_AGENT})
        self._page.goto(f"{BASE}/en/tournament-calendar/", wait_until="domcontentloaded", timeout=60000)
        time.sleep(5)
        try:
            self._page.locator("button:has-text('Accept')").first.click(timeout=3000)
        except Exception:
            pass
        time.sleep(2)

    def fetch_json(self, path: str):
        self._ensure()
        text = self._page.evaluate(
            "async (u) => { const r = await fetch(u, {headers:{accept:'application/json'}});"
            " return r.status + '|' + await r.text(); }",
            path,
        )
        status, _, body = text.partition("|")
        if status != "200":
            raise RuntimeError(f"itftennis API {status} for {path[:80]}")
        return json.loads(body)

    def close(self):
        for closer in (lambda: self._browser.close(), lambda: self._pw.stop()):
            try:
                closer()
            except Exception:
                pass
        self._page = self._browser = self._pw = None


# ---------------------------------------------------------------------------
# Parsers (pure — fixture-testable without a browser)
# ---------------------------------------------------------------------------

def parse_calendar(payload: dict) -> pd.DataFrame:
    """Calendar items → one row per event."""
    rows = []
    for it in payload.get("items", []):
        rows.append({
            "event": str(it.get("tournamentName", "")).strip(),
            "key": str(it.get("tournamentKey", "")).lower(),
            "location": it.get("location", ""),
            "host_nation": it.get("hostNationCode", ""),
            "category": it.get("category", ""),
            "surface": str(it.get("surfaceDesc", "") or "").split()[0].title() if it.get("surfaceDesc") else "",
            "start_date": str(it.get("startDate", ""))[:10],
            "end_date": str(it.get("endDate", ""))[:10],
            "link": it.get("tournamentLink", ""),
        })
    return pd.DataFrame(rows)


def _round_code(match: dict) -> Optional[str]:
    """ITF roundCode + classification → our round codes (Q1/Q2/R32/.../F)."""
    classification = str(match.get("eventClassificationDesc", "") or "")
    group = str(match.get("roundGroupDesc", "") or "").lower()
    code = str(match.get("roundCode", "") or "")
    if "qualifying" in classification.lower():
        for n, q in (("1st", "Q1"), ("2nd", "Q2"), ("3rd", "Q3"), ("final", "Q2")):
            if n in group:
                return q
        return "Q1"
    if code in ("128", "64", "32", "16"):
        return f"R{code}"
    if "quarter" in group:
        return "QF"
    if "semi" in group:
        return "SF"
    if "final" in group:
        return "F"
    return None


def _team_name(team: dict) -> str:
    players = [p for p in (team.get("players") or []) if p]
    if not players:
        return ""
    p = players[0]
    return f"{p.get('givenName','')} {p.get('familyName','')}".strip()


def _score_string(winner_team: dict, loser_team: dict) -> str:
    """Winner-first set scores, TA convention, tiebreak points in parens."""
    sets = []
    for ws, ls in zip(winner_team.get("scores") or [], loser_team.get("scores") or []):
        if not ws or not ls or ws.get("score") is None or ls.get("score") is None:
            continue
        s = f"{ws['score']}-{ls['score']}"
        tb = ws.get("losingScore") if ws.get("losingScore") is not None else ls.get("losingScore")
        if tb is not None:
            s += f"({tb})"
        sets.append(s)
    return " ".join(sets)


def parse_oop_matches(payload: list, play_date: str, singles_only: bool = True) -> pd.DataFrame:
    """One day's order-of-play → match rows (completed and scheduled).

    Columns: date, round, p1, p2, p1_id, p2_id, winner (1|2|None), score,
    completed (bool), classification.
    """
    rows = []
    for court in payload or []:
        for m in court.get("matches", []):
            # MS = main-draw singles, MSQ = qualifying singles (MD/XD = doubles)
            if singles_only and not str(m.get("matchDescription", "")).startswith(("MS", "WS")):
                continue
            teams = m.get("teams") or []
            if len(teams) != 2:
                continue
            names = [_team_name(t) for t in teams]
            if not names[0] or not names[1]:
                continue
            completed = "completed" in str(m.get("playStatusDesc", "") or "").lower()
            winner = None
            score = ""
            if completed:
                w = [bool(t.get("isWinner")) for t in teams]
                if w[0] != w[1]:
                    winner = 1 if w[0] else 2
                    score = _score_string(teams[winner - 1], teams[2 - winner])
            ids = []
            for t in teams:
                players = [p for p in (t.get("players") or []) if p]
                ids.append(players[0].get("playerId") if players else None)
            rows.append({
                "date": play_date,
                "round": _round_code(m),
                "p1": names[0], "p2": names[1],
                "p1_id": ids[0], "p2_id": ids[1],
                "winner": winner, "score": score,
                "completed": completed,
                "classification": m.get("eventClassificationDesc", ""),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# High-level fetchers
# ---------------------------------------------------------------------------

def get_calendar(client: ItfClient, date_from: str, date_to: str, take: int = 60) -> pd.DataFrame:
    payload = client.fetch_json(
        "/tennis/api/TournamentApi/GetCalendar?circuitCode=MT&searchString="
        f"&skip=0&take={take}&dateFrom={date_from}&dateTo={date_to}"
        "&isOrderAscending=true&orderField=startDate"
    )
    return parse_calendar(payload)


def get_event_matches(client: ItfClient, tournament_key: str) -> pd.DataFrame:
    """All order-of-play matches (completed + scheduled) for one event."""
    days = client.fetch_json(
        f"/tennis/api/TournamentApi/GetOrderOfPlayDays?tournamentKey={tournament_key.lower()}"
    )
    frames = []
    for d in days or []:
        day_id = d.get("orderOfPlayDayId")
        play_date = str(d.get("playDate", ""))[:10]
        if not day_id:
            continue
        payload = client.fetch_json(
            f"/tennis/api/TournamentApi/GetOrderOfPlay?orderOfPlayDayId={day_id}"
        )
        frames.append(parse_oop_matches(payload, play_date))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)
