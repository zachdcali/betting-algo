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
import os
import re
import time
import unicodedata
from hashlib import sha256
from html import unescape
from html.parser import HTMLParser
from typing import Optional
from urllib.parse import urljoin, urlparse

import pandas as pd

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

BASE = "https://www.itftennis.com"
DEFAULT_PROFILE_SESSION_BATCH_SIZE = 4
DEFAULT_PROFILE_FETCH_ATTEMPTS = 2
DEFAULT_PROFILE_HTML_FALLBACK_ATTEMPTS = 1


class ItfClient:
    """Persistent browser session that can call itftennis JSON APIs."""

    def __init__(self):
        self._pw = None
        self._browser = None
        self._page = None

    def _ensure(self):
        if self._page is not None:
            return
        try:
            from .browser_session import new_page
        except ImportError:  # pragma: no cover - legacy production/ on sys.path
            from browser_session import new_page
        self._page = new_page()
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

    def fetch_text(self, path: str) -> str:
        """Fetch an official ITF HTML page through the accepted browser session.

        Player profiles are server-rendered, so an in-page fetch is both faster
        and less failure-prone than navigating the shared page once per player.
        """
        self._ensure()
        text = self._page.evaluate(
            "async (u) => { const r = await fetch(u, {headers:{accept:'text/html'}});"
            " return r.status + '|' + await r.text(); }",
            path,
        )
        status, _, body = text.partition("|")
        if status != "200":
            raise RuntimeError(f"itftennis profile {status} for {path[:80]}")
        return body

    def close(self):
        try:
            if self._page is not None:
                self._page.close()   # page only — the browser is shared
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


def _team_player(team: dict) -> dict:
    """Return the first real singles player from an ITF team payload."""
    players = [p for p in (team.get("players") or []) if p]
    return players[0] if players else {}


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

    Columns include date/round, player names, stable ITF IDs, official profile
    links/nationalities, winner (1|2|None), score, completion, classification.
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
            players = [_team_player(t) for t in teams]
            rows.append({
                "date": play_date,
                "round": _round_code(m),
                "p1": names[0], "p2": names[1],
                "p1_id": players[0].get("playerId"),
                "p2_id": players[1].get("playerId"),
                "p1_profile_url": players[0].get("profileLink") or "",
                "p2_profile_url": players[1].get("profileLink") or "",
                "p1_nationality": players[0].get("nationality") or "",
                "p2_nationality": players[1].get("nationality") or "",
                "winner": winner, "score": score,
                "completed": completed,
                "classification": m.get("eventClassificationDesc", ""),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Player profiles (official identity + handedness; ITF does not publish height)
# ---------------------------------------------------------------------------

def _identity_key(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", str(value or "").casefold())
    return "".join(
        char for char in normalized
        if not unicodedata.combining(char) and char.isascii() and char.isalnum()
    )


def _player_id_string(value) -> str:
    """Normalize JSON/pandas numeric IDs without ever accepting a guess."""
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return ""
    text = str(value).strip()
    if re.fullmatch(r"\d+", text):
        return text
    if re.fullmatch(r"\d+\.0", text):
        return text[:-2]
    return ""


class _ItfProfileParser(HTMLParser):
    """Extract the three identity/profile fields without optional packages."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.name = ""
        self.canonical_url = ""
        self.hand_text: list[str] = []
        self._inside_hand = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, Optional[str]]]) -> None:
        values = {str(key).casefold(): str(value or "") for key, value in attrs}
        lowered_tag = tag.casefold()
        if lowered_tag == "meta" and values.get("name", "").casefold() == "keywords":
            self.name = values.get("content", "").strip()
        elif lowered_tag == "link" and "canonical" in values.get("rel", "").casefold():
            self.canonical_url = values.get("href", "").strip()
        if values.get("id", "").casefold() == "ga__player-plays-hand":
            self._inside_hand = True

    def handle_endtag(self, tag: str) -> None:
        if self._inside_hand and tag.casefold() == "span":
            self._inside_hand = False

    def handle_data(self, data: str) -> None:
        if self._inside_hand:
            self.hand_text.append(data)


def profile_refs_for_names(
    event_frames: dict[str, pd.DataFrame], player_names: list[str],
) -> dict[str, dict]:
    """Resolve exact slate names to one stable ITF player ID/profile URL.

    The order-of-play payload is the identity bridge: it binds the displayed
    full name, numeric ``playerId``, nationality, and official ``profileLink``
    in one response. Conflicting IDs or URLs fail closed instead of selecting
    the first match.
    """
    names_by_key: dict[str, list[str]] = {}
    for name in player_names:
        key = _identity_key(name)
        if key:
            names_by_key.setdefault(key, []).append(str(name))

    candidates: dict[str, set[tuple[str, str, str, str]]] = {
        key: set() for key, names in names_by_key.items() if len(set(names)) == 1
    }
    for frame in (event_frames or {}).values():
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            continue
        for _, row in frame.iterrows():
            for side in (1, 2):
                observed_name = str(row.get(f"p{side}") or "").strip()
                key = _identity_key(observed_name)
                if key not in candidates:
                    continue
                player_id = _player_id_string(row.get(f"p{side}_id"))
                profile_url = str(row.get(f"p{side}_profile_url") or "").strip()
                nationality = str(row.get(f"p{side}_nationality") or "").strip().upper()
                if not player_id or not profile_url:
                    continue
                candidates[key].add((player_id, profile_url, observed_name, nationality))

    resolved: dict[str, dict] = {}
    for key, refs in candidates.items():
        ids = {ref[0] for ref in refs}
        urls = {ref[1] for ref in refs}
        if len(ids) != 1 or len(urls) != 1:
            continue
        display_name = names_by_key[key][0]
        player_id, profile_url, observed_name, nationality = sorted(refs)[0]
        resolved[display_name] = {
            "itf_player_id": player_id,
            "profile_url": profile_url,
            "observed_name": observed_name,
            "nationality": nationality,
        }
    return resolved


def parse_player_profile(
    html: str, *, expected_name: str, expected_player_id: str | int,
) -> dict:
    """Parse one ITF profile and prove both its name and numeric identity."""
    body = str(html or "")
    content_hash = sha256(body.encode("utf-8")).hexdigest() if body else ""
    parser = _ItfProfileParser()
    parser.feed(body)
    observed_name = unescape(parser.name).strip()
    canonical_url = unescape(parser.canonical_url).strip()
    path_parts = [part for part in urlparse(canonical_url).path.split("/") if part]
    observed_id = ""
    try:
        player_index = [part.casefold() for part in path_parts].index("players")
        observed_id = path_parts[player_index + 2]
    except (ValueError, IndexError):
        pass

    identity_matches = (
        bool(observed_name)
        and _identity_key(observed_name) == _identity_key(expected_name)
        and observed_id == str(expected_player_id)
    )
    # Imperva can return a non-profile HTML body with HTTP 200.  That is a
    # transient source failure, not evidence that the ITF identity binding is
    # wrong.  Only a fully rendered canonical player identity can prove a real
    # conflict.
    identity_conflict = (
        bool(observed_name)
        and bool(observed_id)
        and (
            _identity_key(observed_name) != _identity_key(expected_name)
            or observed_id != str(expected_player_id)
        )
    )
    hand_match = re.search(
        r"\b(Right|Left)\s+Handed\b",
        " ".join(parser.hand_text),
        flags=re.IGNORECASE,
    )
    hand = None
    if identity_matches and hand_match:
        hand = "R" if hand_match.group(1).casefold() == "right" else "L"
    return {
        "status": (
            "resolved" if hand is not None
            else "not_found" if identity_matches
            else "identity_mismatch" if identity_conflict
            else "fetch_error"
        ),
        "name": observed_name,
        "itf_player_id": observed_id,
        "profile_url": canonical_url,
        "hand": hand,
        "height_cm": None,
        "source_content_sha256": content_hash,
    }


def parse_player_details(
    payload: dict, *, expected_name: str, expected_player_id: str | int,
) -> dict:
    """Parse the official structured player-details identity and hand."""
    data = payload if isinstance(payload, dict) else {}
    observed_name = str(data.get("FullName") or "").strip()
    observed_id = _player_id_string(data.get("playerId"))
    expected_id = _player_id_string(expected_player_id)
    identity_matches = (
        bool(observed_name)
        and _identity_key(observed_name) == _identity_key(expected_name)
        and observed_id == expected_id
    )
    identity_conflict = (
        bool(observed_name)
        and bool(observed_id)
        and (
            _identity_key(observed_name) != _identity_key(expected_name)
            or observed_id != expected_id
        )
    )
    play_hand = str(data.get("playHand") or "").strip()
    hand_match = re.search(r"\b(Right|Left)\s+Handed\b", play_hand, re.IGNORECASE)
    hand = None
    if identity_matches and hand_match:
        hand = "R" if hand_match.group(1).casefold() == "right" else "L"
    canonical_payload = json.dumps(
        data, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    )
    return {
        "status": (
            "resolved" if hand is not None
            else "not_found" if identity_matches
            else "identity_mismatch" if identity_conflict
            else "fetch_error"
        ),
        "name": observed_name,
        "itf_player_id": observed_id,
        "profile_url": str(data.get("playerProfileLink") or "").strip(),
        "nationality": str(data.get("playerNationalityCode") or "").strip().upper(),
        "hand": hand,
        "height_cm": None,
        "source_content_sha256": (
            sha256(canonical_payload.encode("utf-8")).hexdigest() if data else ""
        ),
        "source_kind": "itf_player_details_api",
    }


def _profile_circuit_code(profile_url: str) -> str:
    supported = {"MT", "WT", "JT", "WCT", "VT", "BT"}
    for part in urlparse(str(profile_url or "")).path.split("/"):
        candidate = part.strip().upper()
        if candidate in supported:
            return candidate
    return "MT"


def get_player_details(client: ItfClient, refs_by_name: dict[str, dict]) -> dict[str, dict]:
    """Fetch official structured identity/hand data before touching HTML."""
    results: dict[str, dict] = {}
    for display_name, ref in refs_by_name.items():
        player_id = str(ref.get("itf_player_id") or "").strip()
        circuit_code = _profile_circuit_code(str(ref.get("profile_url") or ""))
        endpoint = (
            "/tennis/api/PlayerApi/GetHeadToHeadPlayerDetails"
            f"?circuitCode={circuit_code}&playerId={player_id}"
        )
        source_uri = urljoin(BASE, endpoint)
        try:
            payload = client.fetch_json(endpoint)
            parsed = parse_player_details(
                payload,
                expected_name=display_name,
                expected_player_id=player_id,
            )
        except Exception as exc:
            parsed = {
                "status": "fetch_error",
                "name": "",
                "itf_player_id": player_id,
                "profile_url": str(ref.get("profile_url") or ""),
                "hand": None,
                "height_cm": None,
                "source_content_sha256": "",
                "source_kind": "itf_player_details_api",
                "error": str(exc),
            }
        parsed["source_uri"] = source_uri
        parsed["nationality"] = (
            str(parsed.get("nationality") or ref.get("nationality") or "")
            .strip()
            .upper()
        )
        results[display_name] = parsed
    return results


def get_player_profiles(client: ItfClient, refs_by_name: dict[str, dict]) -> dict[str, dict]:
    """Fetch an exact-ID-bound set of official ITF player profiles."""
    results: dict[str, dict] = {}
    for display_name, ref in refs_by_name.items():
        profile_url = str(ref.get("profile_url") or "").strip()
        player_id = str(ref.get("itf_player_id") or "").strip()
        source_uri = urljoin(BASE, profile_url)
        try:
            html = client.fetch_text(profile_url)
            parsed = parse_player_profile(
                html,
                expected_name=display_name,
                expected_player_id=player_id,
            )
        except Exception as exc:
            parsed = {
                "status": "fetch_error",
                "name": "",
                "itf_player_id": player_id,
                "profile_url": source_uri,
                "hand": None,
                "height_cm": None,
                "source_content_sha256": "",
                "error": str(exc),
            }
        parsed["source_uri"] = source_uri
        parsed["nationality"] = str(ref.get("nationality") or "").strip().upper()
        parsed["source_kind"] = "itf_player_profile_html"
        results[display_name] = parsed
    return results


def _positive_int_env(name: str, default: int) -> int:
    try:
        return max(1, int(os.environ.get(name, str(default))))
    except (TypeError, ValueError, OverflowError):
        return default


def get_player_profiles_resilient(
    refs_by_name: dict[str, dict],
    *,
    batch_size: Optional[int] = None,
    max_attempts: Optional[int] = None,
    client_factory=None,
) -> dict[str, dict]:
    """Fetch profiles through bounded, clean ITF browser sessions.

    Prefer ITF's structured player-details endpoint, which returns the stable
    player ID, full name, nationality, and ``playHand`` in one response. Rotate
    to an isolated page/cookie jar after a small batch and retry only transient
    ``fetch_error`` rows. After structured attempts are exhausted, retain one
    bounded HTML fallback pass. A proven identity mismatch or an official
    ``Unknown`` hand is never retried or softened.
    """
    if not refs_by_name:
        return {}
    resolved_batch_size = batch_size or _positive_int_env(
        "ITF_PROFILE_SESSION_BATCH_SIZE", DEFAULT_PROFILE_SESSION_BATCH_SIZE,
    )
    resolved_attempts = max_attempts or _positive_int_env(
        "ITF_PROFILE_FETCH_ATTEMPTS", DEFAULT_PROFILE_FETCH_ATTEMPTS,
    )
    html_attempts = _positive_int_env(
        "ITF_PROFILE_HTML_FALLBACK_ATTEMPTS",
        DEFAULT_PROFILE_HTML_FALLBACK_ATTEMPTS,
    )
    resolved_batch_size = max(1, int(resolved_batch_size))
    resolved_attempts = max(1, int(resolved_attempts))
    client_factory = client_factory or ItfClient
    ordered_names = list(refs_by_name)
    results: dict[str, dict] = {}
    attempt_counts = {name: 0 for name in ordered_names}

    fetch_plan = (
        (get_player_details, resolved_attempts),
        (get_player_profiles, html_attempts),
    )
    for fetcher, source_attempts in fetch_plan:
        for _attempt in range(source_attempts):
            pending_names = [
                name for name in ordered_names
                if name not in results or results[name].get("status") == "fetch_error"
            ]
            if not pending_names:
                break
            for start in range(0, len(pending_names), resolved_batch_size):
                names = pending_names[start:start + resolved_batch_size]
                refs = {name: refs_by_name[name] for name in names}
                client = client_factory()
                try:
                    fetched = fetcher(client, refs)
                except Exception as exc:  # one session must not abort the slate
                    fetched = {
                        name: {
                            "status": "fetch_error",
                            "name": "",
                            "itf_player_id": str(
                                refs[name].get("itf_player_id") or ""
                            ),
                            "profile_url": str(refs[name].get("profile_url") or ""),
                            "source_uri": urljoin(
                                BASE, str(refs[name].get("profile_url") or ""),
                            ),
                            "hand": None,
                            "height_cm": None,
                            "source_content_sha256": "",
                            "error": str(exc),
                        }
                        for name in names
                    }
                finally:
                    try:
                        client.close()
                    except Exception:
                        pass
                for name in names:
                    attempt_counts[name] += 1
                    result = dict(fetched.get(name) or {})
                    if not result:
                        result = {
                            "status": "fetch_error",
                            "hand": None,
                            "height_cm": None,
                            "source_content_sha256": "",
                            "error": "profile batch omitted requested player",
                        }
                    result["attempt_count"] = attempt_counts[name]
                    results[name] = result

    return results


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
