#!/usr/bin/env python3
"""
Tennis Abstract Scraper - JavaScript Variable Parser
Extracts player profile and match data from embedded JavaScript variables
on tennisabstract.com (Jeff Sackmann's site - same data source as training)
"""

import re
import time
import json
import hashlib
import requests
from pathlib import Path
from typing import Optional, Dict, List
from datetime import datetime
import pandas as pd
import ast

# Rate limiting
DEFAULT_DELAY = 4.0  # 4s between requests to avoid 429 errors
MAX_RETRIES = 3

# Cache directories
CACHE_DIR = Path(__file__).parent.parent.parent / "data" / "ta"
PLAYERS_DIR = CACHE_DIR / "players"

# Ensure cache dirs exist
PLAYERS_DIR.mkdir(parents=True, exist_ok=True)


class TennisAbstractScraper:
    """Scraper for Tennis Abstract - parses JavaScript variables from HTML"""

    def __init__(self, rate_limit_delay: float = DEFAULT_DELAY):
        self.rate_limit_delay = rate_limit_delay
        self.last_request_time = 0
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })

    def _rate_limit(self):
        """Enforce rate limiting between requests"""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)
        self.last_request_time = time.time()

    def _fetch_with_retry(self, url: str, retries: int = MAX_RETRIES) -> Optional[str]:
        """Fetch URL with retries and rate limiting"""
        import requests as _requests
        for attempt in range(retries):
            try:
                self._rate_limit()
                response = self.session.get(url, timeout=15)
                response.raise_for_status()
                return response.text
            except _requests.exceptions.HTTPError as e:
                if e.response is not None and e.response.status_code == 429:
                    wait = 30 + (attempt * 15)  # 30s, 45s, 60s
                    print(f"  429 rate limited — waiting {wait}s before retry {attempt + 1}/{retries}...")
                    time.sleep(wait)
                    if attempt == retries - 1:
                        print(f"Failed to fetch {url} after {retries} attempts: {e}")
                        return None
                else:
                    if attempt == retries - 1:
                        print(f"Failed to fetch {url} after {retries} attempts: {e}")
                        return None
                    time.sleep(2 ** attempt)
            except Exception as e:
                if attempt == retries - 1:
                    print(f"Failed to fetch {url} after {retries} attempts: {e}")
                    return None
                time.sleep(2 ** attempt)
        return None

    @staticmethod
    def _extract_js_var(html: str, var_name: str) -> Optional[str]:
        """Extract a JavaScript variable value from HTML"""
        # Match: var varname = value;
        pattern = rf"var {var_name}\s*=\s*(.+?);"
        match = re.search(pattern, html, re.MULTILINE)
        if match:
            return match.group(1).strip()
        return None

    @staticmethod
    def _extract_career_record(html: str) -> Optional[dict]:
        """Extract career record from HTML tablelabel (e.g., 'Matches (197-124) > Time Span: Career')"""
        # Pattern: Matches (wins-losses) > Time Span: Career
        # This appears in <td id="tablelabel"><b>Matches (197-124) > Time Span: Career</b></td>
        pattern = r'Matches\s+\((\d+)-(\d+)\)\s+>\s+Time Span:\s+Career'
        match = re.search(pattern, html)
        if match:
            wins = int(match.group(1))
            losses = int(match.group(2))
            return {
                'career_wins': wins,
                'career_losses': losses,
                'career_total': wins + losses
            }
        return None

    @staticmethod
    def _extract_js_array(html: str, array_name: str) -> Optional[List]:
        """Extract a JavaScript array from HTML and parse to Python list"""
        # Find the start of the array
        start_pattern = rf"var {array_name}\s*=\s*\["
        start_match = re.search(start_pattern, html)
        if not start_match:
            return None

        start_pos = start_match.end() - 1  # Include the opening bracket

        # Find the matching closing bracket and semicolon
        # Track bracket depth to handle nested arrays
        depth = 0
        i = start_pos
        in_string = False
        string_char = None

        while i < len(html):
            char = html[i]

            # Track string boundaries
            if char in ('"', "'") and (i == 0 or html[i-1] != '\\'):
                if not in_string:
                    in_string = True
                    string_char = char
                elif char == string_char:
                    in_string = False
                    string_char = None

            # Track bracket depth (only outside strings)
            if not in_string:
                if char == '[':
                    depth += 1
                elif char == ']':
                    depth -= 1
                    if depth == 0:
                        # Found the closing bracket - extract array
                        end_pos = i + 1
                        array_str = html[start_pos:end_pos]
                        break
            i += 1
        else:
            print(f"Could not find closing bracket for {array_name}")
            return None

        # Convert JavaScript syntax to JSON
        # TA uses single-quoted strings with apostrophes: 'Queen's Club'
        # Replace single quotes only at string boundaries
        # Pattern: after comma/bracket (,' or [') and before comma/bracket (', or '])

        # Replace opening single quotes: ,' → ,"  or [' → ["
        array_str = re.sub(r"([,\[])\s*'", r'\1"', array_str)
        # Replace closing single quotes: ', → ",  or '] → "]
        array_str = re.sub(r"'(\s*[,\]])", r'"\1', array_str)

        # Remove trailing commas (JS allows, JSON doesn't)
        array_str = re.sub(r',\s*]', ']', array_str)

        try:
            return json.loads(array_str)
        except Exception as e:
            print(f"Failed to parse {array_name}: {e}")
            # Save problematic string for debugging
            debug_file = CACHE_DIR / f"debug_{array_name}.txt"
            with open(debug_file, 'w') as f:
                f.write(array_str)
            print(f"  Saved debug output to {debug_file}")
            return None

    def _get_cached_html(
        self,
        slug: str,
        force_refresh: bool = True,
        session_cache: Optional[Dict] = None
    ) -> Optional[str]:
        """
        Get player HTML with caching to avoid duplicate requests.
        Both profile and matches use the same HTML.
        """
        # Check session cache for HTML first
        if session_cache is not None:
            html_cache = session_cache.setdefault('html', {})
            if slug in html_cache:
                return html_cache[slug]

        # Fetch from TA
        url = f"https://www.tennisabstract.com/cgi-bin/player-classic.cgi?p={slug}&f=ACareerqq"
        html = self._fetch_with_retry(url)

        # Cache the HTML for reuse
        if html and session_cache is not None:
            html_cache[slug] = html

        return html

    def get_player_profile(
        self,
        slug: str,
        force_refresh: bool = True,
        persist: bool = False,
        session_cache: Optional[Dict] = None
    ) -> Optional[Dict]:
        """
        Fetch player profile from JavaScript variables.

        Args:
            slug: Tennis Abstract player slug
            force_refresh: If True, always fetch from TA (default: True for fresh data)
            persist: If True, read/write disk cache (default: False)
            session_cache: Dict for in-run memoization (default: None)

        Returns: {name, country, dob, age, height_cm, hand, current_rank, peak_rank}
        """
        # Check session cache first (in-run memoization)
        if session_cache is not None:
            profiles = session_cache.setdefault('profiles', {})
            if slug in profiles:
                return profiles[slug]

        cache_file = PLAYERS_DIR / slug / "profile.json"

        # Check disk cache only if persist=True and not forcing refresh
        if persist and not force_refresh and cache_file.exists():
            age_days = (datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)).days
            if age_days < 7:
                with open(cache_file, 'r') as f:
                    profile = json.load(f)
                    # Store in session cache
                    if session_cache is not None:
                        profiles[slug] = profile
                    return profile

        # Get HTML (cached to avoid duplicate requests)
        html = self._get_cached_html(slug, force_refresh, session_cache)
        if not html:
            return None

        # Extract JavaScript variables
        profile = {
            'slug': slug,
            'name': self._extract_js_var(html, 'fullname'),
            'country': self._extract_js_var(html, 'country'),
            'dob': self._extract_js_var(html, 'dob'),  # YYYYMMDD format
            'height_cm': self._extract_js_var(html, 'ht'),
            'hand': self._extract_js_var(html, 'hand'),  # 'R' or 'L'
            'current_rank': self._extract_js_var(html, 'currentrank'),
            'peak_rank': self._extract_js_var(html, 'peakrank'),
            'last_updated': datetime.now().isoformat()
        }

        # Clean up values
        if profile['name']:
            profile['name'] = profile['name'].strip("'\"")
        if profile['country']:
            profile['country'] = profile['country'].strip("'\"")
        if profile['hand']:
            profile['hand'] = profile['hand'].strip("'\"")

        # Convert numeric fields
        for field in ['height_cm', 'current_rank', 'peak_rank']:
            if profile[field]:
                try:
                    profile[field] = int(profile[field])
                except:
                    profile[field] = None

        # Convert DOB from YYYYMMDD to ISO format and calculate age
        if profile['dob']:
            try:
                dob_str = str(profile['dob'])
                profile['birthdate'] = f"{dob_str[:4]}-{dob_str[4:6]}-{dob_str[6:8]}"
                dob_dt = datetime(int(dob_str[:4]), int(dob_str[4:6]), int(dob_str[6:8]))
                today = datetime.now()
                profile['age'] = today.year - dob_dt.year - (
                    1 if (today.month, today.day) < (dob_dt.month, dob_dt.day) else 0
                )
            except:
                profile['birthdate'] = None
                profile['age'] = None

        # Save to disk cache only if persist=True
        if persist:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_file, 'w') as f:
                json.dump(profile, f, indent=2)

        # Store in session cache for in-run memoization
        if session_cache is not None:
            profiles = session_cache.setdefault('profiles', {})
            profiles[slug] = profile

        return profile

    def get_player_matches(
        self,
        slug: str,
        years: Optional[List[int]] = None,
        force_refresh: bool = True,
        persist: bool = False,
        session_cache: Optional[Dict] = None
    ) -> pd.DataFrame:
        """
        Fetch player match history from JavaScript matchmx array.

        Args:
            slug: Tennis Abstract player slug
            years: List of years to filter (default: current year)
            force_refresh: If True, always fetch from TA (default: True for fresh data)
            persist: If True, read/write disk cache (default: False)
            session_cache: Dict for in-run memoization (default: None)

        Returns:
            DataFrame with columns compatible with LiveFeatureEngine
        """
        # years=None means current year only
        # years=[] means ALL years (for career stats, H2H, etc.)
        if years is None:
            years = [datetime.now().year]

        # Check session cache first (in-run memoization)
        if session_cache is not None:
            matches_cache = session_cache.setdefault('matches', {})
            cache_key = f"{slug}::{','.join(map(str, sorted(years)))}"
            if cache_key in matches_cache:
                return matches_cache[cache_key].copy()  # Return copy to avoid mutation

        cache_file = PLAYERS_DIR / slug / "matches.csv"

        # Check disk cache only if persist=True and not forcing refresh
        if persist and not force_refresh and cache_file.exists():
            age_hours = (datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)).total_seconds() / 3600
            if age_hours < 24:
                df = pd.read_csv(cache_file)
                # Filter by years if specified
                if years and 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'], errors='coerce')
                    df = df[df['date'].dt.year.isin(years)]

                # Store in session cache
                if session_cache is not None:
                    matches_cache[cache_key] = df

                return df

        # Get HTML (cached to avoid duplicate requests)
        html = self._get_cached_html(slug, force_refresh, session_cache)
        if not html:
            return pd.DataFrame()

        # Extract matchhead (column names) and matchmx (match data)
        matchhead = self._extract_js_array(html, 'matchhead')
        matchmx = self._extract_js_array(html, 'matchmx')

        if not matchhead or not matchmx:
            print(f"Failed to extract match data for {slug}")
            return pd.DataFrame()

        # Handle column mismatch: TA sometimes has more data columns than header columns
        # Add dummy column names for any extra columns
        if matchmx and len(matchmx[0]) > len(matchhead):
            extra_cols = len(matchmx[0]) - len(matchhead)
            for i in range(extra_cols):
                matchhead.append(f'_extra_{i}')

        # Convert to DataFrame
        df = pd.DataFrame(matchmx, columns=matchhead)

        # Normalize columns to what LiveFeatureEngine expects
        df_normalized = pd.DataFrame()

        # Date: YYYYMMDD → YYYY-MM-DD
        if 'date' in df.columns:
            df_normalized['date'] = pd.to_datetime(df['date'], format='%Y%m%d', errors='coerce').dt.strftime('%Y-%m-%d')

        # Event/Tournament
        df_normalized['event'] = df.get('tourn', '')

        # Surface: keep as-is (Hard/Clay/Grass/Carpet)
        df_normalized['surface'] = df.get('surf', '').str.title()

        # Round: normalize codes (R32, QF, SF, F, etc.)
        df_normalized['round'] = df.get('round', '')

        # Level: G/M/A/C/25/15/etc
        df_normalized['level'] = df.get('level', '')

        # Player rank at match
        df_normalized['rank'] = pd.to_numeric(df.get('rank', ''), errors='coerce')

        # Opponent info
        df_normalized['opp_name'] = df.get('opp', '')
        df_normalized['opp_rank'] = pd.to_numeric(df.get('orank', ''), errors='coerce')
        df_normalized['opp_hand'] = df.get('ohand', '')
        df_normalized['opp_country'] = df.get('ocountry', '')

        # Score
        df_normalized['score'] = df.get('score', '')

        # Result: W/L from 'wl' column
        df_normalized['result'] = df.get('wl', '').str.upper()

        # Filter out upcoming/unfinished matches (result='U')
        df_normalized = df_normalized[df_normalized['result'].isin(['W', 'L'])].copy()

        # Generate deterministic match_id
        def generate_match_id(row):
            key = f"{row.get('date', '')}|{row.get('event', '')}|{row.get('round', '')}|{row.get('opp_name', '')}".lower()
            return f"ta::{hashlib.sha1(key.encode()).hexdigest()[:16]}"

        df_normalized['match_id'] = df_normalized.apply(generate_match_id, axis=1)

        # Filter by years
        if years:
            df_normalized['date'] = pd.to_datetime(df_normalized['date'], errors='coerce')
            df_normalized = df_normalized[df_normalized['date'].dt.year.isin(years)]

        # Sort by date (newest first)
        df_normalized = df_normalized.sort_values('date', ascending=False)

        # Save to disk cache only if persist=True
        if persist and not df_normalized.empty:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            df_normalized.to_csv(cache_file, index=False)

        # Store in session cache for in-run memoization
        if session_cache is not None:
            matches_cache = session_cache.setdefault('matches', {})
            cache_key = f"{slug}::{','.join(map(str, sorted(years)))}"
            matches_cache[cache_key] = df_normalized

        return df_normalized

    def get_upcoming_match(
        self,
        slug: str,
        opp_name: str,
        session_cache: Optional[Dict] = None
    ) -> Optional[Dict]:
        """
        Look for an upcoming match (result='U') in the player's matchmx against opp_name.
        Uses cached HTML so no extra HTTP request if get_player_matches was already called.

        Returns dict with keys: round, surface, event, date (all strings) or None.
        """
        html = self._get_cached_html(slug, force_refresh=False, session_cache=session_cache)
        if not html:
            return None

        matchhead = self._extract_js_array(html, 'matchhead')
        matchmx = self._extract_js_array(html, 'matchmx')
        if not matchhead or not matchmx:
            return None

        if matchmx and len(matchmx[0]) > len(matchhead):
            for i in range(len(matchmx[0]) - len(matchhead)):
                matchhead.append(f'_extra_{i}')

        df = pd.DataFrame(matchmx, columns=matchhead)

        # Only upcoming rows
        wl_col = df.get('wl', pd.Series(dtype=str))
        upcoming = df[wl_col.str.upper() == 'U'].copy()
        if upcoming.empty:
            return None

        # Match opponent name (last-name fallback)
        opp_last = opp_name.lower().split()[-1] if opp_name else ''
        for _, row in upcoming.iterrows():
            ta_opp = str(row.get('opp', '')).lower()
            if opp_last and opp_last in ta_opp:
                return {
                    'round': str(row.get('round', '')),
                    'surface': str(row.get('surf', '')).title(),
                    'event': str(row.get('tourn', '')),
                    'date': str(row.get('date', '')),
                }

        return None

    @staticmethod
    def name_to_slug(name: str) -> str:
        """
        Convert player name to most-likely TA slug.
        TA uses CamelCase: 'Arthur Cazaux' → 'ArthurCazaux'
        Handles diacritics and hyphens.
        """
        import unicodedata
        # Strip diacritics
        name = ''.join(
            c for c in unicodedata.normalize('NFKD', name)
            if not unicodedata.combining(c)
        )
        # Title-case each word, remove spaces/hyphens/dots
        return ''.join(p.capitalize() for p in name.replace('-', ' ').replace('.', '').split())

    def search_player(self, name: str) -> Optional[str]:
        """
        Search for player slug by trying common slug formats.
        TA uses CamelCase FirstnameLastname (e.g., 'ArthurCazaux').
        Returns slug if found, None otherwise.
        """
        parts = name.strip().split()
        if len(parts) < 2:
            return None

        # CamelCase is the primary TA format; try first+initial as fallback
        camel = self.name_to_slug(name)
        first_initial = parts[0][0].upper() + self.name_to_slug(' '.join(parts[1:]))
        candidates = [camel, first_initial]

        # Strip common name suffixes (Jr, Sr, II, III) and retry
        _SUFFIXES = {'jr', 'sr', 'ii', 'iii', 'iv'}
        clean_parts = [p for p in parts if p.lower().rstrip('.') not in _SUFFIXES]
        if len(clean_parts) < len(parts) and len(clean_parts) >= 2:
            clean_name = ' '.join(clean_parts)
            camel_clean = self.name_to_slug(clean_name)
            first_initial_clean = clean_parts[0][0].upper() + self.name_to_slug(' '.join(clean_parts[1:]))
            for c in (camel_clean, first_initial_clean):
                if c not in candidates:
                    candidates.append(c)

        for slug in candidates:
            url = f"https://www.tennisabstract.com/cgi-bin/player-classic.cgi?p={slug}"
            html = self._fetch_with_retry(url)
            if html and "Player not found" not in html and len(html) > 5000:
                return slug

        return None


def main():
    """Test the scraper"""
    scraper = TennisAbstractScraper()

    test_slug = "ArthurCazaux"
    print(f"Testing with slug: {test_slug}\n")

    # Get profile
    print("1. Fetching profile...")
    profile = scraper.get_player_profile(test_slug, force_refresh=True)
    if profile:
        print("✅ Profile:")
        for key, val in profile.items():
            print(f"   {key}: {val}")

    # Get matches
    print("\n2. Fetching matches (2025)...")
    matches = scraper.get_player_matches(test_slug, years=[2025], force_refresh=True)
    if not matches.empty:
        print(f"✅ Found {len(matches)} matches")
        print("\nFirst 3 matches:")
        print(matches[['date', 'event', 'surface', 'round', 'rank', 'opp_name', 'result']].head(3).to_string(index=False))

    print("\n✅ Test complete!")


if __name__ == "__main__":
    main()
