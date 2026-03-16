import pandas as pd
from pathlib import Path
import re
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime

ROOT = Path(__file__).resolve().parents[2]
SACKMANN = ROOT / "data" / "JeffSackmann" / "jeffsackmann_ml_ready_LEAK_FREE.csv"
OUT = ROOT / "data" / "tournaments_map.csv"
CONFLICTS = ROOT / "data" / "tournaments_conflicts.csv"
OVERRIDES = ROOT / "data" / "tournaments_overrides.csv"
BOVADA_LATEST = ROOT / "production" / "odds" / "data" / "bovada_tennis_latest.csv"  # created by fetch_bovada.py

LEVEL_MAP = {
    "G":"G", "M":"M", "A":"A", "C":"C",
    "25":"25", "15":"15", "F":"F"  # keep F for generic ITF/Futures if present
}

# Seed aliases for recurring ITF tournaments with naming variations
SEED_ALIASES = {
    # ITF M15 tournaments - only M15 aliases
    "M15 Monastir": ["ITF Men's - ITF M15 Monastir", "ITF M15 Monastir", "Monastir MD"],
    "M15 Hurghada": ["ITF Men's - ITF M15 Hurghada", "ITF M15 Hurghada", "Hurghada MD"],
    "M15 Maanshan": ["ITF Men's - ITF M15 Maanshan", "ITF M15 Maanshan", "Maanshan MD"],
    "M15 Kursumlijska Banja": ["ITF Men's - ITF M15 Kursumlijska Banja"],
    "M15 Madrid": ["ITF Men's - ITF M15 Madrid"],
    
    # ITF M25 tournaments - only M25 aliases
    "M25 Cuiaba": ["ITF Men's - ITF M25 Cuiaba", "ITF M25 Cuiaba", "ITF M25 Cuiabá", "Cuiaba MD", "Cuiabá MD"],
    "M25 Tamworth": ["ITF Men's - ITF M25 Tamworth", "ITF M25 Tamworth", "Tamworth MD"],
    "M25 Sapporo": ["ITF Men's - ITF M25 Sapporo", "ITF M25 Sapporo", "Sapporo MD"],
    "M25 Bali": ["ITF Men's - ITF M25 Bali"],
    "M25 Plaisir": ["ITF Men's - ITF M25 Plaisir"],
    "M25 Pozzuoli": ["ITF Men's - ITF M25 Pozzuoli"],
    "M25 Biella": ["ITF Men's - ITF M25 Biella", "M25 Biella"],
    
    # Challenger tournaments - only Challenger aliases
    "Biella CH": ["Challenger - Biella", "Biella CH"],
}

def norm_title(s: str) -> str:
    """Enhanced normalization for tournament titles"""
    s = str(s or "").strip()
    
    # Remove diacritics
    s = unicodedata.normalize('NFD', s)
    s = ''.join(c for c in s if unicodedata.category(c) != 'Mn')
    
    # Strip common tournament prefixes/suffixes and noise
    s = re.sub(r"^challenger\s*-\s*", "", s, flags=re.I)
    s = re.sub(r"^itf\s+men(?:'s)?\s*-\s*(?:itf\s*)?", "", s, flags=re.I)
    s = re.sub(r"^itf\s*-\s*", "", s, flags=re.I)
    s = re.sub(r"-\s*men(?:'s)?\s*singles?$", "", s, flags=re.I)
    
    # Remove tournament round counts like "(11)", "(12)", etc.
    s = re.sub(r"\(\d+\)$", "", s)
    
    # Remove "MD", "Main Draw", "Qualifying" variations
    s = re.sub(r"\b(?:md|main\s+draw|qualifying|qualifiers|qualies)\b", "", s, flags=re.I)
    
    # Clean up spaces and lowercase
    s = re.sub(r"\s+", " ", s).strip().lower()
    
    return s

def norm(s: str) -> str:
    """Simple normalization for backward compatibility"""
    s = str(s or "").strip()
    s = re.sub(r"\s+", " ", s)
    return s

def city_sig_from_title(s: str) -> str:
    """Extract city signature from tournament title (place words only)"""
    toks = re.findall(r"[a-z0-9]+", norm_title(s))
    # Drop level tokens, organizational words, and numbers
    drop = {"atp","challenger","ch","m15","m25","itf","men","mens","tennis",
            "open","championships","grand","slam","masters","1000","250","500",
            "tour","singles","qualifying","qualifiers","qualies","md","main","draw"}
    place = [t for t in toks if t not in drop and not t.isdigit()]
    return " ".join(place)

def load_bovada_events():
    if not BOVADA_LATEST.exists():
        print(f"⚠️ No Bovada data found at {BOVADA_LATEST}")
        return []
    df = pd.read_csv(BOVADA_LATEST)
    return sorted(set(df["event"].dropna().map(norm)))

def load_overrides(latest_year):
    """Load tournament overrides that are valid for the current year"""
    if not OVERRIDES.exists():
        return {}
    
    try:
        df = pd.read_csv(OVERRIDES)
        overrides = {}
        for _, row in df.iterrows():
            canonical_name = str(row['canonical_name']).strip()
            valid_from_year = int(row.get('valid_from_year', 0))
            
            if valid_from_year <= latest_year:
                # Handle pandas NaN values properly
                surface_val = row.get('surface', '')
                level_val = row.get('level', '')
                
                surface = str(surface_val).strip() if pd.notna(surface_val) and str(surface_val).strip() else None
                level = str(level_val).strip() if pd.notna(level_val) and str(level_val).strip() else None
                
                # Additional guard against string "nan"
                if surface and surface.lower() == 'nan':
                    surface = None
                if level and level.lower() == 'nan':
                    level = None
                    
                overrides[canonical_name] = {
                    'surface': surface,
                    'level': level,
                    'draw_size': int(row['draw_size']) if pd.notna(row.get('draw_size')) else None,
                    'valid_from_year': valid_from_year
                }
        
        if overrides:
            print(f"📋 Loaded {len(overrides)} tournament overrides")
        return overrides
    except Exception as e:
        print(f"⚠️ Error loading overrides: {e}")
        return {}

def weighted_mode(series, weights):
    """Get the most common value weighted by weights"""
    if series.empty:
        return None
    
    weighted_counts = {}
    for val, weight in zip(series, weights):
        if pd.notna(val):
            weighted_counts[val] = weighted_counts.get(val, 0) + weight
    
    if not weighted_counts:
        return None
    
    return max(weighted_counts.items(), key=lambda x: x[1])[0]

def get_year_weight(year, latest_year):
    """Calculate recency weight for a given year"""
    years_ago = latest_year - year
    if years_ago <= 4:  # last 5 seasons (0-4 years ago)
        return 3.0
    elif years_ago <= 9:  # seasons 6-10 (5-9 years ago) 
        return 1.0
    else:  # older than 10 years
        return 0.5

def main():
    assert SACKMANN.exists(), f"Missing {SACKMANN}"
    df = pd.read_csv(SACKMANN, low_memory=False)

    # Try common column names
    cols = {c.lower(): c for c in df.columns}
    name_col = cols.get("tournament") or cols.get("tourney_name") or cols.get("tourney") or "tourney_name"
    surface_col = cols.get("surface") or "surface"
    level_col = cols.get("level") or cols.get("tourney_level") or "tourney_level"
    draw_col = cols.get("draw_size") or cols.get("draw") or "draw_size"
    date_col = cols.get("tourney_date") or cols.get("date") or "tourney_date"

    for needed in (name_col, surface_col, level_col):
        assert needed in df.columns, f"Column '{needed}' not found in Sackmann file. Available: {list(df.columns)}"

    print(f"Using columns: name='{name_col}', surface='{surface_col}', level='{level_col}', draw='{draw_col}', date='{date_col}'")

    # Clean and add year column for recency weighting
    df[name_col] = df[name_col].map(norm)
    df[surface_col] = df[surface_col].astype(str).str.title().map({"Hard":"Hard","Clay":"Clay","Grass":"Grass","Carpet":"Carpet"})
    df[level_col] = df[level_col].astype(str).str.upper()

    if draw_col in df.columns:
        df[draw_col] = pd.to_numeric(df[draw_col], errors="coerce").fillna(32).astype(int)
    else:
        df[draw_col] = 32

    # Add year column for recency weighting
    if date_col in df.columns:
        df['year'] = pd.to_datetime(df[date_col], errors='coerce').dt.year
        latest_year = int(df['year'].max())
        df['weight'] = df['year'].apply(lambda y: get_year_weight(y, latest_year) if pd.notna(y) else 1.0)
    else:
        print("⚠️ No date column found, using equal weighting")
        df['year'] = 2024  # fallback
        df['weight'] = 1.0
        latest_year = 2024

    print(f"📅 Using recency weighting with latest year: {latest_year}")

    # Load any tournament overrides
    overrides = load_overrides(latest_year)

    groups = df.groupby(name_col)
    rows = []
    conflicts = []

    for tname, g in groups:
        if not tname or pd.isna(tname):
            continue

        # Get recent window (last 5 seasons) vs weighted counts
        def get_recent_window_counts(field, min_recent_obs=3):
            """Get counts from recent window (last 5 seasons) if sufficient data"""
            recent_cutoff = latest_year - 4  # last 5 seasons
            recent_data = g[g['year'] >= recent_cutoff][field].dropna()
            
            if len(recent_data) >= min_recent_obs:
                # Sufficient recent data - use simple mode
                recent_counts = recent_data.value_counts().to_dict()
                return sorted(recent_counts.items(), key=lambda x: x[1], reverse=True), "recent_window"
            else:
                # Insufficient recent data - fall back to weighted
                counts = {}
                for val, weight in zip(g[field].dropna(), g.loc[g[field].notna(), 'weight']):
                    counts[val] = counts.get(val, 0) + weight
                return sorted(counts.items(), key=lambda x: x[1], reverse=True), "weighted"

        def get_weighted_counts(field):
            """Get all-time weighted counts for conflict reporting"""
            counts = {}
            for val, weight in zip(g[field].dropna(), g.loc[g[field].notna(), 'weight']):
                counts[val] = counts.get(val, 0) + weight
            return sorted(counts.items(), key=lambda x: x[1], reverse=True)

        # Get best choices using recent-window mode
        surf_counts, surf_source = get_recent_window_counts(surface_col)
        lvl_counts, lvl_source = get_recent_window_counts(level_col)
        draw_counts, draw_source = get_recent_window_counts(draw_col)

        # Get full weighted counts for conflict reporting
        surf_weighted = get_weighted_counts(surface_col)
        lvl_weighted = get_weighted_counts(level_col)
        draw_weighted = get_weighted_counts(draw_col)

        # Check for conflicts and record detailed history
        has_conflicts = (len(surf_weighted) > 1) or (len(lvl_weighted) > 1) or (len(draw_weighted) > 1)
        
        if has_conflicts:
            conflicts.append({
                "canonical_name": tname,
                "surface_hist": ";".join(f"{k}:{int(v)}" for k, v in surf_weighted[:3]),
                "level_hist": ";".join(f"{k}:{int(v)}" for k, v in lvl_weighted[:3]),
                "draw_hist": ";".join(f"{int(k)}:{int(v)}" for k, v in draw_weighted[:3]),
                "chosen_surface": surf_counts[0][0] if surf_counts else "Hard",
                "chosen_level": LEVEL_MAP.get(lvl_counts[0][0] if lvl_counts else "A", "A"),
                "chosen_draw": int(draw_counts[0][0]) if draw_counts else 32,
                "surface_source": surf_source,
                "level_source": lvl_source,
                "draw_source": draw_source
            })

        # Choose best values (recent window or weighted fallback)
        surface = surf_counts[0][0] if surf_counts else "Hard"
        level_raw = lvl_counts[0][0] if lvl_counts else "A"
        level = LEVEL_MAP.get(level_raw, "A")
        draw_size = int(draw_counts[0][0]) if draw_counts else 32
        
        # Apply manual overrides if they exist
        override_applied = False
        override_details = []
        if tname in overrides:
            override = overrides[tname]
            if override['surface']:
                surface = override['surface']
                override_applied = True
                override_details.append(f"surface={surface}")
            if override['level']:
                level = LEVEL_MAP.get(override['level'], override['level'])
                override_applied = True
                override_details.append(f"level={level}")
            if override['draw_size']:
                draw_size = override['draw_size']
                override_applied = True
                override_details.append(f"draw={draw_size}")
            
            if override_applied:
                print(f"🔧 Applied override for {tname}: {', '.join(override_details)}")

        rows.append({
            "canonical_name": tname,
            "aliases": "",  # fill later with seed aliases and Bovada matches
            "surface": surface,
            "level": level,
            "draw_size": draw_size,
            "city_sig": city_sig_from_title(tname),
            "override_applied": override_applied,
            "surface_source": "override" if override_applied and tname in overrides and overrides[tname]['surface'] else surf_source,
            "level_source": "override" if override_applied and tname in overrides and overrides[tname]['level'] else lvl_source,
            "draw_source": "override" if override_applied and tname in overrides and overrides[tname]['draw_size'] else draw_source
        })

    base = pd.DataFrame(rows).sort_values("canonical_name").reset_index(drop=True)
    print(f"✅ Built base map with {len(base)} tournaments from Sackmann data (recency-weighted)")

    # Add seed aliases first for known recurring tournaments
    aliases_map = defaultdict(set)
    seed_count = 0
    
    def find_canonical_by_tokens(search_canonical):
        """Find canonical tournament using token signature matching with level awareness"""
        search_tokens = set(re.findall(r"[a-z0-9]+", norm_title(search_canonical)))
        
        # Extract level hint from search canonical
        search_level_tokens = search_tokens & {"m15", "m25", "challenger", "ch"}
        
        best_match = None
        best_overlap = 0
        
        for _, row in base.iterrows():
            canonical_name = row["canonical_name"]
            canonical_level = str(row["level"]).upper()
            canonical_tokens = set(re.findall(r"[a-z0-9]+", norm_title(canonical_name)))
            
            # Check if level tokens are compatible
            level_compatible = True
            if search_level_tokens:
                if "m15" in search_level_tokens and canonical_level != "15":
                    level_compatible = False
                elif "m25" in search_level_tokens and canonical_level != "25":
                    level_compatible = False
                elif ("challenger" in search_level_tokens or "ch" in search_level_tokens) and canonical_level != "C":
                    level_compatible = False
            
            if not level_compatible:
                continue
                
            # Calculate place name overlap (excluding level tokens)
            place_search = search_tokens - {"m15", "m25", "challenger", "ch", "itf", "mens", "men", "atp"}
            place_canonical = canonical_tokens - {"m15", "m25", "challenger", "ch", "itf", "mens", "men", "atp"}
            overlap = len(place_search & place_canonical)
            
            # Require good place name overlap
            if overlap > best_overlap and overlap >= max(1, len(place_search) * 0.6):
                best_match = canonical_name
                best_overlap = overlap
                
        return best_match
    
    for seed_canonical, alias_list in SEED_ALIASES.items():
        # Find the canonical tournament using token signature
        matched_canonical = find_canonical_by_tokens(seed_canonical)
        if matched_canonical:
            for alias in alias_list:
                aliases_map[matched_canonical].add(alias)
                seed_count += 1
        else:
            print(f"⚠️ No match found for seed canonical: {seed_canonical}")
    
    print(f"✅ Added {seed_count} seed aliases for recurring tournaments")

    # Enhanced Bovada alias enrichment with level+city awareness
    ENABLE_BOVADA_ALIAS_ENRICHMENT = True  # Set to False to disable auto-aliasing
    
    if ENABLE_BOVADA_ALIAS_ENRICHMENT:
        bovada_events = load_bovada_events()

        def sig(s): 
            """Create token signature using enhanced normalization"""
            return set(re.findall(r"[a-z0-9]+", norm_title(s)))
        
        def level_hint_from_title_builder(title: str) -> str:
            """Parse level hint from Bovada title (same logic as resolver)"""
            if not title:
                return None
            t = title.lower()
            if "challenger" in t:
                return "C"
            if re.search(r"\bm25\b|\b25k\b", t):
                return "25"
            if re.search(r"\bm15\b|\b15k\b", t):
                return "15"
            if "masters" in t or "1000" in t:
                return "M"
            if any(slam in t for slam in ["grand slam", "australian open", "roland garros", 
                                         "french open", "wimbledon", "us open"]):
                return "G"
            if "atp" in t and "challenger" not in t:
                return "A"
            return None

        safe_aliases_added = 0
        blocked_cross_level = 0
        
        # Level+city-aware Bovada event matching
        for ev in bovada_events:
            hint = level_hint_from_title_builder(ev)
            ev_sig = sig(ev)
            ev_city = city_sig_from_title(ev)
            
            # Find level+city-compatible candidates
            candidates = []
            for _, row in base.iterrows():
                row_city = str(row.get("city_sig", "")).strip()
                row_level = str(row["level"]).upper()
                canonical_name = row["canonical_name"]
                
                # Skip if no city info
                if not ev_city or not row_city:
                    continue
                    
                # Must be same city
                if row_city != ev_city:
                    continue
                    
                # Must be same level if hint exists
                if hint and row_level != hint:
                    blocked_cross_level += 1
                    continue
                
                tn_sig = sig(canonical_name)
                overlap = len(ev_sig & tn_sig)
                candidates.append((overlap, canonical_name))
            
            if not candidates:
                # No safe candidates found (city ambiguous or level mismatch)
                continue
                
            # Pick best candidate
            candidates.sort(reverse=True)
            best_overlap, best_name = candidates[0]
            
            # Require decent place overlap to learn (≥2 tokens)
            if best_overlap >= 2:
                aliases_map[best_name].add(ev)
                safe_aliases_added += 1

        print(f"✅ Added {safe_aliases_added} level+city-safe aliases from Bovada")
        if blocked_cross_level > 0:
            print(f"🚫 Blocked {blocked_cross_level} cross-level alias attempts")
    else:
        print("⚠️ Bovada alias enrichment disabled - using resolver auto-learning only")

    # Merge aliases and normalize
    final_aliases = {}
    for canonical_name in base["canonical_name"]:
        aliases = aliases_map.get(canonical_name, set())
        # De-duplicate by normalized form
        normalized_aliases = set()
        for alias in aliases:
            norm_alias = norm_title(alias)
            if norm_alias and norm_alias != norm_title(canonical_name):
                normalized_aliases.add(alias)
        final_aliases[canonical_name] = "|".join(sorted(normalized_aliases))

    base["aliases"] = base["canonical_name"].map(lambda n: final_aliases.get(n, ""))
    
    total_with_aliases = len([a for a in final_aliases.values() if a])
    print(f"✅ Added aliases for {total_with_aliases} tournaments (seed + Bovada)")

    # Save outputs
    OUT.parent.mkdir(parents=True, exist_ok=True)
    base.to_csv(OUT, index=False)
    print(f"✅ Wrote {len(base)} rows -> {OUT}")

    if conflicts:
        conflicts_df = pd.DataFrame(conflicts)
        conflicts_df.to_csv(CONFLICTS, index=False)
        print(f"⚠️ Found {len(conflicts)} tournaments with surface/level/draw conflicts -> {CONFLICTS}")
        print(f"   Top conflicts by surface changes:")
        for _, row in conflicts_df.head(5).iterrows():
            print(f"     {row['canonical_name']}: surface={row['surface_hist']}")
    else:
        print("✅ No conflicts found")

if __name__ == "__main__":
    main()