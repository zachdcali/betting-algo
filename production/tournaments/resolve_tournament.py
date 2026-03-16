from dataclasses import dataclass
from typing import Optional, Dict, Tuple
import pandas as pd
import re
import unicodedata
import csv
from pathlib import Path

@dataclass
class TournamentMeta:
    surface: str      # "Hard"|"Clay"|"Grass"|"Carpet"
    level: str        # "A"|"M"|"G"|"C"|"F"|"25"|"15"
    draw_size: int
    round_code: Optional[str] = None

ROUND_ALIASES: Dict[str,str] = {
    "F": r"\bfinal\b|^f$",
    "SF": r"\bsemi(?:-)?finals?\b|^sf$",
    "QF": r"\bquarter(?:-)?finals?\b|^qf$",
    "R128": r"\br(?:ound)?\s?of\s?128\b|^r128$|^128$",
    "R64": r"\br(?:ound)?\s?of\s?64\b|^r64$|^64$",
    "R32": r"\br(?:ound)?\s?of\s?32\b|^r32$|^32$",
    "R16": r"\br(?:ound)?\s?of\s?16\b|^r16$|^16$",
    "Q4": r"\bq4\b", "Q3": r"\bq3\b", "Q2": r"\bq2\b", "Q1": r"\bq1\b",
    "RR": r"\bround\s?robin\b|\bgroup\b|^rr$", "ER": r"\bearly\s?rounds?\b|^er$",
    "BR": r"\bbronze\b|^br$|third\s?place",
}

# Stopwords to filter out for fuzzy matching
STOPWORDS = {
    "the","men","men's","mens","singles","tennis","open","championships",
    "atp","challenger","itf","m15","m25","m10","w15","w25","qualifying",
    "qualifiers","qualies","q","q1","q2","q3","q4","1","2","3","4"
}

LEVEL_TOKENS = {
    "g": {"grand","slam","australian","roland","garros","wimbledon","us"},
    "m": {"masters","1000"},
    "a": {"atp","tour","500","250"},
    "25": {"m25","25k"},
    "15": {"m15","15k"},
    "c": {"challenger"},
}

def norm_title(s: str) -> str:
    """Enhanced normalization for tournament titles (for fuzzy matching - strips level tokens)"""
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

def norm_exact(s: str) -> str:
    """Stricter normalizer for exact/alias matching (preserves level tokens)"""
    s = str(s or "").strip()
    
    # Remove diacritics
    s = unicodedata.normalize('NFD', s)
    s = ''.join(c for c in s if unicodedata.category(c) != 'Mn')
    
    # Only strip trailing Bovada count like "(11)" - keep level tokens
    s = re.sub(r"\(\d+\)$", "", s).strip()
    
    # Convert to lowercase and normalize whitespace
    s = s.lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s

def level_hint_from_title(title: str) -> Optional[str]:
    """Parse tournament level from Bovada title and return Sackmann code"""
    if not title:
        return None
        
    t = title.lower()
    
    # Grand Slams
    if any(slam in t for slam in ["grand slam", "australian open", "roland garros", 
                                 "french open", "wimbledon", "us open"]):
        return "G"
    
    # Masters 1000
    if "masters" in t or "1000" in t:
        return "M"
    
    # Challenger
    if "challenger" in t:
        return "C"
    
    # ITF Futures M25
    if re.search(r"\bm25\b|\b25k\b", t):
        return "25"
    
    # ITF Futures M15  
    if re.search(r"\bm15\b|\b15k\b", t):
        return "15"
    
    # ATP (500/250) - only if it explicitly says ATP and not Challenger
    if "atp" in t and "challenger" not in t:
        return "A"
    
    # Return None for unclear cases (safer)
    return None

def city_sig_from_title(s: str) -> str:
    """Extract city signature from tournament title (place words only)"""
    toks = re.findall(r"[a-z0-9]+", norm_title(s))
    # Drop level tokens, organizational words, and numbers
    drop = {"atp","challenger","ch","m15","m25","itf","men","mens","tennis",
            "open","championships","grand","slam","masters","1000","250","500",
            "tour","singles","qualifying","qualifiers","qualies","md","main","draw"}
    place = [t for t in toks if t not in drop and not t.isdigit()]
    return " ".join(place)

def _norm(s: str) -> str:
    """Legacy normalization - now uses enhanced norm_title"""
    return norm_title(s)

def _tokens(s: str) -> set:
    return set(re.findall(r"[a-z0-9]+", norm_title(s)))

def _sig_tokens(s: str) -> set:
    """Get tokens minus stopwords for fuzzy matching"""
    toks = set(re.findall(r"[a-z0-9]+", norm_title(s)))
    return {t for t in toks if t not in STOPWORDS}

def _jaccard(a: set, b: set) -> float:
    """Calculate Jaccard similarity between two token sets"""
    if not a or not b: return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0

def _round_from_text(txt: Optional[str]) -> Optional[str]:
    if not txt: return None
    t = _norm(txt)
    for canon, pat in ROUND_ALIASES.items():
        if re.search(pat, t): return canon
    m = re.match(r"^r(128|64|32|16|8)$", t)
    return f"R{m.group(1)}" if m else None

class TournamentResolver:
    """CSV schema: canonical_name,aliases,surface,level,draw_size"""
    def __init__(self, csv_path: str):
        self.csv_path = Path(csv_path)
        self.df = pd.read_csv(csv_path)
        self._alias_row, self._sig_row = {}, {}
        self._rebuild_indices()
        
        # Precompute signature tokens for fuzzy matching
        self._name_tokens = []
        for i, row in self.df.iterrows():
            name_tokens = _sig_tokens(str(row["canonical_name"]))
            self._name_tokens.append((name_tokens, i))
        
        # Precompute ambiguous cities (cities with multiple levels)
        levels_by_city = {}
        for _, row in self.df.iterrows():
            city_sig = str(row.get("city_sig", "")).strip()
            if city_sig:  # Only track non-empty city signatures
                levels_by_city.setdefault(city_sig, set()).add(str(row["level"]).upper())
        
        self._ambiguous_cities = {sig for sig, lvls in levels_by_city.items() if len(lvls) > 1}

        # Commented out verbose tournament warning list for cleaner dry-run logs
        # if self._ambiguous_cities:
        #     print(f"⚠️  Found {len(self._ambiguous_cities)} cities with multiple tournament levels:")
        #     for city in sorted(self._ambiguous_cities):
        #         city_levels = levels_by_city[city]
        #         print(f"   {city}: {', '.join(sorted(city_levels))}")
        # else:
        #     print("✅ No ambiguous cities detected")

    def _rebuild_indices(self):
        """Rebuild the internal indices after dataframe changes"""
        self._alias_row, self._sig_row = {}, {}
        for i, row in self.df.iterrows():
            names = [str(row["canonical_name"])]
            aliases = str(row.get("aliases","") or "")
            names += [a.strip() for a in aliases.split("|") if a.strip()]
            for n in names: 
                self._alias_row[norm_exact(n)] = i
            
            # Exact-signature should also preserve level tokens
            sig = tuple(sorted(re.findall(r"[a-z0-9]+", norm_exact(row["canonical_name"]))))
            if sig: 
                self._sig_row[sig] = i

    def best_match(self, event_name: str, level_hint: Optional[str] = None) -> Tuple[Optional[int], float]:
        """Find best matching tournament row with city+level-aware disambiguation"""
        
        # 1) Fast path: exact alias or exact signature match (preserves level tokens)
        idx = self._alias_row.get(norm_exact(event_name))
        if idx is not None:
            return int(idx), 1.0  # perfect match
            
        sig = tuple(sorted(re.findall(r"[a-z0-9]+", norm_exact(event_name))))
        idx = self._sig_row.get(sig)
        if idx is not None:
            return int(idx), 1.0  # perfect match

        # 2) City+level-aware fuzzy matching
        query_tokens = _sig_tokens(event_name)
        event_city_sig = city_sig_from_title(event_name)
        
        # Step 2: Level-compatible, same-city candidates (highest priority)
        if level_hint and event_city_sig:
            same_city_level_candidates = []
            for name_tokens, i in self._name_tokens:
                row = self.df.iloc[i]
                row_level = str(row["level"]).upper()
                row_city_sig = str(row.get("city_sig", "")).strip()
                
                if (row_level == level_hint and 
                    row_city_sig == event_city_sig and
                    row_city_sig):  # non-empty city sig
                    
                    base_score = _jaccard(query_tokens, name_tokens)
                    overlap = len(query_tokens & name_tokens)
                    
                    # Minimal requirements for same-city, same-level
                    if base_score >= 0.25 or overlap >= 1:
                        same_city_level_candidates.append((i, base_score))
            
            if same_city_level_candidates:
                # Sort candidates by score (best first)
                same_city_level_candidates.sort(key=lambda x: x[1], reverse=True)
                best_idx, best_score = same_city_level_candidates[0]
                
                # Tie-breaking guard: if top two candidates are very close, treat as ambiguous
                if len(same_city_level_candidates) > 1:
                    second_score = same_city_level_candidates[1][1]
                    if (best_score - second_score) <= 0.05:  # tie threshold
                        print(f"⚠️ Tie in same-city+level candidates for '{event_name}' — returning None")
                        return None, 0.0
                
                return int(best_idx), best_score
            
            # Critical guard: if no same-city+same-level candidates found, block cross-level (unconditional)
            else:
                print(f"⚠️ No same-city+same-level for '{event_name}' (level={level_hint}, city='{event_city_sig}') — returning None")
                return None, 0.0
        
        # Step 3: Non-ambiguous city, level-agnostic fallback with penalties
        candidates = []
        for name_tokens, i in self._name_tokens:
            row = self.df.iloc[i]
            row_level = str(row["level"]).upper()
            
            # Calculate base Jaccard score
            base_score = _jaccard(query_tokens, name_tokens)
            
            # Apply level penalties for cross-level matches
            adjusted_score = base_score
            if level_hint is not None:
                if row_level == level_hint:
                    adjusted_score += 0.20  # bonus for level match
                elif (level_hint in {"A", "C", "M", "G"} and 
                      row_level in {"15", "25"}):
                    adjusted_score -= 0.50  # strong penalty: ATP/Challenger/Masters → ITF
                elif (level_hint in {"15", "25"} and 
                      row_level in {"A", "C", "M", "G"}):
                    adjusted_score -= 0.30  # moderate penalty: ITF → ATP/Challenger/Masters
                else:
                    adjusted_score -= 0.20  # general level mismatch penalty
            
            # Small bonus for "CH" token when event says Challenger
            if (level_hint == "C" and 
                row_level == "C" and 
                "ch" in name_tokens):
                adjusted_score += 0.05
            
            candidates.append({
                'idx': i,
                'name_tokens': name_tokens,
                'base_score': base_score,
                'adjusted_score': adjusted_score,
                'row_level': row_level
            })
        
        # Sort by adjusted score
        candidates.sort(key=lambda x: x['adjusted_score'], reverse=True)
        
        if not candidates:
            return None, 0.0
            
        best_candidate = candidates[0]
        
        # Higher threshold for cross-level matches
        overlap = len(query_tokens & best_candidate['name_tokens'])
        required_score = 0.50 if level_hint and best_candidate['row_level'] != level_hint else 0.40
        required_overlap = 2
        
        if best_candidate['adjusted_score'] >= required_score or overlap >= required_overlap:
            return int(best_candidate['idx']), best_candidate['base_score']

        return None, 0.0
    

    def resolve(self, event_name: str, round_hint: Optional[str]=None, default_draw: int=32):
        """Original resolve method (exact matching only)"""
        idx = self._alias_row.get(_norm(event_name))
        if idx is None:
            idx = self._sig_row.get(tuple(sorted(_tokens(event_name))))
        if idx is None:
            return None
        row = self.df.iloc[idx]
        surface = str(row["surface"]).strip().title()
        level = str(row["level"]).strip()
        draw = int(row.get("draw_size", default_draw))
        rc = _round_from_text(round_hint) if round_hint else None
        return TournamentMeta(surface, level, draw, rc)

    def resolve_soft(self, event_name: str, round_hint: Optional[str]=None, default_draw: int=32):
        """Resolve with level-aware fuzzy matching fallback"""
        # Parse level hint from event name
        level_hint = level_hint_from_title(event_name)
        
        # Try to match with level awareness
        idx, score = self.best_match(event_name, level_hint)
        if idx is None:
            return None
            
        row = self.df.iloc[idx]
        surface = str(row["surface"]).strip().title()
        level = str(row["level"]).strip()
        draw = int(row.get("draw_size", default_draw))
        rc = _round_from_text(round_hint) if round_hint else None
        return TournamentMeta(surface, level, draw, rc), score

    def learn_alias(self, event_name: str, canonical_idx: int, score: float, save_to_csv: bool = True) -> bool:
        """
        Learn a new alias for a tournament after successful fuzzy matching.
        
        Args:
            event_name: The Bovada event name that matched
            canonical_idx: The index of the matched canonical tournament
            score: The match score
            save_to_csv: Whether to save the updated CSV immediately
            
        Returns:
            True if alias was learned, False if not (failed checks)
        """
        # Safety checks
        if canonical_idx >= len(self.df):
            return False
            
        # Guardrails - only learn high-confidence matches
        if score < 0.55:
            return False
            
        row = self.df.iloc[canonical_idx]
        canonical_name = str(row["canonical_name"])
        
        # Check level token agreement
        event_tokens = set(re.findall(r"[a-z0-9]+", event_name.lower()))
        canonical_tokens = set(re.findall(r"[a-z0-9]+", canonical_name.lower()))
        
        # Enhanced level agreement check using Sackmann codes
        row = self.df.iloc[canonical_idx]
        canonical_level = str(row["level"]).upper()
        
        event_level_tokens = event_tokens & {"challenger", "atp", "m25", "m15", "masters", "grand", "slam", "1000", "500", "250"}
        
        # Map event tokens to expected Sackmann codes
        level_compatible = True
        if event_level_tokens:
            expected_codes = set()
            if "challenger" in event_level_tokens:
                expected_codes.add("C")
            if "m25" in event_level_tokens or "25k" in event_tokens:
                expected_codes.add("25")
            if "m15" in event_level_tokens or "15k" in event_tokens:
                expected_codes.add("15")
            if any(t in event_level_tokens for t in ["atp", "500", "250"]):
                expected_codes.add("A")
            if any(t in event_level_tokens for t in ["masters", "1000"]):
                expected_codes.add("M")
            if any(t in event_level_tokens for t in ["grand", "slam"]):
                expected_codes.add("G")
                
            if expected_codes and canonical_level not in expected_codes:
                level_compatible = False
        
        if not level_compatible:
            return False
        
        # Place token overlap check (at least 2 meaningful place tokens)
        place_tokens = event_tokens & canonical_tokens - STOPWORDS - event_level_tokens
        if len(place_tokens) < 2:
            return False
            
        # Normalize the event name before adding as alias
        normalized_event = norm_title(event_name)
        normalized_canonical = norm_title(canonical_name)
        
        # Don't add if it's essentially the same as canonical
        if normalized_event == normalized_canonical:
            return False
            
        # Get current aliases
        current_aliases = str(row.get("aliases", "") or "")
        aliases_list = [a.strip() for a in current_aliases.split("|") if a.strip()]
        
        # Check if this alias (or normalized version) already exists
        existing_normalized = {norm_title(a) for a in aliases_list}
        if normalized_event in existing_normalized:
            return False
            
        # Add the new alias
        aliases_list.append(event_name)
        new_aliases_str = "|".join(sorted(set(aliases_list)))
        
        # Update the dataframe
        self.df.at[canonical_idx, "aliases"] = new_aliases_str
        
        # Rebuild indices to include the new alias
        self._rebuild_indices()
        
        # Save to CSV if requested
        if save_to_csv:
            self.df.to_csv(self.csv_path, index=False)
            
            # Log the learning event
            log_path = self.csv_path.parent / "tournaments_alias_log.csv"
            from datetime import datetime
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "event_raw": event_name,
                "canonical": canonical_name,
                "score": f"{score:.3f}",
                "level_check_passed": "true",
                "place_overlap": len(place_tokens),
                "canonical_level": canonical_level,
                "last_seen": datetime.now().isoformat()
            }
            
            # Append to log file
            file_exists = log_path.exists()
            with open(log_path, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=log_entry.keys())
                if not file_exists:
                    writer.writeheader()
                writer.writerow(log_entry)
        
        return True