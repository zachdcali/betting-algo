"""
Purge bad cross-level aliases that may have been created before level-aware enrichment.

This removes aliases like "Challenger - Winston Salem" from ITF M15 tournament rows.
"""

import pandas as pd
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MAP = ROOT / "data" / "tournaments_map.csv"

def purge_cross_level_aliases():
    """Remove aliases that don't match the tournament level"""
    
    if not MAP.exists():
        print(f"❌ No tournaments map found at {MAP}")
        return
    
    df = pd.read_csv(MAP)
    
    def get_alias_level_hint(alias):
        """Get level hint from alias text"""
        alias_lower = alias.lower()
        if "challenger" in alias_lower:
            return "C"
        if re.search(r"\bm25\b|\b25k\b", alias_lower):
            return "25"
        if re.search(r"\bm15\b|\b15k\b", alias_lower):
            return "15"
        if "masters" in alias_lower or "1000" in alias_lower:
            return "M"
        if any(slam in alias_lower for slam in ["grand slam", "australian", "roland", "french", "wimbledon", "us open"]):
            return "G"
        if "atp" in alias_lower and "challenger" not in alias_lower:
            return "A"
        return None
    
    def clean_aliases(row):
        """Remove aliases that don't match row level"""
        aliases_str = str(row["aliases"]) if pd.notna(row["aliases"]) else ""
        if not aliases_str:
            return aliases_str
            
        row_level = str(row["level"]).upper()
        aliases_list = [a.strip() for a in aliases_str.split("|") if a.strip()]
        
        clean_aliases_list = []
        purged_count = 0
        
        for alias in aliases_list:
            alias_level = get_alias_level_hint(alias)
            
            # Keep alias if no level hint or if it matches row level
            if alias_level is None or alias_level == row_level:
                clean_aliases_list.append(alias)
            else:
                print(f"🚫 Purging cross-level alias: '{alias}' from {row['canonical_name']} (level {row_level})")
                purged_count += 1
        
        return "|".join(clean_aliases_list)
    
    # Apply cleaning
    original_aliases = df["aliases"].copy()
    df["aliases"] = df.apply(clean_aliases, axis=1)
    
    # Count changes
    changed_rows = (df["aliases"] != original_aliases).sum()
    
    if changed_rows > 0:
        # Save cleaned version
        df.to_csv(MAP, index=False)
        print(f"✅ Cleaned {changed_rows} tournament rows with cross-level aliases")
        print(f"💾 Updated {MAP}")
    else:
        print("✅ No cross-level aliases found to purge")

if __name__ == "__main__":
    purge_cross_level_aliases()