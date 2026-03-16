#!/usr/bin/env python3
"""
Player Name Matching Utilities
Matches Bovada player names to UTR player IDs
"""

import pandas as pd
import unicodedata
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Tuple, Optional

def normalize_name(name_str: str) -> str:
    """Normalize player name for matching"""
    if not name_str or pd.isna(name_str):
        return ""
    
    # Remove accents and diacritics
    name = unicodedata.normalize('NFKD', str(name_str))
    name = ''.join(ch for ch in name if not unicodedata.combining(ch))
    
    # Remove non-alphabetic characters except spaces and hyphens
    name = re.sub(r'[^a-zA-Z\s\-]', '', name).lower().strip()
    
    # Normalize spaces
    name = re.sub(r'\s+', ' ', name)
    
    return name

def similarity_score(name1: str, name2: str) -> float:
    """Calculate similarity score between two names"""
    return SequenceMatcher(None, name1, name2).ratio()

def extract_name_variants(full_name: str) -> List[str]:
    """Extract possible name variants (first last, last first, etc.)"""
    normalized = normalize_name(full_name)
    parts = normalized.split()
    
    variants = [normalized]  # Full normalized name
    
    if len(parts) >= 2:
        # First Last
        variants.append(f"{parts[0]} {parts[-1]}")
        # Last First  
        variants.append(f"{parts[-1]} {parts[0]}")
        # Last, First
        variants.append(f"{parts[-1]}, {parts[0]}")
        
    return list(set(variants))  # Remove duplicates

class PlayerNameMatcher:
    """Match player names between different sources"""
    
    def __init__(self, data_dir: str = "../../data"):
        self.data_dir = Path(data_dir)
        self.mapping_file = self.data_dir / "player_mapping.csv"
        self.manual_mappings = self._load_manual_mappings()
        self.utr_players = self._load_utr_players()
        
    def _load_manual_mappings(self) -> Dict[str, str]:
        """Load UTR player mappings with name variants"""
        mappings = {}
        if self.mapping_file.exists():
            try:
                df = pd.read_csv(self.mapping_file)
                
                # Handle the UTR scraper format (utr_id, primary_name, name_variants)
                if 'utr_id' in df.columns and 'name_variants' in df.columns:
                    for _, row in df.iterrows():
                        utr_id = str(row['utr_id'])
                        name_variants = str(row['name_variants']) if pd.notna(row['name_variants']) else ""
                        
                        # Split variants and normalize each one
                        if name_variants:
                            variants = name_variants.split('|')
                            for variant in variants:
                                normalized = normalize_name(variant.strip())
                                if normalized:
                                    mappings[normalized] = utr_id
                
                # Also handle old format (bovada_name, utr_id) for backward compatibility
                elif 'bovada_name' in df.columns and 'utr_id' in df.columns:
                    mappings.update(dict(zip(df['bovada_name'].str.lower(), df['utr_id'])))
                    
                print(f"✅ Loaded {len(mappings)} player name mappings")
                    
            except Exception as e:
                print(f"⚠️  Error loading player mappings: {e}")
        return mappings
    
    def _load_utr_players(self) -> Dict[str, str]:
        """Load known UTR players from match data"""
        utr_players = {}
        
        # Look for player files in matches directory
        matches_dir = self.data_dir / "matches"
        if matches_dir.exists():
            for player_file in matches_dir.glob("player_*_matches.csv"):
                # Extract player ID from filename
                player_id = player_file.name.split('_')[1]
                
                # Try to read player name from file (if available)
                try:
                    df = pd.read_csv(player_file, nrows=5)  # Just read first few rows
                    if 'player_name' in df.columns:
                        player_name = df['player_name'].iloc[0]
                        normalized_name = normalize_name(player_name)
                        utr_players[normalized_name] = player_id
                except:
                    pass  # Skip files we can't read
        
        return utr_players
    
    def find_best_match(self, bovada_name: str, threshold: float = 0.8) -> Optional[str]:
        """Find best matching UTR player ID for a Bovada name"""
        normalized_bovada = normalize_name(bovada_name)
        
        # Check manual mappings first
        if normalized_bovada in self.manual_mappings:
            return self.manual_mappings[normalized_bovada]
        
        # Try fuzzy matching with known UTR players
        best_match = None
        best_score = 0.0
        
        for utr_name, utr_id in self.utr_players.items():
            # Generate variants of both names
            bovada_variants = extract_name_variants(bovada_name)
            utr_variants = extract_name_variants(utr_name)
            
            # Check all combinations
            for bov_variant in bovada_variants:
                for utr_variant in utr_variants:
                    score = similarity_score(bov_variant, utr_variant)
                    if score > best_score:
                        best_score = score
                        best_match = utr_id
        
        if best_score >= threshold:
            # Save this mapping for future use
            self._save_mapping(normalized_bovada, best_match)
            return best_match
        
        return None
    
    def _save_mapping(self, bovada_name: str, utr_id: str):
        """Save a new player mapping"""
        try:
            # Load existing mappings
            if self.mapping_file.exists():
                df = pd.read_csv(self.mapping_file)
            else:
                df = pd.DataFrame(columns=['bovada_name', 'utr_id'])
            
            # Add new mapping if not already present
            if bovada_name not in df['bovada_name'].str.lower().values:
                new_row = pd.DataFrame([{'bovada_name': bovada_name, 'utr_id': utr_id}])
                df = pd.concat([df, new_row], ignore_index=True)
                df.to_csv(self.mapping_file, index=False)
                print(f"💾 Saved mapping: {bovada_name} -> {utr_id}")
        except Exception as e:
            print(f"⚠️  Error saving mapping: {e}")
    
    def batch_match_players(self, player_names: List[str]) -> Dict[str, Optional[str]]:
        """Match a batch of player names"""
        results = {}
        
        for name in player_names:
            utr_id = self.find_best_match(name)
            results[name] = utr_id
            
            if utr_id:
                print(f"✅ Matched: {name} -> {utr_id}")
            else:
                print(f"❌ No match: {name}")
        
        return results

def create_sample_mappings():
    """Create sample player mappings file"""
    sample_mappings = [
        {'bovada_name': 'novak djokovic', 'utr_id': '100001'},
        {'bovada_name': 'rafael nadal', 'utr_id': '100002'}, 
        {'bovada_name': 'roger federer', 'utr_id': '100003'},
        {'bovada_name': 'carlos alcaraz', 'utr_id': '1013703'},
        {'bovada_name': 'jannik sinner', 'utr_id': '1017623'}
    ]
    
    df = pd.DataFrame(sample_mappings)
    mapping_file = Path("../../data/player_mapping.csv")
    mapping_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(mapping_file, index=False)
    print(f"📝 Created sample mappings at {mapping_file}")

def main():
    """Test the name matcher"""
    matcher = PlayerNameMatcher()
    
    # Test some player names
    test_names = [
        "Novak Djokovic",
        "Rafael Nadal", 
        "Carlos Alcaraz",
        "Unknown Player"
    ]
    
    results = matcher.batch_match_players(test_names)
    
    print("\n📊 Matching Results:")
    for name, utr_id in results.items():
        status = "✅ Found" if utr_id else "❌ Not found"
        print(f"  {name}: {utr_id} ({status})")

if __name__ == "__main__":
    # Create sample mappings if they don't exist
    mapping_file = Path("../../data/player_mapping.csv")
    if not mapping_file.exists():
        create_sample_mappings()
    
    main()