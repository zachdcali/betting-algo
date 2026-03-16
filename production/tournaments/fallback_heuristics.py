"""
Fallback heuristics for unmatched tournaments in the live betting pipeline.

When a Bovada event can't be matched to any Sackmann tournament, these heuristics
parse the event name to extract likely surface/level/draw so the pipeline doesn't stall.
"""

import re
from typing import Optional, Dict
from dataclasses import dataclass

@dataclass
class FallbackTournamentMeta:
    surface: Optional[str] = None  # None means unknown (all surface one-hots = 0)
    level: str = "A"  # Default to ATP level
    draw_size: int = 32  # Default draw size
    confidence: str = "fallback"  # Always mark as fallback

# Draw size overrides for known venues with non-standard draws
DRAW_OVERRIDES = {
    "winston-salem": 64,
    "winston salem": 64,
    "atp finals": 8,
    "tour finals": 8,
    "masters cup": 8,
    "year-end championships": 8,
    # Add more as needed
}

def parse_level_from_title(title: str) -> str:
    """
    Parse tournament level from Bovada title and map to Sackmann codes.
    
    Returns:
        Sackmann level code: G, M, A, C, 25, 15
    """
    title_lower = title.lower()
    
    # Grand Slams
    if any(slam in title_lower for slam in ["australian open", "roland garros", "french open", 
                                           "wimbledon", "us open", "grand slam"]):
        return "G"
    
    # Masters 1000
    if any(masters in title_lower for masters in ["masters", "1000", "indian wells", "miami open",
                                                 "monte carlo", "madrid", "rome", "canada", "toronto",
                                                 "montreal", "cincinnati", "shanghai", "paris masters"]):
        return "M"
    
    # Challenger
    if "challenger" in title_lower:
        return "C"
    
    # ITF Futures M25
    if re.search(r"\bm25\b|\b25k\b", title_lower):
        return "25"
    
    # ITF Futures M15  
    if re.search(r"\bm15\b|\b15k\b", title_lower):
        return "15"
    
    # ATP (500/250) - default for most tournaments
    return "A"

def parse_draw_size(title: str, level: str) -> int:
    """
    Parse likely draw size from title and level.
    
    Args:
        title: Tournament title
        level: Parsed level code
        
    Returns:
        Likely draw size
    """
    title_lower = title.lower()
    
    # Check for explicit overrides
    for venue, draw in DRAW_OVERRIDES.items():
        if venue in title_lower:
            return draw
    
    # Level-based defaults
    if level == "G":  # Grand Slams
        return 128
    elif level == "M":  # Masters 1000
        return 64 if "indian wells" in title_lower or "miami" in title_lower else 64
    elif level == "C":  # Challenger
        return 32
    elif level in ["25", "15"]:  # ITF Futures
        return 32
    else:  # ATP 500/250
        return 32

def parse_surface_hint(title: str) -> Optional[str]:
    """
    Try to extract surface hints from tournament title.
    
    Returns:
        Surface hint or None if unknown (safer for training alignment)
    """
    title_lower = title.lower()
    
    # Explicit surface mentions
    if any(clay in title_lower for clay in ["clay", "terre"]):
        return "Clay"
    if any(grass in title_lower for grass in ["grass", "lawn"]):
        return "Grass"
    if any(hard in title_lower for hard in ["hard court", "hard"]):
        return "Hard"
    
    # Venue-based surface knowledge (only add if very confident)
    venue_surfaces = {
        "roland garros": "Clay",
        "french open": "Clay", 
        "wimbledon": "Grass",
        "monte carlo": "Clay",
        "rome": "Clay",
        "madrid": "Clay",  # Note: was hard court before 2009
        "barcelona": "Clay",
        "hamburg": "Clay",
        # Add more only if 100% confident
    }
    
    for venue, surface in venue_surfaces.items():
        if venue in title_lower:
            return surface
    
    # Return None for unknown (safer - lets feature engine set all surface one-hots to 0)
    return None

def get_fallback_tournament_meta(event_name: str) -> FallbackTournamentMeta:
    """
    Generate fallback tournament metadata for unmatched events.
    
    Args:
        event_name: Raw Bovada event name
        
    Returns:
        FallbackTournamentMeta with parsed level/draw and optional surface
    """
    level = parse_level_from_title(event_name)
    draw_size = parse_draw_size(event_name, level)
    surface = parse_surface_hint(event_name)  # May be None
    
    return FallbackTournamentMeta(
        surface=surface,
        level=level,
        draw_size=draw_size,
        confidence="fallback"
    )

def log_fallback_usage(event_name: str, meta: FallbackTournamentMeta, log_path: str = None):
    """
    Log when fallback heuristics are used for monitoring.
    
    Args:
        event_name: The unmatched event name
        meta: The generated fallback metadata
        log_path: Optional custom log path
    """
    import csv
    from datetime import datetime
    from pathlib import Path
    
    if log_path is None:
        log_path = Path(__file__).parent.parent.parent / "data" / "fallback_usage.csv"
    else:
        log_path = Path(log_path)
    
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "event_name": event_name,
        "parsed_level": meta.level,
        "parsed_draw": meta.draw_size,
        "parsed_surface": meta.surface or "unknown",
        "confidence": meta.confidence
    }
    
    # Append to log file
    file_exists = log_path.exists()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(log_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=log_entry.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(log_entry)

# Test function
if __name__ == "__main__":
    test_events = [
        "ITF Men's - ITF M15 Hurghada (12)",
        "Challenger - Biella (11)", 
        "ATP Masters 1000 - Indian Wells",
        "Wimbledon - Men's Singles",
        "ITF Men's - ITF M25 Cuiaba (3)",
        "Unknown City Tournament"
    ]
    
    print("FALLBACK HEURISTICS TEST:")
    print("=" * 60)
    
    for event in test_events:
        meta = get_fallback_tournament_meta(event)
        print(f"{event:<35} → Level:{meta.level}, Draw:{meta.draw_size}, Surface:{meta.surface or 'unknown'}")