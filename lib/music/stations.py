"""
Station Registry

Provides station lookup from local cache with curated fallback.
Expects clean, normalized inputs from upstream (intent classifier or LLM).

Resolution order:
1. Local cache (~/.kin_radio_cache.json) - sorted by quality (votes)
2. Curated fallback - reliable, manually tested stations
"""

import os
import json
from typing import Optional, Dict, List
from dataclasses import dataclass


CACHE_FILE = os.path.expanduser("~/.kin_radio_cache.json")


@dataclass
class Station:
    """Radio station for playback"""
    name: str
    url: str
    tags: List[str]
    votes: int
    bitrate: int
    source: str  # "cache" or "curated"
    
    def __str__(self) -> str:
        return f"{self.name} ({self.source})"
    
    @property
    def quality_score(self) -> float:
        """Quality score based on votes and bitrate"""
        # Votes are primary, bitrate is secondary
        # Normalize bitrate: 128=1.0, 192=1.5, 320=2.5
        bitrate_factor = min(self.bitrate / 128, 2.5) if self.bitrate > 0 else 1.0
        return self.votes * bitrate_factor


# Curated fallback stations - reliable, tested, no geo-restrictions
CURATED_STATIONS: Dict[str, List[Station]] = {
    "default": [
        Station("SomaFM Groove Salad", "https://ice2.somafm.com/groovesalad-128-mp3", ["ambient", "chill"], 1000, 128, "curated"),
        Station("SomaFM Secret Agent", "https://ice2.somafm.com/secretagent-128-mp3", ["lounge", "spy"], 800, 128, "curated"),
    ],
    "jazz": [
        Station("SomaFM Secret Agent", "https://ice2.somafm.com/secretagent-128-mp3", ["jazz", "lounge"], 800, 128, "curated"),
    ],
    "classical": [
        Station("SomaFM Drone Zone", "https://ice2.somafm.com/dronezone-128-mp3", ["ambient", "classical"], 700, 128, "curated"),
    ],
    "ambient": [
        Station("SomaFM Groove Salad", "https://ice2.somafm.com/groovesalad-128-mp3", ["ambient", "chill"], 1000, 128, "curated"),
        Station("SomaFM Deep Space One", "https://ice1.somafm.com/deepspaceone-128-mp3", ["ambient", "space"], 600, 128, "curated"),
    ],
    "electronic": [
        Station("SomaFM DEF CON Radio", "https://ice1.somafm.com/defcon-128-mp3", ["electronic", "hacker"], 500, 128, "curated"),
    ],
    "rock": [
        Station("Radio Bob", "https://streams.radiobob.de/bob-live/mp3-192/mediaplayer", ["rock"], 400, 192, "curated"),
    ],
}


class StationRegistry:
    """
    Station lookup from cache with curated fallback.
    Expects clean inputs (upstream handles semantic mapping).
    
    Usage:
        registry = StationRegistry()
        station = registry.find_by_genre("jazz")      # Best jazz station
        station = registry.find_by_query("BBC Radio") # Search by name
        station = registry.get_default()              # Best popular station
    """
    
    def __init__(self):
        self._cache = self._load_cache()
    
    def _load_cache(self) -> Dict[str, List[Dict]]:
        """Load cached stations from file"""
        if not os.path.exists(CACHE_FILE):
            return {}
        try:
            with open(CACHE_FILE, 'r') as f:
                return json.load(f)
        except Exception:
            return {}
    
    def reload_cache(self):
        """Reload cache from disk"""
        self._cache = self._load_cache()
    
    def _station_from_cache(self, data: Dict) -> Station:
        """Create Station from cache entry"""
        return Station(
            name=data.get("name", "Unknown"),
            url=data.get("url", ""),
            tags=data.get("tags", []),
            votes=data.get("votes", 0),
            bitrate=data.get("bitrate", 128),
            source="cache"
        )
    
    def _get_best_station(self, stations: List[Dict]) -> Optional[Station]:
        """Get highest quality station from list (by votes * bitrate)"""
        if not stations:
            return None
        
        # Convert to Station objects and sort by quality
        station_objs = [self._station_from_cache(s) for s in stations]
        station_objs.sort(key=lambda s: s.quality_score, reverse=True)
        
        return station_objs[0]
    
    def find_by_genre(self, genre: str) -> Optional[Station]:
        """
        Find best station by genre tag.
        Returns highest-quality station from matching genre.
        """
        genre_lower = genre.lower().strip()
        
        # 1. Try cache first (best quality station)
        if genre_lower in self._cache and self._cache[genre_lower]:
            return self._get_best_station(self._cache[genre_lower])
        
        # 2. Try curated fallback (first = best)
        if genre_lower in CURATED_STATIONS:
            return CURATED_STATIONS[genre_lower][0]
        
        # 3. Return default
        return self.get_default()
    
    def find_by_query(self, query: str) -> Optional[Station]:
        """
        Find station by name search.
        Returns best matching station or falls back to genre search.
        """
        query_lower = query.lower()
        matches = []
        
        # Search all cached stations for name match
        for stations in self._cache.values():
            for entry in stations:
                if query_lower in entry.get("name", "").lower():
                    matches.append(entry)
        
        # Return best match
        if matches:
            return self._get_best_station(matches)
        
        # No match - try as genre
        return self.find_by_genre(query)
    
    def get_default(self) -> Station:
        """Get best default station (highest quality popular)"""
        # Try popular stations from cache first
        if "popular" in self._cache and self._cache["popular"]:
            return self._get_best_station(self._cache["popular"])
        
        # Fall back to curated default
        return CURATED_STATIONS["default"][0]
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics"""
        return {genre: len(stations) for genre, stations in self._cache.items()}
