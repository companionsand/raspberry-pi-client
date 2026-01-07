#!/usr/bin/env python3
"""
Radio Cache Update Script

Fetches radio stations from Radio Browser API and saves verified ones to local cache.
Run this overnight via cron to keep the cache fresh.

Usage:
    python scripts/update_radio_cache.py

Cron setup (run at 3am daily):
    0 3 * * * /path/to/venv/bin/python /path/to/scripts/update_radio_cache.py

The cache file is saved to ~/.kin_radio_cache.json
"""

import os
import sys
import json
import time

# Add project root to path for imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from lib.music.radio_browser import RadioBrowserClient, verify_stream

# Cache configuration
CACHE_FILE = os.path.expanduser("~/.kin_radio_cache.json")
STATIONS_PER_GENRE = 15  # How many stations to cache per genre
VERIFY_TIMEOUT = 3.0     # Seconds to wait for stream verification

# Genres to cache (common user requests)
GENRES = [
    "jazz",
    "classical", 
    "rock",
    "pop",
    "country",
    "blues",
    "electronic",
    "ambient",
    "folk",
    "soul",
]


def fetch_genre(client: RadioBrowserClient, genre: str, verify: bool = True) -> list:
    """Fetch and optionally verify stations for a genre"""
    print(f"  {genre}...", end=" ", flush=True)
    
    try:
        stations = client.search_by_tag(genre, limit=50)
    except Exception as e:
        print(f"✗ API error: {e}")
        return []
    
    if not verify:
        # Skip verification, take top N
        result = [s.to_dict() for s in stations[:STATIONS_PER_GENRE]]
        print(f"✓ {len(result)} stations (unverified)")
        return result
    
    # Verify streams and collect working ones
    verified = []
    for station in stations:
        if verify_stream(station.url, timeout=VERIFY_TIMEOUT):
            verified.append(station.to_dict())
            if len(verified) >= STATIONS_PER_GENRE:
                break
    
    print(f"✓ {len(verified)} verified")
    return verified


def fetch_popular(client: RadioBrowserClient, limit: int = 30) -> list:
    """Fetch top popular stations"""
    print("  popular...", end=" ", flush=True)
    
    try:
        stations = client.get_top_stations(limit=limit * 2)
        # Take top N without verification (popular = likely working)
        result = [s.to_dict() for s in stations[:limit]]
        print(f"✓ {len(result)} stations")
        return result
    except Exception as e:
        print(f"✗ API error: {e}")
        return []


def main():
    """Main entry point"""
    start_time = time.time()
    print("=" * 60)
    print("RADIO CACHE UPDATE")
    print("=" * 60)
    print()
    
    client = RadioBrowserClient()
    cache = {}
    
    # Fetch popular stations (for "Play music" default)
    print("Fetching stations:")
    cache["popular"] = fetch_popular(client)
    
    # Fetch by genre
    for genre in GENRES:
        cache[genre] = fetch_genre(client, genre, verify=False)  # Skip verify for speed
        time.sleep(0.5)  # Rate limiting courtesy
    
    # Save cache
    print()
    try:
        with open(CACHE_FILE, 'w') as f:
            json.dump(cache, f, indent=2)
        
        total = sum(len(v) for v in cache.values())
        elapsed = time.time() - start_time
        
        print("=" * 60)
        print(f"✓ Cache saved: {CACHE_FILE}")
        print(f"  Total stations: {total}")
        print(f"  Genres: {len(cache)}")
        print(f"  Time: {elapsed:.1f}s")
        print("=" * 60)
        
    except Exception as e:
        print(f"✗ Failed to save cache: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

