#!/usr/bin/env python3
"""
Radio Stream Test Script

Tests the full radio architecture: StationRegistry → MusicPlayer → mpv

Usage:
    python test_radio.py "jazz"           # Play jazz station
    python test_radio.py "rock music"     # Play rock station
    python test_radio.py "Frank Sinatra"  # Search for station
    python test_radio.py                  # Play default station
    python test_radio.py --stats          # Show cache statistics
    python test_radio.py --refresh        # Refresh cache from API

Examples:
    python test_radio.py "jazz"
    python test_radio.py "classical music"
    python test_radio.py "BBC Radio"
"""

import os
import sys
import time
import shutil
import signal

# Add project root to path for imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)


def check_mpv() -> bool:
    """Check if mpv is installed"""
    return shutil.which("mpv") is not None


def show_stats():
    """Show cache statistics"""
    from lib.music.stations import StationRegistry, CACHE_FILE
    
    print("\n" + "=" * 60)
    print("CACHE STATISTICS")
    print("=" * 60)
    print()
    
    registry = StationRegistry()
    stats = registry.get_cache_stats()
    
    if not stats:
        print("  Cache is empty!")
        print()
        print("  Run this to populate cache:")
        print("    python scripts/update_radio_cache.py")
    else:
        print(f"  Cache file: {CACHE_FILE}")
        print()
        total = 0
        for genre, count in sorted(stats.items()):
            print(f"    {genre:15} {count:3} stations")
            total += count
        print()
        print(f"  Total: {total} stations")
    
    print()
    print("=" * 60)


def refresh_cache():
    """Refresh cache from Radio Browser API"""
    print("\n" + "=" * 60)
    print("REFRESHING CACHE")
    print("=" * 60)
    print()
    
    from lib.music.radio_browser import RadioBrowserClient
    from lib.music.stations import CACHE_FILE
    import json
    
    client = RadioBrowserClient()
    cache = {}
    
    genres = ["jazz", "classical", "rock", "pop", "country", "blues", "electronic", "ambient"]
    
    # Fetch popular
    print("Fetching popular...", end=" ", flush=True)
    try:
        popular = client.get_top_stations(limit=20)
        cache["popular"] = [s.to_dict() for s in popular]
        print(f"✓ {len(cache['popular'])} stations")
    except Exception as e:
        print(f"✗ {e}")
    
    # Fetch genres
    for genre in genres:
        print(f"Fetching {genre}...", end=" ", flush=True)
        try:
            stations = client.search_by_tag(genre, limit=15)
            cache[genre] = [s.to_dict() for s in stations]
            print(f"✓ {len(cache[genre])} stations")
        except Exception as e:
            print(f"✗ {e}")
        time.sleep(0.3)  # Rate limiting
    
    # Save
    try:
        with open(CACHE_FILE, 'w') as f:
            json.dump(cache, f, indent=2)
        total = sum(len(v) for v in cache.values())
        print()
        print(f"✓ Saved {total} stations to {CACHE_FILE}")
    except Exception as e:
        print(f"✗ Failed to save: {e}")
    
    print()
    print("=" * 60)


def play_music(query: str = None):
    """Play music with the full architecture"""
    from lib.music.player import MusicPlayer
    
    print("\n" + "=" * 60)
    if query:
        print(f"PLAYING: {query}")
    else:
        print("PLAYING: Default station")
    print("=" * 60)
    print()
    
    player = MusicPlayer()
    
    # Handle Ctrl+C gracefully
    def signal_handler(sig, frame):
        print("\n")
        player.stop()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    # Start playback
    if query:
        # Try as genre first, then as search query
        success = player.play_genre(query)
    else:
        success = player.play_default()
    
    if not success:
        print("\n✗ Failed to start playback")
        sys.exit(1)
    
    # Wait and show status
    print()
    print("Press Ctrl+C to stop")
    print()
    
    try:
        while player.is_active():
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    
    player.stop()
    print()
    print("=" * 60)


def main():
    """Main entry point"""
    # Check mpv
    if not check_mpv():
        print("✗ mpv not found!")
        print()
        print("Install mpv:")
        print("  Mac:   brew install mpv")
        print("  Linux: sudo apt install mpv")
        sys.exit(1)
    
    # Parse arguments
    if len(sys.argv) < 2:
        # No arguments - play default
        play_music()
        return
    
    arg = sys.argv[1]
    
    if arg == "--stats" or arg == "-s":
        show_stats()
    elif arg == "--refresh" or arg == "-r":
        refresh_cache()
    elif arg == "--help" or arg == "-h":
        print(__doc__)
    else:
        # Treat as genre/query
        query = " ".join(sys.argv[1:])
        play_music(query)


if __name__ == "__main__":
    main()
