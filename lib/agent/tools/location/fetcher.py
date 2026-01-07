"""Location fetcher using WiFi triangulation via Google Geolocation API"""

import asyncio
import json
import os
from typing import Optional, Dict
from .wifi_location import fetch_location_from_wifi


# Cache file path
CACHE_FILE = "/tmp/device_location_cache.json"


async def fetch_location(
    google_api_key: Optional[str] = None,
    interface: str = "wlan0",
    timeout: float = 10.0
) -> Optional[Dict[str, any]]:
    """
    Fetch current location using WiFi triangulation
    
    Args:
        google_api_key: Google API key (required for WiFi geolocation)
        interface: WiFi interface name (default: wlan0)
        timeout: Request timeout in seconds (default: 10.0)
    
    Returns:
        Dictionary with location data or None if fetch fails:
        {
            'latitude': 28.5355,
            'longitude': 77.3910,
            'accuracy': 25.0,
            'city': 'Noida',
            'state': 'Uttar Pradesh',
            'country': 'India'
        }
    """
    # If no API key, try to load cached location
    if not google_api_key:
        return _load_cached_location()
    
    try:
        # Fetch location via WiFi triangulation
        location = await fetch_location_from_wifi(
            google_api_key=google_api_key,
            interface=interface,
            timeout=timeout
        )
        
        if location:
            # Cache the location for future use
            _save_cached_location(location)
            return location
        else:
            # If fetch fails, try cached location
            return _load_cached_location()
            
    except Exception:
        # On any error, try cached location
        return _load_cached_location()


def _save_cached_location(location: Dict) -> None:
    """Save location to cache file"""
    try:
        with open(CACHE_FILE, 'w') as f:
            json.dump(location, f)
    except Exception:
        pass  # Silently fail if can't write cache


def _load_cached_location() -> Optional[Dict]:
    """Load location from cache file"""
    try:
        if os.path.exists(CACHE_FILE):
            with open(CACHE_FILE, 'r') as f:
                return json.load(f)
    except Exception:
        pass  # Silently fail if can't read cache
    
    return None
