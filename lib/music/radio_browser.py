"""
Radio Browser API Client

Fetches radio stations from the free Radio Browser API (https://api.radio-browser.info/).
Used by the overnight cache job to populate local station cache.
"""

import random
import requests
from typing import List, Dict, Optional
from dataclasses import dataclass


# Available API servers (randomize to distribute load)
# Per API docs: use DNS lookup in production, but hardcoded list is fine for MVP
API_SERVERS = [
    "https://de2.api.radio-browser.info",
    "https://fi1.api.radio-browser.info",
]


@dataclass
class RadioStation:
    """Radio station data from API"""
    name: str
    url: str
    tags: List[str]
    country: str
    bitrate: int
    votes: int
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            "name": self.name,
            "url": self.url,
            "tags": self.tags,
            "country": self.country,
            "bitrate": self.bitrate,
            "votes": self.votes
        }
    
    @classmethod
    def from_api_response(cls, data: Dict) -> "RadioStation":
        """Create from API response"""
        return cls(
            name=data.get("name", "Unknown"),
            url=data.get("url_resolved") or data.get("url", ""),
            tags=[t.strip() for t in data.get("tags", "").split(",") if t.strip()],
            country=data.get("countrycode", ""),
            bitrate=data.get("bitrate", 0),
            votes=data.get("votes", 0)
        )


class RadioBrowserClient:
    """
    Client for Radio Browser API.
    
    Usage:
        client = RadioBrowserClient()
        jazz_stations = client.search_by_tag("jazz", limit=20)
        bbc_stations = client.search_by_name("BBC Radio", limit=10)
    """
    
    def __init__(self, timeout: int = 10):
        self.base_url = random.choice(API_SERVERS)
        self.timeout = timeout
        self.headers = {"User-Agent": "Kin-AI-Companion/1.0"}
    
    def _get(self, endpoint: str, params: Optional[Dict] = None) -> List[Dict]:
        """Make GET request to API"""
        url = f"{self.base_url}{endpoint}"
        response = requests.get(
            url,
            params=params,
            headers=self.headers,
            timeout=self.timeout
        )
        response.raise_for_status()
        return response.json()
    
    def search_by_tag(self, tag: str, limit: int = 50) -> List[RadioStation]:
        """
        Search stations by genre tag (e.g., 'jazz', 'classical', 'rock').
        Returns stations sorted by votes (popularity).
        """
        data = self._get(
            f"/json/stations/bytag/{tag}",
            params={
                "limit": limit,
                "order": "votes",
                "reverse": "true",
                "hidebroken": "true"
            }
        )
        return [RadioStation.from_api_response(s) for s in data if s.get("url_resolved") or s.get("url")]
    
    def search_by_name(self, name: str, limit: int = 20) -> List[RadioStation]:
        """
        Search stations by name (e.g., 'BBC Radio', 'Frank Sinatra').
        Useful for artist or specific station searches.
        """
        data = self._get(
            "/json/stations/search",
            params={
                "name": name,
                "limit": limit,
                "order": "votes",
                "reverse": "true",
                "hidebroken": "true"
            }
        )
        return [RadioStation.from_api_response(s) for s in data if s.get("url_resolved") or s.get("url")]
    
    def get_top_stations(self, limit: int = 100) -> List[RadioStation]:
        """Get most popular stations overall (sorted by votes)"""
        data = self._get(f"/json/stations/topvote/{limit}")
        return [RadioStation.from_api_response(s) for s in data if s.get("url_resolved") or s.get("url")]


def verify_stream(url: str, timeout: float = 3.0) -> bool:
    """
    Quick check if a stream URL is accessible.
    Used during cache refresh to filter out broken streams.
    """
    try:
        response = requests.head(url, timeout=timeout, allow_redirects=True)
        return response.status_code < 400
    except Exception:
        return False

