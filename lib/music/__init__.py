"""Music playback module for Kin AI client"""

from .player import MusicPlayer, DEFAULT_STREAM_URL
from .stations import StationRegistry, Station

# StopDetector has heavy dependencies (onnxruntime, elevenlabs)
# Import it lazily only when needed
def get_stop_detector():
    """Lazy import of StopDetector to avoid heavy dependencies"""
    from .stop_detector import StopDetector
    return StopDetector

__all__ = [
    "MusicPlayer",
    "StationRegistry", 
    "Station",
    "get_stop_detector",
    "DEFAULT_STREAM_URL"
]
