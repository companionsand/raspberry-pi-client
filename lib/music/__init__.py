"""Music playback module for Kin AI client"""

from .player import MusicPlayer, DEFAULT_STREAM_URL, BBC_RADIO_6_MUSIC_URL
from .stop_detector import StopDetector

__all__ = ["MusicPlayer", "StopDetector", "DEFAULT_STREAM_URL", "BBC_RADIO_6_MUSIC_URL"]

