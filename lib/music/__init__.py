"""Music playback module for Kin AI client"""

from .player import MusicPlayer, DEFAULT_STREAM_URL
from .stations import StationRegistry, Station

# VoiceCommandDetector has heavy dependencies (onnxruntime, elevenlabs)
# Import it lazily only when needed
def get_voice_command_detector():
    """Lazy import of VoiceCommandDetector to avoid heavy dependencies"""
    from .command_detector import VoiceCommandDetector
    return VoiceCommandDetector

def get_music_command():
    """Lazy import of MusicCommand enum"""
    from .command_detector import MusicCommand
    return MusicCommand

# Backwards compatibility
get_stop_detector = get_voice_command_detector

__all__ = [
    "MusicPlayer",
    "StationRegistry", 
    "Station",
    "get_voice_command_detector",
    "get_music_command",
    "get_stop_detector",  # Backwards compatibility
    "DEFAULT_STREAM_URL"
]
