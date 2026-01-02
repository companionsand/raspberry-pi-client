"""Detection module for wake word and human presence detection"""

from .wake_word import WakeWordDetector
from .presence import HumanPresenceDetector
from .activity import ActivityMonitor  # Legacy - not currently used, kept for reference

__all__ = ["WakeWordDetector", "HumanPresenceDetector", "ActivityMonitor"]

