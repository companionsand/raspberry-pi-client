"""Detection module for wake word and human presence detection"""

from .wake_word import WakeWordDetector
from .presence import HumanPresenceDetector
# ActivityMonitor is legacy/unused - kept in .activity.py for reference only

__all__ = ["WakeWordDetector", "HumanPresenceDetector"]

