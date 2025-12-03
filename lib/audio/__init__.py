"""Audio device detection and LED control"""

from .device_detection import get_audio_devices
from .led_controller import LEDController

__all__ = ["get_audio_devices", "LEDController"]

