"""
Signal dataclasses for the unified signal system.

Three types of signals:
- AudioSignal: Continuous audio waveform data (accessed via RingBufferHub)
- ScalarSignal: Numeric time series (VAD probability, YAMNET scores, RMS)
- TextSignal: Discrete text events (wake word, conversation state, transcripts)
"""

import time
from dataclasses import dataclass, field
from enum import Enum


class SignalLevel(Enum):
    """Log level for TextSignals."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass(slots=True)
class Signal:
    """Base class for all signals."""
    timestamp: float = field(default_factory=time.monotonic)
    source: str = ""  # e.g., "wake_word_detector", "audio_manager"
    
    def __post_init__(self):
        if self.timestamp == 0:
            self.timestamp = time.monotonic()


@dataclass(slots=True)
class AudioSignal(Signal):
    """
    Signal indicating new audio data is available in a ring buffer.
    
    The actual audio data is accessed via RingBufferHub.get_window(),
    not stored in this signal (to avoid copying).
    
    Attributes:
        stream_name: Name of the audio stream ("aec_input", "agent_output", "raw_input")
        sample_count: Number of new samples available
    """
    stream_name: str = ""  # "aec_input", "agent_output", "raw_input"
    sample_count: int = 0  # Number of new samples written


@dataclass(slots=True)
class ScalarSignal(Signal):
    """
    Signal for numeric time series data.
    
    Used for continuous metrics like VAD probability, YAMNET scores, audio RMS.
    These are visualized as line graphs (oscilloscope-style).
    
    Attributes:
        name: Identifier for this scalar ("vad_probability", "yamnet_speech", "input_rms")
        value: The scalar value (typically 0.0 - 1.0 for probabilities)
    """
    name: str = ""  # "vad_probability", "yamnet_speech", "input_rms"
    value: float = 0.0


@dataclass(slots=True)
class TextSignal(Signal):
    """
    Signal for discrete text events.
    
    Used for events like wake word detection, conversation state changes,
    transcripts, LED state changes. Visualized as a timeline.
    
    Attributes:
        category: Event category ("wake_word", "conversation", "transcript", "led")
        message: Human-readable message
        level: Severity level (debug, info, warning, error)
    """
    category: str = ""  # "wake_word", "conversation", "transcript", "led"
    message: str = ""
    level: str = "info"  # "debug", "info", "warning", "error"
    
    @property
    def level_enum(self) -> SignalLevel:
        """Get level as enum."""
        try:
            return SignalLevel(self.level)
        except ValueError:
            return SignalLevel.INFO


# Type alias for any signal type
AnySignal = Signal | AudioSignal | ScalarSignal | TextSignal

