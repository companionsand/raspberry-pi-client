"""
SignalPublisher mixin for components that publish signals.

Provides convenience methods to publish TextSignal, ScalarSignal, and AudioSignal
without manually constructing the dataclasses each time.
"""

import time
from typing import TYPE_CHECKING, Optional

from lib.signals.base import AudioSignal, ScalarSignal, TextSignal

if TYPE_CHECKING:
    from lib.signals.bus import SignalBus


class SignalPublisher:
    """
    Mixin class for components that publish signals.
    
    Provides convenience methods for publishing signals with consistent
    source naming and timestamp handling.
    
    Usage:
        class MyComponent(SignalPublisher):
            def __init__(self, signal_bus: SignalBus):
                SignalPublisher.__init__(self, signal_bus, "my_component")
            
            def do_something(self):
                self.publish_text("event", "Something happened")
                self.publish_scalar("metric", 0.5)
    """
    
    def __init__(
        self,
        signal_bus: Optional["SignalBus"] = None,
        source_name: str = ""
    ):
        """
        Initialize the publisher mixin.
        
        Args:
            signal_bus: SignalBus to publish to (can be set later)
            source_name: Name of this component for signal attribution
        """
        self._signal_bus: Optional["SignalBus"] = signal_bus
        self._source_name = source_name
    
    def set_signal_bus(self, signal_bus: "SignalBus") -> None:
        """Set or update the signal bus."""
        self._signal_bus = signal_bus
    
    def set_source_name(self, source_name: str) -> None:
        """Set or update the source name."""
        self._source_name = source_name
    
    @property
    def has_signal_bus(self) -> bool:
        """Check if a signal bus is configured."""
        return self._signal_bus is not None
    
    def publish_text(
        self,
        category: str,
        message: str,
        level: str = "info"
    ) -> bool:
        """
        Publish a text signal.
        
        Args:
            category: Event category (e.g., "wake_word", "conversation")
            message: Human-readable message
            level: Log level ("debug", "info", "warning", "error")
            
        Returns:
            True if published, False if no bus or publish failed
        """
        if self._signal_bus is None:
            return False
        
        signal = TextSignal(
            timestamp=time.monotonic(),
            source=self._source_name,
            category=category,
            message=message,
            level=level
        )
        return self._signal_bus.publish(signal)
    
    def publish_scalar(
        self,
        name: str,
        value: float
    ) -> bool:
        """
        Publish a scalar signal.
        
        Args:
            name: Metric name (e.g., "vad_probability", "input_rms")
            value: Scalar value
            
        Returns:
            True if published, False if no bus or publish failed
        """
        if self._signal_bus is None:
            return False
        
        signal = ScalarSignal(
            timestamp=time.monotonic(),
            source=self._source_name,
            name=name,
            value=value
        )
        return self._signal_bus.publish(signal)
    
    def publish_audio(
        self,
        stream_name: str,
        sample_count: int
    ) -> bool:
        """
        Publish an audio signal (notification of new data in ring buffer).
        
        Args:
            stream_name: Name of the audio stream
            sample_count: Number of samples written
            
        Returns:
            True if published, False if no bus or publish failed
        """
        if self._signal_bus is None:
            return False
        
        signal = AudioSignal(
            timestamp=time.monotonic(),
            source=self._source_name,
            stream_name=stream_name,
            sample_count=sample_count
        )
        return self._signal_bus.publish(signal)

