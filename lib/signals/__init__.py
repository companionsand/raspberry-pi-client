"""
Signal system for unified audio and event streaming.

This module provides:
- Signal dataclasses (AudioSignal, ScalarSignal, TextSignal)
- RingBuffer for zero-copy audio access
- SignalBus for pub/sub event distribution
- SignalPublisher mixin for components
"""

from lib.signals.base import (
    Signal,
    AudioSignal,
    ScalarSignal,
    TextSignal,
    SignalLevel,
    AnySignal,
)
from lib.signals.ring_buffer import RingBuffer
from lib.signals.bus import SignalBus
from lib.signals.publisher import SignalPublisher

__all__ = [
    # Base signals
    "Signal",
    "AudioSignal",
    "ScalarSignal",
    "TextSignal",
    "SignalLevel",
    "AnySignal",
    # Infrastructure
    "RingBuffer",
    "SignalBus",
    "SignalPublisher",
]

