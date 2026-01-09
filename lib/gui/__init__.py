"""
GUI components for Kin AI visualization.

This module provides PyQt5-based widgets for real-time visualization of:
- Audio spectrograms (FFT-based frequency analysis)
- Scalar signals (VAD probability, YAMNET scores, RMS)
- Text events (wake word, conversation state, transcripts)
"""

from lib.gui.app import KinGUIApp

__all__ = ["KinGUIApp"]

