"""
Setup Module for Raspberry Pi Client

Handles WiFi access point creation, network scanning, and connection setup
when the device doesn't have internet connectivity.
"""

from .manager import SetupManager
from .startup import run_startup_sequence
__all__ = ['SetupManager', 'run_startup_sequence']

