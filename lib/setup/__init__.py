"""
Setup Module for Raspberry Pi Client

Handles device setup including WiFi access point creation, network scanning,
connection setup, and device pairing when the device doesn't have internet
connectivity or is unpaired.
"""

from .manager import SetupManager
from .connectivity import ConnectivityChecker
from .startup import run_startup_sequence, StartupResult, StartupFailureReason

__all__ = [
    'SetupManager',
    'ConnectivityChecker',
    'run_startup_sequence',
    'StartupResult',
    'StartupFailureReason',
]

