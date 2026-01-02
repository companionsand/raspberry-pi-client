"""
Setup Module for Raspberry Pi Client

Handles device setup including WiFi access point creation, network scanning,
connection setup, and device pairing when the device doesn't have internet
connectivity or is unpaired.
"""

from .manager import SetupManager

__all__ = ['SetupManager']

