"""
Setup Module for Raspberry Pi Client

Handles WiFi access point creation, network scanning, and connection setup
when the device doesn't have internet connectivity.
"""

from .manager import SetupManager
__all__ = ['SetupManager']

