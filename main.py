#!/usr/bin/env python3
"""
Kin AI Raspberry Pi Client (v2)
================================
Modular, refactored client with ALSA-only audio, full telemetry, and LED feedback.

Features:
- Wake word detection using Porcupine
- Real-time conversation via ElevenLabs WebSocket API
- ReSpeaker hardware AEC (with fallback to default devices)
- Communication with conversation-orchestrator via WebSocket
- Device authentication via provisioned Ed25519 credentials
- OpenTelemetry observability (traces, spans, logs - no metrics)
- LED visual feedback for device states
- Setup mode for device pairing and WiFi configuration when needed

Usage:
    python main.py

Requirements:
    - Raspberry Pi OS with ALSA
    - Audio device (ReSpeaker recommended, fallback to any mic/speaker)
    - Environment variables: see Config class
"""

import asyncio
import logging

# Configure logging BEFORE any other imports (critical for telemetry)
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    force=True,
)

from lib.config import Config

# Import telemetry (optional - graceful degradation)
try:
    from lib.telemetry import setup_stdout_redirect, cleanup_stdout_redirect
    TELEMETRY_AVAILABLE = True
except ImportError:
    TELEMETRY_AVAILABLE = False
    def setup_stdout_redirect(name=None):
        return False
    def cleanup_stdout_redirect():
        pass

# Setup stdout/stderr redirection EARLY (before any print statements)
# This captures all print() output and sends it to OTEL/BetterStack
# Skip in MAC_MODE to avoid unnecessary telemetry overhead
if TELEMETRY_AVAILABLE and not Config.MAC_MODE:
    setup_stdout_redirect("raspberry-pi-client")


def main():
    """Main application entry point using KinEngine."""
    from lib.engine import KinEngine
    from lib.signals import TextSignal
    
    # Create the engine
    engine = KinEngine()
    
    # Start signal bus dispatcher
    engine.signal_bus.start()
    
    # Subscribe to TextSignals for console output
    def cli_text_handler(signal: TextSignal):
        """Print TextSignals to console with appropriate formatting."""
        prefixes = {
            "debug": "🔍",
            "info": "ℹ️ ",
            "warning": "⚠️ ",
            "error": "✗",
        }
        prefix = prefixes.get(signal.level, "•")
        print(f"{prefix} [{signal.category}] {signal.message}")
    
    engine.signal_bus.subscribe(
        signal_type=TextSignal,
        callback=cli_text_handler
    )
    
    # Run the engine
    try:
        asyncio.run(engine.run())
    finally:
        # Cleanup stdout/stderr redirection
        if TELEMETRY_AVAILABLE:
            cleanup_stdout_redirect()


if __name__ == "__main__":
    main()
