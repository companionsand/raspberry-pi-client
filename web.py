#!/usr/bin/env python3
"""
Kin AI Web Dashboard - Real-time Audio and Signal Visualization
================================================================

A web-based interface for visualizing the Kin AI conversational system.
Accessible from any device on the same network via http://kin.local:8080

Features:
- Real-time spectrograms of audio streams (computed client-side)
- Scalar signal plots (VAD probability, RMS levels, YAMNET scores)
- Text event timeline (wake word, conversation events, transcripts)
- Wake word injection for testing

Usage:
    uv run web.py

Environment Variables:
    WEB_HOSTNAME: mDNS hostname (default: "kin" -> kin.local)
    WEB_PORT: Server port (default: 8080)

Requirements:
    - fastapi
    - uvicorn
    - zeroconf
    - All Kin AI dependencies
"""

import asyncio
import logging
import sys
import threading

# Configure logging before other imports
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    force=True,
)


def main():
    """Main entry point for the web dashboard."""
    # Check for required dependencies
    try:
        import fastapi
        import uvicorn
        import zeroconf
    except ImportError as e:
        print("=" * 60)
        print("ERROR: Missing web dashboard dependencies")
        print("=" * 60)
        print(f"\nMissing: {e}")
        print("\nTo install web dashboard dependencies:")
        print("  uv sync")
        print("=" * 60)
        sys.exit(1)
    
    # Import engine and web server
    from lib.config import Config
    from lib.engine import KinEngine
    from lib.signals import TextSignal
    from lib.web import WebDashboardServer
    
    print("=" * 60)
    print("🌐 Kin AI - Web Dashboard")
    print("=" * 60)
    print()
    print("Features:")
    print("  • Real-time spectrograms (client-side FFT)")
    print("  • Scalar signal plots (VAD, RMS, YAMNET)")
    print("  • Event timeline (wake word, conversation, transcripts)")
    print("  • Wake word injection button for testing")
    print()
    print(f"Access the dashboard at:")
    print(f"  • http://{Config.WEB_HOSTNAME}.local:{Config.WEB_PORT}")
    print(f"  • http://localhost:{Config.WEB_PORT}")
    print()
    print("=" * 60)
    
    # Create engine
    engine = KinEngine()
    
    # Start signal bus early (so subscriptions work before engine.run())
    engine.signal_bus.start()
    
    # Also subscribe for CLI output (print events to console)
    def cli_handler(signal: TextSignal):
        """Print signals to console as backup."""
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
        callback=cli_handler
    )
    
    # Create web server
    server = WebDashboardServer(engine)
    
    # Run engine in background thread
    def run_engine():
        """Run the engine in a background thread."""
        try:
            asyncio.run(engine.run())
        except Exception as e:
            print(f"✗ Engine error: {e}")
    
    engine_thread = threading.Thread(target=run_engine, daemon=True)
    engine_thread.start()
    
    # Run web server (blocking)
    try:
        asyncio.run(server.start())
    except KeyboardInterrupt:
        print("\n🛑 Shutting down web dashboard...")
    
    sys.exit(0)


if __name__ == "__main__":
    main()

