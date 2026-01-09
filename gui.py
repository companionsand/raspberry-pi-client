#!/usr/bin/env python3
"""
Kin AI GUI - Real-time Audio and Signal Visualization
======================================================

A graphical interface for visualizing the Kin AI conversational system.

Features:
- Real-time spectrograms of audio streams (echo-cancelled input, agent output)
- Scalar signal plots (VAD probability, RMS levels, YAMNET scores)
- Text event timeline (wake word, conversation events, transcripts)
- Wake word injection for testing

Usage:
    python gui.py

Requirements:
    - PyQt5
    - pyqtgraph
    - numpy
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
    """Main entry point for the GUI application."""
    # Check for required dependencies
    try:
        from PyQt5.QtWidgets import QApplication
        import pyqtgraph
    except ImportError as e:
        print("=" * 60)
        print("ERROR: Missing GUI dependencies")
        print("=" * 60)
        print(f"\nMissing: {e}")
        print("\nTo install GUI dependencies:")
        print("  uv add PyQt5 pyqtgraph")
        print("=" * 60)
        sys.exit(1)
    
    # Import engine and GUI
    from lib.engine import KinEngine
    from lib.gui.app import KinGUIApp
    from lib.signals import TextSignal
    
    print("=" * 60)
    print("🎙️  Kin AI - Audio & Signal Visualization")
    print("=" * 60)
    print()
    print("Features:")
    print("  • Real-time spectrograms (FFT-based frequency analysis)")
    print("  • Scalar signal plots (VAD, RMS, YAMNET)")
    print("  • Event timeline (wake word, conversation, transcripts)")
    print("  • Wake word injection button for testing")
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
    
    # Create Qt application
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle("Fusion")
    
    # Set a sensible default font (avoid Monospace warning on macOS)
    from PyQt5.QtGui import QFont
    if sys.platform == "darwin":
        app.setFont(QFont("Monaco", 12))
    else:
        app.setFont(QFont("DejaVu Sans Mono", 10))
    
    # Create and show main window
    window = KinGUIApp(engine)
    window.show()
    
    # Run engine in background thread
    def run_engine():
        """Run the engine in a background thread."""
        try:
            asyncio.run(engine.run())
        except Exception as e:
            print(f"✗ Engine error: {e}")
    
    engine_thread = threading.Thread(target=run_engine, daemon=True)
    engine_thread.start()
    
    # Run Qt event loop
    exit_code = app.exec_()
    
    # Cleanup
    print("\n🛑 Shutting down GUI...")
    
    sys.exit(exit_code)


if __name__ == "__main__":
    main()

