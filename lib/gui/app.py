"""
Main GUI application for Kin AI visualization.

Provides real-time visualization of audio streams and signals from KinEngine.
"""

import asyncio
import sys
import threading
from typing import TYPE_CHECKING

from PyQt5.QtCore import QTimer, Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from lib.signals import TextSignal, ScalarSignal

if TYPE_CHECKING:
    from lib.engine import KinEngine

# Import GUI widgets
from lib.gui.spectrogram import SpectrogramWidget
from lib.gui.scalar_plot import ScalarPlotWidget
from lib.gui.text_timeline import TextTimelineWidget


class KinGUIApp(QMainWindow):
    """
    Main GUI window for Kin AI visualization.
    
    Features:
    - Two configurable spectrogram displays (audio streams)
    - Scalar signal plot (VAD probability, RMS, etc.)
    - Text event timeline
    - Wake word injection button
    """
    
    # Qt signals for thread-safe updates from SignalBus callbacks
    text_signal_received = pyqtSignal(object)
    scalar_signal_received = pyqtSignal(object)
    
    AUDIO_STREAMS = [
        ("aec_input", "Echo-Cancelled Input"),
        ("agent_output", "Agent Output"),
        ("raw_input", "Raw Input (6ch)"),
    ]
    
    def __init__(self, engine: "KinEngine", parent=None):
        """
        Initialize the GUI application.
        
        Args:
            engine: KinEngine instance to visualize
            parent: Parent widget (optional)
        """
        super().__init__(parent)
        
        self.engine = engine
        self._setup_ui()
        self._setup_signal_subscriptions()
        self._setup_timers()
        
        self.setWindowTitle("Kin AI - Audio & Signal Visualization")
        self.resize(1200, 800)
    
    def _setup_ui(self) -> None:
        """Setup the UI layout."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout(central_widget)
        
        # Control bar at top
        control_bar = self._create_control_bar()
        main_layout.addWidget(control_bar)
        
        # Main content splitter (vertical)
        splitter = QSplitter(Qt.Vertical)
        main_layout.addWidget(splitter, stretch=1)
        
        # Top section: Two spectrograms side by side
        spectrogram_container = QWidget()
        spec_layout = QHBoxLayout(spectrogram_container)
        spec_layout.setContentsMargins(0, 0, 0, 0)
        
        # Left spectrogram
        self.spectrogram_left = SpectrogramWidget(
            self.engine,
            stream_name="aec_input",
            title="Echo-Cancelled Input"
        )
        spec_layout.addWidget(self.spectrogram_left)
        
        # Right spectrogram
        self.spectrogram_right = SpectrogramWidget(
            self.engine,
            stream_name="agent_output",
            title="Agent Output"
        )
        spec_layout.addWidget(self.spectrogram_right)
        
        splitter.addWidget(spectrogram_container)
        
        # Middle section: Scalar plot
        self.scalar_plot = ScalarPlotWidget()
        splitter.addWidget(self.scalar_plot)
        
        # Bottom section: Text timeline
        self.text_timeline = TextTimelineWidget()
        splitter.addWidget(self.text_timeline)
        
        # Set initial splitter sizes (40% spectrograms, 30% scalars, 30% text)
        splitter.setSizes([400, 200, 200])
    
    def _create_control_bar(self) -> QWidget:
        """Create the control bar with stream selectors and buttons."""
        control_bar = QWidget()
        layout = QHBoxLayout(control_bar)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Left spectrogram selector
        layout.addWidget(QLabel("Left Stream:"))
        self.stream_selector_left = QComboBox()
        for stream_id, stream_name in self.AUDIO_STREAMS:
            self.stream_selector_left.addItem(stream_name, stream_id)
        self.stream_selector_left.setCurrentIndex(0)  # aec_input
        self.stream_selector_left.currentIndexChanged.connect(self._on_left_stream_changed)
        layout.addWidget(self.stream_selector_left)
        
        layout.addSpacing(20)
        
        # Right spectrogram selector
        layout.addWidget(QLabel("Right Stream:"))
        self.stream_selector_right = QComboBox()
        for stream_id, stream_name in self.AUDIO_STREAMS:
            self.stream_selector_right.addItem(stream_name, stream_id)
        self.stream_selector_right.setCurrentIndex(1)  # agent_output
        self.stream_selector_right.currentIndexChanged.connect(self._on_right_stream_changed)
        layout.addWidget(self.stream_selector_right)
        
        layout.addStretch()
        
        # Wake word injection button
        self.wake_word_button = QPushButton("🎤 Inject Wake Word")
        self.wake_word_button.clicked.connect(self._on_inject_wake_word)
        layout.addWidget(self.wake_word_button)
        
        return control_bar
    
    def _setup_signal_subscriptions(self) -> None:
        """Subscribe to signals from KinEngine."""
        # Connect Qt signals for thread-safe updates
        self.text_signal_received.connect(self._handle_text_signal)
        self.scalar_signal_received.connect(self._handle_scalar_signal)
        
        # Subscribe to TextSignals
        self.engine.signal_bus.subscribe(
            signal_type=TextSignal,
            callback=lambda s: self.text_signal_received.emit(s)
        )
        
        # Subscribe to ScalarSignals
        self.engine.signal_bus.subscribe(
            signal_type=ScalarSignal,
            callback=lambda s: self.scalar_signal_received.emit(s)
        )
    
    def _setup_timers(self) -> None:
        """Setup update timers for visualizations."""
        # Spectrogram update timer (30 FPS)
        self.spectrogram_timer = QTimer()
        self.spectrogram_timer.timeout.connect(self._update_spectrograms)
        self.spectrogram_timer.start(33)  # ~30 FPS
    
    def _on_left_stream_changed(self, index: int) -> None:
        """Handle left stream selector change."""
        stream_id = self.stream_selector_left.currentData()
        stream_name = self.stream_selector_left.currentText()
        self.spectrogram_left.set_stream(stream_id, stream_name)
    
    def _on_right_stream_changed(self, index: int) -> None:
        """Handle right stream selector change."""
        stream_id = self.stream_selector_right.currentData()
        stream_name = self.stream_selector_right.currentText()
        self.spectrogram_right.set_stream(stream_id, stream_name)
    
    def _on_inject_wake_word(self) -> None:
        """Handle wake word injection button click."""
        self.engine.inject_wake_word()
    
    def _update_spectrograms(self) -> None:
        """Update spectrogram displays."""
        self.spectrogram_left.update_display()
        self.spectrogram_right.update_display()
    
    def _handle_text_signal(self, signal: TextSignal) -> None:
        """Handle incoming TextSignal (thread-safe via Qt signal)."""
        self.text_timeline.add_signal(signal)
    
    def _handle_scalar_signal(self, signal: ScalarSignal) -> None:
        """Handle incoming ScalarSignal (thread-safe via Qt signal)."""
        self.scalar_plot.add_signal(signal)


def run_gui_with_engine(engine: "KinEngine") -> None:
    """
    Run the GUI application with the given engine.
    
    This function starts the Qt event loop and runs the engine
    in a background thread.
    
    Args:
        engine: KinEngine instance to visualize
    """
    app = QApplication(sys.argv)
    
    # Create main window
    window = KinGUIApp(engine)
    window.show()
    
    # Run engine in background thread
    def run_engine():
        asyncio.run(engine.run())
    
    engine_thread = threading.Thread(target=run_engine, daemon=True)
    engine_thread.start()
    
    # Run Qt event loop
    sys.exit(app.exec_())

