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
from lib.gui.transcript import TranscriptWidget


class CollapsiblePanel(QWidget):
    """
    A collapsible panel widget with a +/- button that properly handles
    splitter sizing when collapsed/expanded.
    """
    
    def __init__(self, title: str, content: QWidget, parent=None, collapsed: bool = False):
        """
        Initialize collapsible panel.
        
        Args:
            title: Panel title
            content: Widget to wrap
            parent: Parent widget
            collapsed: Whether panel starts collapsed
        """
        super().__init__(parent)
        self.content = content
        self._collapsed = collapsed
        self._saved_size = None
        self._splitter = None
        self._splitter_index = -1
        
        self._setup_ui(title)
    
    def _setup_ui(self, title: str) -> None:
        """Setup the UI with header and content."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Header with button and title
        self.header = QWidget()
        header_layout = QHBoxLayout(self.header)
        header_layout.setContentsMargins(5, 5, 5, 5)
        
        self.toggle_button = QPushButton("−" if not self._collapsed else "+")
        self.toggle_button.setFixedSize(20, 20)
        self.toggle_button.setStyleSheet("""
            QPushButton {
                border: 1px solid #555;
                border-radius: 3px;
                background-color: #333;
                color: #fff;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #444;
            }
        """)
        self.toggle_button.clicked.connect(self._toggle)
        header_layout.addWidget(self.toggle_button)
        
        title_label = QLabel(title)
        title_label.setStyleSheet("font-weight: bold; font-size: 12px; padding: 2px;")
        header_layout.addWidget(title_label)
        header_layout.addStretch()
        
        layout.addWidget(self.header)
        layout.addWidget(self.content)
        
        # Set minimum height to header height so header is always visible
        self.header.setMinimumHeight(30)
        self.setMinimumHeight(30)
        
        # Set initial state
        self.content.setVisible(not self._collapsed)
    
    def _toggle(self) -> None:
        """Toggle collapsed state."""
        self._collapsed = not self._collapsed
        self.toggle_button.setText("−" if not self._collapsed else "+")
        self.content.setVisible(not self._collapsed)
        
        # Update splitter size
        if self._splitter is not None and self._splitter_index >= 0:
            sizes = self._splitter.sizes()
            header_height = 30  # Height of header in pixels
            
            if self._collapsed:
                # Save current size (if it's larger than header height)
                if self._saved_size is None:
                    current_size = sizes[self._splitter_index]
                    # Only save if it's larger than header, otherwise use default
                    self._saved_size = current_size if current_size > header_height else 200
                # Set to header height so header remains visible
                sizes[self._splitter_index] = header_height
            else:
                # Restore saved size or use default
                if self._saved_size is not None:
                    sizes[self._splitter_index] = self._saved_size
                    self._saved_size = None
                else:
                    # Default size if we don't have a saved one
                    sizes[self._splitter_index] = 200
            self._splitter.setSizes(sizes)
    
    def set_splitter_info(self, splitter: QSplitter, index: int) -> None:
        """
        Set the splitter and index for this panel.
        
        Args:
            splitter: The QSplitter containing this panel
            index: Index of this panel in the splitter
        """
        self._splitter = splitter
        self._splitter_index = index
        
        # If collapsed, set size to header height immediately
        if self._collapsed:
            sizes = splitter.sizes()
            header_height = 30
            if self._saved_size is None:
                current_size = sizes[index]
                # Only save if it's larger than header, otherwise use default
                self._saved_size = current_size if current_size > header_height else 200
            sizes[index] = header_height
            splitter.setSizes(sizes)


class KinGUIApp(QMainWindow):
    """
    Main GUI window for Kin AI visualization.
    
    Features:
    - Two configurable spectrogram displays (audio streams)
    - Scalar signal plot (VAD probability, RMS, etc.)
    - Conversation transcript display (input/output)
    - Text event timeline
    - Wake word injection button
    """
    
    # Qt signals for thread-safe updates from SignalBus callbacks
    text_signal_received = pyqtSignal(object)
    scalar_signal_received = pyqtSignal(object)
    
    AUDIO_STREAMS = [
        ("aec_input", "Echo-Cancelled Input"),
        ("speaker_loopback", "Speaker Output (Loopback)"),
        ("agent_output", "Agent Output (play() only)"),
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
        self.main_splitter = QSplitter(Qt.Vertical)
        main_layout.addWidget(self.main_splitter, stretch=1)
        
        # Top section: Two spectrograms side by side (collapsible)
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
        
        spectrogram_panel = CollapsiblePanel("Audio Spectrograms", spectrogram_container, collapsed=False)
        self.main_splitter.addWidget(spectrogram_panel)
        spectrogram_panel.set_splitter_info(self.main_splitter, 0)
        
        # Transcript section: Input and output transcripts (collapsible)
        self.transcript_widget = TranscriptWidget()
        transcript_panel = CollapsiblePanel("Conversation Transcripts", self.transcript_widget, collapsed=False)
        self.main_splitter.addWidget(transcript_panel)
        transcript_panel.set_splitter_info(self.main_splitter, 1)
        
        # Scalar plot section (collapsible, collapsed by default)
        self.scalar_plot = ScalarPlotWidget()
        scalar_panel = CollapsiblePanel("Signal Plots", self.scalar_plot, collapsed=True)
        self.main_splitter.addWidget(scalar_panel)
        scalar_panel.set_splitter_info(self.main_splitter, 2)
        
        # Text timeline section (collapsible, collapsed by default)
        self.text_timeline = TextTimelineWidget()
        timeline_panel = CollapsiblePanel("Event Timeline", self.text_timeline, collapsed=True)
        self.main_splitter.addWidget(timeline_panel)
        timeline_panel.set_splitter_info(self.main_splitter, 3)
        
        # Set initial splitter sizes (30% spectrograms, 25% transcript, header height for collapsed panels)
        # Use 30px (header height) for collapsed panels so header remains visible
        self.main_splitter.setSizes([300, 200, 30, 30])
    
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
        self.stream_selector_right.setCurrentIndex(2)  # agent_output
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
        # Connect Qt signals with QueuedConnection for cross-thread safety
        self.text_signal_received.connect(self._handle_text_signal, Qt.QueuedConnection)
        self.scalar_signal_received.connect(self._handle_scalar_signal, Qt.QueuedConnection)
        
        # Subscribe to signals from engine
        self.engine.signal_bus.subscribe(
            signal_type=TextSignal,
            callback=lambda s: self.text_signal_received.emit(s)
        )
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
        # Route transcript signals to transcript widget
        if signal.category in ("transcript_input", "transcript_output"):
            self.transcript_widget.add_transcript(signal)
        # All text signals go to timeline
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

