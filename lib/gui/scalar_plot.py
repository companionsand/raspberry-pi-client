"""
Scalar signal plot widget for real-time visualization.

Displays scalar signals (VAD probability, YAMNET scores, RMS)
as scrolling line graphs using pyqtgraph.
"""

import time
from collections import defaultdict, deque
from typing import Dict, Optional

import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QVBoxLayout, QWidget, QLabel

from lib.signals import ScalarSignal


class ScalarPlotWidget(QWidget):
    """
    Real-time line plot for scalar signals.
    
    Displays multiple scalar signals as overlapping line graphs
    with automatic legend and color coding.
    """
    
    # Colors for different signals (cycling)
    COLORS = [
        (0, 255, 0),      # Green - input_rms
        (255, 165, 0),    # Orange - vad_probability
        (0, 191, 255),    # Deep sky blue - yamnet
        (255, 0, 255),    # Magenta
        (255, 255, 0),    # Yellow
        (0, 255, 255),    # Cyan
    ]
    
    # Known signal name to friendly name mapping
    SIGNAL_NAMES = {
        "input_rms": "Input RMS",
        "vad_probability": "VAD Probability",
        "yamnet_speech": "YAMNET Speech Score",
        "output_rms": "Output RMS",
    }
    
    def __init__(
        self,
        display_seconds: float = 10.0,
        max_points: int = 300,
        parent=None
    ):
        """
        Initialize scalar plot widget.
        
        Args:
            display_seconds: Width of display in seconds
            max_points: Maximum data points per signal
            parent: Parent widget
        """
        super().__init__(parent)
        
        self.display_seconds = display_seconds
        self.max_points = max_points
        
        # Data storage: signal_name -> deque of (timestamp, value)
        self.signal_data: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=max_points)
        )
        
        # Plot curves: signal_name -> PlotDataItem
        self.curves: Dict[str, pg.PlotDataItem] = {}
        
        # Color assignment: signal_name -> color
        self.signal_colors: Dict[str, tuple] = {}
        self._next_color_idx = 0
        
        # Reference time for x-axis (set on first signal)
        self._start_time: Optional[float] = None
        
        self._setup_ui()
    
    def _setup_ui(self) -> None:
        """Setup the UI layout."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        
        # Title label
        self.title_label = QLabel("Scalar Signals")
        self.title_label.setStyleSheet("font-weight: bold; padding: 2px;")
        layout.addWidget(self.title_label)
        
        # pyqtgraph plot widget
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setLabel('left', 'Value')
        self.plot_widget.setLabel('bottom', 'Time', units='s')
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.setYRange(0, 1)  # Most scalars are 0-1
        self.plot_widget.addLegend(offset=(70, 30))
        layout.addWidget(self.plot_widget)
    
    def add_signal(self, signal: ScalarSignal) -> None:
        """
        Add a scalar signal data point.
        
        Args:
            signal: ScalarSignal to add
        """
        # Initialize start time on first signal
        if self._start_time is None:
            self._start_time = signal.timestamp
        
        # Calculate relative time
        rel_time = signal.timestamp - self._start_time
        
        # Store data point
        self.signal_data[signal.name].append((rel_time, signal.value))
        
        # Create curve if new signal
        if signal.name not in self.curves:
            self._create_curve(signal.name)
        
        # Update curve data
        self._update_curve(signal.name)
        
        # Update x-axis range to show last display_seconds
        if rel_time > self.display_seconds:
            self.plot_widget.setXRange(
                rel_time - self.display_seconds,
                rel_time
            )
    
    def _create_curve(self, signal_name: str) -> None:
        """
        Create a new plot curve for a signal.
        
        Args:
            signal_name: Name of the signal
        """
        # Assign color
        color = self.COLORS[self._next_color_idx % len(self.COLORS)]
        self._next_color_idx += 1
        self.signal_colors[signal_name] = color
        
        # Get friendly name
        friendly_name = self.SIGNAL_NAMES.get(signal_name, signal_name)
        
        # Create pen
        pen = pg.mkPen(color=color, width=2)
        
        # Create curve
        curve = self.plot_widget.plot(
            [],
            [],
            pen=pen,
            name=friendly_name
        )
        self.curves[signal_name] = curve
    
    def _update_curve(self, signal_name: str) -> None:
        """
        Update the data for a curve.
        
        Args:
            signal_name: Name of the signal to update
        """
        if signal_name not in self.curves:
            return
        
        data = self.signal_data[signal_name]
        if not data:
            return
        
        # Convert to numpy arrays
        times = np.array([t for t, v in data])
        values = np.array([v for t, v in data])
        
        # Update curve
        self.curves[signal_name].setData(times, values)
    
    def clear(self) -> None:
        """Clear all data and curves."""
        self.signal_data.clear()
        
        for curve in self.curves.values():
            self.plot_widget.removeItem(curve)
        self.curves.clear()
        
        self.signal_colors.clear()
        self._next_color_idx = 0
        self._start_time = None

