"""
Text timeline widget for displaying text events.

Shows TextSignals as a scrolling list with timestamps and category-based
color coding.
"""

from typing import Optional

from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (
    QListWidget,
    QListWidgetItem,
    QVBoxLayout,
    QWidget,
    QLabel,
)

from lib.signals import TextSignal


class TextTimelineWidget(QWidget):
    """
    Scrolling text event timeline widget.
    
    Displays TextSignals as a list with:
    - Timestamps
    - Category-based color coding
    - Log level indicators
    - Auto-scroll to newest events
    """
    
    # Category to color mapping
    CATEGORY_COLORS = {
        "system": "#808080",      # Gray
        "wake_word": "#00FF00",   # Green
        "conversation": "#00BFFF", # Deep sky blue
        "orchestrator": "#FFA500", # Orange
        "transcript": "#FFFFFF",   # White
        "led": "#FF00FF",          # Magenta
        "audio": "#FFFF00",        # Yellow
    }
    
    # Level to prefix/color mapping
    LEVEL_PREFIXES = {
        "debug": ("🔍", "#666666"),
        "info": ("ℹ️", "#FFFFFF"),
        "warning": ("⚠️", "#FFA500"),
        "error": ("✗", "#FF0000"),
    }
    
    def __init__(
        self,
        max_items: int = 500,
        parent=None
    ):
        """
        Initialize text timeline widget.
        
        Args:
            max_items: Maximum number of items to keep
            parent: Parent widget
        """
        super().__init__(parent)
        
        self.max_items = max_items
        self._start_time: Optional[float] = None
        
        self._setup_ui()
    
    def _setup_ui(self) -> None:
        """Setup the UI layout."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        
        # Title label
        self.title_label = QLabel("Event Timeline")
        self.title_label.setStyleSheet("font-weight: bold; padding: 2px;")
        layout.addWidget(self.title_label)
        
        # List widget for events
        self.list_widget = QListWidget()
        self.list_widget.setStyleSheet("""
            QListWidget {
                background-color: #1e1e1e;
                color: #ffffff;
                font-family: Monaco, Menlo, monospace;
                font-size: 11px;
            }
            QListWidget::item {
                padding: 2px 5px;
                border-bottom: 1px solid #333333;
            }
            QListWidget::item:selected {
                background-color: #3d3d3d;
            }
        """)
        layout.addWidget(self.list_widget)
    
    def add_signal(self, signal: TextSignal) -> None:
        """
        Add a TextSignal to the timeline.
        
        Args:
            signal: TextSignal to display
        """
        # Initialize start time on first signal
        if self._start_time is None:
            self._start_time = signal.timestamp
        
        # Calculate relative time
        rel_time = signal.timestamp - self._start_time
        
        # Get prefix and level color
        prefix, level_color = self.LEVEL_PREFIXES.get(
            signal.level,
            ("•", "#FFFFFF")
        )
        
        # Get category color
        category_color = self.CATEGORY_COLORS.get(
            signal.category,
            "#FFFFFF"
        )
        
        # Format timestamp
        timestamp_str = f"{rel_time:8.3f}s"
        
        # Format category
        category_str = f"[{signal.category}]"
        
        # Create item text
        text = f"{timestamp_str} {prefix} {category_str:15} {signal.message}"
        
        # Create list item
        item = QListWidgetItem(text)
        
        # Set foreground color based on level
        if signal.level == "error":
            item.setForeground(QColor("#FF0000"))
        elif signal.level == "warning":
            item.setForeground(QColor("#FFA500"))
        else:
            item.setForeground(QColor(category_color))
        
        # Add to list
        self.list_widget.addItem(item)
        
        # Limit items
        while self.list_widget.count() > self.max_items:
            self.list_widget.takeItem(0)
        
        # Auto-scroll to bottom
        self.list_widget.scrollToBottom()
    
    def clear(self) -> None:
        """Clear all items."""
        self.list_widget.clear()
        self._start_time = None

