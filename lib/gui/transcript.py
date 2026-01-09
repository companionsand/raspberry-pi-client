"""
Transcript widget for displaying conversation input and output transcripts.

Shows user input and agent output transcripts in separate, dedicated sections.
"""

from typing import Optional

from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QVBoxLayout,
    QWidget,
)

from lib.signals import TextSignal


class TranscriptWidget(QWidget):
    """
    Widget for displaying conversation transcripts.
    
    Shows input (user) and output (agent) transcripts in separate sections
    with timestamps and auto-scrolling.
    """
    
    def __init__(
        self,
        max_items: int = 100,
        parent=None
    ):
        """
        Initialize transcript widget.
        
        Args:
            max_items: Maximum number of items to keep per section
            parent: Parent widget
        """
        super().__init__(parent)
        
        self.max_items = max_items
        self._start_time: Optional[float] = None
        
        self._setup_ui()
    
    def _setup_ui(self) -> None:
        """Setup the UI layout."""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(10)
        
        # Left side: Input transcripts (user)
        input_container = QWidget()
        input_layout = QVBoxLayout(input_container)
        input_layout.setContentsMargins(0, 0, 0, 0)
        input_layout.setSpacing(2)
        
        input_title = QLabel("👤 Input Transcript")
        input_title.setStyleSheet("font-weight: bold; padding: 2px; color: #00BFFF;")
        input_layout.addWidget(input_title)
        
        self.input_list = QListWidget()
        self.input_list.setStyleSheet("""
            QListWidget {
                background-color: #1e1e1e;
                color: #00BFFF;
                font-family: Monaco, Menlo, monospace;
                font-size: 12px;
                border: 1px solid #333333;
            }
            QListWidget::item {
                padding: 4px 8px;
                border-bottom: 1px solid #2a2a2a;
            }
            QListWidget::item:selected {
                background-color: #2a4a5a;
            }
        """)
        input_layout.addWidget(self.input_list)
        
        # Right side: Output transcripts (agent)
        output_container = QWidget()
        output_layout = QVBoxLayout(output_container)
        output_layout.setContentsMargins(0, 0, 0, 0)
        output_layout.setSpacing(2)
        
        output_title = QLabel("🤖 Output Transcript")
        output_title.setStyleSheet("font-weight: bold; padding: 2px; color: #FFA500;")
        output_layout.addWidget(output_title)
        
        self.output_list = QListWidget()
        self.output_list.setStyleSheet("""
            QListWidget {
                background-color: #1e1e1e;
                color: #FFA500;
                font-family: Monaco, Menlo, monospace;
                font-size: 12px;
                border: 1px solid #333333;
            }
            QListWidget::item {
                padding: 4px 8px;
                border-bottom: 1px solid #2a2a2a;
            }
            QListWidget::item:selected {
                background-color: #5a4a2a;
            }
        """)
        output_layout.addWidget(self.output_list)
        
        # Add both containers to main layout with equal stretch
        layout.addWidget(input_container, stretch=1)
        layout.addWidget(output_container, stretch=1)
    
    def add_transcript(self, signal: TextSignal) -> None:
        """
        Add a transcript signal to the appropriate section.
        
        Args:
            signal: TextSignal with category "transcript_input" or "transcript_output"
        """
        # Initialize start time on first signal
        if self._start_time is None:
            self._start_time = signal.timestamp
        
        # Calculate relative time
        rel_time = signal.timestamp - self._start_time
        
        # Format timestamp
        timestamp_str = f"{rel_time:8.3f}s"
        
        # Determine which list to add to based on category
        if signal.category == "transcript_input":
            target_list = self.input_list
            color = QColor("#00BFFF")  # Deep sky blue
        elif signal.category == "transcript_output":
            target_list = self.output_list
            color = QColor("#FFA500")  # Orange
        else:
            # Unknown category, skip
            return
        
        # Create item text with timestamp and message
        text = f"[{timestamp_str}] {signal.message}"
        
        # Create list item
        item = QListWidgetItem(text)
        item.setForeground(color)
        
        # Add to appropriate list
        target_list.addItem(item)
        
        # Limit items
        while target_list.count() > self.max_items:
            target_list.takeItem(0)
        
        # Auto-scroll to bottom
        target_list.scrollToBottom()
    
    def clear(self) -> None:
        """Clear all transcripts."""
        self.input_list.clear()
        self.output_list.clear()
        self._start_time = None

