"""
Spectrogram widget for real-time audio visualization.

Uses pyqtgraph for efficient rendering and a worker thread for FFT computation.
"""

import queue
import time
from typing import TYPE_CHECKING

import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import QObject, QThread, pyqtSignal
from PyQt5.QtWidgets import QVBoxLayout, QWidget, QLabel

if TYPE_CHECKING:
    from lib.engine import KinEngine


class FFTWorker(QObject):
    """
    Worker for computing FFT in a separate thread.
    
    Receives audio chunks, applies windowing, computes FFT,
    and emits the magnitude spectrum in dB.
    """
    
    # Signal emitted when FFT result is ready
    result_ready = pyqtSignal(np.ndarray, float)  # (magnitude_db, timestamp)
    
    def __init__(self, window_size: int = 512, hop_size: int = 256, sample_rate: int = 16000):
        """
        Initialize FFT worker.
        
        Args:
            window_size: FFT window size in samples (~32ms at 16kHz)
            hop_size: Hop size for overlapping windows
            sample_rate: Audio sample rate
        """
        super().__init__()
        
        self.window_size = window_size
        self.hop_size = hop_size
        self.sample_rate = sample_rate
        
        # Pre-compute Hann window
        self.window_fn = np.hanning(window_size).astype(np.float32)
        
        # Job queue
        self._queue: queue.Queue = queue.Queue(maxsize=10)
        self._running = True
    
    def submit(self, audio: np.ndarray, timestamp: float) -> bool:
        """
        Submit audio for FFT processing.
        
        Args:
            audio: Audio samples (int16 or float32)
            timestamp: Timestamp for this chunk
            
        Returns:
            True if submitted, False if queue full
        """
        try:
            self._queue.put_nowait((audio.copy(), timestamp))
            return True
        except queue.Full:
            return False
    
    def stop(self) -> None:
        """Stop the worker."""
        self._running = False
    
    def run(self) -> None:
        """Main processing loop (runs in worker thread)."""
        while self._running:
            try:
                # Get audio from queue with timeout
                try:
                    audio, timestamp = self._queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                
                # Convert to float32 if needed
                if audio.dtype == np.int16:
                    audio_float = audio.astype(np.float32) / 32768.0
                else:
                    audio_float = audio.astype(np.float32)
                
                # Ensure we have enough samples
                if len(audio_float) < self.window_size:
                    # Pad with zeros
                    audio_float = np.pad(
                        audio_float,
                        (0, self.window_size - len(audio_float)),
                        mode='constant'
                    )
                
                # Take most recent window_size samples
                audio_windowed = audio_float[-self.window_size:] * self.window_fn
                
                # Compute FFT
                spectrum = np.fft.rfft(audio_windowed)
                
                # Convert to magnitude in dB (avoid log(0))
                magnitude = np.abs(spectrum)
                magnitude_db = 20 * np.log10(magnitude + 1e-10)
                
                # Clamp to reasonable range
                magnitude_db = np.clip(magnitude_db, -80, 0)
                
                # Emit result
                self.result_ready.emit(magnitude_db.astype(np.float32), timestamp)
                
            except Exception as e:
                print(f"⚠️  FFTWorker error: {e}")


class SpectrogramWidget(QWidget):
    """
    Real-time spectrogram display widget.
    
    Uses pyqtgraph ImageItem for efficient rendering of the
    scrolling spectrogram display.
    """
    
    def __init__(
        self,
        engine: "KinEngine",
        stream_name: str = "aec_input",
        title: str = "Spectrogram",
        window_size: int = 512,
        display_seconds: float = 5.0,
        parent=None
    ):
        """
        Initialize spectrogram widget.
        
        Args:
            engine: KinEngine for audio access
            stream_name: Audio stream to visualize
            title: Display title
            window_size: FFT window size
            display_seconds: Width of display in seconds
            parent: Parent widget
        """
        super().__init__(parent)
        
        self.engine = engine
        self.stream_name = stream_name
        self.title = title
        self.window_size = window_size
        self.display_seconds = display_seconds
        self.sample_rate = 16000  # Matches Config.SAMPLE_RATE
        
        # Calculate spectrogram dimensions
        # Number of frequency bins = window_size // 2 + 1
        self.freq_bins = window_size // 2 + 1
        
        # Number of time columns (update rate ~30fps, 5 seconds = 150 columns)
        self.time_columns = int(display_seconds * 30)
        
        # Spectrogram data buffer (freq_bins x time_columns)
        self.spectrogram_data = np.zeros(
            (self.freq_bins, self.time_columns),
            dtype=np.float32
        ) - 80  # Initialize to -80 dB (silence)
        
        # Track when we last processed data (for detecting stale buffers)
        self._last_processed_write_time: float = 0.0
        
        self._setup_ui()
        self._setup_fft_worker()
    
    def _setup_ui(self) -> None:
        """Setup the UI layout."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        
        # Title label
        self.title_label = QLabel(self.title)
        self.title_label.setStyleSheet("font-weight: bold; padding: 2px;")
        layout.addWidget(self.title_label)
        
        # pyqtgraph plot widget
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setLabel('left', 'Frequency', units='Hz')
        self.plot_widget.setLabel('bottom', 'Time', units='s')
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        layout.addWidget(self.plot_widget)
        
        # Create ImageItem for spectrogram
        self.image_item = pg.ImageItem()
        self.plot_widget.addItem(self.image_item)
        
        # Configure colormap (viridis-like)
        colormap = pg.colormap.get('viridis')
        self.image_item.setColorMap(colormap)
        
        # Set axis scaling
        # X: 0 to display_seconds
        # Y: 0 to Nyquist frequency (sample_rate / 2)
        nyquist = self.sample_rate / 2
        self.plot_widget.setXRange(0, self.display_seconds)
        self.plot_widget.setYRange(0, nyquist)
        
        # Configure image item transform
        # Scale: width = display_seconds, height = nyquist
        scale_x = self.display_seconds / self.time_columns
        scale_y = nyquist / self.freq_bins
        self.image_item.setTransform(
            pg.QtGui.QTransform.fromScale(scale_x, scale_y)
        )
        
        # Initial display
        self.image_item.setImage(self.spectrogram_data.T)
    
    def _setup_fft_worker(self) -> None:
        """Setup the FFT worker thread."""
        self.fft_worker = FFTWorker(
            window_size=self.window_size,
            sample_rate=self.sample_rate
        )
        
        self.fft_thread = QThread()
        self.fft_worker.moveToThread(self.fft_thread)
        
        # Connect signals
        self.fft_thread.started.connect(self.fft_worker.run)
        self.fft_worker.result_ready.connect(self._on_fft_result)
        
        # Start thread
        self.fft_thread.start()
    
    def set_stream(self, stream_name: str, display_name: str) -> None:
        """
        Change the audio stream being visualized.
        
        Args:
            stream_name: New stream identifier
            display_name: Human-readable name for title
        """
        self.stream_name = stream_name
        self.title = display_name
        self.title_label.setText(display_name)
        
        # Clear FFT worker queue (discard pending work for old stream)
        while not self.fft_worker._queue.empty():
            try:
                self.fft_worker._queue.get_nowait()
            except queue.Empty:
                break
        
        # Clear spectrogram data
        self.spectrogram_data.fill(-80)
        
        # Immediately update display with cleared data
        self.image_item.setImage(self.spectrogram_data.T)
        
        # Reset tracking for the new stream
        self._last_processed_write_time = 0.0
    
    def update_display(self) -> None:
        """
        Update the spectrogram display with latest audio.
        
        Called by timer in main app (~30 FPS).
        Continuously scrolls to keep timelines synchronized across streams.
        """
        # Check if there's fresh data
        last_write_time = self.engine.get_stream_last_write_time(self.stream_name)
        
        if last_write_time <= self._last_processed_write_time:
            # No new data - scroll a silence column to keep timeline moving
            self._scroll_silence()
            return
        
        # Have fresh data - update tracking
        self._last_processed_write_time = last_write_time
        
        # Get recent audio from engine (enough for one FFT window + some extra)
        audio = self.engine.get_audio_window(self.stream_name, 0.1)
        
        if len(audio) > 0:
            # Submit to FFT worker
            self.fft_worker.submit(audio, time.monotonic())
        else:
            # No audio available - show silence
            self._scroll_silence()
    
    def _scroll_silence(self) -> None:
        """
        Scroll the spectrogram and add a silence column.
        
        Used when no new audio data is available to keep the timeline
        moving and synchronized with other streams.
        """
        # Shift spectrogram left (scroll)
        self.spectrogram_data = np.roll(self.spectrogram_data, -1, axis=1)
        
        # Fill rightmost column with silence (-80 dB)
        self.spectrogram_data[:, -1] = -80
        
        # Update display
        self.image_item.setImage(self.spectrogram_data.T)
    
    def _on_fft_result(self, magnitude_db: np.ndarray, timestamp: float) -> None:
        """
        Handle FFT result from worker thread.
        
        Args:
            magnitude_db: FFT magnitude in dB
            timestamp: When the audio was captured
        """
        # Ensure correct size
        if len(magnitude_db) != self.freq_bins:
            # Resize if needed
            if len(magnitude_db) > self.freq_bins:
                magnitude_db = magnitude_db[:self.freq_bins]
            else:
                magnitude_db = np.pad(
                    magnitude_db,
                    (0, self.freq_bins - len(magnitude_db)),
                    constant_values=-80
                )
        
        # Shift spectrogram left (scroll)
        self.spectrogram_data = np.roll(self.spectrogram_data, -1, axis=1)
        
        # Add new column on the right
        self.spectrogram_data[:, -1] = magnitude_db
        
        # Update display
        self.image_item.setImage(self.spectrogram_data.T)
    
    def cleanup(self) -> None:
        """Clean up resources."""
        self.fft_worker.stop()
        self.fft_thread.quit()
        self.fft_thread.wait(1000)

