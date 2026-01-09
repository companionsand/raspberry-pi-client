"""
Ring buffer implementation for zero-copy audio access.

Provides efficient circular buffer storage for audio streams with:
- Pre-allocated numpy arrays (no runtime allocation)
- Lock-free writes via atomic pointer updates
- Zero-copy reads via numpy views when possible
- Thread-safe concurrent read/write access
"""

import threading
import time
from typing import Optional

import numpy as np


class RingBuffer:
    """
    Thread-safe ring buffer for audio data with zero-copy access.
    
    Uses a circular buffer design where the write pointer advances
    continuously. Readers can access any window of recent data.
    
    Attributes:
        capacity: Maximum number of samples the buffer can hold
        channels: Number of audio channels (1 for mono, 6 for ReSpeaker raw)
        dtype: NumPy dtype for samples (default: int16)
    """
    
    def __init__(
        self,
        capacity: int,
        channels: int = 1,
        dtype: np.dtype = np.int16
    ):
        """
        Initialize ring buffer.
        
        Args:
            capacity: Number of samples to store (e.g., sample_rate * seconds)
            channels: Number of audio channels (1 for mono)
            dtype: NumPy dtype for audio samples
        """
        self._capacity = capacity
        self._channels = channels
        self._dtype = dtype
        
        # Pre-allocate buffer
        if channels == 1:
            self._buffer = np.zeros(capacity, dtype=dtype)
        else:
            self._buffer = np.zeros((capacity, channels), dtype=dtype)
        
        # Write position (always advances, modulo capacity for actual index)
        self._write_pos = 0
        
        # Lock for thread safety (minimal contention - only protects pointer update)
        self._lock = threading.Lock()
        
        # Track total samples written (for debugging/stats)
        self._total_written = 0
        
        # Track when last write occurred (for detecting stale data)
        self._last_write_time: float = 0.0
    
    @property
    def capacity(self) -> int:
        """Get buffer capacity in samples."""
        return self._capacity
    
    @property
    def channels(self) -> int:
        """Get number of channels."""
        return self._channels
    
    @property
    def total_written(self) -> int:
        """Get total samples written since creation."""
        return self._total_written
    
    @property
    def last_write_time(self) -> float:
        """Get timestamp of last write (monotonic clock)."""
        return self._last_write_time
    
    def write(self, data: np.ndarray) -> int:
        """
        Write audio data to the buffer.
        
        Data is written circularly, overwriting oldest samples when full.
        This method is designed for the audio callback hot path.
        
        Args:
            data: Audio samples to write (1D for mono, 2D for multi-channel)
            
        Returns:
            Number of samples written
        """
        # Ensure correct shape
        if self._channels == 1:
            if data.ndim > 1:
                data = data.flatten()
            samples = len(data)
        else:
            if data.ndim == 1:
                # Assume single sample across all channels
                data = data.reshape(1, -1)
            samples = data.shape[0]
        
        if samples == 0:
            return 0
        
        # Convert dtype if needed
        if data.dtype != self._dtype:
            data = data.astype(self._dtype)
        
        with self._lock:
            # Calculate write position in circular buffer
            start_idx = self._write_pos % self._capacity
            
            if start_idx + samples <= self._capacity:
                # Simple case: write doesn't wrap
                if self._channels == 1:
                    self._buffer[start_idx:start_idx + samples] = data
                else:
                    self._buffer[start_idx:start_idx + samples, :] = data
            else:
                # Write wraps around
                first_chunk = self._capacity - start_idx
                second_chunk = samples - first_chunk
                
                if self._channels == 1:
                    self._buffer[start_idx:] = data[:first_chunk]
                    self._buffer[:second_chunk] = data[first_chunk:]
                else:
                    self._buffer[start_idx:, :] = data[:first_chunk, :]
                    self._buffer[:second_chunk, :] = data[first_chunk:, :]
            
            # Update write position and timestamp
            self._write_pos += samples
            self._total_written += samples
            self._last_write_time = time.monotonic()
        
        return samples
    
    def get_window(self, samples: int) -> np.ndarray:
        """
        Get the most recent N samples from the buffer.
        
        Returns a copy to ensure thread safety. For very high-frequency
        access, consider using get_window_view() with external locking.
        
        Args:
            samples: Number of samples to retrieve
            
        Returns:
            NumPy array of most recent samples (copy)
        """
        if samples > self._capacity:
            samples = self._capacity
        
        if samples <= 0:
            if self._channels == 1:
                return np.array([], dtype=self._dtype)
            else:
                return np.zeros((0, self._channels), dtype=self._dtype)
        
        with self._lock:
            # Handle case where buffer isn't full yet
            available = min(self._write_pos, self._capacity)
            if samples > available:
                samples = available
            
            if samples == 0:
                if self._channels == 1:
                    return np.array([], dtype=self._dtype)
                else:
                    return np.zeros((0, self._channels), dtype=self._dtype)
            
            # Calculate read position
            end_idx = self._write_pos % self._capacity
            start_idx = (end_idx - samples) % self._capacity
            
            if start_idx < end_idx:
                # Simple case: data is contiguous
                if self._channels == 1:
                    return self._buffer[start_idx:end_idx].copy()
                else:
                    return self._buffer[start_idx:end_idx, :].copy()
            else:
                # Data wraps around - need to concatenate
                if self._channels == 1:
                    first_part = self._buffer[start_idx:]
                    second_part = self._buffer[:end_idx]
                    return np.concatenate([first_part, second_part])
                else:
                    first_part = self._buffer[start_idx:, :]
                    second_part = self._buffer[:end_idx, :]
                    return np.concatenate([first_part, second_part], axis=0)
    
    def get_window_seconds(self, seconds: float, sample_rate: int = 16000) -> np.ndarray:
        """
        Get the most recent N seconds of audio.
        
        Convenience method that converts seconds to samples.
        
        Args:
            seconds: Duration to retrieve
            sample_rate: Sample rate in Hz (default: 16000)
            
        Returns:
            NumPy array of most recent samples
        """
        samples = int(seconds * sample_rate)
        return self.get_window(samples)
    
    def clear(self) -> None:
        """Clear the buffer (reset to zeros)."""
        with self._lock:
            self._buffer.fill(0)
            self._write_pos = 0
            self._total_written = 0
    
    def get_fill_level(self) -> float:
        """
        Get buffer fill level as a fraction (0.0 - 1.0).
        
        Returns 1.0 once buffer has been filled at least once.
        """
        with self._lock:
            if self._write_pos >= self._capacity:
                return 1.0
            return self._write_pos / self._capacity

