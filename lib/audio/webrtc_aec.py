"""WebRTC AEC3 (Acoustic Echo Cancellation) processor for real-time audio streaming.

This module provides a wrapper around the aec-audio-processing library (WebRTC AEC3)
to process audio in real-time streaming chunks. It handles:
- Echo cancellation using reference signal (playback loopback)
- Noise suppression (configurable 0-3)
- Automatic Gain Control (configurable adaptive/fixed)
- 320-sample chunks (20ms @ 16kHz) split into 2x 160-sample WebRTC frames

Architecture:
- Input: 320-sample chunks (mic + reference)
- Processing: Split into 2x 160-sample chunks for WebRTC (10ms frames)
- Output: 320-sample processed chunk

Reference implementation: pipipi/tests/test_aec_webrtc.py
"""

import numpy as np
from typing import Optional

try:
    from aec_audio_processing import AudioProcessor
except ImportError:
    AudioProcessor = None


class WebRTCAECProcessor:
    """
    Real-time WebRTC AEC3 processor for streaming audio.
    
    Processes audio in 320-sample chunks (20ms @ 16kHz), splitting internally
    into 2x 160-sample chunks for WebRTC AEC3 (which requires 10ms frames).
    
    Usage:
        processor = WebRTCAECProcessor(stream_delay_ms=100, ns_level=1, agc_mode=2)
        processor.start()  # Initialize processor state
        
        # In audio loop:
        processed_chunk = processor.process_chunk(mic_chunk, ref_chunk)
        
        processor.stop()  # Clean up state
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        stream_delay_ms: int = 100,
        ns_level: int = 1,
        agc_mode: int = 2,
        enable_vad: bool = True,
    ):
        """
        Initialize WebRTC AEC processor.
        
        Args:
            sample_rate: Sample rate in Hz (default: 16000)
            channels: Number of channels (default: 1, mono)
            stream_delay_ms: Delay between playback and capture in milliseconds (50-200ms typical for USB)
            ns_level: Noise suppression level [0-3] (0=off, 1=moderate, 3=max)
            agc_mode: AGC mode [1=adaptive digital, 2=fixed digital]
            enable_vad: Enable Voice Activity Detection (default: True)
        """
        if AudioProcessor is None:
            raise ImportError(
                "aec-audio-processing package not installed. "
                "Install with: pip install aec-audio-processing"
            )
        
        self.sample_rate = sample_rate
        self.channels = channels
        self.stream_delay_ms = stream_delay_ms
        self.ns_level = ns_level
        self.agc_mode = agc_mode
        self.enable_vad = enable_vad
        
        # WebRTC requires 10ms frames (160 samples @ 16kHz)
        self.webrtc_frame_size = 160
        
        # We process 320-sample chunks (20ms) = 2x WebRTC frames
        self.chunk_size = 320
        
        # Initialize processor (will be set in start())
        self.processor: Optional[AudioProcessor] = None
        self.is_started = False
        
        print(f"✓ WebRTC AEC initialized:")
        print(f"   - Sample rate: {sample_rate}Hz")
        print(f"   - Chunk size: {self.chunk_size} samples (20ms)")
        print(f"   - WebRTC frame size: {self.webrtc_frame_size} samples (10ms)")
        print(f"   - Stream delay: {stream_delay_ms}ms")
        print(f"   - Noise suppression level: {ns_level}")
        print(f"   - AGC mode: {agc_mode} ({'adaptive' if agc_mode == 1 else 'fixed digital'})")
    
    def start(self):
        """
        Start the AEC processor and initialize internal state.
        Must be called before processing audio.
        """
        if self.is_started:
            print("⚠️  WebRTC AEC already started, skipping initialization")
            return
        
        print("🎯 Starting WebRTC AEC processor...")
        
        # Initialize AudioProcessor with AEC3 + NS + AGC
        self.processor = AudioProcessor(
            enable_aec=True,
            enable_ns=(self.ns_level > 0),
            ns_level=self.ns_level,
            enable_agc=True,
            agc_mode=self.agc_mode,
            enable_vad=self.enable_vad,
        )
        
        # Configure stream formats (mono 16kHz)
        self.processor.set_stream_format(self.sample_rate, self.channels)
        self.processor.set_reverse_stream_format(self.sample_rate, self.channels)
        
        # Set stream delay for proper echo alignment
        self.processor.set_stream_delay(self.stream_delay_ms)
        
        self.is_started = True
        print(f"✓ WebRTC AEC started (delay={self.stream_delay_ms}ms, ns={self.ns_level}, agc={self.agc_mode})")
    
    def stop(self):
        """
        Stop the AEC processor and clean up state.
        """
        if not self.is_started:
            return
        
        print("🛑 Stopping WebRTC AEC processor...")
        self.processor = None
        self.is_started = False
        print("✓ WebRTC AEC stopped")
    
    def process_chunk(self, mic_chunk: np.ndarray, ref_chunk: np.ndarray) -> np.ndarray:
        """
        Process a single 320-sample chunk (20ms @ 16kHz) through WebRTC AEC.
        
        Internally splits the chunk into 2x 160-sample frames for WebRTC processing.
        
        Args:
            mic_chunk: Microphone audio chunk (320 samples, int16)
            ref_chunk: Reference audio chunk (320 samples, int16) - playback loopback
        
        Returns:
            Processed audio chunk (320 samples, int16) with echo cancelled
        
        Raises:
            RuntimeError: If processor not started or chunk size mismatch
        """
        if not self.is_started or self.processor is None:
            raise RuntimeError("WebRTC AEC processor not started. Call start() first.")
        
        # Validate chunk sizes
        if len(mic_chunk) != self.chunk_size:
            raise ValueError(
                f"mic_chunk size mismatch: expected {self.chunk_size}, got {len(mic_chunk)}"
            )
        if len(ref_chunk) != self.chunk_size:
            raise ValueError(
                f"ref_chunk size mismatch: expected {self.chunk_size}, got {len(ref_chunk)}"
            )
        
        # Process in 2x 160-sample chunks (10ms frames for WebRTC)
        processed_frames = []
        
        for i in range(0, self.chunk_size, self.webrtc_frame_size):
            # Extract 160-sample frame from mic and ref
            mic_frame = mic_chunk[i : i + self.webrtc_frame_size]
            ref_frame = ref_chunk[i : i + self.webrtc_frame_size]
            
            # Convert to bytes (160 samples * 2 bytes = 320 bytes)
            mic_frame_bytes = mic_frame.tobytes()
            ref_frame_bytes = ref_frame.tobytes()
            
            # Process reference signal first (for AEC adaptation)
            self.processor.process_reverse_stream(ref_frame_bytes)
            
            # Process microphone signal (AEC cancels echo based on reference)
            processed_bytes = self.processor.process_stream(mic_frame_bytes)
            
            # Convert back to numpy array
            processed_frame = np.frombuffer(processed_bytes, dtype=np.int16)
            processed_frames.append(processed_frame)
        
        # Concatenate 2x 160-sample frames back into 320-sample chunk
        processed_chunk = np.concatenate(processed_frames)
        
        return processed_chunk
    
    def __enter__(self):
        """Context manager entry: start processor"""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit: stop processor"""
        self.stop()
        return False

