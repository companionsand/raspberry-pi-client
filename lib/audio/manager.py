"""
AudioManager - Unified audio capture, playback, and echo cancellation.

This module provides a single interface for all audio operations:
- Opens one duplex stream (6ch capture, 1ch playback for ReSpeaker)
- Handles WebRTC AEC internally using Ch0 (mic) and Ch5 (reference)
- Distributes processed audio to registered consumers via callbacks
- Accepts playback audio from multiple sources via queue

Usage:
    manager = AudioManager(use_webrtc_aec=True)
    manager.start()
    
    # Register consumer to receive processed audio
    manager.register_consumer(my_callback)
    
    # Queue audio for playback
    manager.play(audio_array)
    
    manager.stop()
"""

import queue
import threading
import time
from typing import Callable, List, Optional

import numpy as np
import sounddevice as sd

from lib.config import Config

# WebRTC AEC (optional)
try:
    from lib.audio.webrtc_aec import WebRTCAECProcessor
    WEBRTC_AEC_AVAILABLE = True
except ImportError as e:
    WebRTCAECProcessor = None
    WEBRTC_AEC_AVAILABLE = False
    WEBRTC_AEC_IMPORT_ERROR = str(e)


class AudioManager:
    """
    Unified audio manager for capture, playback, and echo cancellation.
    
    Opens a single duplex stream shared by all consumers:
    - WakeWordDetector
    - VoiceCommandDetector  
    - ElevenLabsConversationClient
    
    Handles AEC transparently - consumers receive clean, echo-cancelled audio.
    """
    
    def __init__(self, use_webrtc_aec: bool = None):
        """
        Initialize AudioManager.
        
        Args:
            use_webrtc_aec: Enable WebRTC AEC. If None, uses Config.USE_WEBRTC_AEC.
        """
        # AEC configuration
        if use_webrtc_aec is None:
            use_webrtc_aec = Config.USE_WEBRTC_AEC
        self._use_webrtc_aec = use_webrtc_aec and WEBRTC_AEC_AVAILABLE
        
        # Stream state
        self._stream: Optional[sd.Stream] = None
        self._is_running = False
        self._lock = threading.Lock()
        
        # AEC processor
        self._aec_processor: Optional[WebRTCAECProcessor] = None
        
        # Consumer callbacks: list of (callback, include_raw) tuples
        # include_raw: if True, callback receives (processed, raw) tuple
        self._consumers: List[Callable[[np.ndarray], None]] = []
        self._consumers_lock = threading.Lock()
        
        # Output queue for playback
        self._output_queue: queue.Queue = queue.Queue()
        self._output_remainder: Optional[np.ndarray] = None
        
        # Device detection state
        self._has_respeaker = False
        self._input_channels = Config.CHANNELS
        self._respeaker_device_index: Optional[int] = None
        self._respeaker_alsa_card: Optional[int] = None
        
        # Debug logging
        self._last_debug_log = 0
        self._debug_interval = 3.0  # Log every 3 seconds
        
        # Logger
        self._logger = Config.LOGGER
    
    @property
    def is_running(self) -> bool:
        """Check if audio manager is running."""
        return self._is_running
    
    @property
    def has_respeaker(self) -> bool:
        """Check if ReSpeaker device is available."""
        return self._has_respeaker
    
    @property
    def has_webrtc_aec(self) -> bool:
        """Check if WebRTC AEC is active."""
        return self._use_webrtc_aec and self._aec_processor is not None
    
    def start(self) -> bool:
        """
        Start the audio manager.
        
        Opens duplex stream, initializes AEC if enabled.
        
        Returns:
            True if started successfully, False otherwise.
        """
        with self._lock:
            if self._is_running:
                print("⚠️  AudioManager already running")
                return True
            
            try:
                # Detect ReSpeaker
                self._detect_respeaker()
                
                # Initialize WebRTC AEC if enabled and ReSpeaker detected
                if self._use_webrtc_aec and self._has_respeaker:
                    self._init_webrtc_aec()
                
                # Open duplex stream
                self._open_stream()
                
                self._is_running = True
                print("✓ AudioManager started")
                
                if self._logger:
                    self._logger.info(
                        "audio_manager_started",
                        extra={
                            "has_respeaker": self._has_respeaker,
                            "webrtc_aec_enabled": self.has_webrtc_aec,
                            "input_channels": self._input_channels,
                            "device_id": Config.DEVICE_ID
                        }
                    )
                
                return True
                
            except Exception as e:
                print(f"✗ AudioManager start failed: {e}")
                if self._logger:
                    self._logger.error(
                        "audio_manager_start_failed",
                        extra={"error": str(e), "device_id": Config.DEVICE_ID},
                        exc_info=True
                    )
                self._cleanup()
                return False
    
    def stop(self) -> None:
        """Stop the audio manager and clean up resources."""
        with self._lock:
            if not self._is_running:
                return
            
            print("🛑 Stopping AudioManager...")
            self._is_running = False
            self._cleanup()
            print("✓ AudioManager stopped")
            
            if self._logger:
                self._logger.info(
                    "audio_manager_stopped",
                    extra={"device_id": Config.DEVICE_ID}
                )
    
    def register_consumer(self, callback: Callable[[np.ndarray], None]) -> None:
        """
        Register a callback to receive processed audio.
        
        Callback receives numpy array of int16 audio samples (mono, AEC-processed).
        Called from audio thread - keep processing minimal.
        
        Args:
            callback: Function that accepts numpy array of audio samples.
        """
        with self._consumers_lock:
            if callback not in self._consumers:
                self._consumers.append(callback)
                print(f"✓ AudioManager: Consumer registered ({len(self._consumers)} total)")
    
    def unregister_consumer(self, callback: Callable) -> None:
        """
        Remove a consumer callback.
        
        Args:
            callback: Previously registered callback function.
        """
        with self._consumers_lock:
            if callback in self._consumers:
                self._consumers.remove(callback)
                print(f"✓ AudioManager: Consumer unregistered ({len(self._consumers)} remaining)")
    
    def play(self, audio: np.ndarray) -> None:
        """
        Queue audio for playback.
        
        Args:
            audio: numpy array of int16 audio samples (mono).
        """
        if not self._is_running:
            return
        
        # Ensure correct dtype
        if audio.dtype != np.int16:
            audio = audio.astype(np.int16)
        
        # Ensure 1D
        if audio.ndim > 1:
            audio = audio.flatten()
        
        self._output_queue.put(audio)
    
    def clear_playback_queue(self) -> int:
        """
        Clear all pending playback audio.
        
        Returns:
            Number of chunks cleared.
        """
        cleared = 0
        while not self._output_queue.empty():
            try:
                self._output_queue.get_nowait()
                cleared += 1
            except queue.Empty:
                break
        self._output_remainder = None
        return cleared
    
    def _detect_respeaker(self) -> None:
        """Detect ReSpeaker device and configure channels."""
        devices = sd.query_devices()
        
        for idx, dev in enumerate(devices):
            if any(kw in dev['name'].lower() for kw in ['respeaker', 'arrayuac10', 'uac1.0']):
                # Always capture all 6 channels for simplicity and flexibility
                # - Hardware AEC: extract Ch0 (AEC-processed)
                # - WebRTC AEC: extract Ch0 (mic) and Ch5 (reference)
                # Check that device supports both input (6ch) and output (1ch)
                if (dev['max_input_channels'] >= Config.RESPEAKER_CHANNELS and 
                    dev['max_output_channels'] >= Config.CHANNELS):
                    self._has_respeaker = True
                    # Always use 6 channels - simpler and works for both AEC modes
                    self._input_channels = Config.RESPEAKER_CHANNELS
                    self._respeaker_device_index = idx
                    
                    # Get ALSA card number for direct ALSA device string access
                    from lib.audio.device_detection import get_alsa_card_number
                    self._respeaker_alsa_card = get_alsa_card_number(dev['name'])
                    
                    print(f"✓ ReSpeaker detected: {dev['name']}")
                    print(f"   PortAudio device index: {idx}")
                    if self._respeaker_alsa_card is not None:
                        print(f"   ALSA card number: {self._respeaker_alsa_card}")
                    print(f"   Input channels: {self._input_channels} (always capture all channels)")
                    print(f"   Output channels: {dev['max_output_channels']}")
                    return
        
        # Fallback to mono
        self._has_respeaker = False
        self._input_channels = Config.CHANNELS
        self._respeaker_device_index = None
        self._respeaker_alsa_card = None
        print("⚠️  ReSpeaker not detected - using default audio device")
    
    def _init_webrtc_aec(self) -> None:
        """Initialize WebRTC AEC processor."""
        if not WEBRTC_AEC_AVAILABLE:
            print(f"⚠️  WebRTC AEC not available: library not installed")
            return
        
        try:
            print("🎯 Initializing WebRTC AEC...")
            self._aec_processor = WebRTCAECProcessor(
                sample_rate=Config.SAMPLE_RATE,
                channels=Config.CHANNELS,
                stream_delay_ms=Config.WEBRTC_AEC_STREAM_DELAY_MS,
                ns_level=Config.WEBRTC_AEC_NS_LEVEL,
                agc_mode=Config.WEBRTC_AEC_AGC_MODE,
            )
            self._aec_processor.start()
            
            print(f"✓ WebRTC AEC enabled:")
            print(f"   Stream delay: {Config.WEBRTC_AEC_STREAM_DELAY_MS}ms")
            print(f"   Noise suppression: {Config.WEBRTC_AEC_NS_LEVEL}")
            print(f"   AGC mode: {Config.WEBRTC_AEC_AGC_MODE}")
            
            # Apply ReSpeaker tuning for WebRTC mode
            self._configure_respeaker_for_webrtc()
            
        except Exception as e:
            print(f"⚠️  WebRTC AEC init failed: {e}")
            self._aec_processor = None
    
    def _configure_respeaker_for_webrtc(self) -> None:
        """
        Configure ReSpeaker DSP for WebRTC AEC mode (disable hardware AEC).
        
        Note: ReSpeaker configuration is NOT persistent - settings are lost when:
        - Device is unplugged/replugged
        - System reboots
        - Device is power cycled
        
        This configuration must be applied each time the application starts.
        """
        try:
            from lib.audio.respeaker import find as find_respeaker
            respeaker_dev = find_respeaker()
            
            if respeaker_dev:
                print("📊 Configuring ReSpeaker for WebRTC AEC mode...")
                # Disable hardware processing - WebRTC handles it
                respeaker_dev.write('ECHOONOFF', 0)
                respeaker_dev.write('AGCONOFF', 0)
                respeaker_dev.write('CNIONOFF', 0)
                respeaker_dev.write('STATNOISEONOFF', 0)
                respeaker_dev.write('NONSTATNOISEONOFF', 0)
                respeaker_dev.write('FREEZEONOFF', 0)  # Keep beamforming enabled
                respeaker_dev.close()
                print("✓ ReSpeaker tuned for WebRTC AEC")
                # Small delay to ensure USB device is fully released before opening audio stream
                time.sleep(0.1)
        except Exception as e:
            print(f"⚠️  Could not configure ReSpeaker: {e}")
    
    def _open_stream(self) -> None:
        """Open duplex audio stream."""
        # Use ALSA default device (None) - routes through /etc/asound.conf
        # For WebRTC AEC, we need hardware device to access all 6 channels
        # CRITICAL: Use ALSA device strings (hw:CARD,DEVICE) instead of PortAudio indices
        # to avoid PortAudio's card mapping issues
        input_device = None
        output_device = None
        
        if self._has_respeaker:
            # Always use ALSA default (None) for both input and output when ReSpeaker is detected.
            # This routes through /etc/asound.conf which:
            # - Uses respeaker_6ch for input (always captures all 6 channels via dsnoop)
            # - Uses respeaker_out/dmix for output (allows mpv to share the device)
            # 
            # Always capturing 6 channels simplifies the code:
            # - Hardware AEC: extract Ch0 (AEC-processed audio)
            # - WebRTC AEC: extract Ch0 (mic) and Ch5 (reference)
            # 
            # Using ALSA default allows both AudioManager and mpv to share the device.
            input_device = None
            output_device = None
            print(f"   Using ALSA default routing (through /etc/asound.conf)")
            print(f"   Input: 6-channel capture via respeaker_6ch (always all channels)")
            print(f"   Output: 1-channel playback via respeaker_out/dmix (allows mpv sharing)")
        else:
            # Use ALSA default routing (None, None) - goes through /etc/asound.conf
            # This works for hardware AEC mode where ALSA extracts Ch0
            input_device = None
            output_device = None
            print(f"   Using ALSA default routing (through /etc/asound.conf)")
        
        self._stream = sd.Stream(
            device=(input_device, output_device),
            samplerate=Config.SAMPLE_RATE,
            channels=(self._input_channels, Config.CHANNELS),
            dtype='int16',
            blocksize=Config.CHUNK_SIZE,
            callback=self._audio_callback
        )
        self._stream.start()
        
        print(f"✓ Audio stream opened:")
        print(f"   Sample rate: {Config.SAMPLE_RATE} Hz")
        print(f"   Block size: {Config.CHUNK_SIZE} samples")
        print(f"   Input channels: {self._input_channels}")
        print(f"   Output channels: {Config.CHANNELS}")
    
    def _audio_callback(self, indata, outdata, frames, time_info, status):
        """
        PortAudio callback for audio processing.
        
        1. Capture input → extract channels → AEC → distribute to consumers
        2. Output queue → speaker
        """
        if status:
            print(f"⚠️  AudioManager status: {status}")
        
        # =====================================================================
        # INPUT PROCESSING
        # =====================================================================
        processed_audio = self._process_input(indata)
        
        # Distribute to consumers
        with self._consumers_lock:
            for callback in self._consumers:
                try:
                    callback(processed_audio)
                except Exception as e:
                    print(f"⚠️  Consumer callback error: {e}")
        
        # =====================================================================
        # OUTPUT PROCESSING
        # =====================================================================
        self._process_output(outdata)
    
    def _process_input(self, indata: np.ndarray) -> np.ndarray:
        """
        Process input audio: channel extraction and AEC.
        
        Args:
            indata: Raw input from PortAudio (may be 6ch for ReSpeaker)
            
        Returns:
            Mono int16 array, AEC-processed if enabled.
        """
        # Extract channels for ReSpeaker
        if self._has_respeaker and indata.ndim == 2 and indata.shape[1] >= Config.RESPEAKER_CHANNELS:
            mic_channel = indata[:, Config.RESPEAKER_AEC_CHANNEL].copy()
            ref_channel = indata[:, Config.RESPEAKER_REFERENCE_CHANNEL].copy()
            
            # Debug logging
            self._log_channel_debug(indata, mic_channel, ref_channel)
            
            # WebRTC AEC processing
            if self._aec_processor is not None:
                try:
                    processed = self._aec_processor.process_chunk(mic_channel, ref_channel)
                    return processed
                except Exception as e:
                    print(f"⚠️  AEC processing error: {e}")
                    return mic_channel
            else:
                # Hardware AEC only - return Ch0
                return mic_channel
        else:
            # Non-ReSpeaker: return first channel as mono
            if indata.ndim == 2:
                return indata[:, 0].astype(np.int16)
            return indata.flatten().astype(np.int16)
    
    def _process_output(self, outdata: np.ndarray) -> None:
        """
        Process output: fill buffer from playback queue.
        
        Args:
            outdata: Output buffer to fill (modified in place).
        """
        outdata[:] = 0  # Start with silence
        frames_written = 0
        
        # Use remainder from previous callback
        if self._output_remainder is not None:
            chunk = self._output_remainder
            self._output_remainder = None
            frames_written = self._write_chunk_to_output(chunk, outdata, frames_written)
        
        # Fill from queue
        while frames_written < len(outdata):
            try:
                chunk = self._output_queue.get_nowait()
                frames_written = self._write_chunk_to_output(chunk, outdata, frames_written)
            except queue.Empty:
                break
    
    def _write_chunk_to_output(self, chunk: np.ndarray, outdata: np.ndarray, 
                                frames_written: int) -> int:
        """Write chunk to output buffer, save remainder if needed."""
        remaining = len(outdata) - frames_written
        frames_to_copy = min(len(chunk), remaining)
        
        outdata[frames_written:frames_written + frames_to_copy, 0] = chunk[:frames_to_copy]
        frames_written += frames_to_copy
        
        # Save remainder for next callback
        if len(chunk) > frames_to_copy:
            self._output_remainder = chunk[frames_to_copy:]
        
        return frames_written
    
    def _log_channel_debug(self, indata: np.ndarray, mic: np.ndarray, ref: np.ndarray) -> None:
        """Log channel RMS for debugging (every 3 seconds)."""
        now = time.time()
        if now - self._last_debug_log < self._debug_interval:
            return
        self._last_debug_log = now
        
        if not Config.SHOW_AEC_DEBUG_LOGS:
            return
        
        # Calculate RMS for each channel
        ch_rms = []
        for ch in range(min(6, indata.shape[1])):
            rms = np.sqrt(np.mean(indata[:, ch].astype(float) ** 2)) / 32768.0
            ch_rms.append(rms)
        
        aec_mode = "WebRTC" if self._aec_processor else "HW AEC"
        has_output = not self._output_queue.empty()
        status = "PLAYING" if has_output else "IDLE"
        
        print(f"📊 [AudioManager] [{status}] [{aec_mode}] " +
              " | ".join(f"Ch{i}={rms:.4f}" for i, rms in enumerate(ch_rms)))
    
    def _cleanup(self) -> None:
        """Clean up resources."""
        if self._stream:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        
        if self._aec_processor:
            try:
                self._aec_processor.stop()
            except Exception:
                pass
            self._aec_processor = None
        
        # Clear queues
        self.clear_playback_queue()
        
        # Clear consumers
        with self._consumers_lock:
            self._consumers.clear()

