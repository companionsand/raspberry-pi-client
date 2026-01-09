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
from typing import TYPE_CHECKING, Callable, List, Optional

import numpy as np
import sounddevice as sd

from lib.config import Config
from lib.signals.ring_buffer import RingBuffer

if TYPE_CHECKING:
    from lib.signals.bus import SignalBus

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
    
    def __init__(self, use_webrtc_aec: bool = None, signal_bus: Optional["SignalBus"] = None):
        """
        Initialize AudioManager.
        
        Args:
            use_webrtc_aec: Enable WebRTC AEC. If None, uses Config.USE_WEBRTC_AEC.
            signal_bus: Optional SignalBus for publishing audio signals.
        """
        # AEC configuration
        if use_webrtc_aec is None:
            use_webrtc_aec = Config.USE_WEBRTC_AEC
        self._use_webrtc_aec = use_webrtc_aec and WEBRTC_AEC_AVAILABLE
        
        # Stream state (separate streams for ALSA asym compatibility)
        self._input_stream: Optional[sd.InputStream] = None
        self._output_stream: Optional[sd.OutputStream] = None
        self._is_running = False
        self._lock = threading.Lock()
        
        # Watchdog state
        self._last_callback_time = 0.0
        self._watchdog_thread = None
        self._stop_watchdog_event = threading.Event()
        
        # AEC processor
        self._aec_processor: Optional[WebRTCAECProcessor] = None
        
        # Consumer callbacks: list of (callback, include_raw) tuples
        # include_raw: if True, callback receives (processed, raw) tuple
        self._consumers: List[Callable[[np.ndarray], None]] = []
        self._consumers_lock = threading.Lock()
        
        # Audio distribution queue and thread (non-blocking consumer callbacks)
        self._audio_distribution_queue: queue.Queue = queue.Queue(maxsize=10)  # Limit queue size to prevent memory buildup
        self._distribution_thread: Optional[threading.Thread] = None
        self._stop_distribution_event = threading.Event()
        
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
        self._first_frame_debug = True  # One-time debug for audio format
        
        # Status warning rate limiting
        self._last_input_status_warning = 0
        self._last_output_status_warning = 0
        self._status_warning_interval = 5.0  # Only log status warnings every 5 seconds
        self._status_warning_count = 0  # Count warnings between logs
        
        # Logger
        self._logger = Config.LOGGER
        
        # Signal bus for publishing audio events (optional)
        self._signal_bus: Optional["SignalBus"] = signal_bus
        
        # Ring buffers for visualization (10 seconds of audio each)
        # These store audio for GUI/debugging access without affecting the main audio flow
        buffer_size = Config.SAMPLE_RATE * 10  # 10 seconds at 16kHz = 160,000 samples
        self._aec_input_buffer = RingBuffer(buffer_size, channels=1, dtype=np.int16)
        self._agent_output_buffer = RingBuffer(buffer_size, channels=1, dtype=np.int16)
        self._raw_input_buffer: Optional[RingBuffer] = None  # Lazily initialized if ReSpeaker detected
        
        # RMS calculation for scalar signals (computed in distribution thread)
        self._last_rms_publish_time = 0.0
        self._rms_publish_interval = 0.05  # Publish RMS every 50ms
    
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
    
    def set_signal_bus(self, signal_bus: "SignalBus") -> None:
        """Set or update the signal bus."""
        self._signal_bus = signal_bus
    
    def get_audio_window(self, stream: str, seconds: float) -> np.ndarray:
        """
        Get recent audio from a stream's ring buffer.
        
        This provides zero-copy access (when possible) to recent audio
        for visualization or analysis.
        
        Args:
            stream: Stream name:
                - "aec_input": Echo-cancelled microphone input (mono)
                - "agent_output": Audio sent via play() method (mono)
                - "raw_input": Raw 6-channel ReSpeaker input
                - "speaker_loopback": Channel 5 from ReSpeaker (actual speaker output)
            seconds: Duration of audio to retrieve
            
        Returns:
            NumPy array of audio samples
        """
        # Handle speaker_loopback specially - extract channel 5 from raw_input
        if stream == "speaker_loopback":
            if self._raw_input_buffer is None:
                return np.array([], dtype=np.int16)
            
            raw_data = self._raw_input_buffer.get_window_seconds(seconds, Config.SAMPLE_RATE)
            if len(raw_data) == 0 or raw_data.ndim != 2:
                return np.array([], dtype=np.int16)
            
            # Extract channel 5 (playback loopback from ReSpeaker)
            return raw_data[:, Config.RESPEAKER_REFERENCE_CHANNEL].copy()
        
        buffers = {
            "aec_input": self._aec_input_buffer,
            "agent_output": self._agent_output_buffer,
            "raw_input": self._raw_input_buffer,
        }
        
        buffer = buffers.get(stream)
        if buffer is None:
            return np.array([], dtype=np.int16)
        
        return buffer.get_window_seconds(seconds, Config.SAMPLE_RATE)
    
    def get_stream_last_write_time(self, stream: str) -> float:
        """
        Get the timestamp of when data was last written to a stream.
        
        Useful for detecting stale data (e.g., agent_output when not speaking).
        
        Args:
            stream: Stream name (aec_input, agent_output, raw_input, speaker_loopback)
            
        Returns:
            Monotonic timestamp of last write, or 0.0 if no writes yet
        """
        if stream == "speaker_loopback":
            stream = "raw_input"  # speaker_loopback is derived from raw_input
        
        buffers = {
            "aec_input": self._aec_input_buffer,
            "agent_output": self._agent_output_buffer,
            "raw_input": self._raw_input_buffer,
        }
        
        buffer = buffers.get(stream)
        if buffer is None:
            return 0.0
        
        return buffer.last_write_time
    
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
                
                # Open separate input/output streams
                self._open_stream()
                
                self._is_running = True
                
                # Start watchdog
                self._start_watchdog()
                
                # Start audio distribution thread (non-blocking consumer callbacks)
                self._start_distribution_thread()
                
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
            # Stop watchdog and distribution thread FIRST to prevent race conditions during cleanup
            self._is_running = False
            self._stop_watchdog()
            self._stop_distribution_thread()
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
        
        # Queue for playback (buffer write happens in _process_output for accurate timing)
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
        # Retry detection a few times to handle intermittent USB initialization delays
        max_attempts = 3
        
        for attempt in range(max_attempts):
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
                        
                        # Initialize raw input ring buffer for 6-channel capture
                        buffer_size = Config.SAMPLE_RATE * 10  # 10 seconds
                        self._raw_input_buffer = RingBuffer(buffer_size, channels=6, dtype=np.int16)
                        
                        print(f"✓ ReSpeaker detected: {dev['name']}")
                        print(f"   PortAudio device index: {idx}")
                        if self._respeaker_alsa_card is not None:
                            print(f"   ALSA card number: {self._respeaker_alsa_card}")
                        print(f"   Input channels: {self._input_channels} (always capture all channels)")
                        print(f"   Output channels: {dev['max_output_channels']}")
                        return
            
            # If not found, wait briefly before retrying
            if attempt < max_attempts - 1:
                # Only log retry if we're actually looking for it
                print(f"ℹ️  ReSpeaker not found, retrying detection ({attempt + 1}/{max_attempts})...")
                time.sleep(1.0)
        
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
                chunk_size=Config.CHUNK_SIZE,
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
        
        # Open separate input and output streams (ALSA asym compatibility)
        print("📡 Opening separate input/output streams for ALSA asym compatibility...")
        
        # Input stream
        self._input_stream = sd.InputStream(
            device=input_device,
            samplerate=Config.SAMPLE_RATE,
            channels=self._input_channels,
            dtype='int16',
            blocksize=Config.CHUNK_SIZE,
            latency='high',  # Prevent input overflow with larger buffer
            callback=self._input_callback
        )
        self._input_stream.start()
        print(f"✓ Input stream opened: {self._input_channels} channels @ {Config.SAMPLE_RATE} Hz (latency=high)")
        
        # Output stream
        self._output_stream = sd.OutputStream(
            device=output_device,
            samplerate=Config.SAMPLE_RATE,
            channels=Config.CHANNELS,
            dtype='int16',
            blocksize=Config.CHUNK_SIZE,
            latency='high',  # Prevent output underflow with larger buffer
            callback=self._output_callback
        )
        self._output_stream.start()
        print(f"✓ Output stream opened: {Config.CHANNELS} channels @ {Config.SAMPLE_RATE} Hz (latency=high)")
    
    def _input_callback(self, indata, frames, time_info, status):
        """Input stream callback (capture)."""
        # DIAGNOSTIC: Print on first callback
        if not hasattr(self, '_callback_count'):
            self._callback_count = 0
        self._callback_count += 1
        if self._callback_count == 1:
            print(f"🎤 Input callback started (indata.shape={indata.shape})")
        
        # Rate-limit status warnings to avoid console spam (which worsens the problem)
        if status:
            now = time.time()
            self._status_warning_count += 1
            
            # Only log every N seconds to avoid slowing down the callback
            if now - self._last_input_status_warning >= self._status_warning_interval:
                if self._status_warning_count > 1:
                    print(f"⚠️  AudioManager input status: {status} (occurred {self._status_warning_count} times in last {self._status_warning_interval:.0f}s)")
                else:
                    print(f"⚠️  AudioManager input status: {status}")
                self._last_input_status_warning = now
                self._status_warning_count = 0
        
        # Update watchdog timestamp
        self._last_callback_time = time.time()
        
        # Write raw input to ring buffer (if ReSpeaker with 6 channels)
        if self._raw_input_buffer is not None and indata.ndim == 2:
            self._raw_input_buffer.write(indata)
        
        # Process input
        processed_audio = self._process_input(indata)
        
        # Enqueue for non-blocking distribution to consumers
        # Use non-blocking put to avoid blocking the audio callback if queue is full
        try:
            self._audio_distribution_queue.put_nowait(processed_audio.copy())
        except queue.Full:
            # Queue is full - drop this frame to prevent blocking
            # This is better than blocking the audio callback
            pass
    
    def _output_callback(self, outdata, frames, time_info, status):
        """Output stream callback (playback)."""
        # Rate-limit status warnings to avoid console spam
        if status:
            now = time.time()
            # Only log every N seconds
            if now - self._last_output_status_warning >= self._status_warning_interval:
                print(f"⚠️  AudioManager output status: {status}")
                self._last_output_status_warning = now
        
        # Fill output buffer
        self._process_output(outdata)
    
    def _audio_callback(self, indata, outdata, frames, time_info, status):
        """
        PortAudio callback for audio processing.
        
        1. Capture input → extract channels → AEC → distribute to consumers
        2. Output queue → speaker
        """
        # DIAGNOSTIC: Print on first callback to confirm it's being called
        if not hasattr(self, '_callback_count'):
            self._callback_count = 0
        self._callback_count += 1
        if self._callback_count == 1:
            print(f"🎤 DIAGNOSTIC: Audio callback IS being called (indata.shape={indata.shape})")
        
        if status:
            print(f"⚠️  AudioManager status: {status}")
        
        # Update watchdog timestamp
        self._last_callback_time = time.time()
        
        # =====================================================================
        # INPUT PROCESSING
        # =====================================================================
        processed_audio = self._process_input(indata)
        
        # Enqueue for non-blocking distribution to consumers
        try:
            self._audio_distribution_queue.put_nowait(processed_audio.copy())
        except queue.Full:
            # Queue is full - drop this frame to prevent blocking
            pass
        
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
        start_time = time.time()
        
        # Diagnostic logging for first frame (to debug intermittent missing stats)
        if self._first_frame_debug:
            self._first_frame_debug = False
            print(f"🎤 First audio frame: shape={indata.shape}, dtype={indata.dtype}, "
                  f"respeaker={self._has_respeaker}, channels={self._input_channels}")
            
            if self._has_respeaker and (indata.ndim != 2 or indata.shape[1] < Config.RESPEAKER_CHANNELS):
                print(f"⚠️  ReSpeaker detected but audio format incorrect for channel extraction!")
        
        # Extract channels for ReSpeaker
        if self._has_respeaker and indata.ndim == 2 and indata.shape[1] >= Config.RESPEAKER_CHANNELS:
            mic_channel = indata[:, Config.RESPEAKER_AEC_CHANNEL].copy()
            ref_channel = indata[:, Config.RESPEAKER_REFERENCE_CHANNEL].copy()
            
            # Debug logging
            self._log_channel_debug(indata, mic_channel, ref_channel)
            
            # WebRTC AEC processing
            result = None
            if self._aec_processor is not None:
                try:
                    result = self._aec_processor.process_chunk(mic_channel, ref_channel)
                except Exception as e:
                    print(f"⚠️  AEC processing error: {e}")
                    result = mic_channel
            else:
                # Hardware AEC only - return Ch0
                result = mic_channel
            
            # Performance check
            duration = (time.time() - start_time) * 1000  # ms
            if duration > 15.0:  # Warning threshold (block size is 20ms)
                print(f"⚠️  AudioManager CPU spike: input processing took {duration:.1f}ms")
            
            return result
        else:
            # Non-ReSpeaker: return first channel as mono
            result = None
            if indata.ndim == 2:
                result = indata[:, 0].astype(np.int16)
            else:
                result = indata.flatten().astype(np.int16)
            
            # Performance check
            duration = (time.time() - start_time) * 1000  # ms
            if duration > 15.0:
                print(f"⚠️  AudioManager CPU spike: input processing took {duration:.1f}ms")
            
            return result
    
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
        
        # Write actual output to agent_output buffer for visualization
        # This captures exactly what's going to the speaker (audio or silence)
        if outdata.ndim > 1:
            # Extract mono channel for buffer
            self._agent_output_buffer.write(outdata[:, 0].astype(np.int16))
        else:
            self._agent_output_buffer.write(outdata.astype(np.int16))
    
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
    
    def _start_watchdog(self) -> None:
        """Start the audio stream watchdog thread."""
        self._stop_watchdog_event.clear()
        self._last_callback_time = time.time()  # Reset time
        self._watchdog_thread = threading.Thread(target=self._watchdog_loop, daemon=True)
        self._watchdog_thread.start()
        print("✓ AudioManager watchdog started")
    
    def _start_distribution_thread(self) -> None:
        """Start the audio distribution thread for non-blocking consumer callbacks."""
        self._stop_distribution_event.clear()
        self._distribution_thread = threading.Thread(target=self._distribution_loop, daemon=True)
        self._distribution_thread.start()
        print("✓ AudioManager distribution thread started")
    
    def _stop_distribution_thread(self) -> None:
        """Stop the audio distribution thread."""
        if self._distribution_thread:
            self._stop_distribution_event.set()
            self._distribution_thread.join(timeout=1.0)
            self._distribution_thread = None
    
    def _distribution_loop(self) -> None:
        """Pull audio from queue and distribute to consumers (non-blocking)."""
        while not self._stop_distribution_event.is_set():
            try:
                # Get audio from queue with timeout to allow checking stop event
                try:
                    processed_audio = self._audio_distribution_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                
                # Write to AEC input ring buffer for visualization
                self._aec_input_buffer.write(processed_audio)
                
                # Publish RMS signal periodically (every 50ms)
                now = time.time()
                if self._signal_bus is not None and (now - self._last_rms_publish_time) >= self._rms_publish_interval:
                    self._last_rms_publish_time = now
                    # Calculate RMS (normalized to 0-1 range for int16)
                    rms = np.sqrt(np.mean(processed_audio.astype(np.float32) ** 2)) / 32768.0
                    from lib.signals.base import ScalarSignal
                    self._signal_bus.publish(ScalarSignal(
                        timestamp=time.monotonic(),
                        source="audio_manager",
                        name="input_rms",
                        value=float(rms)
                    ))
                
                # Distribute to all consumers
                with self._consumers_lock:
                    for callback in self._consumers:
                        try:
                            callback(processed_audio)
                        except Exception as e:
                            print(f"⚠️  Consumer callback error: {e}")
                            
            except Exception as e:
                # Log but continue - don't let one error stop distribution
                print(f"⚠️  Audio distribution error: {e}")
                time.sleep(0.01)  # Brief pause before retrying

    def _stop_watchdog(self) -> None:
        """Stop the watchdog thread."""
        if self._watchdog_thread:
            self._stop_watchdog_event.set()
            self._watchdog_thread.join(timeout=1.0)
            self._watchdog_thread = None

    def _watchdog_loop(self) -> None:
        """Monitor audio stream health and restart if frozen."""
        TIMEOUT_SECONDS = 10.0  # Increased from 5.0s to avoid false positives
        CHECK_INTERVAL = 1.0
        
        while not self._stop_watchdog_event.is_set():
            time.sleep(CHECK_INTERVAL)
            
            # Check if stream is frozen
            time_since_last = time.time() - self._last_callback_time
            if time_since_last > TIMEOUT_SECONDS and self._is_running:
                print(f"⚠️  AudioManager frozen (no callback for {time_since_last:.1f}s) - Restarting stream...")
                
                # Double check we haven't been stopped while waiting
                if self._stop_watchdog_event.is_set() or not self._is_running:
                    return
                
                # Restart stream logic
                # Note: We can't call stop()/start() directly because of lock recursion potential
                # and we're in a background thread.
                try:
                    # 1. Close existing streams
                    if self._input_stream:
                        try:
                            self._input_stream.stop()
                            self._input_stream.close()
                        except Exception:
                            pass
                        self._input_stream = None
                    
                    if self._output_stream:
                        try:
                            self._output_stream.stop()
                            self._output_stream.close()
                        except Exception:
                            pass
                        self._output_stream = None
                    
                    # 2. Re-detect device (in case it was unplugged/reset)
                    try:
                        self._detect_respeaker()
                    except Exception as e:
                        print(f"⚠️  Watchdog re-detection failed: {e}")
                    
                    # 3. Re-open streams
                    self._open_stream()
                    self._last_callback_time = time.time()  # Reset timer
                    print("✓ AudioManager streams restarted by watchdog")
                except Exception as e:
                    print(f"✗ Watchdog failed to restart stream: {e}")
                    # Back off to avoid tight loop
                    time.sleep(5.0)

    def _cleanup(self) -> None:
        """Clean up resources."""
        if self._input_stream:
            try:
                self._input_stream.stop()
                self._input_stream.close()
            except Exception:
                pass
            self._input_stream = None
        
        if self._output_stream:
            try:
                self._output_stream.stop()
                self._output_stream.close()
            except Exception:
                pass
            self._output_stream = None
        
        if self._aec_processor:
            try:
                self._aec_processor.stop()
            except Exception:
                pass
            self._aec_processor = None
        
        # Clear queues
        self.clear_playback_queue()
        
        # Clear audio distribution queue
        while not self._audio_distribution_queue.empty():
            try:
                self._audio_distribution_queue.get_nowait()
            except queue.Empty:
                break
        
        # Clear consumers
        with self._consumers_lock:
            self._consumers.clear()

