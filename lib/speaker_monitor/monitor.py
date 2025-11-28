"""
Speaker Monitor - Monitors actual speaker output for agent speech detection.

On Linux/Raspberry Pi with ALSA loopback configured:
  - Reads from the 'speaker_monitor' ALSA device
  - This device captures audio being sent to the speaker via loopback

To enable, run the setup script:
  sudo raspberry-pi-client-wrapper/speaker-monitor/install-loopback.sh

Then set: SPEAKER_MONITOR_MODE=loopback
"""

import os
import sys
import time
import threading
import queue
from typing import Optional, Callable, List
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import onnxruntime as ort
import sounddevice as sd


@dataclass
class SpeakerTurn:
    """A detected speech segment from the speaker"""
    start_time: float
    end_time: Optional[float] = None
    
    @property
    def duration(self) -> Optional[float]:
        if self.end_time is None:
            return None
        return self.end_time - self.start_time


class SpeakerMonitor:
    """
    Monitors speaker output for agent speech using ALSA loopback.
    
    On Linux, requires ALSA loopback configuration (snd-aloop module).
    The install script creates a 'speaker_monitor' device that captures
    audio being sent to the speaker.
    
    To enable: Set SPEAKER_MONITOR_MODE=loopback
    """
    
    # Silero VAD requires 512 samples at 16kHz
    VAD_CHUNK_SIZE = 512
    
    def __init__(
        self,
        sample_rate: int = 16000,
        vad_threshold: float = 0.6,  # Higher for loopback noise rejection
        silence_timeout: float = 0.8,  # Shorter - agent pauses are brief
        min_turn_duration: float = 0.15,
        min_speech_frames: int = 2,  # Faster onset detection
        energy_threshold: float = 0.001,  # Minimum energy to consider
        loopback_device_name: str = "speaker_monitor",  # ALSA device name
        on_turn_start: Optional[Callable[[float], None]] = None,
        on_turn_end: Optional[Callable[[float, float], None]] = None,
    ):
        """
        Initialize speaker monitor.
        
        Args:
            sample_rate: Audio sample rate for VAD processing
            vad_threshold: VAD probability threshold (higher = less sensitive)
            silence_timeout: Seconds of silence before turn ends
            min_turn_duration: Minimum turn duration to record
            min_speech_frames: Consecutive speech frames to start turn
            energy_threshold: Minimum audio energy to process (filters noise)
            loopback_device_name: Name of ALSA loopback device
            on_turn_start: Callback when agent starts speaking (timestamp)
            on_turn_end: Callback when agent stops speaking (start, end)
        """
        self.sample_rate = sample_rate
        self.vad_threshold = vad_threshold
        self.silence_timeout = silence_timeout
        self.min_turn_duration = min_turn_duration
        self.min_speech_frames = min_speech_frames
        self.energy_threshold = energy_threshold
        self.loopback_device_name = loopback_device_name
        self.on_turn_start = on_turn_start
        self.on_turn_end = on_turn_end
        
        # Debug logging (set SPEAKER_MONITOR_DEBUG=1 to enable)
        self.debug = os.environ.get("SPEAKER_MONITOR_DEBUG", "").lower() in ("1", "true", "yes")
        
        # State
        self.enabled = False
        self.running = False
        self._thread: Optional[threading.Thread] = None
        self._audio_queue: queue.Queue = queue.Queue()
        self._loopback_device_index: Optional[int] = None
        
        # VAD state
        self._vad_session: Optional[ort.InferenceSession] = None
        self._vad_state = np.zeros((2, 1, 128), dtype=np.float32)
        self._audio_buffer = np.array([], dtype=np.float32)
        
        # Turn tracking state
        self._is_speaking = False
        self._turn_start: Optional[float] = None
        self._last_speech_time: Optional[float] = None
        self._consecutive_speech_frames = 0
        self._consecutive_silence_frames = 0
        self._speech_onset_time: Optional[float] = None
        
        # Recorded turns
        self.turns: List[SpeakerTurn] = []
        
        # Try to find loopback device
        self._loopback_device_index = self._find_loopback_device()
        
        if self._loopback_device_index is not None:
            self._load_vad_model()
            if self._vad_session is not None:
                self.enabled = True
                print(f"✓ SpeakerMonitor enabled (using ALSA {loopback_device_name})")
            else:
                print("⚠ SpeakerMonitor: VAD model not found, disabled")
        else:
            print(f"⚠ SpeakerMonitor: ALSA loopback device '{loopback_device_name}' not found")
            print(f"   For accurate agent timing, run:")
            print(f"   sudo ./raspberry-pi-client-wrapper/speaker-monitor/install-loopback.sh")
    
    def _find_loopback_device(self) -> Optional[int]:
        """Find the ALSA loopback device by name"""
        try:
            devices = sd.query_devices()
            for i, device in enumerate(devices):
                device_name = device['name'].lower()
                # Look for our speaker_monitor device or generic loopback
                if self.loopback_device_name.lower() in device_name:
                    if device['max_input_channels'] > 0:
                        return i
                # Also try to find "loopback" in the name
                if 'loopback' in device_name and device['max_input_channels'] > 0:
                    # Check if it's the capture side (subdevice 1)
                    if 'hw:' in device['name'] and ',1' in device['name']:
                        return i
        except Exception as e:
            if self.debug:
                print(f"  [SPKR] Error querying devices: {e}")
        return None
    
    def _load_vad_model(self):
        """Load Silero VAD model"""
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        model_path = os.path.join(project_root, 'models', 'silero_vad.onnx')
        
        if os.path.exists(model_path):
            try:
                self._vad_session = ort.InferenceSession(model_path)
            except Exception as e:
                print(f"⚠ SpeakerMonitor: Failed to load VAD: {e}")
    
    def start(self):
        """Start monitoring speaker output"""
        if not self.enabled:
            return
        
        if self.running:
            return
        
        self.running = True
        self.turns = []
        self._reset_state()
        
        # Start audio capture in background thread
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        print(f"[{timestamp}] 🔊 SpeakerMonitor started")
    
    def stop(self):
        """Stop monitoring"""
        if not self.running:
            return
        
        self.running = False
        
        # Finalize any ongoing turn
        self._finalize_turn()
        
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
    
    def _reset_state(self):
        """Reset VAD and turn tracking state"""
        self._vad_state = np.zeros((2, 1, 128), dtype=np.float32)
        self._audio_buffer = np.array([], dtype=np.float32)
        self._is_speaking = False
        self._turn_start = None
        self._last_speech_time = None
        self._consecutive_speech_frames = 0
        self._consecutive_silence_frames = 0
        self._speech_onset_time = None
    
    def _capture_loop(self):
        """Background thread that captures audio from loopback device"""
        try:
            def audio_callback(indata, frames, time_info, status):
                if not self.running:
                    return
                current_time = time.time()
                # Convert to mono float32
                audio = indata[:, 0].astype(np.float32) if indata.ndim > 1 else indata.flatten().astype(np.float32)
                self._audio_queue.put((current_time, audio))
            
            with sd.InputStream(
                device=self._loopback_device_index,
                samplerate=self.sample_rate,
                channels=1,
                dtype='float32',
                blocksize=self.VAD_CHUNK_SIZE,
                callback=audio_callback
            ):
                while self.running:
                    try:
                        timestamp, audio = self._audio_queue.get(timeout=0.1)
                        
                        # Add to buffer and process
                        self._audio_buffer = np.concatenate([self._audio_buffer, audio])
                        
                        while len(self._audio_buffer) >= self.VAD_CHUNK_SIZE:
                            chunk = self._audio_buffer[:self.VAD_CHUNK_SIZE]
                            self._audio_buffer = self._audio_buffer[self.VAD_CHUNK_SIZE:]
                            self._process_chunk(chunk, timestamp)
                            
                    except queue.Empty:
                        self._check_silence_timeout()
                        
        except Exception as e:
            print(f"✗ SpeakerMonitor capture error: {e}")
            self.running = False
    
    def _process_chunk(self, audio_chunk: np.ndarray, chunk_time: float):
        """Process a single audio chunk through VAD"""
        if self._vad_session is None:
            return
        
        # Energy gating - skip very quiet chunks (loopback noise)
        energy = np.mean(np.abs(audio_chunk))
        if energy < self.energy_threshold:
            if self.debug and self._is_speaking:
                print(f"  [SPKR] energy={energy:.6f} < threshold, treating as silence")
            # Treat as silence
            self._handle_silence(chunk_time)
            return
        
        # Run VAD inference
        try:
            ort_inputs = {
                'input': audio_chunk.reshape(1, -1).astype(np.float32),
                'sr': np.array([self.sample_rate], dtype=np.int64),
                'state': self._vad_state
            }
            outs = self._vad_session.run(None, ort_inputs)
            speech_prob = outs[0][0][0]
            self._vad_state = outs[1]
        except Exception:
            return
        
        is_speech = speech_prob > self.vad_threshold
        
        if self.debug:
            state_str = "SPEAKING" if self._is_speaking else ("ONSET" if self._speech_onset_time else "IDLE")
            speech_str = "SPEECH" if is_speech else "silence"
            print(f"  [SPKR] energy={energy:.4f} prob={speech_prob:.3f} {speech_str} | state={state_str} frames=+{self._consecutive_speech_frames}/-{self._consecutive_silence_frames}")
        
        if is_speech:
            self._last_speech_time = chunk_time
            self._consecutive_speech_frames += 1
            self._consecutive_silence_frames = 0
            
            if not self._is_speaking:
                if self._speech_onset_time is None:
                    self._speech_onset_time = chunk_time
                
                # Start turn after enough consecutive speech frames
                if self._consecutive_speech_frames >= self.min_speech_frames:
                    self._is_speaking = True
                    self._turn_start = self._speech_onset_time
                    self._speech_onset_time = None
                    self._consecutive_speech_frames = 0
                    
                    # Log and callback
                    timestamp = datetime.fromtimestamp(self._turn_start).strftime("%H:%M:%S.%f")[:-3]
                    print(f"[{timestamp}] 🔊 SPEAKER_START")
                    
                    if self.on_turn_start:
                        self.on_turn_start(self._turn_start)
        else:
            self._handle_silence(chunk_time)
    
    def _handle_silence(self, chunk_time: float):
        """Handle a silence frame"""
        self._consecutive_silence_frames += 1
        self._consecutive_speech_frames = 0
        
        # Reset onset after sustained silence
        if self._speech_onset_time is not None and not self._is_speaking:
            if self._consecutive_silence_frames >= 2:
                if self.debug:
                    print(f"  [SPKR] onset reset after {self._consecutive_silence_frames} silence frames")
                self._speech_onset_time = None
        
        # End turn after silence timeout
        if self._is_speaking and self._last_speech_time is not None:
            silence_duration = chunk_time - self._last_speech_time
            if self.debug and self._consecutive_silence_frames % 10 == 0:
                print(f"  [SPKR] silence for {silence_duration:.2f}s (timeout={self.silence_timeout}s)")
            if silence_duration > self.silence_timeout:
                self._end_turn()
    
    def _check_silence_timeout(self):
        """Check if silence timeout exceeded (called periodically)"""
        if not self._is_speaking or self._last_speech_time is None:
            return
        
        silence_duration = time.time() - self._last_speech_time
        
        if silence_duration > self.silence_timeout:
            self._end_turn()
    
    def _end_turn(self):
        """End the current turn"""
        if not self._is_speaking or self._turn_start is None:
            return
        
        end_time = self._last_speech_time or time.time()
        duration = end_time - self._turn_start
        
        if duration >= self.min_turn_duration:
            turn = SpeakerTurn(start_time=self._turn_start, end_time=end_time)
            self.turns.append(turn)
            
            # Log and callback
            timestamp = datetime.fromtimestamp(end_time).strftime("%H:%M:%S.%f")[:-3]
            print(f"[{timestamp}] 🔊 SPEAKER_END ({duration:.2f}s)")
            
            if self.on_turn_end:
                self.on_turn_end(self._turn_start, end_time)
        
        # Reset state
        self._is_speaking = False
        self._turn_start = None
    
    def _finalize_turn(self):
        """Finalize any ongoing turn at stop"""
        if self._is_speaking and self._turn_start is not None:
            self._end_turn()
        elif self._speech_onset_time is not None and self._last_speech_time is not None:
            # Capture onset-phase turn
            duration = self._last_speech_time - self._speech_onset_time
            if duration >= self.min_turn_duration * 0.5:
                turn = SpeakerTurn(
                    start_time=self._speech_onset_time,
                    end_time=self._last_speech_time
                )
                self.turns.append(turn)
                
                timestamp = datetime.fromtimestamp(self._last_speech_time).strftime("%H:%M:%S.%f")[:-3]
                print(f"[{timestamp}] 🔊 SPEAKER_END (finalized) ({duration:.2f}s)")
    
    def get_turns(self) -> List[SpeakerTurn]:
        """Get all detected speaker turns"""
        return self.turns.copy()

