"""
Voice Command Detector for Music Playback Control

Detects voice commands using VAD + Scribe ASR + fuzzy matching.
Supports: stop, pause, resume, volume up, volume down

Uses a rolling buffer to capture trailing audio for better transcription.
"""

import os
import time
import asyncio
import base64
from collections import deque
from typing import Optional, Callable
from enum import Enum
import numpy as np
import onnxruntime as ort
import sounddevice as sd
from elevenlabs import (
    AudioFormat,
    CommitStrategy,
    ElevenLabs,
    RealtimeAudioOptions,
)
from elevenlabs.realtime.connection import RealtimeEvents
from lib.config import Config

# Try to import rapidfuzz, fall back to fuzzywuzzy
try:
    from rapidfuzz import fuzz, process
    FUZZY_LIB = "rapidfuzz"
except ImportError:
    try:
        from fuzzywuzzy import fuzz, process
        FUZZY_LIB = "fuzzywuzzy"
    except ImportError:
        fuzz = None
        process = None
        FUZZY_LIB = None


class MusicCommand(Enum):
    """Voice commands for music control"""
    STOP = "stop"
    PAUSE = "pause"
    RESUME = "resume"
    VOLUME_UP = "volume_up"
    VOLUME_DOWN = "volume_down"
    NONE = "none"


# Command patterns for fuzzy matching
# Each command has multiple phrases that map to it
COMMAND_PATTERNS = {
    MusicCommand.STOP: ["stop", "stop music", "stop the music", "stop playing", "turn it off", "off"],
    MusicCommand.PAUSE: ["pause", "pause music", "pause the music", "hold", "wait"],
    MusicCommand.RESUME: ["resume", "play", "continue", "unpause", "keep playing", "go on"],
    MusicCommand.VOLUME_UP: ["volume up", "louder", "turn it up", "raise volume", "increase volume", "more volume"],
    MusicCommand.VOLUME_DOWN: ["volume down", "quieter", "turn it down", "lower volume", "decrease volume", "less volume"],
}

# Minimum fuzzy match score to accept (0-100)
FUZZY_THRESHOLD = 60


class VoiceCommandDetector:
    """
    Detects voice commands for music control using VAD-gated Scribe transcription.
    
    This detector:
    1. Maintains a rolling 2s audio buffer
    2. Uses VAD to detect speech
    3. When speech ends, sends trailing 2s to Scribe
    4. Fuzzy matches transcript against command patterns
    """
    
    def __init__(
        self,
        mic_device_index: Optional[int] = None,
        on_command: Optional[Callable[[MusicCommand], None]] = None
    ):
        """
        Initialize voice command detector.
        
        Args:
            mic_device_index: Microphone device index (None for default)
            on_command: Callback function when command is detected
        """
        self.mic_device_index = mic_device_index
        self.on_command = on_command
        self.running = False
        self.last_command = MusicCommand.NONE
        self.audio_stream = None
        self.logger = Config.LOGGER
        
        # Audio settings
        self.sample_rate = Config.SAMPLE_RATE  # 16kHz
        self.chunk_size = Config.CHUNK_SIZE  # 512 samples
        
        # ReSpeaker AEC detection
        self._use_respeaker_aec = False
        
        # -------------------------------------------------------------------------
        # Rolling buffer: 2s of audio (for trailing context)
        # At 16kHz, 2s = 32000 samples
        # With 512 samples/chunk, that's ~63 chunks
        # -------------------------------------------------------------------------
        self._rolling_buffer_seconds = 2.0
        self._rolling_buffer_samples = int(self.sample_rate * self._rolling_buffer_seconds)
        self._rolling_buffer_chunks = self._rolling_buffer_samples // self.chunk_size + 1
        self._rolling_buffer = deque(maxlen=self._rolling_buffer_chunks)
        
        # -------------------------------------------------------------------------
        # Silero VAD initialization
        # -------------------------------------------------------------------------
        self._vad_session = None
        self._vad_enabled = False
        
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        model_path = os.path.join(project_root, 'models', 'silero_vad.onnx')
        
        if os.path.exists(model_path):
            try:
                self._vad_session = ort.InferenceSession(model_path)
                self._vad_enabled = True
            except Exception as e:
                print(f"⚠️  VAD init failed: {e}")
        
        # VAD state (Silero VAD v5)
        self._vad_state = np.zeros((2, 1, 128), dtype=np.float32)
        self._vad_threshold = 0.4  # Slightly higher for noisy music environment
        
        # Speech state tracking
        self._is_speaking = False
        self._silence_frames = 0
        self._silence_threshold_frames = 15  # ~500ms of silence ends speech
        self._min_speech_frames = 6  # Minimum frames before processing (~200ms)
        self._speech_frame_count = 0
        
        # Scribe configuration
        self._scribe_timeout = 3.0
    
    async def start(self) -> MusicCommand:
        """
        Start the voice command detector.
        
        This method blocks until a "stop" command is detected or stop() is called.
        Other commands (pause, resume, volume) are handled via callback and continue listening.
        
        Returns:
            The command that caused the detector to stop (usually STOP or NONE if stopped externally)
        """
        if self.running:
            return MusicCommand.NONE
        
        self.running = True
        self.last_command = MusicCommand.NONE
        
        print("🎤 Voice command detector started")
        print("   Commands: stop, pause, resume, volume up, volume down")
        
        if self.logger:
            self.logger.info(
                "voice_command_detector_started",
                extra={"device_id": Config.DEVICE_ID}
            )
        
        try:
            # Detect ReSpeaker for AEC channel extraction
            input_channels = Config.CHANNELS
            try:
                devices = sd.query_devices()
                input_dev = None
                for dev in devices:
                    if any(kw in dev['name'].lower() for kw in ['respeaker', 'arrayuac10', 'uac1.0']):
                        if dev['max_input_channels'] >= Config.RESPEAKER_CHANNELS:
                            input_dev = dev
                            break
                
                if input_dev and input_dev['max_input_channels'] >= Config.RESPEAKER_CHANNELS:
                    self._use_respeaker_aec = True
                    input_channels = Config.RESPEAKER_CHANNELS
                    print(f"   ✓ Using ReSpeaker Ch{Config.RESPEAKER_AEC_CHANNEL} for voice detection")
            except Exception as e:
                print(f"   ⚠️  Device detection error: {e}")
            
            # Start audio stream
            self.audio_stream = sd.InputStream(
                device=self.mic_device_index,
                channels=input_channels,
                samplerate=self.sample_rate,
                blocksize=self.chunk_size,
                dtype='int16'
            )
            self.audio_stream.start()
            
            # Main detection loop
            while self.running:
                # Read audio frame
                audio_data, _ = self.audio_stream.read(self.chunk_size)
                
                # Extract channel for ReSpeaker
                if self._use_respeaker_aec and audio_data.ndim == 2 and audio_data.shape[1] >= Config.RESPEAKER_CHANNELS:
                    audio_frame = audio_data[:, Config.RESPEAKER_AEC_CHANNEL].astype(np.int16)
                else:
                    audio_frame = audio_data[:, 0].astype(np.int16) if audio_data.ndim == 2 else audio_data.astype(np.int16)
                
                # Always add to rolling buffer
                self._rolling_buffer.append(audio_frame.copy())
                
                # Run VAD
                speech_detected = await self._process_vad(audio_frame)
                
                if speech_detected:
                    self._speech_frame_count += 1
                    self._silence_frames = 0
                    
                    if not self._is_speaking:
                        self._is_speaking = True
                        print("   🎙️  Speech detected...")
                else:
                    if self._is_speaking:
                        self._silence_frames += 1
                        
                        # Check if speech segment ended
                        if self._silence_frames >= self._silence_threshold_frames:
                            if self._speech_frame_count >= self._min_speech_frames:
                                # Process the trailing audio from rolling buffer
                                print(f"   📝 Processing speech...")
                                command = await self._process_command()
                                
                                if command != MusicCommand.NONE:
                                    self.last_command = command
                                    
                                    # Call callback
                                    if self.on_command:
                                        self.on_command(command)
                                    
                                    # Stop detector on STOP command
                                    if command == MusicCommand.STOP:
                                        self.running = False
                                        return command
                            
                            # Reset state
                            self._is_speaking = False
                            self._speech_frame_count = 0
                
                # Small delay to prevent busy loop
                await asyncio.sleep(0.001)
            
            return self.last_command
            
        except Exception as e:
            print(f"✗ Voice command detector error: {e}")
            if self.logger:
                self.logger.error(
                    "voice_command_detector_error",
                    extra={
                        "error": str(e),
                        "device_id": Config.DEVICE_ID
                    }
                )
            return MusicCommand.NONE
            
        finally:
            self._cleanup_stream()
    
    async def _process_vad(self, audio_frame: np.ndarray) -> bool:
        """Run VAD on audio frame and return True if speech detected."""
        if not self._vad_enabled or self._vad_session is None:
            # Fallback: simple energy detection
            return np.abs(audio_frame).mean() > 500
        
        try:
            # Convert to float32 normalized
            audio_float = audio_frame.astype(np.float32) / 32768.0
            
            # Skip very quiet audio
            if np.max(np.abs(audio_float)) < 0.01:
                return False
            
            # Run VAD inference
            ort_inputs = {
                'input': audio_float.reshape(1, -1),
                'sr': np.array([self.sample_rate], dtype=np.int64),
                'state': self._vad_state
            }
            
            outs = self._vad_session.run(None, ort_inputs)
            speech_prob = outs[0][0][0]
            self._vad_state = outs[1]
            
            return speech_prob >= self._vad_threshold
            
        except Exception as e:
            # Fallback on error
            return np.abs(audio_frame).mean() > 500
    
    async def _process_command(self) -> MusicCommand:
        """Transcribe rolling buffer audio and match to command."""
        if not Config.ELEVENLABS_API_KEY:
            print("   ⚠️  No ElevenLabs API key - cannot transcribe")
            return MusicCommand.NONE
        
        if len(self._rolling_buffer) == 0:
            return MusicCommand.NONE
        
        try:
            # Concatenate rolling buffer (trailing 2s of audio)
            audio_data = np.concatenate(list(self._rolling_buffer))
            audio_bytes = audio_data.tobytes()
            audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
            
            # Initialize ElevenLabs client
            client = ElevenLabs(api_key=Config.ELEVENLABS_API_KEY)
            
            transcript_text = None
            
            async def transcribe():
                nonlocal transcript_text
                connection = await client.speech_to_text.realtime.connect(
                    RealtimeAudioOptions(
                        model_id="scribe_v2_realtime",
                        audio_format=AudioFormat.PCM_16000,
                        sample_rate=self.sample_rate,
                        commit_strategy=CommitStrategy.MANUAL,
                        language_code="en",
                    )
                )
                
                loop = asyncio.get_event_loop()
                transcript_future = loop.create_future()
                
                def handle_committed(data):
                    nonlocal transcript_text
                    transcript_text = data.get("text") or data.get("transcript") or ""
                    if not transcript_future.done():
                        transcript_future.set_result(transcript_text)
                
                def handle_error(data):
                    err = data.get("error") or data.get("message") or "Unknown error"
                    if not transcript_future.done():
                        transcript_future.set_exception(RuntimeError(err))
                
                connection.on(RealtimeEvents.COMMITTED_TRANSCRIPT, handle_committed)
                connection.on(RealtimeEvents.COMMITTED_TRANSCRIPT_WITH_TIMESTAMPS, handle_committed)
                connection.on(RealtimeEvents.ERROR, handle_error)
                connection.on(RealtimeEvents.AUTH_ERROR, handle_error)
                connection.on(RealtimeEvents.QUOTA_EXCEEDED, handle_error)
                
                try:
                    # Send audio in chunks
                    chunk_size = 4096
                    for i in range(0, len(audio_base64), chunk_size):
                        chunk = audio_base64[i:i+chunk_size]
                        await connection.send({
                            "audio_base_64": chunk,
                            "sample_rate": self.sample_rate
                        })
                    
                    # Commit and wait for result
                    await connection.commit()
                    await asyncio.wait_for(transcript_future, timeout=self._scribe_timeout)
                    await connection.close()
                    
                except asyncio.TimeoutError:
                    print("   ⚠️  Transcription timeout")
                    await connection.close()
            
            # Run transcription
            await asyncio.wait_for(transcribe(), timeout=self._scribe_timeout + 1.0)
            
            if transcript_text:
                print(f"   📝 Heard: '{transcript_text}'")
                
                # Match transcript to command using fuzzy matching
                command = self._match_command(transcript_text)
                
                if command != MusicCommand.NONE:
                    print(f"   ✓ Command: {command.value}")
                    
                    if self.logger:
                        self.logger.info(
                            "voice_command_detected",
                            extra={
                                "transcript": transcript_text,
                                "command": command.value,
                                "device_id": Config.DEVICE_ID
                            }
                        )
                
                return command
            else:
                print("   ⚠️  Empty transcript")
                return MusicCommand.NONE
                
        except Exception as e:
            print(f"   ⚠️  Transcription error: {e}")
            if self.logger:
                self.logger.error(
                    "voice_command_transcription_error",
                    extra={
                        "error": str(e),
                        "device_id": Config.DEVICE_ID
                    }
                )
            return MusicCommand.NONE
    
    def _match_command(self, transcript: str) -> MusicCommand:
        """
        Match transcript to command using fuzzy matching.
        
        Args:
            transcript: The transcribed text
            
        Returns:
            Matched MusicCommand or NONE if no match
        """
        if not transcript:
            return MusicCommand.NONE
        
        transcript_lower = transcript.lower().strip()
        
        # First try exact/substring match
        for command, patterns in COMMAND_PATTERNS.items():
            for pattern in patterns:
                if pattern in transcript_lower or transcript_lower in pattern:
                    return command
        
        # Fall back to fuzzy matching if available
        if fuzz is not None and process is not None:
            # Build flat list of (pattern, command) pairs
            all_patterns = []
            pattern_to_command = {}
            for command, patterns in COMMAND_PATTERNS.items():
                for pattern in patterns:
                    all_patterns.append(pattern)
                    pattern_to_command[pattern] = command
            
            # Find best fuzzy match
            result = process.extractOne(transcript_lower, all_patterns, scorer=fuzz.ratio)
            
            if result:
                matched_pattern, score = result[0], result[1]
                if score >= FUZZY_THRESHOLD:
                    return pattern_to_command[matched_pattern]
        
        return MusicCommand.NONE
    
    def stop(self):
        """Stop the detector (called externally)."""
        self.running = False
        self._cleanup_stream()
    
    def _cleanup_stream(self):
        """Clean up audio stream."""
        if self.audio_stream:
            try:
                self.audio_stream.stop()
                self.audio_stream.close()
            except:
                pass
            self.audio_stream = None
        
        # Reset state
        self._vad_state = np.zeros((2, 1, 128), dtype=np.float32)
        self._rolling_buffer.clear()
        self._is_speaking = False
        self._speech_frame_count = 0
        self._silence_frames = 0


# Backwards compatibility alias
StopDetector = VoiceCommandDetector

