"""Wake word detection using openWakeWord with Silero VAD gating"""

import os
import io
import time
import asyncio
import base64
from collections import deque
from difflib import SequenceMatcher
from datetime import datetime
from typing import Optional
import wave
import sounddevice as sd
import numpy as np
import onnxruntime as ort
from openwakeword.model import Model as OpenWakeWordModel
from elevenlabs import (
    AudioFormat,
    CommitStrategy,
    ElevenLabs,
    RealtimeAudioOptions,
)
from elevenlabs.realtime.connection import RealtimeEvents
from lib.config import Config

# Import telemetry (optional - graceful fallback if not available)
try:
    from lib.telemetry.telemetry import get_meter
    TELEMETRY_AVAILABLE = True
except ImportError:
    TELEMETRY_AVAILABLE = False


class WakeWordDetector:
    """openWakeWord-based wake word detection with telemetry"""
    
    def __init__(self, mic_device_index=None, orchestrator_client=None):
        """
        Initialize wake word detector.
        
        Args:
            mic_device_index: Audio device index for microphone (None for default)
            orchestrator_client: OrchestratorClient for sending detection data (optional)
        """
        self.oww_model = None
        self.audio_stream = None
        self.detected = False
        self.running = False
        self.mic_device_index = mic_device_index
        self.orchestrator_client = orchestrator_client
        self.logger = Config.LOGGER
        
        # openWakeWord expects 80ms frames (1280 samples at 16kHz)
        # We receive 512 sample frames, so accumulate 2.5 frames
        self.oww_frame_size = 1280
        self.audio_buffer = np.array([], dtype=np.int16)
        
        # Detection threshold (can be tuned - start with 0.5 as recommended)
        self.detection_threshold = 0.5
        
        # -------------------------------------------------------------------------
        # Silero VAD Gate - Reduces false positives by only processing speech
        # -------------------------------------------------------------------------
        self._vad_session = None
        self._vad_enabled = False
        
        # Locate model: project_root/models/silero_vad.onnx
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        vad_model_path = os.path.join(project_root, 'models', 'silero_vad.onnx')
        
        if os.path.exists(vad_model_path):
            try:
                self._vad_session = ort.InferenceSession(vad_model_path)
                self._vad_enabled = True
                print("✓ VAD gate initialized (reduces false wake word triggers)")
            except Exception as e:
                print(f"⚠ VAD gate init failed: {e} - falling back to ungated detection")
        else:
            print(f"⚠ VAD model not found at {vad_model_path} - falling back to ungated detection")
        
        # VAD state: Combined state tensor for Silero VAD v5
        # Shape: (2, 1, 128) for v5 (was separate h/c with shape (2,1,64) in v4)
        self._vad_state = np.zeros((2, 1, 128), dtype=np.float32)
        self._vad_threshold = 0.3  # Lowered to 0.3 to catch softer consonants
        
        # Hangover: Keep gate open for ~1 second after speech stops
        # Prevents wake word from being chopped during natural pauses ("Hey... Mycroft")
        self._hangover_frames = 30  # ~1 second at 32ms/frame
        self._hangover_counter = 0   # Current frames remaining
        
        # Frame counter for one-time VAD disabled warning
        self._vad_frame_count = 0
        
        # -------------------------------------------------------------------------
        # Ring buffer for Scribe v2 verification (stores VAD-passed audio)
        # -------------------------------------------------------------------------
        # At 16kHz, 512 samples per frame = 32ms per frame
        # 3 seconds = 3000ms / 32ms = ~93 frames
        self._ring_buffer_duration_seconds = 3.0
        self._frames_per_second = 16000 / 512  # ~31.25 frames/second
        self._ring_buffer_max_frames = int(self._ring_buffer_duration_seconds * self._frames_per_second)
        self._ring_buffer = deque(maxlen=self._ring_buffer_max_frames)  # Auto-drops oldest frames
        
        # Scribe v2 verification settings
        self._scribe_verification_enabled = True
        self._scribe_pre_trigger_seconds = 1.5  # Audio to send before wake word trigger
        self._scribe_post_trigger_seconds = 0.2  # Audio to capture after trigger
        self._scribe_timeout_seconds = 2.0  # Max time to wait for Scribe response
        
        # Telemetry counters (for local stats)
        self._scribe_verifications_total = 0
        self._scribe_verifications_passed = 0
        self._scribe_verifications_failed = 0
        self._scribe_api_errors = 0
        
        # Initialize OpenTelemetry metrics (if available)
        self._metrics = None
        if TELEMETRY_AVAILABLE and Config.OTEL_ENABLED:
            try:
                meter = get_meter("wake_word_detector")
                self._metrics = {
                    "scribe_verifications": meter.create_counter(
                        name="scribe_verifications_total",
                        description="Total Scribe v2 verification attempts",
                        unit="1",
                    ),
                    "scribe_verifications_passed": meter.create_counter(
                        name="scribe_verifications_passed_total",
                        description="Scribe v2 verifications that passed",
                        unit="1",
                    ),
                    "scribe_verifications_failed": meter.create_counter(
                        name="scribe_verifications_failed_total",
                        description="Scribe v2 verifications that failed",
                        unit="1",
                    ),
                    "scribe_api_errors": meter.create_counter(
                        name="scribe_api_errors_total",
                        description="Scribe v2 API errors",
                        unit="1",
                    ),
                }
            except Exception as e:
                print(f"⚠ Failed to initialize Scribe verification metrics: {e}")
                self._metrics = None
        
        # -------------------------------------------------------------------------
        # Negative detection throttling (once per 15 minutes)
        # -------------------------------------------------------------------------
        self._last_negative_detection_time = 0  # Unix timestamp
        self._negative_detection_throttle_seconds = 15 * 60  # 15 minutes
        
        # Track last Scribe result for detection data sending
        self._last_scribe_transcript = None
        self._last_scribe_error = None
        
    def _convert_frames_to_wav(self, audio_frames: list) -> bytes:
        """Convert list of int16 audio frames to WAV bytes.
        
        Args:
            audio_frames: List of int16 numpy arrays
            
        Returns:
            WAV file bytes
        """
        if not audio_frames:
            return b""
        
        # Concatenate all frames
        audio_data = np.concatenate(audio_frames)
        
        # Create WAV file in memory
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(16000)  # 16kHz
            wav_file.writeframes(audio_data.tobytes())
        
        # Get WAV bytes
        wav_buffer.seek(0)
        return wav_buffer.read()
    
    async def _send_detection_data_async(
        self,
        wake_word_detector_result: bool,
        verification_audio: list,
        asr_result: Optional[bool] = None,
        transcript: Optional[str] = None,
        asr_error: Optional[str] = None
    ):
        """Send wake word detection data to orchestrator (async, non-blocking).
        
        Args:
            wake_word_detector_result: openWakeWord result
            verification_audio: List of audio frames
            asr_result: Scribe v2 result (true/false/null)
            transcript: Actual transcript from Scribe
            asr_error: Error message if Scribe failed
        """
        if not self.orchestrator_client:
            return
        
        # Throttle negative detections (once per 15 minutes)
        if not wake_word_detector_result:
            current_time = time.time()
            if current_time - self._last_negative_detection_time < self._negative_detection_throttle_seconds:
                # Skip this negative detection (within throttle window)
                return
            self._last_negative_detection_time = current_time
        
        # Convert audio frames to WAV
        audio_wav = self._convert_frames_to_wav(verification_audio)
        if not audio_wav:
            return
        
        # Calculate audio duration
        audio_duration_ms = int((len(verification_audio) * 512 / 16000) * 1000) if verification_audio else None
        
        # Generate timestamp in YYYYMMDDHHmmss format
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        
        # Send detection data (fire and forget - async)
        try:
            await self.orchestrator_client.send_wake_word_detection(
                wake_word=Config.WAKE_WORD,
                wake_word_detector_result=wake_word_detector_result,
                asr_result=asr_result,
                audio_data=audio_wav,
                timestamp=timestamp,
                asr_error=asr_error,
                transcript=transcript,
                confidence_score=None,  # Not available from openWakeWord/Scribe
                audio_duration_ms=audio_duration_ms,
                retry_attempts=3
            )
        except Exception as e:
            # Log but don't block wake word detection
            if self.logger:
                self.logger.warning(
                    "send_wake_word_detection_async_failed",
                    extra={
                        "error": str(e),
                        "user_id": Config.USER_ID,
                        "device_id": Config.DEVICE_ID
                    }
                )
    
    def start(self):
        """Initialize openWakeWord and start listening"""
        if self.running:
            return

        # Reset detection flag each time we enter listening mode
        self.detected = False

        print(f"\n🎤 Initializing wake word detection...")
        print(f"   Wake word: '{Config.WAKE_WORD}'")
        
        try:
            # Locate openWakeWord model file
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            model_path = os.path.join(project_root, 'models', 'hey_mycroft.tflite')
            
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"openWakeWord model not found at {model_path}")
            
            # Initialize openWakeWord model
            print(f"   Loading model: {model_path}")
            self.oww_model = OpenWakeWordModel(
                wakeword_models=[model_path],
                inference_framework='tflite'
            )
            print(f"   Model loaded: {list(self.oww_model.models.keys())}")
            
            # Query devices for logging and error messages
            devices = sd.query_devices()
            
            # Log device being used
            if self.mic_device_index is not None:
                dev = devices[self.mic_device_index]
                print(f"   Using microphone: {dev['name']} (index {self.mic_device_index})")
            else:
                print(f"   Using default microphone")
            
            print(f"   Required sample rate: 16000 Hz")
            
            # Start audio stream for wake word detection
            # Use 512 samples per frame (32ms) to match existing system
            self.audio_stream = sd.InputStream(
                device=self.mic_device_index,
                channels=Config.CHANNELS,
                samplerate=16000,
                blocksize=512,  # 32ms frames
                dtype='int16',
                callback=self._audio_callback
            )
            
            self.audio_stream.start()
            self.running = True
            print(f"✓ Listening for wake word...")
            
            if self.logger:
                self.logger.info(
                    "listening_for_wake_word",
                    extra={
                        "wake_word": Config.WAKE_WORD,
                        "user_id": Config.USER_ID,
                        "device_id": Config.DEVICE_ID
                    }
                )
        except sd.PortAudioError as e:
            # Handle audio device errors with helpful message
            error_msg = str(e)
            if "Invalid sample rate" in error_msg or "paInvalidSampleRate" in error_msg:
                # Get device name for error message
                try:
                    device_name = devices[self.mic_device_index]['name'] if self.mic_device_index is not None else 'default'
                except:
                    device_name = f"index {self.mic_device_index}" if self.mic_device_index is not None else "default"
                
                print(f"\n✗ ERROR: Microphone doesn't support required sample rate")
                print(f"   Required: 16000 Hz (openWakeWord requirement)")
                print(f"   Device: {device_name}")
                print(f"\n   Solutions:")
                print(f"   1. Use a different USB microphone that supports 16kHz")
                print(f"   2. Use ReSpeaker 4 Mic Array (recommended - has hardware AEC)")
                print(f"   3. Check device supported rates with: python -m sounddevice")
                
                if self.logger:
                    self.logger.error(
                        "wake_word_unsupported_sample_rate",
                        extra={
                            "error": error_msg,
                            "required_rate": 16000,
                            "device_index": self.mic_device_index,
                            "user_id": Config.USER_ID
                        }
                    )
            else:
                print(f"\n✗ ERROR: Audio device error: {error_msg}")
                if self.logger:
                    self.logger.error(
                        "wake_word_audio_device_error",
                        extra={
                            "error": error_msg,
                            "user_id": Config.USER_ID
                        },
                        exc_info=True
                    )
            
            self.stop()
            raise
        except Exception as e:
            # Ensure partially-initialized resources are cleaned up
            if self.logger:
                self.logger.error(
                    "wake_word_detection_start_failed",
                    extra={
                        "error": str(e),
                        "user_id": Config.USER_ID
                    },
                    exc_info=True
                )
            self.stop()
            raise
        
    def _audio_callback(self, indata, frames, time_info, status):
        """Process audio frames for wake word detection with VAD gating"""
        if status:
            print(f"⚠ Audio status: {status}")
        
        # Convert to the format we need (int16)
        audio_frame = indata[:, 0].astype(np.int16)
        
        # Track frame count for one-time VAD disabled warning
        self._vad_frame_count += 1
        
        # -------------------------------------------------------------------------
        # VAD Gate: Only process with openWakeWord if speech is detected
        # -------------------------------------------------------------------------
        vad_passed = False
        if self._vad_enabled and self._vad_session:
            # Convert to float32 normalized [-1, 1] for Silero VAD
            audio_float = audio_frame.astype(np.float32) / 32768.0
            
            # Optimization: Skip VAD if audio is effectively silent (saves CPU)
            if np.max(np.abs(audio_float)) < 0.001:
                # Decrement hangover counter if active
                if self._hangover_counter > 0:
                    self._hangover_counter -= 1
                # Check gate with hangover
                if self._hangover_counter == 0:
                    return  # Pure silence and no hangover - skip openWakeWord
            else:
                # Run VAD inference (Silero VAD v5 API)
                try:
                    ort_inputs = {
                        'input': audio_float.reshape(1, -1),
                        'sr': np.array([16000], dtype=np.int64),
                        'state': self._vad_state
                    }
                    outs = self._vad_session.run(None, ort_inputs)
                    speech_prob = outs[0][0][0]  # First output is probability
                    self._vad_state = outs[1]     # Second output is new state (CRITICAL)
                    
                    # Hangover logic: Keep gate open for ~1s after speech stops
                    if speech_prob >= self._vad_threshold:
                        # Speech detected - reset hangover counter
                        self._hangover_counter = self._hangover_frames
                        vad_passed = True
                    elif self._hangover_counter > 0:
                        # Below threshold but hangover active - decrement
                        self._hangover_counter -= 1
                        vad_passed = True
                    
                    # Gate: Only pass if probability high OR hangover active
                    if not vad_passed:
                        return  # No speech and no hangover - skip openWakeWord
                    
                except Exception as e:
                    print(f"⚠ VAD inference error: {e}")
                    vad_passed = True  # VAD failed - fall through to openWakeWord anyway
        else:
            # VAD disabled - log warning once
            if self._vad_frame_count == 1:
                print("⚠ VAD gate DISABLED - openWakeWord processing ALL audio (higher false positive risk)")
            vad_passed = True
        
        # -------------------------------------------------------------------------
        # Ring buffer: Store VAD-passed audio for Scribe verification
        # -------------------------------------------------------------------------
        # Store a copy of the frame (avoid reference issues)
        self._ring_buffer.append(audio_frame.copy())
        
        # -------------------------------------------------------------------------
        # openWakeWord: Accumulate frames and check for wake word
        # -------------------------------------------------------------------------
        if vad_passed:
            # Accumulate audio for openWakeWord (needs 1280 samples = 80ms)
            self.audio_buffer = np.concatenate([self.audio_buffer, audio_frame])
            
            # Process when we have enough samples
            while len(self.audio_buffer) >= self.oww_frame_size:
                # Extract one openWakeWord frame
                oww_frame = self.audio_buffer[:self.oww_frame_size]
                self.audio_buffer = self.audio_buffer[self.oww_frame_size:]
                
                # Get predictions from openWakeWord
                # Model expects int16 audio directly
                prediction = self.oww_model.predict(oww_frame)
                
                # Check all models for wake word detection
                # The model returns a dict with model names as keys and scores as values
                for model_name, score in prediction.items():
                    if score >= self.detection_threshold:
                        print(f"\n🎯 Wake word '{Config.WAKE_WORD}' detected by openWakeWord! (score: {score:.3f}, VAD passed)")
                        
                        # -------------------------------------------------------------------------
                        # Scribe v2 Verification Layer
                        # -------------------------------------------------------------------------
                        if self._scribe_verification_enabled and Config.ELEVENLABS_API_KEY:
                            # Calculate frames needed for pre/post trigger window
                            pre_trigger_frames = int(self._scribe_pre_trigger_seconds * self._frames_per_second)
                            post_trigger_frames = int(self._scribe_post_trigger_seconds * self._frames_per_second)
                            
                            # Extract pre-trigger audio from ring buffer
                            buffer_list = list(self._ring_buffer)
                            if len(buffer_list) > post_trigger_frames:
                                # Get pre-trigger audio (everything except the last post_trigger_frames)
                                pre_trigger_audio = buffer_list[-(pre_trigger_frames + post_trigger_frames):-post_trigger_frames] if len(buffer_list) >= (pre_trigger_frames + post_trigger_frames) else buffer_list[:-post_trigger_frames]
                            else:
                                pre_trigger_audio = buffer_list
                            
                            # Collect post-trigger audio (capture a few more frames)
                            post_trigger_audio = []
                            for _ in range(post_trigger_frames):
                                # Small sleep to let audio capture happen
                                time.sleep(0.032)  # ~32ms per frame
                                if len(self._ring_buffer) > 0:
                                    post_trigger_audio.append(self._ring_buffer[-1].copy())
                            
                            # Combine pre and post trigger audio
                            verification_audio = pre_trigger_audio + post_trigger_audio
                            
                            # Run async Scribe verification synchronously (blocks audio callback)
                            try:
                                # Create new event loop for this thread if needed
                                try:
                                    loop = asyncio.get_event_loop()
                                except RuntimeError:
                                    loop = asyncio.new_event_loop()
                                    asyncio.set_event_loop(loop)
                                
                                # If loop is already running elsewhere, use a fresh loop for blocking waits
                                if loop.is_running():
                                    loop = asyncio.new_event_loop()
                                    asyncio.set_event_loop(loop)
                                
                                # Run verification synchronously
                                verified = loop.run_until_complete(self._verify_with_scribe(verification_audio))
                                
                                # Send detection data async (fire and forget - don't block wake word)
                                loop.run_until_complete(self._send_detection_data_async(
                                    wake_word_detector_result=True,  # openWakeWord detected
                                    verification_audio=verification_audio,
                                    asr_result=verified,  # Scribe result
                                    transcript=self._last_scribe_transcript,
                                    asr_error=self._last_scribe_error
                                ))
                                
                                if verified:
                                    print(f"✓ Scribe verification PASSED - accepting wake word")
                                    self.detected = True
                                    
                                    if self.logger:
                                        self.logger.info(
                                            "wake_word_detected",
                                            extra={
                                                "wake_word": Config.WAKE_WORD,
                                                "scribe_verified": True,
                                                "user_id": Config.USER_ID,
                                                "device_id": Config.DEVICE_ID
                                            }
                                        )
                                else:
                                    print(f"✗ Scribe verification FAILED - rejecting wake word (false positive)")
                                    # Do NOT set self.detected = True
                                    
                                    if self.logger:
                                        self.logger.warning(
                                            "wake_word_rejected_by_scribe",
                                            extra={
                                                "wake_word": Config.WAKE_WORD,
                                                "user_id": Config.USER_ID,
                                                "device_id": Config.DEVICE_ID
                                            }
                                        )
                                    
                            except Exception as e:
                                print(f"⚠ Scribe verification exception: {e} - accepting wake word (fail-open)")
                                self.detected = True
                                
                                # Send detection data with exception error
                                try:
                                    loop = asyncio.get_event_loop()
                                except RuntimeError:
                                    loop = asyncio.new_event_loop()
                                    asyncio.set_event_loop(loop)
                                
                                if loop.is_running():
                                    loop = asyncio.new_event_loop()
                                    asyncio.set_event_loop(loop)
                                
                                loop.run_until_complete(self._send_detection_data_async(
                                    wake_word_detector_result=True,
                                    verification_audio=verification_audio if 'verification_audio' in locals() else [],
                                    asr_result=None,  # Error occurred
                                    transcript=None,
                                    asr_error=str(e)
                                ))
                                
                                if self.logger:
                                    self.logger.info(
                                        "wake_word_detected",
                                        extra={
                                            "wake_word": Config.WAKE_WORD,
                                            "scribe_verified": False,
                                            "scribe_error": str(e),
                                            "user_id": Config.USER_ID,
                                            "device_id": Config.DEVICE_ID
                                        }
                                    )
                        else:
                            # Scribe verification disabled or no API key - accept immediately
                            self.detected = True
                            
                            # Send detection data without Scribe verification
                            # Extract audio from ring buffer
                            buffer_list = list(self._ring_buffer)
                            verification_audio = buffer_list[-int(self._frames_per_second * 1.7):] if len(buffer_list) > 0 else []
                            
                            try:
                                loop = asyncio.get_event_loop()
                            except RuntimeError:
                                loop = asyncio.new_event_loop()
                                asyncio.set_event_loop(loop)
                            
                            if loop.is_running():
                                loop = asyncio.new_event_loop()
                                asyncio.set_event_loop(loop)
                            
                            loop.run_until_complete(self._send_detection_data_async(
                                wake_word_detector_result=True,
                                verification_audio=verification_audio,
                                asr_result=None,  # Scribe not enabled
                                transcript=None,
                                asr_error="Scribe verification disabled"
                            ))
                            
                            if self.logger:
                                self.logger.info(
                                    "wake_word_detected",
                                    extra={
                                        "wake_word": Config.WAKE_WORD,
                                        "scribe_verified": False,
                                        "user_id": Config.USER_ID,
                                        "device_id": Config.DEVICE_ID
                                    }
                                )
                        
                        # Break after first detection
                        break
    
    def _fuzzy_match_wake_word(self, transcript: str, wake_word: str, threshold: float = 0.75) -> bool:
        """
        Check if transcript contains the wake word using fuzzy + substring matching.
        
        Args:
            transcript: The transcribed text from Scribe
            wake_word: The expected wake word (from Config.WAKE_WORD)
            threshold: Minimum similarity score (0.0 to 1.0)
            
        Returns:
            True if wake word is found (fuzzy or substring match)
        """
        if not transcript:
            return False
        
        transcript_lower = transcript.lower().strip()
        wake_word_lower = wake_word.lower().strip()
        
        # 1. Exact substring match
        if wake_word_lower in transcript_lower:
            print(f"   ✓ Scribe exact match: '{transcript}' contains '{wake_word}'")
            return True
        
        # 2. Fuzzy match on entire transcript
        similarity = SequenceMatcher(None, transcript_lower, wake_word_lower).ratio()
        if similarity >= threshold:
            print(f"   ✓ Scribe fuzzy match: '{transcript}' ~ '{wake_word}' (similarity: {similarity:.2f})")
            return True
        
        # 3. Fuzzy match on individual words in transcript
        words = transcript_lower.split()
        for word in words:
            word_similarity = SequenceMatcher(None, word, wake_word_lower).ratio()
            if word_similarity >= threshold:
                print(f"   ✓ Scribe word match: '{word}' ~ '{wake_word}' (similarity: {word_similarity:.2f})")
                return True
        
        # No match found
        print(f"   ✗ Scribe no match: '{transcript}' ≠ '{wake_word}' (best similarity: {similarity:.2f})")
        return False
    
    async def _verify_with_scribe(self, audio_frames: list) -> bool:
        """
        Verify wake word detection using ElevenLabs Scribe v2.
        
        Args:
            audio_frames: List of audio frames (int16 numpy arrays) to transcribe
            
        Returns:
            True if verification passes or API fails (fail-open), False if transcript doesn't match
        """
        if not Config.ELEVENLABS_API_KEY:
            print("   ⚠ Scribe verification skipped: No ElevenLabs API key")
            return True  # Fail-open: accept wake word if no API key
        
        if not audio_frames:
            print("   ⚠ Scribe verification skipped: No audio frames")
            return True  # Fail-open
        
        self._scribe_verifications_total += 1
        
        # Reset last result tracking
        self._last_scribe_transcript = None
        self._last_scribe_error = None
        
        # Emit telemetry metric
        if self._metrics:
            self._metrics["scribe_verifications"].add(1, {
                "device_id": Config.DEVICE_ID,
                "wake_word": Config.WAKE_WORD,
            })
        
        try:
            # Concatenate audio frames into single array
            audio_data = np.concatenate(audio_frames)
            
            # Convert int16 to float32 normalized [-1, 1] for processing
            audio_float = audio_data.astype(np.float32) / 32768.0
            
            # Convert to PCM16 bytes for base64 encoding
            audio_bytes = audio_data.tobytes()
            audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
            
            print(f"   🔍 Scribe verification: Sending {len(audio_data)} samples ({len(audio_data)/16000:.2f}s)")
            
            # Initialize ElevenLabs client
            client = ElevenLabs(api_key=Config.ELEVENLABS_API_KEY)
            
            # Create async connection to Scribe v2 Realtime
            transcript_text = None
            
            async def transcribe():
                nonlocal transcript_text
                connection = await client.speech_to_text.realtime.connect(
                    RealtimeAudioOptions(
                        model_id="scribe_v2_realtime",
                        audio_format=AudioFormat.PCM_16000,
                        sample_rate=16000,
                        commit_strategy=CommitStrategy.MANUAL,
                        language_code="en",
                    )
                )
                
                # Use event callbacks to capture the committed transcript
                loop = asyncio.get_event_loop()
                transcript_future = loop.create_future()
                
                def handle_committed(data):
                    nonlocal transcript_text
                    transcript_text = data.get("text") or data.get("transcript") or ""
                    if not transcript_future.done():
                        transcript_future.set_result(transcript_text)
                
                def handle_error(data):
                    err = data.get("error") or data.get("message") or "Unknown Scribe error"
                    self._last_scribe_error = err
                    if not transcript_future.done():
                        transcript_future.set_exception(RuntimeError(err))
                
                connection.on(RealtimeEvents.COMMITTED_TRANSCRIPT, handle_committed)
                connection.on(RealtimeEvents.COMMITTED_TRANSCRIPT_WITH_TIMESTAMPS, handle_committed)
                connection.on(RealtimeEvents.ERROR, handle_error)
                connection.on(RealtimeEvents.AUTH_ERROR, handle_error)
                connection.on(RealtimeEvents.QUOTA_EXCEEDED, handle_error)
                
                try:
                    # Send audio in chunks (simulate streaming)
                    chunk_size = 4096  # ~256ms chunks
                    for i in range(0, len(audio_base64), chunk_size):
                        chunk = audio_base64[i:i+chunk_size]
                        await connection.send({
                            "audio_base_64": chunk,
                            "sample_rate": 16000
                        })
                    
                    # Commit to get final transcript
                    await connection.commit()
                    try:
                        await asyncio.wait_for(transcript_future, timeout=self._scribe_timeout_seconds)
                    except asyncio.TimeoutError:
                        self._last_scribe_error = f"Scribe timeout after {self._scribe_timeout_seconds}s"
                        print(f"   ⚠ {self._last_scribe_error}")
                    await connection.close()
                
                except asyncio.TimeoutError:
                    self._last_scribe_error = f"Scribe timeout after {self._scribe_timeout_seconds}s"
                    print(f"   ⚠ {self._last_scribe_error}")
                    await connection.close()
            
            # Run async transcription with timeout
            await asyncio.wait_for(transcribe(), timeout=self._scribe_timeout_seconds + 1.0)
            
            if transcript_text:
                # Store transcript for detection data sending
                self._last_scribe_transcript = transcript_text
                
                # Check if transcript matches wake word
                matches = self._fuzzy_match_wake_word(transcript_text, Config.WAKE_WORD, Config.WAKE_WORD_ASR_SIMILARITY_THRESHOLD)
                
                if matches:
                    self._scribe_verifications_passed += 1
                    
                    # Emit telemetry metric
                    if self._metrics:
                        self._metrics["scribe_verifications_passed"].add(1, {
                            "device_id": Config.DEVICE_ID,
                            "wake_word": Config.WAKE_WORD,
                        })
                    
                    if self.logger:
                        self.logger.info(
                            "scribe_verification_passed",
                            extra={
                                "transcript": transcript_text,
                                "wake_word": Config.WAKE_WORD,
                                "user_id": Config.USER_ID,
                                "device_id": Config.DEVICE_ID
                            }
                        )
                    return True
                else:
                    self._scribe_verifications_failed += 1
                    
                    # Emit telemetry metric
                    if self._metrics:
                        self._metrics["scribe_verifications_failed"].add(1, {
                            "device_id": Config.DEVICE_ID,
                            "wake_word": Config.WAKE_WORD,
                            "transcript": transcript_text,
                        })
                    
                    if self.logger:
                        self.logger.warning(
                            "scribe_verification_failed",
                            extra={
                                "transcript": transcript_text,
                                "wake_word": Config.WAKE_WORD,
                                "user_id": Config.USER_ID,
                                "device_id": Config.DEVICE_ID
                            }
                        )
                    return False
            else:
                print("   ⚠ Scribe verification: No transcript received")
                return True  # Fail-open: accept wake word if no transcript
                
        except Exception as e:
            self._scribe_api_errors += 1
            
            # Store error for detection data sending
            self._last_scribe_error = str(e)
            
            # Emit telemetry metric
            if self._metrics:
                self._metrics["scribe_api_errors"].add(1, {
                    "device_id": Config.DEVICE_ID,
                    "error": str(e)[:100],  # Truncate long errors
                })
            
            if self.logger:
                self.logger.warning(
                    "scribe_verification_error",
                    extra={
                        "error": str(e),
                        "user_id": Config.USER_ID,
                        "device_id": Config.DEVICE_ID
                    },
                    exc_info=True
                )
            print(f"   ⚠ Scribe verification error: {e}")
            return True  # Fail-open: accept wake word on API error
    
    def stop(self):
        """Stop wake word detection"""
        if self.audio_stream:
            self.audio_stream.stop()
            self.audio_stream.close()
            self.audio_stream = None
        
        if self.oww_model:
            # openWakeWord model cleanup (if needed)
            self.oww_model = None
        
        self.running = False
        
        # Clear audio buffer
        self.audio_buffer = np.array([], dtype=np.int16)
        
        # Scribe verification stats logging intentionally suppressed
    
    def cleanup(self):
        """Clean up resources"""
        self.stop()
