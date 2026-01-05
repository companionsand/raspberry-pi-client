"""Stop keyword detector using VAD + Scribe for music playback control"""

import os
import time
import asyncio
import base64
from collections import deque
from typing import Optional, Callable
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


class StopDetector:
    """
    Detects "stop" keyword using VAD-gated Scribe transcription.
    
    This detector:
    1. Continuously monitors audio via VAD (Voice Activity Detection)
    2. When speech is detected, buffers the audio
    3. When speech ends, transcribes using Scribe v2
    4. Checks transcript for "stop" keyword
    """
    
    # Keywords that trigger stop (case-insensitive)
    STOP_KEYWORDS = ["stop"]
    
    def __init__(
        self,
        mic_device_index: Optional[int] = None,
        on_stop_detected: Optional[Callable[[], None]] = None,
        shared_input_queue=None,
        use_respeaker_aec: bool = False
    ):
        """
        Initialize stop detector.
        
        Args:
            mic_device_index: Microphone device index (None for default)
            on_stop_detected: Callback function when stop is detected
            shared_input_queue: Optional queue to read audio from (for ReSpeaker mode)
                               When provided, won't open a new audio stream
            use_respeaker_aec: Whether ReSpeaker AEC mode is active (for channel extraction)
        """
        self.mic_device_index = mic_device_index
        self.on_stop_detected = on_stop_detected
        self._shared_input_queue = shared_input_queue
        self._shared_respeaker_aec = use_respeaker_aec
        self.running = False
        self.stop_detected = False
        self.audio_stream = None
        self.logger = Config.LOGGER
        
        # Audio settings
        self.sample_rate = Config.SAMPLE_RATE  # 16kHz
        self.chunk_size = Config.CHUNK_SIZE  # 512 samples
        
        # ReSpeaker AEC detection
        self._use_respeaker_aec = False
        
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
        self._vad_threshold = 0.4  # Slightly higher threshold for noisy environment with music
        
        # Speech buffering
        self._speech_buffer = deque(maxlen=100)  # ~3 seconds at 32ms/frame
        self._is_speaking = False
        self._silence_frames = 0
        self._silence_threshold_frames = 15  # ~500ms of silence ends speech
        self._min_speech_frames = 12  # Minimum frames (~384ms) - Scribe requires at least 0.3s
        self._speech_frame_count = 0
        
        # Scribe configuration
        self._scribe_timeout = 3.0  # Timeout for Scribe transcription
        
    async def start(self) -> bool:
        """
        Start the stop detector.
        
        This method blocks until stop is detected or stop() is called.
        
        Returns:
            True if stop was detected, False if stopped externally
        """
        if self.running:
            return False
        
        self.running = True
        self.stop_detected = False
        
        print("🎤 Stop detector started - say 'Stop' to stop music")
        
        if self.logger:
            self.logger.info(
                "stop_detector_started",
                extra={"device_id": Config.DEVICE_ID}
            )
        
        try:
            # Check if we're using a shared input queue (ReSpeaker mode)
            # This avoids "Device unavailable" errors on devices that don't support
            # multiple simultaneous audio streams (like ReSpeaker with ALSA)
            using_shared_queue = self._shared_input_queue is not None
            
            if using_shared_queue:
                # Using shared queue from main conversation client
                self._use_respeaker_aec = self._shared_respeaker_aec
                print(f"   ✓ Using shared audio queue for stop detection (ReSpeaker mode)")
            else:
                # Fallback: Open our own audio stream (works on Mac, may fail on Pi)
                input_channels = Config.CHANNELS
                try:
                    devices = sd.query_devices()
                    if self.mic_device_index is not None:
                        input_dev = devices[self.mic_device_index]
                    else:
                        input_dev = None
                        for dev in devices:
                            if any(kw in dev['name'].lower() for kw in ['respeaker', 'arrayuac10', 'uac1.0']):
                                if dev['max_input_channels'] >= Config.RESPEAKER_CHANNELS:
                                    input_dev = dev
                                    break
                    
                    if input_dev and input_dev['max_input_channels'] >= Config.RESPEAKER_CHANNELS:
                        self._use_respeaker_aec = True
                        input_channels = Config.RESPEAKER_CHANNELS
                        print(f"   ✓ Using ReSpeaker Ch{Config.RESPEAKER_AEC_CHANNEL} for stop detection")
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
            while self.running and not self.stop_detected:
                # Read audio frame (from shared queue or own stream)
                if using_shared_queue:
                    try:
                        # Non-blocking get with timeout to allow checking self.running
                        audio_data = self._shared_input_queue.get(timeout=0.1)
                    except:
                        # Queue empty or timeout - continue loop
                        await asyncio.sleep(0.01)
                        continue
                else:
                    audio_data, _ = self.audio_stream.read(self.chunk_size)
                
                # Extract channel for ReSpeaker
                if self._use_respeaker_aec and audio_data.ndim == 2 and audio_data.shape[1] >= Config.RESPEAKER_CHANNELS:
                    audio_frame = audio_data[:, Config.RESPEAKER_AEC_CHANNEL].astype(np.int16)
                else:
                    audio_frame = audio_data[:, 0].astype(np.int16) if audio_data.ndim == 2 else audio_data.astype(np.int16)
                
                # Run VAD
                speech_detected = await self._process_vad(audio_frame)
                
                if speech_detected:
                    # Add frame to buffer
                    self._speech_buffer.append(audio_frame.copy())
                    self._speech_frame_count += 1
                    self._silence_frames = 0
                    
                    if not self._is_speaking:
                        self._is_speaking = True
                        print("   🎙️  Speech detected...")
                else:
                    if self._is_speaking:
                        self._silence_frames += 1
                        
                        # Still add a few silence frames to buffer for clean audio end
                        if self._silence_frames <= 5:
                            self._speech_buffer.append(audio_frame.copy())
                        
                        # Check if speech segment ended
                        if self._silence_frames >= self._silence_threshold_frames:
                            if self._speech_frame_count >= self._min_speech_frames:
                                # Process the speech segment
                                print(f"   📝 Processing speech ({self._speech_frame_count} frames)...")
                                await self._process_speech_segment()
                            
                            # Reset state
                            self._is_speaking = False
                            self._speech_frame_count = 0
                            self._speech_buffer.clear()
                
                # Small delay to prevent busy loop
                await asyncio.sleep(0.001)
            
            return self.stop_detected
            
        except Exception as e:
            print(f"✗ Stop detector error: {e}")
            if self.logger:
                self.logger.error(
                    "stop_detector_error",
                    extra={
                        "error": str(e),
                        "device_id": Config.DEVICE_ID
                    }
                )
            return False
            
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
    
    async def _process_speech_segment(self):
        """Transcribe buffered speech and check for stop keyword."""
        if not Config.ELEVENLABS_API_KEY:
            print("   ⚠️  No ElevenLabs API key - cannot transcribe")
            return
        
        if len(self._speech_buffer) == 0:
            return
        
        try:
            # Concatenate audio frames
            audio_data = np.concatenate(list(self._speech_buffer))
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
                print(f"   📝 Transcript: '{transcript_text}'")
                
                # Check for stop keyword
                transcript_lower = transcript_text.lower().strip()
                for keyword in self.STOP_KEYWORDS:
                    if keyword in transcript_lower:
                        print(f"   🛑 Stop keyword detected!")
                        self.stop_detected = True
                        
                        if self.logger:
                            self.logger.info(
                                "stop_keyword_detected",
                                extra={
                                    "transcript": transcript_text,
                                    "device_id": Config.DEVICE_ID
                                }
                            )
                        
                        # Call callback if provided
                        if self.on_stop_detected:
                            self.on_stop_detected()
                        
                        return
            else:
                print("   ⚠️  Empty transcript")
                
        except Exception as e:
            print(f"   ⚠️  Transcription error: {e}")
            if self.logger:
                self.logger.error(
                    "stop_detector_transcription_error",
                    extra={
                        "error": str(e),
                        "device_id": Config.DEVICE_ID
                    }
                )
    
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
        
        # Reset VAD state
        self._vad_state = np.zeros((2, 1, 128), dtype=np.float32)
        self._speech_buffer.clear()
        self._is_speaking = False
        self._speech_frame_count = 0
        self._silence_frames = 0

