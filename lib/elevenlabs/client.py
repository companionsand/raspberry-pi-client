"""ElevenLabs conversation client with telemetry"""

import json
import base64
import asyncio
import time
import uuid
import ssl
import os
import websockets
import certifi
import sounddevice as sd
import numpy as np
import onnxruntime as ort
import queue
from scipy import signal
from typing import Optional
from lib.config import Config
from lib.orchestrator.client import OrchestratorClient
from lib.local_storage import ContextManager


class ElevenLabsConversationClient:
    """WebSocket-based conversation client for ElevenLabs with full telemetry"""
    
    def __init__(self, web_socket_url: str, agent_id: str, 
                 mic_device_index=None, speaker_device_index=None,
                 user_terminate_flag=None, led_controller=None):
        """
        Initialize ElevenLabs conversation client.
        
        Args:
            web_socket_url: ElevenLabs WebSocket URL
            agent_id: Agent ID for this conversation
            mic_device_index: Microphone device index (None for default)
            speaker_device_index: Speaker device index (None for default)
            user_terminate_flag: Mutable flag [False] for user termination
            led_controller: LEDController instance for visual feedback (optional)
        """
        self.web_socket_url = web_socket_url
        self.agent_id = agent_id
        self.mic_device_index = mic_device_index
        self.speaker_device_index = speaker_device_index
        self.websocket = None
        self.input_stream = None  # Microphone input stream
        self.output_stream = None  # Speaker output stream (separate for instant abort)
        self.stream = None  # Duplex stream for ReSpeaker (input_stream/output_stream alias to this)
        self.running = False
        self._use_respeaker_aec = False  # Set to True in start() if ReSpeaker detected
        self._aec_debug_last_log = 0  # Timestamp for periodic AEC debug logging
        self.conversation_id = str(uuid.uuid4())
        self.elevenlabs_conversation_id = None
        self.end_reason = "normal"
        self.last_audio_time = None  # Used for LED state management
        self.user_terminate_flag = user_terminate_flag  # Reference to shared flag
        self.led_controller = led_controller  # LED controller for visual feedback
        self.logger = Config.LOGGER
        
        # -------------------------------------------------------------------------
        # Silero VAD (Voice Activity Detection) - Local silence detection via ONNX
        # -------------------------------------------------------------------------
        self.vad_session = None
        self.vad_enabled = False
        
        # Locate the Silero VAD ONNX model file: project_root/models/silero_vad.onnx
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        model_path = os.path.join(project_root, 'models', 'silero_vad.onnx')
        if os.path.exists(model_path):
            try:
                self.vad_session = ort.InferenceSession(model_path)
                self.vad_enabled = True
                print("✓ Silero VAD initialized (local silence detection)")
            except Exception as e:
                print(f"⚠ Silero VAD init failed: {e}")
        else:
            print("⚠ Silero VAD model not found - using fallback silence detection")
        
        # VAD state: Combined state tensor for Silero VAD v5
        # Shape: (2, 1, 128) for v5 (was separate h/c with shape (2,1,64) in v4)
        self._vad_state = np.zeros((2, 1, 128), dtype=np.float32)
        self.vad_threshold = 0.5  # Probability threshold for speech detection
        
        # THINKING state trigger: track when user stopped speaking
        # After 400ms of silence, transition LED to THINKING state
        # (400ms forgives micro-pauses but still beats server VAD ~1000ms)
        self._last_speech_time = None  # When VAD last detected speech
        self._thinking_threshold_ms = 400  # Silence duration before THINKING state
        
        # -------------------------------------------------------------------------
        # Audio queue and interruption handling
        # -------------------------------------------------------------------------
        # Interruptions are detected server-side by ElevenLabs and sent as 'interruption' messages.
        # The client just needs to stop playback and clear the queue when interrupted.
        self.audio_queue = asyncio.Queue()  # Queue for agent audio chunks
        self.playback_active = False  # Flag to stop current audio playback
        # Callback mode queues (only used when ReSpeaker AEC is active)
        self._input_queue = None   # Filled by PortAudio callback with mic frames
        self._output_queue = None  # Consumed by PortAudio callback for playback
        self._callback_remainder = None  # Store partial chunk for next callback
        
        # Dynamic VAD threshold: stricter during agent playback to filter echo
        # Normal threshold: 0.5 (sensitive, catches user speech quickly)
        # Playback threshold: 0.75 (stricter, filters residual echo from speaker)
        self._vad_threshold_normal = 0.5
        self._vad_threshold_playback = 0.75
        
        # Agent audio chunk counter (for logging and VAD reset detection)
        self._chunk_count = 0
        
        # Periodic status logging: Log conversation health every 10 seconds
        self._last_status_log_time = 0.0
        self._status_log_interval = 10.0  # Log status every 10 seconds
        
        # Max conversation duration failsafe (5 minutes)
        self._max_conversation_duration = 300.0  # 5 minutes
        self._conversation_start_time = None
        
        # ElevenLabs audio sample rate - detect from metadata or assume 16kHz (same as playback)
        # If audio sounds robotic/slow, ElevenLabs may be sending 24kHz - enable resampling
        self._elevenlabs_sample_rate = None  # Will be detected from metadata or default to 16kHz
        self._playback_sample_rate = Config.SAMPLE_RATE  # 16kHz for ReSpeaker
        self._resampling_enabled = False  # Disable by default - enable if audio sounds wrong
        
        
    async def start(self, orchestrator_client: OrchestratorClient):
        """Start a conversation session"""
        logger = self.logger
        print(f"\n💬 Starting conversation...")
        print(f"   Agent ID: {self.agent_id}")
        print(f"   Conversation ID: {self.conversation_id}")
        
        # Add API key to WebSocket URL
        ws_url = f"{self.web_socket_url}&api_key={Config.ELEVENLABS_API_KEY}"
        
        # Output device
        output_device = self.speaker_device_index
        
        # Log device being used
        if self.mic_device_index is not None or self.speaker_device_index is not None:
            devices = sd.query_devices()
            if self.mic_device_index is not None:
                mic_dev = devices[self.mic_device_index]
                print(f"   Microphone: {mic_dev['name']} (index {self.mic_device_index})")
            if self.speaker_device_index is not None:
                speaker_dev = devices[self.speaker_device_index]
                print(f"   Speaker: {speaker_dev['name']} (index {self.speaker_device_index})")
        else:
            print(f"   Audio: Using default devices")
        
        # Create SSL context
        ssl_context = ssl.create_default_context(cafile=certifi.where())
        
        try:
            # -------------------------------------------------------------------------
            # Detect ReSpeaker for 6-channel AEC extraction
            # -------------------------------------------------------------------------
            # Check if input device supports 6 channels (ReSpeaker 4-Mic Array)
            # If so, open all 6 channels and extract Ch0 (AEC-processed) in code
            input_channels = Config.CHANNELS  # Default to mono
            
            # Track device indices for duplex stream
            # ReSpeaker may expose INPUT and OUTPUT as DIFFERENT device indices!
            respeaker_input_idx = self.mic_device_index
            respeaker_output_idx = output_device  # Start with configured output
            
            try:
                devices = sd.query_devices()
                
                # Find ReSpeaker INPUT device (6+ input channels)
                if self.mic_device_index is not None:
                    input_dev = devices[self.mic_device_index]
                    respeaker_input_idx = self.mic_device_index
                else:
                    # Scan for ReSpeaker input device
                    input_dev = None
                    for idx, dev in enumerate(devices):
                        if any(kw in dev['name'].lower() for kw in ['respeaker', 'arrayuac10', 'uac1.0']):
                            if dev['max_input_channels'] >= Config.RESPEAKER_CHANNELS:
                                input_dev = dev
                                respeaker_input_idx = idx
                                break
                
                # Check if ReSpeaker with 6 channels
                if input_dev and input_dev['max_input_channels'] >= Config.RESPEAKER_CHANNELS:
                    self._use_respeaker_aec = True
                    input_channels = Config.RESPEAKER_CHANNELS
                    
                    # Find ReSpeaker OUTPUT device (may be same or different index)
                    # Look for device with same name pattern that has output channels
                    for idx, dev in enumerate(devices):
                        if any(kw in dev['name'].lower() for kw in ['respeaker', 'arrayuac10', 'uac1.0']):
                            if dev['max_output_channels'] >= 1:
                                respeaker_output_idx = idx
                                break
                    
                    print(f"   ✓ ReSpeaker AEC: Opening {input_channels} channels, extracting Ch{Config.RESPEAKER_AEC_CHANNEL} (AEC-processed)")
                    print(f"   ✓ ReSpeaker input device: {respeaker_input_idx}, output device: {respeaker_output_idx}")
            except Exception as e:
                print(f"   ⚠ Could not detect ReSpeaker channels: {e}")
            
            # -------------------------------------------------------------------------
            # Open audio streams: Duplex for ReSpeaker, separate for other devices
            # -------------------------------------------------------------------------
            # ReSpeaker requires a duplex stream for proper multi-channel capture.
            # Separate streams cause Ch1-5 to be zero (hardware loopback issue).
            if self._use_respeaker_aec:
                # Duplex stream: 6-channel input, mono output (fixes AEC reference signal)
                # Input and output may be different device indices!
                # Use callback-based I/O so PortAudio handles full-duplex safely.
                self._input_queue = queue.Queue()
                # Unbounded queue to prevent dropping chunks (which causes audio jumps)
                # Queue will naturally drain as callback consumes audio
                self._output_queue = queue.Queue()

                def audio_callback(indata, outdata, frames, time_info, status):
                    """PortAudio callback for full-duplex ReSpeaker capture/playback."""
                    # Push captured audio to queue for async sender
                    try:
                        self._input_queue.put_nowait(indata.copy())
                    except queue.Full:
                        # Drop oldest frame if we're lagging to keep latency low
                        try:
                            self._input_queue.get_nowait()
                        except queue.Empty:
                            pass
                        self._input_queue.put_nowait(indata.copy())

                    # Pull next playback chunk (mono) or output silence
                    # PortAudio callback needs exactly 'frames' samples (typically 512)
                    outdata[:] = 0  # Start with silence
                    frames_written = 0
                    remainder_chunk = None
                    
                    # First, use any remainder from previous callback if it exists
                    if hasattr(self, '_callback_remainder') and self._callback_remainder is not None:
                        remainder_chunk = self._callback_remainder
                        self._callback_remainder = None
                    
                    while frames_written < len(outdata):
                        # Use remainder first, then get from queue
                        if remainder_chunk is not None:
                            out_chunk = remainder_chunk
                            remainder_chunk = None
                        else:
                            try:
                                out_chunk = self._output_queue.get_nowait()
                            except queue.Empty:
                                break  # No more audio, fill rest with silence
                        
                        # Ensure shape (frames,) for mono
                        if out_chunk.ndim == 2:
                            out_chunk = out_chunk.flatten()
                        elif out_chunk.ndim != 1:
                            out_chunk = out_chunk.flatten()
                        
                        # Copy into PortAudio buffer
                        remaining = len(outdata) - frames_written
                        frames_to_copy = min(len(out_chunk), remaining)
                        outdata[frames_written:frames_written + frames_to_copy, 0] = out_chunk[:frames_to_copy]
                        frames_written += frames_to_copy
                        
                        # If chunk was larger than remaining space, save remainder for next callback
                        if len(out_chunk) > frames_to_copy:
                            self._callback_remainder = out_chunk[frames_to_copy:]

                self.stream = sd.Stream(
                    device=(respeaker_input_idx, respeaker_output_idx),
                    samplerate=Config.SAMPLE_RATE,
                    channels=(input_channels, Config.CHANNELS),
                    dtype='int16',
                    blocksize=Config.CHUNK_SIZE,
                    callback=audio_callback
                )
                self.stream.start()
                # Duplex stream handles BOTH input (6ch) and output (1ch)
                # Using the same stream for I/O avoids "Device unavailable" errors
                # since ReSpeaker can only be opened once
                self.input_stream = self.stream
                self.output_stream = self.stream  # Same stream - duplex handles both
                print(f"   ✓ ReSpeaker: Duplex stream (6ch in, 1ch out) on devices ({respeaker_input_idx}, {respeaker_output_idx}) [callback mode]")
            else:
                # Non-ReSpeaker: Use separate streams (original behavior)
                self.input_stream = sd.InputStream(
                    device=self.mic_device_index,
                    samplerate=Config.SAMPLE_RATE,
                    channels=input_channels,
                    dtype='int16',
                    blocksize=Config.CHUNK_SIZE
                )
                self.input_stream.start()
                
                self.output_stream = sd.OutputStream(
                    device=output_device,
                    samplerate=Config.SAMPLE_RATE,
                    channels=Config.CHANNELS,
                    dtype='int16',
                    blocksize=Config.CHUNK_SIZE
                )
                self.output_stream.start()
            
            # Connect to WebSocket
            async with websockets.connect(ws_url, ssl=ssl_context) as websocket:
                self.websocket = websocket
                self.running = True
                self.last_audio_time = time.time()
                self._conversation_start_time = time.time()  # For max duration failsafe
                
                print("✓ Connected to ElevenLabs")
                
                if logger:
                    logger.info(
                        "elevenlabs_connected",
                        extra={
                            "conversation_id": self.conversation_id,
                            "agent_id": self.agent_id,
                            "user_id": Config.USER_ID,
                            "device_id": Config.DEVICE_ID
                        }
                    )
                
                # Get dynamic variables from context manager
                context_manager = ContextManager()
                dynamic_variables = context_manager.get_dynamic_variables()
                
                # Log dynamic variables being sent
                if logger:
                    logger.info(
                        "sending_dynamic_variables",
                        extra={
                            "conversation_id": self.conversation_id,
                            "has_location": context_manager.has_location_data,
                            "variable_count": len(dynamic_variables)
                        }
                    )
                
                # Send conversation initiation with dynamic variables
                await websocket.send(json.dumps({
                    "type": "conversation_initiation_client_data",
                    "dynamic_variables": dynamic_variables
                }))
                
                print("✓ Conversation started - speak now!")
                if logger:
                    logger.info(
                        "conversation_started",
                        extra={
                            "conversation_id": self.conversation_id,
                            "agent_id": self.agent_id,
                            "user_id": Config.USER_ID,
                            "device_id": Config.DEVICE_ID
                        }
                    )
                
                # Run send, receive, playback, and watchdog tasks concurrently
                send_task = asyncio.create_task(self._send_audio())
                receive_task = asyncio.create_task(self._receive_messages(orchestrator_client))
                playback_task = asyncio.create_task(self._play_audio())
                watchdog_task = asyncio.create_task(self._watchdog())
                
                # Wait for any task to complete
                done, pending = await asyncio.wait(
                    [send_task, receive_task, playback_task, watchdog_task],
                    return_when=asyncio.FIRST_COMPLETED
                )
                
                # Cancel pending tasks
                for task in pending:
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
                
        except Exception as e:
            print(f"✗ Conversation error: {e}")
            if logger:
                logger.error(
                    "conversation_error",
                    extra={
                        "conversation_id": self.conversation_id,
                        "agent_id": self.agent_id,
                        "user_id": Config.USER_ID,
                        "device_id": Config.DEVICE_ID,
                        "error": str(e)
                    },
                    exc_info=True
                )
            self.end_reason = "network_failure"
        finally:
            await self.stop(orchestrator_client)
    
    async def _send_audio(self):
        """Send microphone audio to WebSocket"""
        logger = self.logger
        try:
            while self.running:
                # Read audio from microphone
                if self._use_respeaker_aec:
                    # Callback mode: grab frames from PortAudio callback queue
                    try:
                        audio_data = await asyncio.get_running_loop().run_in_executor(
                            None, self._input_queue.get
                        )
                    except Exception as e:
                        print(f"✗ Input queue error: {e}")
                        self.end_reason = "network_failure"
                        self.running = False
                        break
                else:
                    audio_data, _ = self.input_stream.read(Config.CHUNK_SIZE)
                
                # -------------------------------------------------------------------------
                # AEC Debug Logging: Show RMS for all 6 channels (every 3 seconds)
                # -------------------------------------------------------------------------
                if self._use_respeaker_aec and audio_data.ndim == 2 and audio_data.shape[1] >= Config.RESPEAKER_CHANNELS:
                    now = time.time()
                    if now - self._aec_debug_last_log >= 3.0:
                        self._aec_debug_last_log = now
                        # Debug: Log actual audio data shape and dtype
                        print(f"🔍 [AEC DEBUG] audio_data shape={audio_data.shape}, dtype={audio_data.dtype}, ndim={audio_data.ndim}")
                        # Calculate RMS for each channel
                        ch_rms = []
                        for ch in range(min(6, audio_data.shape[1])):
                            rms = np.sqrt(np.mean(audio_data[:, ch].astype(float) ** 2)) / 32768.0
                            ch_rms.append(rms)
                        # Format like diagnostic script
                        is_playing = self.playback_active or self.audio_queue.qsize() > 0
                        status = "PLAYING" if is_playing else "IDLE"
                        print(f"📊 [AEC DEBUG] [{status}] Ch0={ch_rms[0]:.4f} | Ch1={ch_rms[1]:.4f} | Ch2={ch_rms[2]:.4f} | Ch3={ch_rms[3]:.4f} | Ch4={ch_rms[4]:.4f} | Ch5={ch_rms[5]:.4f} | Using: Ch{Config.RESPEAKER_AEC_CHANNEL}")
                
                # -------------------------------------------------------------------------
                # ReSpeaker AEC: Extract Channel 0 (AEC-processed audio)
                # -------------------------------------------------------------------------
                # If using ReSpeaker with 6 channels, extract only Ch0 (echo-cancelled)
                # This bypasses ALSA routing issues and directly accesses the AEC output
                if self._use_respeaker_aec and audio_data.ndim == 2 and audio_data.shape[1] >= Config.RESPEAKER_CHANNELS:
                    # Extract channel 0 (AEC-processed) and reshape to (samples, 1)
                    audio_data = audio_data[:, Config.RESPEAKER_AEC_CHANNEL:Config.RESPEAKER_AEC_CHANNEL+1].copy()
                
                # -------------------------------------------------------------------------
                # Voice Activity Detection (VAD) - Detect if user is speaking
                # -------------------------------------------------------------------------
                is_speech = False
                speech_prob = 0.0
                
                # Dynamic threshold: stricter during playback to filter echo/residual state
                is_agent_active = self.playback_active or self.audio_queue.qsize() > 0
                active_threshold = self._vad_threshold_playback if is_agent_active else self._vad_threshold_normal
                
                if self.vad_enabled and self.vad_session:
                    # Run Silero VAD inference
                    # Convert int16 audio to float32 normalized [-1, 1]
                    audio_float = audio_data.flatten().astype(np.float32) / 32768.0
                    
                    # Prepare ONNX inputs (Silero VAD v5 API)
                    ort_inputs = {
                        'input': audio_float.reshape(1, -1),
                        'sr': np.array([Config.SAMPLE_RATE], dtype=np.int64),
                        'state': self._vad_state
                    }
                    
                    # Run inference and update state
                    try:
                        outs = self.vad_session.run(None, ort_inputs)
                        speech_prob = outs[0][0][0]  # First output is probability
                        self._vad_state = outs[1]     # Second output is new state
                        is_speech = speech_prob > active_threshold
                    except Exception:
                        # Fallback to energy detection if VAD fails
                        is_speech = np.abs(audio_data).mean() > 100
                else:
                    # Fallback: simple energy detection
                    is_speech = np.abs(audio_data).mean() > 100
                
                # Calculate mic RMS for status logging
                mic_rms = np.sqrt(np.mean(audio_data.astype(float) ** 2))
                
                # -------------------------------------------------------------------------
                # LED State Management: LISTENING ↔ THINKING transitions
                # -------------------------------------------------------------------------
                # Calculate actual conversation inactivity (time since last transcript/agent audio)
                actual_silence_secs = (time.time() - self.last_audio_time) if self.last_audio_time else 0
                
                if is_speech:
                    # User is speaking - update speech time for LED transitions
                    # NOTE: We intentionally do NOT update last_audio_time here
                    # last_audio_time is for silence timeout (no conversation activity)
                    # and should only be reset on actual transcripts/responses
                    self._last_speech_time = time.time()
                    
                    # Ensure LED is in LISTENING state (STATE_CONVERSATION)
                    # BUT only if there's been recent conversation activity (within 5s)
                    # This prevents VAD false positives from keeping LED in LISTENING forever
                    if self.led_controller and actual_silence_secs < 5.0:
                        current = self.led_controller.current_state
                        # Only switch if we're in THINKING (not SPEAKING - agent might be talking)
                        if current == self.led_controller.STATE_THINKING:
                            self.led_controller.set_state(self.led_controller.STATE_CONVERSATION)
                else:
                    # User is silent - check if we should transition to THINKING
                    if self._last_speech_time and self.led_controller:
                        silence_ms = (time.time() - self._last_speech_time) * 1000
                        current = self.led_controller.current_state
                        
                        # Transition to THINKING after 400ms of silence
                        # Only if we're in CONVERSATION (listening) state, not SPEAKING
                        if silence_ms > self._thinking_threshold_ms:
                            if current == self.led_controller.STATE_CONVERSATION:
                                self.led_controller.set_state(self.led_controller.STATE_THINKING)
                
                # Fallback: Force THINKING state if no actual conversation activity for 5+ seconds
                # This handles VAD false positives that keep detecting "speech"
                if self.led_controller and actual_silence_secs >= 5.0:
                    current = self.led_controller.current_state
                    if current == self.led_controller.STATE_CONVERSATION:
                        self.led_controller.set_state(self.led_controller.STATE_THINKING)
                
                # Encode as base64 and send (with timeout to prevent blocking on dead connection)
                audio_b64 = base64.b64encode(audio_data.tobytes()).decode('utf-8')
                message = {"user_audio_chunk": audio_b64}
                
                # Debug: Log what we're sending to ElevenLabs (every 3 seconds, synced with AEC debug)
                now = time.time()
                if self._use_respeaker_aec and (now - self._aec_debug_last_log) < 0.1:  # Log right after AEC debug
                    sent_rms = np.sqrt(np.mean(audio_data.astype(float) ** 2)) / 32768.0
                    print(f"📤 [SENT TO 11LABS] shape={audio_data.shape} RMS={sent_rms:.4f} (extracted from Ch{Config.RESPEAKER_AEC_CHANNEL})")
                
                try:
                    await asyncio.wait_for(
                        self.websocket.send(json.dumps(message)),
                        timeout=5.0  # 5 second timeout for send
                    )
                except asyncio.TimeoutError:
                    print(f"\n⏱️ WebSocket send timeout - connection may be dead")
                    if logger:
                        logger.warning(
                            "websocket_send_timeout",
                            extra={
                                "conversation_id": self.conversation_id,
                                "user_id": Config.USER_ID,
                                "device_id": Config.DEVICE_ID
                            }
                        )
                    self.end_reason = "network_failure"
                    self.running = False
                    break
                
                # Check for user termination
                if self.user_terminate_flag and self.user_terminate_flag[0]:
                    print("\n🛑 User termination - ending conversation")
                    if logger:
                        logger.info(
                            "conversation_user_terminated",
                            extra={
                                "conversation_id": self.conversation_id,
                                "user_id": Config.USER_ID,
                                "device_id": Config.DEVICE_ID
                            }
                        )
                    self.end_reason = "user_terminated"
                    self.running = False
                    break
                
                # Check for max conversation duration failsafe
                if self._conversation_start_time:
                    conversation_duration = time.time() - self._conversation_start_time
                    if conversation_duration > self._max_conversation_duration:
                        print(f"\n⏱️ Max conversation duration ({self._max_conversation_duration}s) reached - ending")
                        if logger:
                            logger.warning(
                                "conversation_max_duration_reached",
                                extra={
                                    "conversation_id": self.conversation_id,
                                    "duration_seconds": conversation_duration,
                                    "max_duration": self._max_conversation_duration,
                                    "user_id": Config.USER_ID,
                                    "device_id": Config.DEVICE_ID
                                }
                            )
                        self.end_reason = "max_duration"
                        self.running = False
                        break
                
                # Periodic status logging for debugging stuck conversations
                now = time.time()
                if now - self._last_status_log_time > self._status_log_interval:
                    self._last_status_log_time = now
                    silence_duration = now - self.last_audio_time if self.last_audio_time else 0
                    conversation_duration = now - self._conversation_start_time if self._conversation_start_time else 0
                    led_state = self.led_controller.current_state if self.led_controller else "unknown"
                    vad_info = f"VAD={speech_prob:.2f}" if self.vad_enabled else "VAD=off"
                    print(f"📊 [STATUS] duration={conversation_duration:.1f}s, silence={silence_duration:.1f}s, LED={led_state}, {vad_info}, RMS={mic_rms:.0f}, queue={self.audio_queue.qsize()}")
                    
                    # If in prolonged silence (>30s) with no transcripts, reset VAD state
                    # to clear any accumulated false positive state
                    if silence_duration > 30.0:
                        self._vad_state = np.zeros((2, 1, 128), dtype=np.float32)
                        print(f"   ↳ VAD state reset due to prolonged silence ({silence_duration:.0f}s)")
                
                # Small delay to prevent overwhelming the connection
                await asyncio.sleep(0.01)
                
        except websockets.exceptions.ConnectionClosedOK as e:
            # Normal close from the server; avoid treating it as an error
            print("\n✓ Conversation closed (send loop)")
            if logger:
                logger.info(
                    "send_connection_closed_ok",
                    extra={
                        "conversation_id": self.conversation_id,
                        "close_code": e.code,
                        "close_reason": e.reason,
                        "user_id": Config.USER_ID
                    }
                )
            self.running = False
        except websockets.exceptions.ConnectionClosedError as e:
            print(f"✗ Send connection closed unexpectedly: {e}")
            if logger:
                logger.error(
                    "audio_send_connection_closed",
                    extra={
                        "conversation_id": self.conversation_id,
                        "close_code": e.code,
                        "close_reason": e.reason,
                        "user_id": Config.USER_ID
                    }
                )
            self.end_reason = "network_failure"
            self.running = False
        except asyncio.CancelledError:
            self.end_reason = "user_terminated"
            self.running = False
        except Exception as e:
            print(f"✗ Send error: {e}")
            if logger:
                logger.error(
                    "audio_send_error",
                    extra={
                        "conversation_id": self.conversation_id,
                        "error": str(e),
                        "user_id": Config.USER_ID
                    }
                )
            self.end_reason = "network_failure"
            self.running = False
    
    async def _play_audio(self):
        """Play audio chunks from queue to output stream"""
        logger = self.logger
        try:
            while self.running:
                # Get next audio chunk from queue (with timeout to check running flag)
                try:
                    audio_array = await asyncio.wait_for(
                        self.audio_queue.get(), 
                        timeout=0.1
                    )
                except asyncio.TimeoutError:
                    # Queue empty - if agent was speaking, mark turn as ended
                    # Reset chunk count so next agent turn triggers VAD state reset
                    if self._chunk_count > 0 and self.audio_queue.qsize() == 0:
                        print(f"📤 AGENT TURN END: {self._chunk_count} chunks played, resetting for next turn")
                        self._chunk_count = 0  # Critical: allows VAD reset on next agent turn
                    continue
                
                # Play the audio chunk in small sub-chunks for fast interruption
                self.playback_active = True
                
                # Write in small chunks (256 samples = ~16ms at 16kHz) for fast response
                # Small chunks also help with callback buffering (callback needs 512 samples per call)
                sub_chunk_size = 256
                interrupted = False
                
                # Get event loop for running blocking write in executor
                loop = asyncio.get_running_loop()
                
                for i in range(0, len(audio_array), sub_chunk_size):
                    # Check if barge-in occurred
                    if not self.playback_active:
                        interrupted = True
                        break
                    
                    # Check if conversation ended
                    if not self.running:
                        interrupted = True
                        break
                    
                    # Write small chunk - run in executor to avoid blocking event loop
                    chunk = audio_array[i:i+sub_chunk_size]
                    try:
                        if self.output_stream and self.output_stream.active:
                            if self._use_respeaker_aec:
                                # Callback mode: queue chunk for PortAudio callback
                                # Ensure chunk is 1D array (mono)
                                if chunk.ndim > 1:
                                    chunk = chunk.flatten()
                                # Queue chunk for callback (unbounded queue, no dropping)
                                # This ensures sequential playback without jumps
                                self._output_queue.put_nowait(chunk)
                                
                                # Small yield to ensure queue stays filled (callback consumes at ~512 samples/call)
                                await asyncio.sleep(0)
                            else:
                                # Run blocking write in thread pool with timeout
                                await asyncio.wait_for(
                                    loop.run_in_executor(None, self.output_stream.write, chunk),
                                    timeout=2.0  # 2 second timeout for audio write
                                )
                    except asyncio.TimeoutError:
                        print(f"⚠️ Audio write timeout - audio device may be stuck")
                        if logger:
                            logger.warning(
                                "audio_write_timeout",
                                extra={
                                    "conversation_id": self.conversation_id,
                                    "user_id": Config.USER_ID,
                                    "device_id": Config.DEVICE_ID
                                }
                            )
                        # Don't break - try to continue, the device might recover
                    except Exception as e:
                        if logger:
                            logger.debug(
                                "playback_write_error",
                                extra={
                                    "error": str(e),
                                    "conversation_id": self.conversation_id
                                }
                            )
                        break
                    
                    # Yield to event loop (already yielded in run_in_executor, but be explicit)
                    await asyncio.sleep(0)
                
                if interrupted:
                    print(f"   [DEBUG] Audio playback interrupted mid-chunk")
                
                self.playback_active = False
                
        except asyncio.CancelledError:
            self.running = False
        except Exception as e:
            print(f"✗ Playback error: {e}")
            if logger:
                logger.error(
                    "audio_playback_error",
                    extra={
                        "conversation_id": self.conversation_id,
                        "error": str(e),
                        "user_id": Config.USER_ID,
                        "device_id": Config.DEVICE_ID
                    }
                )
            self.end_reason = "playback_error"
            self.running = False
    
    async def _watchdog(self):
        """Watchdog task that monitors conversation health and terminates if stuck.
        
        Checks for:
        1. Audio queue growing without being processed (playback stuck)
        2. No activity for extended period (all tasks stuck)
        """
        logger = self.logger
        check_interval = 5.0  # Check every 5 seconds
        max_queue_stall_time = 30.0  # Max time audio can sit in queue
        last_queue_check_time = time.time()
        last_queue_size = 0
        queue_stall_start = None
        
        try:
            while self.running:
                await asyncio.sleep(check_interval)
                
                if not self.running:
                    break
                
                current_queue_size = self.audio_queue.qsize()
                now = time.time()
                
                # Check for queue stall (queue has items that aren't being processed)
                if current_queue_size > 0:
                    if current_queue_size >= last_queue_size and last_queue_size > 0:
                        # Queue is not shrinking - might be stalled
                        if queue_stall_start is None:
                            queue_stall_start = now
                        
                        stall_duration = now - queue_stall_start
                        if stall_duration > max_queue_stall_time:
                            print(f"\n🚨 WATCHDOG: Audio queue stalled for {stall_duration:.1f}s (queue={current_queue_size}) - terminating")
                            if logger:
                                logger.error(
                                    "watchdog_queue_stall",
                                    extra={
                                        "conversation_id": self.conversation_id,
                                        "queue_size": current_queue_size,
                                        "stall_duration": stall_duration,
                                        "user_id": Config.USER_ID,
                                        "device_id": Config.DEVICE_ID
                                    }
                                )
                            self.end_reason = "watchdog_queue_stall"
                            self.running = False
                            break
                    else:
                        # Queue is shrinking - reset stall timer
                        queue_stall_start = None
                else:
                    # Queue is empty - reset stall timer
                    queue_stall_start = None
                
                last_queue_size = current_queue_size
                last_queue_check_time = now
                
        except asyncio.CancelledError:
            pass
        except Exception as e:
            if logger:
                logger.error(
                    "watchdog_error",
                    extra={
                        "conversation_id": self.conversation_id,
                        "error": str(e),
                        "user_id": Config.USER_ID,
                        "device_id": Config.DEVICE_ID
                    }
                )
    
    async def _receive_messages(self, orchestrator_client: OrchestratorClient):
        """Receive and process messages from WebSocket"""
        logger = self.logger
        
        # Receive timeout: If no messages for 15 seconds, consider connection dead
        # ElevenLabs sends pings every ~5 seconds, so 15s means 3 missed pings
        receive_timeout = 15.0
        
        try:
            while self.running:
                try:
                    message = await asyncio.wait_for(
                        self.websocket.recv(),
                        timeout=receive_timeout
                    )
                except asyncio.TimeoutError:
                    print(f"\n⏱️ WebSocket receive timeout ({receive_timeout}s) - connection may be dead")
                    if logger:
                        logger.warning(
                            "websocket_receive_timeout",
                            extra={
                                "conversation_id": self.conversation_id,
                                "timeout_seconds": receive_timeout,
                                "user_id": Config.USER_ID,
                                "device_id": Config.DEVICE_ID
                            }
                        )
                    self.end_reason = "network_failure"
                    self.running = False
                    break
                
                # Parse JSON message
                data = json.loads(message)
                
                # Handle different message types
                if 'conversation_initiation_metadata_event' in data:
                    metadata = data['conversation_initiation_metadata_event']
                    self.elevenlabs_conversation_id = metadata.get('conversation_id', None)
                    # Check if sample rate is in metadata
                    if 'sample_rate' in metadata:
                        self._elevenlabs_sample_rate = metadata['sample_rate']
                        self._resampling_enabled = (self._elevenlabs_sample_rate != self._playback_sample_rate)
                        print(f"   ElevenLabs sample rate: {self._elevenlabs_sample_rate}Hz (resampling: {self._resampling_enabled})")
                    else:
                        # Default to 16kHz (no resampling) - assume same as playback
                        if self._elevenlabs_sample_rate is None:
                            self._elevenlabs_sample_rate = self._playback_sample_rate
                            self._resampling_enabled = False
                            print(f"   ElevenLabs sample rate: assuming {self._elevenlabs_sample_rate}Hz (no resampling)")
                    print(f"   ElevenLabs Conversation ID: {self.elevenlabs_conversation_id}")
                    
                    # Send conversation start notification
                    await orchestrator_client.send_conversation_start(
                        conversation_id=self.conversation_id,
                        elevenlabs_conversation_id=self.elevenlabs_conversation_id or "",
                        agent_id=self.agent_id,
                    )
                
                elif 'user_transcription_event' in data:
                    transcript = data['user_transcription_event'].get('user_transcript', '')
                    if transcript:
                        print(f"👤 You: {transcript}")
                        self.last_audio_time = time.time()  # Reset silence timer
                
                elif 'agent_response_event' in data:
                    response = data['agent_response_event'].get('agent_response', '')
                    if response:
                        print(f"🤖 Agent: {response}")
                
                elif 'audio_event' in data:
                    # Decode and queue agent audio (don't play directly)
                    audio_b64 = data['audio_event'].get('audio_base_64', '')
                    if audio_b64:
                        audio_bytes = base64.b64decode(audio_b64)
                        audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
                        
                        # Resample if sample rates differ (only if resampling is enabled)
                        if self._resampling_enabled and self._elevenlabs_sample_rate and self._elevenlabs_sample_rate != self._playback_sample_rate:
                            # Convert to float for resampling
                            audio_float = audio_array.astype(np.float32) / 32768.0
                            # Resample: 24kHz -> 16kHz
                            num_samples_16k = int(len(audio_float) * self._playback_sample_rate / self._elevenlabs_sample_rate)
                            audio_resampled = signal.resample(audio_float, num_samples_16k)
                            # Convert back to int16
                            audio_array = (audio_resampled * 32768.0).astype(np.int16)
                        
                        # Track chunk count
                        self._chunk_count += 1
                        
                        # First chunk of NEW agent turn: Reset VAD state to clear residual from user turn
                        # This helps prevent false positives in VAD during agent speech
                        if self._chunk_count == 1:
                            self._vad_state = np.zeros((2, 1, 128), dtype=np.float32)
                            print(f"📥 AGENT TURN START: first chunk ({len(audio_array)} samples @ {self._playback_sample_rate}Hz) [VAD state RESET]")
                        elif self._chunk_count % 10 == 0:
                            print(f"📥 AGENT audio: chunk #{self._chunk_count} ({len(audio_array)} samples), queue={self.audio_queue.qsize()}")
                        
                        # Update LEDs synchronized with agent speech (audio-reactive)
                        if self.led_controller:
                            self.led_controller.update_speaking_leds(audio_array)
                        
                        # Queue audio for playback
                        await self.audio_queue.put(audio_array)
                        self.last_audio_time = time.time()  # Reset silence timer
                
                elif data.get('type') == 'interruption' or 'interruption' in data:
                    # ElevenLabs detected user interruption - stop playback immediately
                    interruption_event = data.get('interruption', data)
                    print(f"🛑 ElevenLabs interruption detected - clearing audio queue")
                    if logger:
                        logger.info(
                            "elevenlabs_interruption_received",
                            extra={
                                "conversation_id": self.conversation_id,
                                "queue_size_before": self.audio_queue.qsize(),
                                "user_id": Config.USER_ID,
                                "device_id": Config.DEVICE_ID,
                                "raw_event": data
                            }
                        )
                    
                    # Stop playback immediately
                    self.playback_active = False
                    
                    # Clear the audio queue
                    cleared_count = 0
                    while not self.audio_queue.empty():
                        try:
                            self.audio_queue.get_nowait()
                            cleared_count += 1
                        except asyncio.QueueEmpty:
                            break
                    
                    # Clear callback queue if using ReSpeaker AEC (callback mode)
                    callback_cleared = 0
                    if self._use_respeaker_aec and self._output_queue:
                        while not self._output_queue.empty():
                            try:
                                self._output_queue.get_nowait()
                                callback_cleared += 1
                            except queue.Empty:
                                break
                        # Also clear any remainder chunk
                        self._callback_remainder = None
                    
                    # Reset chunk count for next agent turn
                    self._chunk_count = 0
                    
                    # Reset LED to conversation/listening state
                    if self.led_controller:
                        self.led_controller.set_state(self.led_controller.STATE_CONVERSATION)
                    
                    if callback_cleared > 0:
                        print(f"   ✓ Cleared {cleared_count} queued audio chunks + {callback_cleared} callback chunks, ready for user input")
                    else:
                        print(f"   ✓ Cleared {cleared_count} queued audio chunks, ready for user input")
                    self.last_audio_time = time.time()  # Reset silence timer
                
                elif data.get('type') == 'agent_response_correction_event' or 'agent_response_correction_event' in data:
                    # Agent response was corrected (often happens with interruptions)
                    print(f"📝 Agent response corrected")
                    if logger:
                        logger.info(
                            "agent_response_correction",
                            extra={
                                "conversation_id": self.conversation_id,
                                "user_id": Config.USER_ID,
                                "raw_event": data
                            }
                        )
                
                elif data.get('type') == 'ping':
                    # Respond to ping to keep connection alive
                    ping_event = data.get('ping_event', {})
                    event_id = ping_event.get('event_id')
                    if event_id is not None:
                        await self.websocket.send(json.dumps({
                            'type': 'pong',
                            'event_id': event_id
                        }))
                
                elif data.get('type') == 'error':
                    error_msg = data.get('message', 'Unknown error')
                    print(f"✗ Server error: {error_msg}")
                    if logger:
                        logger.error(
                            "elevenlabs_server_error",
                            extra={
                                "conversation_id": self.conversation_id,
                                "error_message": error_msg,
                                "user_id": Config.USER_ID
                            }
                        )
                    self.end_reason = "error"
                    self.running = False
                
                # Handle conversation termination from ElevenLabs
                elif data.get('type') == 'conversation_end' or 'conversation_end_event' in data:
                    end_event = data.get('conversation_end_event', data)
                    termination_reason = end_event.get('termination_reason', end_event.get('reason', 'unknown'))
                    print(f"\n✓ ElevenLabs ended conversation: {termination_reason}")
                    if logger:
                        logger.info(
                            "elevenlabs_conversation_terminated",
                            extra={
                                "conversation_id": self.conversation_id,
                                "termination_reason": termination_reason,
                                "user_id": Config.USER_ID,
                                "device_id": Config.DEVICE_ID,
                                "raw_event": data
                            }
                        )
                    self.end_reason = f"elevenlabs_{termination_reason}" if termination_reason != "unknown" else "elevenlabs_terminated"
                    self.running = False
                
                else:
                    # Log unknown message types for debugging
                    msg_type = data.get('type', 'unknown')
                    keys = list(data.keys())
                    # Print unknown messages so we can see what we're missing
                    if msg_type not in ('pong',):
                        print(f"📨 [UNKNOWN MSG] type={msg_type}, keys={keys}")
                        if logger:
                            logger.info(
                                "elevenlabs_unknown_message",
                                extra={
                                    "conversation_id": self.conversation_id,
                                    "message_type": msg_type,
                                    "message_keys": keys,
                                    "raw_data": str(data)[:500]  # First 500 chars for debugging
                                }
                            )
                
                # Check for user termination
                if self.user_terminate_flag and self.user_terminate_flag[0]:
                    print("\n🛑 User termination - ending conversation")
                    self.end_reason = "user_terminated"
                    self.running = False
                    break
                    
        except websockets.exceptions.ConnectionClosed:
            print("\n✓ Conversation ended (connection closed)")
            if self.user_terminate_flag and self.user_terminate_flag[0]:
                self.end_reason = "user_terminated"
            else:
                self.end_reason = "normal"
            if logger:
                logger.info(
                    "conversation_connection_closed",
                    extra={
                        "conversation_id": self.conversation_id,
                        "end_reason": self.end_reason,
                        "user_id": Config.USER_ID,
                        "device_id": Config.DEVICE_ID
                    }
                )
            self.running = False
        except asyncio.CancelledError:
            self.end_reason = "user_terminated"
            self.running = False
        except Exception as e:
            print(f"✗ Receive error: {e}")
            if logger:
                logger.error(
                    "receive_error",
                    extra={
                        "conversation_id": self.conversation_id,
                        "error": str(e),
                        "user_id": Config.USER_ID
                    }
                )
            self.end_reason = "network_failure"
            self.running = False
    
    async def stop(self, orchestrator_client: OrchestratorClient):
        """Stop the conversation"""
        self.running = False
        
        # Stop and close audio streams
        # Handle duplex case (ReSpeaker) where input_stream and output_stream are the same object
        if self.input_stream:
            self.input_stream.stop()
            self.input_stream.close()
        
        if self.output_stream and self.output_stream is not self.input_stream:
            # Only close output_stream if it's a separate object (non-ReSpeaker case)
            self.output_stream.stop()
            self.output_stream.close()
        
        # Send conversation end notification
        await orchestrator_client.send_conversation_end(
            conversation_id=self.conversation_id,
            elevenlabs_conversation_id=self.elevenlabs_conversation_id or "",
            agent_id=self.agent_id,
            end_reason=self.end_reason,
        )
        
        if self.logger:
            self.logger.info(
                "conversation_ended",
                extra={
                    "conversation_id": self.conversation_id,
                    "elevenlabs_conversation_id": self.elevenlabs_conversation_id or "",
                    "agent_id": self.agent_id,
                    "end_reason": self.end_reason,
                    "user_id": Config.USER_ID,
                    "device_id": Config.DEVICE_ID
                }
            )
        
        if self.websocket:
            await self.websocket.close()
