"""ElevenLabs conversation client with telemetry and turn tracking"""

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
from typing import Optional
from lib.config import Config
from lib.orchestrator.client import OrchestratorClient
from lib.local_storage import ContextManager
from lib.turn_tracker import TurnTracker
from lib.speaker_monitor import SpeakerMonitor


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
        self.running = False
        self.conversation_id = str(uuid.uuid4())
        self.elevenlabs_conversation_id = None
        self.end_reason = "normal"
        self.silence_timeout = 30.0  # seconds
        self.last_audio_time = None
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
        # Barge-in: Audio queue and interruption handling
        # -------------------------------------------------------------------------
        self.audio_queue = asyncio.Queue()  # Queue for agent audio chunks
        self.playback_active = False  # Flag to stop current audio playback
        self.barge_in_active = False  # Flag indicating barge-in occurred
        self.barge_in_enabled = False  # Barge-in feature disabled
        
        # Sustained speech detection for barge-in
        # Lower threshold = more sensitive (catches short words like "Stop")
        # Higher threshold = less sensitive (ignores brief noise)
        self._consecutive_speech_frames = 0
        self._barge_in_speech_threshold = 2  # Frames (2 = ~20ms, catches even short words)
        
        # -------------------------------------------------------------------------
        # Barge-in protection: Prevent false triggers from VAD state persistence
        # -------------------------------------------------------------------------
        # Guard period: Ignore VAD spikes within Xms after user stops speaking
        # This prevents residual VAD state from triggering false barge-in
        self._user_speech_end_time = None  # Timestamp when user stopped speaking
        self._barge_in_guard_period_ms = 300  # Ignore VAD for 300ms after user stops
        
        # Elevated threshold during playback: Require higher confidence when agent speaking
        # Normal threshold: 0.5 (sensitive, catches user speech quickly)
        # Playback threshold: 0.75 (stricter, filters residual echo/state)
        self._vad_threshold_normal = 0.5
        self._vad_threshold_playback = 0.75
        
        # Agent audio chunk counter (for logging and VAD reset detection)
        self._chunk_count = 0
        
        # Debug logging: periodic RMS/VAD logging
        self._last_debug_log_time = 0.0
        self._debug_log_interval = 0.5  # Log every 500ms during playback
        
        # -------------------------------------------------------------------------
        # Turn Tracker - Tracks user and agent speech turns
        # -------------------------------------------------------------------------
        self.turn_tracker: Optional[TurnTracker] = None
        self.turn_tracking_enabled = not Config.SKIP_TURN_TRACKING
        
        if self.turn_tracking_enabled:
            self.turn_tracker = TurnTracker(
                sample_rate=Config.SAMPLE_RATE,
                vad_threshold=Config.TURN_TRACKER_VAD_THRESHOLD,
                user_silence_timeout=Config.TURN_TRACKER_USER_SILENCE_TIMEOUT,
                user_min_turn_duration=Config.TURN_TRACKER_USER_MIN_TURN_DURATION,
                user_min_speech_onset=Config.TURN_TRACKER_USER_MIN_SPEECH_ONSET,
                agent_silence_timeout=Config.TURN_TRACKER_AGENT_SILENCE_TIMEOUT,
                agent_min_turn_duration=Config.TURN_TRACKER_AGENT_MIN_TURN_DURATION,
                agent_min_speech_onset=Config.TURN_TRACKER_AGENT_MIN_SPEECH_ONSET,
                debounce_window=Config.TURN_TRACKER_DEBOUNCE_WINDOW,
            )
        
        # -------------------------------------------------------------------------
        # Speaker Monitor - Optional loopback-based agent turn detection
        # -------------------------------------------------------------------------
        self.speaker_monitor: Optional[SpeakerMonitor] = None
        self.use_speaker_monitor = self.turn_tracking_enabled and Config.SPEAKER_MONITOR_MODE == "loopback"
        
        if self.use_speaker_monitor:
            # SpeakerMonitor has tuned defaults for loopback
            self.speaker_monitor = SpeakerMonitor(
                sample_rate=Config.SAMPLE_RATE,
                loopback_device_name=Config.SPEAKER_MONITOR_LOOPBACK_DEVICE,
            )
            # If loopback device not found, fall back to estimation
            if not self.speaker_monitor.enabled:
                self.use_speaker_monitor = False
                self.speaker_monitor = None
        
    async def start(self, orchestrator_client: OrchestratorClient):
        """Start a conversation session"""
        logger = self.logger
        print(f"\n💬 Starting conversation...")
        print(f"   Agent ID: {self.agent_id}")
        print(f"   Conversation ID: {self.conversation_id}")
        
        # Add API key to WebSocket URL
        ws_url = f"{self.web_socket_url}&api_key={Config.ELEVENLABS_API_KEY}"
        
        # Determine output device
        # When speaker monitoring is enabled, ALSA default is configured to route
        # through 'speaker_with_monitor' which outputs to both speaker AND loopback
        output_device = self.speaker_device_index
        if self.use_speaker_monitor:
            # Use default device (None) - ALSA config routes it to both speaker + loopback
            output_device = None
            print(f"   Speaker Monitor: Enabled (ALSA default routes to speaker + loopback)")
        
        # Log device being used
        if self.mic_device_index is not None or self.speaker_device_index is not None:
            devices = sd.query_devices()
            if self.mic_device_index is not None:
                mic_dev = devices[self.mic_device_index]
                print(f"   Microphone: {mic_dev['name']} (index {self.mic_device_index})")
            if not self.use_speaker_monitor and self.speaker_device_index is not None:
                speaker_dev = devices[self.speaker_device_index]
                print(f"   Speaker: {speaker_dev['name']} (index {self.speaker_device_index})")
        else:
            print(f"   Audio: Using default devices")
        
        # Create SSL context
        ssl_context = ssl.create_default_context(cafile=certifi.where())
        
        try:
            # Open separate input and output streams for independent control
            # Input stream: microphone only
            self.input_stream = sd.InputStream(
                device=self.mic_device_index,
                samplerate=Config.SAMPLE_RATE,
                channels=Config.CHANNELS,
                dtype='int16',
                blocksize=Config.CHUNK_SIZE
            )
            self.input_stream.start()
            
            # Output stream: speaker only (can be aborted immediately on barge-in)
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
                
                print("✓ Connected to ElevenLabs")
                
                # Start turn tracking (if enabled)
                if self.turn_tracker:
                    self.turn_tracker.start()
                
                # Start speaker monitor if enabled (for loopback-based agent detection)
                if self.speaker_monitor:
                    self.speaker_monitor.start()
                
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
                
                # Run send, receive, and playback tasks concurrently
                send_task = asyncio.create_task(self._send_audio())
                receive_task = asyncio.create_task(self._receive_messages(orchestrator_client))
                playback_task = asyncio.create_task(self._play_audio())
                
                # Wait for any task to complete
                done, pending = await asyncio.wait(
                    [send_task, receive_task, playback_task],
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
                audio_data, _ = self.input_stream.read(Config.CHUNK_SIZE)
                
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
                
                # -------------------------------------------------------------------------
                # Calculate mic RMS for debugging
                # -------------------------------------------------------------------------
                mic_rms = np.sqrt(np.mean(audio_data.astype(float) ** 2))
                
                # -------------------------------------------------------------------------
                # Barge-in Detection: With guard period and state protection (DISABLED)
                # -------------------------------------------------------------------------
                # Barge-in is currently disabled - user speech during agent playback will not interrupt
                if self.barge_in_enabled:
                    # Guard period: Ignore VAD spikes shortly after user stops speaking
                    # This prevents residual VAD state from triggering false barge-in
                    in_guard_period = False
                    guard_time_ms = 0.0
                    if self._user_speech_end_time:
                        guard_time_ms = (time.time() - self._user_speech_end_time) * 1000
                        in_guard_period = guard_time_ms < self._barge_in_guard_period_ms
                    
                    # -------------------------------------------------------------------------
                    # Debug logging: RMS, VAD prob, guard state (every 500ms during agent speech)
                    # -------------------------------------------------------------------------
                    now = time.time()
                    if is_agent_active and (now - self._last_debug_log_time > self._debug_log_interval):
                        guard_status = f"GUARD({guard_time_ms:.0f}ms)" if in_guard_period else "NO_GUARD"
                        vad_status = "SPEECH" if is_speech else "silence"
                        print(f"📊 [AGENT_PLAYING] RMS={mic_rms:.0f} VAD={speech_prob:.3f}({vad_status}) thresh={active_threshold} {guard_status} frames={self._consecutive_speech_frames}")
                        self._last_debug_log_time = now
                    
                    if is_speech:
                        self._consecutive_speech_frames += 1
                        
                        # Trigger barge-in on sustained speech (but only once per speech event)
                        # Additional checks: not in guard period, agent must be active
                        if self._consecutive_speech_frames >= self._barge_in_speech_threshold:
                            if not self.barge_in_active:
                                # Check guard period - skip if too close to user's last speech
                                if in_guard_period:
                                    print(f"🛡️ GUARD BLOCKED: VAD spike at {guard_time_ms:.0f}ms after user stopped (prob={speech_prob:.2f}, RMS={mic_rms:.0f})")
                                elif is_agent_active:
                                    # Barge-in detected during agent speech!
                                    print(f"🎤 BARGE-IN: User interrupting agent (prob={speech_prob:.2f}, RMS={mic_rms:.0f}, thresh={active_threshold})")
                                    self.barge_in_active = True
                                    await self._handle_barge_in()
                                # else: agent not active, no need for barge-in (normal turn-taking)
                            # else: already triggered, don't call handler again (debouncing)
                    else:
                        # User stopped speaking - record timestamp for guard period
                        if self._consecutive_speech_frames >= 2:
                            self._user_speech_end_time = time.time()
                            print(f"🎤 USER turn ended: {self._consecutive_speech_frames} frames (~{self._consecutive_speech_frames * 10}ms), guard period started")
                        self._consecutive_speech_frames = 0
                        self.barge_in_active = False  # Reset: ready for next barge-in
                else:
                    # Barge-in disabled - reset counters but don't process barge-in logic
                    if not is_speech:
                        self._consecutive_speech_frames = 0
                        self.barge_in_active = False
                
                # -------------------------------------------------------------------------
                # LED State Management: LISTENING ↔ THINKING transitions
                # -------------------------------------------------------------------------
                if is_speech:
                    # User is speaking - update timers
                    self.last_audio_time = time.time()
                    self._last_speech_time = time.time()
                    
                    # Ensure LED is in LISTENING state (STATE_CONVERSATION)
                    if self.led_controller:
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
                
                # Feed to turn tracker for user VAD (if enabled)
                if self.turn_tracker:
                    self.turn_tracker.process_user_audio(audio_data)
                    
                    # Check for agent silence timeout (skip if using speaker monitor)
                    if not self.use_speaker_monitor:
                        self.turn_tracker.check_agent_silence()
                
                # Encode as base64 and send
                audio_b64 = base64.b64encode(audio_data.tobytes()).decode('utf-8')
                message = {"user_audio_chunk": audio_b64}
                
                await self.websocket.send(json.dumps(message))
                
                # Check for silence timeout
                if self.last_audio_time and (time.time() - self.last_audio_time) > self.silence_timeout:
                    print("\n⏱️  Silence timeout - ending conversation")
                    if logger:
                        logger.info(
                            "conversation_silence_timeout",
                            extra={
                                "conversation_id": self.conversation_id,
                                "user_id": Config.USER_ID,
                                "device_id": Config.DEVICE_ID
                            }
                        )
                    self.end_reason = "silence"
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
    
    async def _handle_barge_in(self):
        """Handle user barge-in: stop playback immediately and clear audio queue"""
        logger = self.logger
        
        print(f"🛑 Barge-in triggered!")  # Always print to confirm it's being called
        
        # Stop current playback immediately
        self.playback_active = False
        
        # Get queue size before clearing
        queue_size = self.audio_queue.qsize()
        
        # Clear the audio queue
        cleared_count = 0
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
                cleared_count += 1
            except asyncio.QueueEmpty:
                break
        
        print(f"   ✓ Stopped playback and cleared {cleared_count} queued audio chunks (queue had {queue_size})")
        
        if logger:
            logger.info(
                "barge_in_triggered",
                extra={
                    "conversation_id": self.conversation_id,
                    "cleared_chunks": cleared_count,
                    "queue_size_before": queue_size,
                    "user_id": Config.USER_ID,
                    "device_id": Config.DEVICE_ID
                }
            )
    
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
                
                # Write in very small chunks (256 samples = ~16ms at 16kHz) for fast response
                sub_chunk_size = 256
                interrupted = False
                
                for i in range(0, len(audio_array), sub_chunk_size):
                    # Check if barge-in occurred
                    if not self.playback_active:
                        interrupted = True
                        break
                    
                    # Check if conversation ended
                    if not self.running:
                        interrupted = True
                        break
                    
                    # Write small chunk
                    chunk = audio_array[i:i+sub_chunk_size]
                    try:
                        if self.output_stream and self.output_stream.active:
                            self.output_stream.write(chunk)
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
                    
                    # Yield to event loop
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
    
    async def _receive_messages(self, orchestrator_client: OrchestratorClient):
        """Receive and process messages from WebSocket"""
        logger = self.logger
        try:
            while self.running:
                message = await self.websocket.recv()
                
                # Parse JSON message
                data = json.loads(message)
                
                # Handle different message types
                if 'conversation_initiation_metadata_event' in data:
                    metadata = data['conversation_initiation_metadata_event']
                    self.elevenlabs_conversation_id = metadata.get('conversation_id', None)
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
                        # Record transcript for turn reconciliation (if enabled)
                        if self.turn_tracker:
                            self.turn_tracker.record_user_transcript(transcript)
                
                elif 'agent_response_event' in data:
                    response = data['agent_response_event'].get('agent_response', '')
                    if response:
                        print(f"🤖 Agent: {response}")
                        # Record agent response for turn reconciliation (if enabled)
                        if self.turn_tracker:
                            self.turn_tracker.record_agent_response(response)
                
                elif 'audio_event' in data:
                    # Decode and queue agent audio (don't play directly)
                    audio_b64 = data['audio_event'].get('audio_base_64', '')
                    if audio_b64:
                        audio_bytes = base64.b64decode(audio_b64)
                        audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
                        
                        # Track chunk count
                        self._chunk_count += 1
                        
                        # First chunk of NEW agent turn: Reset VAD state to clear residual from user turn
                        # This prevents false barge-in triggers from VAD state persistence
                        # BUG FIX: Check _chunk_count == 1 after increment (means this is first chunk of turn)
                        if self._chunk_count == 1:
                            self._vad_state = np.zeros((2, 1, 128), dtype=np.float32)
                            self._consecutive_speech_frames = 0
                            self.barge_in_active = False
                            print(f"📥 AGENT TURN START: first chunk ({len(audio_array)} samples) [VAD state RESET, guard period active]")
                        elif self._chunk_count % 10 == 0:
                            print(f"📥 AGENT audio: chunk #{self._chunk_count} ({len(audio_array)} samples), queue={self.audio_queue.qsize()}")
                        
                        # Update LEDs synchronized with agent speech (audio-reactive)
                        if self.led_controller:
                            self.led_controller.update_speaking_leds(audio_array)
                        
                        # Feed to turn tracker for agent VAD (if enabled, skip if using speaker monitor)
                        if self.turn_tracker and not self.use_speaker_monitor:
                            playback_time = time.time()  # Estimation mode
                            self.turn_tracker.process_agent_audio(audio_array, playback_time)
                        
                        # Queue audio for playback (barge-in safe)
                        await self.audio_queue.put(audio_array)
                        self.last_audio_time = time.time()  # Reset silence timer
                
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
        
        # Stop and close both input and output streams
        if self.input_stream:
            self.input_stream.stop()
            self.input_stream.close()
        
        if self.output_stream:
            self.output_stream.stop()
            self.output_stream.close()
        
        # Stop speaker monitor and transfer its turns to turn tracker (if enabled)
        if self.speaker_monitor:
            self.speaker_monitor.stop()
            # Copy speaker monitor turns to turn tracker's agent VAD
            # This replaces the estimation-based agent turns with ground truth
            if self.turn_tracker:
                from lib.turn_tracker import Turn, Speaker
                if self.turn_tracker.agent_vad:
                    self.turn_tracker.agent_vad.turns = [
                        Turn(speaker=Speaker.AGENT, start_time=t.start_time, end_time=t.end_time)
                        for t in self.speaker_monitor.turns
                    ]
        
        # Finalize turn tracker and print summary (if enabled)
        if self.turn_tracker:
            self.turn_tracker.finalize()
        
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
