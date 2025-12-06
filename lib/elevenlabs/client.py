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
        self.running = False
        self.conversation_id = str(uuid.uuid4())
        self.elevenlabs_conversation_id = None
        self.end_reason = "normal"
        self.silence_timeout = 60.0  # seconds
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
        # Audio queue and interruption handling
        # -------------------------------------------------------------------------
        # Interruptions are detected server-side by ElevenLabs and sent as 'interruption' messages.
        # The client just needs to stop playback and clear the queue when interrupted.
        self.audio_queue = asyncio.Queue()  # Queue for agent audio chunks
        self.playback_active = False  # Flag to stop current audio playback
        
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
            
            # Output stream: speaker only (separate from input for clean interruption handling)
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
                
                # Write in very small chunks (256 samples = ~16ms at 16kHz) for fast response
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
                        
                        # Track chunk count
                        self._chunk_count += 1
                        
                        # First chunk of NEW agent turn: Reset VAD state to clear residual from user turn
                        # This helps prevent false positives in VAD during agent speech
                        if self._chunk_count == 1:
                            self._vad_state = np.zeros((2, 1, 128), dtype=np.float32)
                            print(f"📥 AGENT TURN START: first chunk ({len(audio_array)} samples) [VAD state RESET]")
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
                    
                    # Reset chunk count for next agent turn
                    self._chunk_count = 0
                    
                    # Reset LED to conversation/listening state
                    if self.led_controller:
                        self.led_controller.set_state(self.led_controller.STATE_CONVERSATION)
                    
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
        
        # Stop and close both input and output streams
        if self.input_stream:
            self.input_stream.stop()
            self.input_stream.close()
        
        if self.output_stream:
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
