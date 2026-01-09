"""ElevenLabs conversation client with telemetry.

Now uses AudioManager for unified audio handling with WebRTC AEC support.
"""

import asyncio
import base64
import json
import os
import ssl
import time
import uuid
from typing import TYPE_CHECKING, Any, Dict, Optional

import certifi
import numpy as np
import onnxruntime as ort
import websockets
from scipy import signal

from lib.agent.context import ContextManager
from lib.agent.orchestrator import OrchestratorClient
from lib.config import Config
from lib.signals.publisher import SignalPublisher

if TYPE_CHECKING:
    from lib.audio.manager import AudioManager
    from lib.signals.bus import SignalBus


class ElevenLabsConversationClient(SignalPublisher):
    """WebSocket-based conversation client for ElevenLabs with full telemetry.
    
    Now uses AudioManager for unified audio handling with AEC support.
    """
    
    def __init__(
        self,
        web_socket_url: str,
        agent_id: str,
        audio_manager: "AudioManager",
        user_terminate_flag=None,
        led_controller=None,
        signal_bus: Optional["SignalBus"] = None
    ):
        """
        Initialize ElevenLabs conversation client.
        
        Args:
            web_socket_url: ElevenLabs WebSocket URL
            agent_id: Agent ID for this conversation
            audio_manager: AudioManager for audio input/output with AEC
            user_terminate_flag: Mutable flag [False] for user termination
            led_controller: LEDController instance for visual feedback (optional)
            signal_bus: SignalBus for publishing transcript signals (optional)
        """
        # Initialize SignalPublisher mixin
        SignalPublisher.__init__(self, signal_bus, "elevenlabs")
        
        self.web_socket_url = web_socket_url
        self.agent_id = agent_id
        self._audio_manager = audio_manager
        self.websocket = None
        self.running = False
        self.conversation_id = str(uuid.uuid4())
        self.elevenlabs_conversation_id = None
        self.end_reason = "normal"
        self.last_audio_time = None  # Used for LED state management
        self.user_terminate_flag = user_terminate_flag  # Reference to shared flag
        self.led_controller = led_controller  # LED controller for visual feedback
        self.logger = Config.LOGGER
        
        # Audio input queue (filled by AudioManager callback)
        self._input_queue: asyncio.Queue = None
        
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
        
        # Track conversation start time for status logging
        self._conversation_start_time = None
        
        # Debug logging
        self._aec_debug_last_log = 0
        
        # ElevenLabs audio sample rate - detect from metadata or assume 16kHz (same as playback)
        # If audio sounds robotic/slow, ElevenLabs may be sending 24kHz - enable resampling
        self._elevenlabs_sample_rate = None  # Will be detected from metadata or default to 16kHz
        self._playback_sample_rate = Config.SAMPLE_RATE  # 16kHz for ReSpeaker
        self._resampling_enabled = False  # Disable by default - enable if audio sounds wrong
        
        # -------------------------------------------------------------------------
        # Client Tools - Music playback triggers conversation end
        # -------------------------------------------------------------------------
        # When play_music is called, we end the conversation so main.py can start music mode
        # This is simpler than trying to share audio devices during conversation
        self._music_mode_requested = False  # Set True when play_music tool is called
        self._music_genre = None  # Genre requested for music playback (from tool parameters)
        
        # Agent speech monitoring (detect when agent stops speaking)
        self._agent_speaking = False
        self._agent_silence_start = None
        self._agent_silence_threshold_ms = 500  # Wait 500ms of silence before ending
        
        
    async def start(self, orchestrator_client: OrchestratorClient):
        """Start a conversation session using AudioManager for audio I/O."""
        logger = self.logger
        print(f"\n💬 Starting conversation...")
        print(f"   Agent ID: {self.agent_id}")
        print(f"   Conversation ID: {self.conversation_id}")
        
        # Log AudioManager status
        aec_mode = "WebRTC AEC" if self._audio_manager.has_webrtc_aec else "Hardware AEC"
        print(f"   Audio: via AudioManager ({aec_mode})")
        
        # Add API key to WebSocket URL
        ws_url = f"{self.web_socket_url}&api_key={Config.ELEVENLABS_API_KEY}"
        
        # Create SSL context
        ssl_context = ssl.create_default_context(cafile=certifi.where())
        
        try:
            # Create async queue for receiving audio from AudioManager
            self._input_queue = asyncio.Queue()
            
            # Register callback with AudioManager to receive AEC-processed audio
            def audio_callback(audio_frame: np.ndarray):
                try:
                    self._input_queue.put_nowait(audio_frame.copy())
                except asyncio.QueueFull:
                    pass  # Drop frame if queue is full
            
            self._audio_manager.register_consumer(audio_callback)
            
            # Connect to WebSocket
            async with websockets.connect(ws_url, ssl=ssl_context) as websocket:
                self.websocket = websocket
                self.running = True
                self.last_audio_time = time.time()
                self._conversation_start_time = time.time()  # For status logging
                
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
            if self.end_reason not in ("music_mode",):
                self.end_reason = "network_failure"
        finally:
            await self.stop(orchestrator_client)
    
    async def _send_audio(self):
        """Send microphone audio to WebSocket.
        
        Audio is received from AudioManager via async queue, already AEC-processed.
        """
        logger = self.logger
        try:
            while self.running:
                # Get audio from AudioManager (already AEC-processed)
                try:
                    audio_data = await asyncio.wait_for(
                        self._input_queue.get(),
                        timeout=0.1
                    )
                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    print(f"✗ Input queue error: {e}")
                    self.end_reason = "network_failure"
                    self.running = False
                    break
                
                # Ensure correct shape (mono, int16)
                if audio_data.ndim > 1:
                    audio_data = audio_data.flatten()
                audio_data = audio_data.astype(np.int16)
                
                # Reshape for processing
                audio_data = audio_data.reshape(-1, 1)
                
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
                
                # Debug: Log what we're sending to ElevenLabs (every 3 seconds)
                now = time.time()
                if now - self._aec_debug_last_log >= 3.0:
                    self._aec_debug_last_log = now
                    sent_rms = np.sqrt(np.mean(audio_data.astype(float) ** 2)) / 32768.0
                    if Config.SHOW_ELEVENLABS_AUDIO_CHUNK_LOGS:
                        aec_mode = "WebRTC" if self._audio_manager.has_webrtc_aec else "HW AEC"
                        print(f"📤 [SENT TO 11LABS] shape={audio_data.shape} RMS={sent_rms:.4f} ({aec_mode})")
                
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
                
                # Periodic status logging for debugging stuck conversations
                now = time.time()
                if now - self._last_status_log_time > self._status_log_interval:
                    self._last_status_log_time = now
                    silence_duration = now - self.last_audio_time if self.last_audio_time else 0
                    conversation_duration = now - self._conversation_start_time if self._conversation_start_time else 0
                    led_state = self.led_controller.current_state if self.led_controller else "unknown"
                    vad_info = f"VAD={speech_prob:.2f}" if self.vad_enabled else "VAD=off"
                    if Config.SHOW_CONVERSATION_STATUS_LOGS:
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
            if self.end_reason not in ("music_mode",):
                self.end_reason = "network_failure"
            self.running = False
        except asyncio.CancelledError:
            if self.end_reason not in ("music_mode",):
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
            if self.end_reason not in ("music_mode",):
                self.end_reason = "network_failure"
            self.running = False
    
    async def _play_audio(self):
        """Play audio chunks from queue via AudioManager."""
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
                        if Config.SHOW_AGENT_TURN_LOGS:
                            print(f"📤 AGENT TURN END: {self._chunk_count} chunks played, resetting for next turn")
                        self._chunk_count = 0  # Critical: allows VAD reset on next agent turn
                    continue
                
                # Play the audio chunk in small sub-chunks for fast interruption
                self.playback_active = True
                
                # Write in small chunks (256 samples = ~16ms at 16kHz) for fast response
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
                    
                    # Write small chunk via AudioManager
                    chunk = audio_array[i:i+sub_chunk_size]
                    try:
                        # Ensure chunk is 1D array (mono)
                        if chunk.ndim > 1:
                            chunk = chunk.flatten()
                        
                        # Queue chunk for playback via AudioManager
                        self._audio_manager.play(chunk)
                        
                        # Yield to event loop
                        await asyncio.sleep(0)
                        
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
                        # Publish transcript signal for GUI
                        self.publish_text("transcript_input", transcript)
                        self.last_audio_time = time.time()  # Reset silence timer
                
                elif 'agent_response_event' in data:
                    response = data['agent_response_event'].get('agent_response', '')
                    if response:
                        print(f"🤖 Agent: {response}")
                        # Publish transcript signal for GUI
                        self.publish_text("transcript_output", response)
                
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
                            if Config.SHOW_AGENT_TURN_LOGS:
                                print(f"📥 AGENT TURN START: first chunk ({len(audio_array)} samples @ {self._playback_sample_rate}Hz) [VAD state RESET]")
                        elif self._chunk_count % 10 == 0:
                            print(f"📥 AGENT audio: chunk #{self._chunk_count} ({len(audio_array)} samples), queue={self.audio_queue.qsize()}")
                        
                        # Update LEDs synchronized with agent speech (audio-reactive)
                        if self.led_controller:
                            self.led_controller.update_speaking_leds(audio_array)
                        
                        # Queue audio for playback
                        await self.audio_queue.put(audio_array)
                        self.last_audio_time = time.time()  # Reset silence timer
                
                elif data.get('type') == 'client_tool_call' or 'client_tool_call' in data:
                    # Handle client-side tool calls from ElevenLabs agent
                    await self._handle_client_tool_call(data, orchestrator_client)
                
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
                    
                    # Clear AudioManager playback queue
                    am_cleared = self._audio_manager.clear_playback_queue()
                    
                    # Reset chunk count for next agent turn
                    self._chunk_count = 0
                    
                    # Reset LED to conversation/listening state
                    if self.led_controller:
                        self.led_controller.set_state(self.led_controller.STATE_CONVERSATION)
                    
                    print(f"   ✓ Cleared {cleared_count} queued audio chunks + {am_cleared} playback chunks, ready for user input")
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
            elif self.end_reason not in ("music_mode",):
                # Don't overwrite specific end reasons like music_mode
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
            if self.end_reason not in ("music_mode",):
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
            if self.end_reason not in ("music_mode",):
                self.end_reason = "network_failure"
            self.running = False
    
    async def _handle_client_tool_call(self, data: Dict[str, Any], orchestrator_client: OrchestratorClient):
        """
        Handle client-side tool calls from ElevenLabs agent.
        
        Supported tools:
        - play_music: Play BBC Radio 6 Music, with VAD+Scribe stop detection
        """
        logger = self.logger
        
        # Extract tool call details
        tool_call = data.get('client_tool_call', data)
        tool_name = tool_call.get('tool_name', '')
        tool_call_id = tool_call.get('tool_call_id', '')
        parameters = tool_call.get('parameters', {})
        
        print(f"\n🔧 Client tool call: {tool_name}")
        print(f"   Tool call ID: {tool_call_id}")
        print(f"   Parameters: {parameters}")
        
        if logger:
            logger.info(
                "client_tool_call_received",
                extra={
                    "tool_name": tool_name,
                    "tool_call_id": tool_call_id,
                    "parameters": parameters,
                    "conversation_id": self.conversation_id,
                    "device_id": Config.DEVICE_ID
                }
            )
        
        if tool_name == "play_music":
            await self._handle_play_music_tool(tool_call_id, parameters)
        else:
            # Unknown tool - send error response
            print(f"   ⚠️  Unknown client tool: {tool_name}")
            await self._send_client_tool_result(
                tool_call_id=tool_call_id,
                result=f"Error: Unknown tool '{tool_name}'",
                is_error=True
            )
    
    async def _handle_play_music_tool(self, tool_call_id: str, parameters: Dict[str, Any]):
        """
        Handle the play_music client tool.
        
        Simpler approach:
        1. Wait for agent to finish speaking (monitor Ch5 RMS for 500ms silence)
        2. End the conversation (frees up audio device)
        3. main.py will detect end_reason="music_mode" and start music playback
        
        This avoids complex audio device sharing - conversation and music are separate modes.
        """
        logger = self.logger
        
        # Extract genre from parameters (if provided)
        genre = parameters.get("genre") if parameters else None
        
        print(f"\n🎵 Music requested{f' (genre: {genre})' if genre else ''} - waiting for agent to finish speaking...")
        
        # Mark that music mode was requested and store genre
        self._music_mode_requested = True
        self._music_genre = genre
        
        if logger:
            logger.info(
                "music_mode_requested",
                extra={
                    "tool_call_id": tool_call_id,
                    "genre": genre,
                    "conversation_id": self.conversation_id,
                    "device_id": Config.DEVICE_ID
                }
            )
        
        # Wait for agent to stop speaking by monitoring Ch5 RMS
        # The audio callback will set _agent_speaking based on Ch5 levels
        # We wait until 500ms of silence
        await self._wait_for_agent_silence()
        
        print("   ✓ Agent finished speaking")
        print("   ✓ Ending conversation to start music mode...")
        
        # Set end reason so main.py knows to start music
        self.end_reason = "music_mode"
        
        # End the conversation
        self.running = False
    
    async def _wait_for_agent_silence(self):
        """
        Wait for agent to stop speaking by monitoring Ch5 (playback reference) RMS.
        
        Returns when 500ms of silence is detected on Ch5.
        """
        # Give agent time to start speaking (the "Let me play some music" response)
        await asyncio.sleep(0.2)
        
        silence_start = None
        
        while self.running:
            # Check if agent is currently speaking (based on recent audio output)
            # playback_active is True when we're outputting agent audio
            if self.playback_active:
                # Agent is speaking, reset silence counter
                silence_start = None
                self._agent_speaking = True
            else:
                # No audio being played
                if self._agent_speaking:
                    # Transition from speaking to silence
                    if silence_start is None:
                        silence_start = time.time()
                    
                    # Check if we've had 500ms of silence
                    silence_duration_ms = (time.time() - silence_start) * 1000
                    if silence_duration_ms >= self._agent_silence_threshold_ms:
                        self._agent_speaking = False
                        return
                else:
                    # Already in silence mode
                    if silence_start is None:
                        silence_start = time.time()
                    
                    silence_duration_ms = (time.time() - silence_start) * 1000
                    if silence_duration_ms >= self._agent_silence_threshold_ms:
                        return
            
            # Small delay to avoid busy loop
            await asyncio.sleep(0.05)
    
    @property
    def music_genre(self) -> Optional[str]:
        """Get the requested music genre (if any) from the play_music tool call."""
        return self._music_genre
    
    async def _send_client_tool_result(self, tool_call_id: str, result: str, is_error: bool = False):
        """
        Send client tool result back to ElevenLabs.
        
        Args:
            tool_call_id: The tool call ID from the original request
            result: Result message to send back
            is_error: Whether this is an error response
        """
        if not self.websocket or not self.running:
            print("⚠️  Cannot send tool result - WebSocket not connected")
            return
        
        message = {
            "type": "client_tool_result",
            "tool_call_id": tool_call_id,
            "result": result,
            "is_error": is_error
        }
        
        print(f"📤 Sending client tool result: {result[:50]}{'...' if len(result) > 50 else ''}")
        
        try:
            await self.websocket.send(json.dumps(message))
            
            if self.logger:
                self.logger.info(
                    "client_tool_result_sent",
                    extra={
                        "tool_call_id": tool_call_id,
                        "is_error": is_error,
                        "conversation_id": self.conversation_id,
                        "device_id": Config.DEVICE_ID
                    }
                )
        except Exception as e:
            print(f"✗ Failed to send tool result: {e}")
            if self.logger:
                self.logger.error(
                    "client_tool_result_send_failed",
                    extra={
                        "error": str(e),
                        "tool_call_id": tool_call_id,
                        "device_id": Config.DEVICE_ID
                    }
                )

    async def stop(self, orchestrator_client: OrchestratorClient):
        """Stop the conversation"""
        self.running = False
        
        # Clear AudioManager playback queue to stop any pending audio
        if self._audio_manager:
            self._audio_manager.clear_playback_queue()
        
        # Note: We don't unregister from AudioManager here because
        # the callback reference is local to start() and will be garbage collected.
        # AudioManager continues running for other consumers (wake word, etc.)
        
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
