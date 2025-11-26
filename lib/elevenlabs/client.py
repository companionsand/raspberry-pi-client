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
        self.audio_stream = None
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
        
        # VAD state: hidden states for LSTM (required for stateful inference)
        # Shape: (2, 1, 64) for h and c states in the Silero LSTM
        self._vad_h = np.zeros((2, 1, 64), dtype=np.float32)
        self._vad_c = np.zeros((2, 1, 64), dtype=np.float32)
        self.vad_threshold = 0.5  # Probability threshold for speech detection
        
        # THINKING state trigger: track when user stopped speaking
        # After 400ms of silence, transition LED to THINKING state
        # (400ms forgives micro-pauses but still beats server VAD ~1000ms)
        self._last_speech_time = None  # When VAD last detected speech
        self._thinking_threshold_ms = 400  # Silence duration before THINKING state
        
    async def start(self, orchestrator_client: OrchestratorClient):
        """Start a conversation session"""
        logger = self.logger
        print(f"\n💬 Starting conversation...")
        print(f"   Agent ID: {self.agent_id}")
        print(f"   Conversation ID: {self.conversation_id}")
        
        # Add API key to WebSocket URL
        ws_url = f"{self.web_socket_url}&api_key={Config.ELEVENLABS_API_KEY}"
        
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
            # Open audio stream for conversation
            self.audio_stream = sd.Stream(
                device=(self.mic_device_index, self.speaker_device_index),
                samplerate=Config.SAMPLE_RATE,
                channels=Config.CHANNELS,
                dtype='int16',
                blocksize=Config.CHUNK_SIZE
            )
            self.audio_stream.start()
            
            # Connect to WebSocket
            async with websockets.connect(ws_url, ssl=ssl_context) as websocket:
                self.websocket = websocket
                self.running = True
                self.last_audio_time = time.time()
                
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
                            "has_weather": context_manager.has_weather_data,
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
                
                # Run send and receive tasks concurrently
                send_task = asyncio.create_task(self._send_audio())
                receive_task = asyncio.create_task(self._receive_messages(orchestrator_client))
                
                # Wait for either task to complete
                done, pending = await asyncio.wait(
                    [send_task, receive_task],
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
                audio_data, _ = self.audio_stream.read(Config.CHUNK_SIZE)
                
                # -------------------------------------------------------------------------
                # Voice Activity Detection (VAD) - Detect if user is speaking
                # -------------------------------------------------------------------------
                is_speech = False
                
                if self.vad_enabled and self.vad_session:
                    # Run Silero VAD inference
                    # Convert int16 audio to float32 normalized [-1, 1]
                    audio_float = audio_data.flatten().astype(np.float32) / 32768.0
                    
                    # Prepare ONNX inputs
                    ort_inputs = {
                        'input': audio_float.reshape(1, -1),
                        'sr': np.array([Config.SAMPLE_RATE], dtype=np.int64),
                        'h': self._vad_h,
                        'c': self._vad_c
                    }
                    
                    # Run inference and update hidden states
                    try:
                        output, self._vad_h, self._vad_c = self.vad_session.run(None, ort_inputs)
                        speech_prob = output[0][0]
                        is_speech = speech_prob > self.vad_threshold
                    except Exception:
                        # Fallback to energy detection if VAD fails
                        is_speech = np.abs(audio_data).mean() > 100
                else:
                    # Fallback: simple energy detection
                    is_speech = np.abs(audio_data).mean() > 100
                
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
                        
                        # Transition to THINKING after 200ms of silence
                        # Only if we're in CONVERSATION (listening) state, not SPEAKING
                        if silence_ms > self._thinking_threshold_ms:
                            if current == self.led_controller.STATE_CONVERSATION:
                                self.led_controller.set_state(self.led_controller.STATE_THINKING)
                
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
                
                elif 'agent_response_event' in data:
                    response = data['agent_response_event'].get('agent_response', '')
                    if response:
                        print(f"🤖 Agent: {response}")
                
                elif 'audio_event' in data:
                    # Decode and play agent audio
                    audio_b64 = data['audio_event'].get('audio_base_64', '')
                    if audio_b64:
                        audio_bytes = base64.b64decode(audio_b64)
                        audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
                        
                        # Update LEDs synchronized with agent speech (audio-reactive)
                        if self.led_controller:
                            self.led_controller.update_speaking_leds(audio_array)
                        
                        self.audio_stream.write(audio_array)
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
        
        if self.audio_stream:
            self.audio_stream.stop()
            self.audio_stream.close()
        
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

