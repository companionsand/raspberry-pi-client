#!/usr/bin/env python3
"""
Kin AI Raspberry Pi Client
===========================
Minimalistic wake word detection + conversation client for Raspberry Pi.

Features:
- Wake word detection using Porcupine ("Porcupine" keyword)
- Real-time conversation via ElevenLabs WebSocket API
- PipeWire echo cancellation for barge-in capability
- Communication with conversation-orchestrator via WebSocket
- Supabase authentication on startup

Usage:
    python main.py

Requirements:
    - Raspberry Pi OS with PipeWire
    - USB microphone and speaker
    - Environment variables: DEVICE_ID, SUPABASE_URL, SUPABASE_ANON_KEY, EMAIL, PASSWORD, 
                            CONVERSATION_ORCHESTRATOR_URL, ELEVENLABS_API_KEY, PICOVOICE_ACCESS_KEY
"""

import os
import sys
import signal
import time
import json
import base64
import asyncio
import subprocess
import uuid
import logging
from datetime import datetime, timezone
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging so OTEL handler captures console output.
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    force=True,
)

# Import required packages
try:
    import pvporcupine
    import sounddevice as sd
    import numpy as np
    import websockets
    import certifi
    import ssl
    from supabase import create_client, Client
except ImportError as e:
    print(f"✗ Missing required package: {e}")
    print("Install dependencies: pip install -r requirements.txt")
    sys.exit(1)

# Import telemetry (optional - graceful degradation)
try:
    from telemetry import (
        setup_telemetry,
        get_tracer,
        get_logger as get_otel_logger,
        create_client_metrics,
        add_span_attributes,
        add_span_event,
        create_span,
        create_conversation_trace,
        inject_trace_context,
        extract_trace_context,
        record_exception
    )
    TELEMETRY_AVAILABLE = True
except ImportError:
    print("⚠️  Telemetry module not available - running without telemetry")
    TELEMETRY_AVAILABLE = False
    # Provide no-op fallbacks
    def get_otel_logger(name, device_id=None):
        import logging
        return logging.getLogger(name)


# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    """Application configuration from environment variables"""
    
    # Device credentials
    DEVICE_ID = os.getenv("DEVICE_ID")
    
    # Logger (will be set after device_id is validated)
    LOGGER = None
    
    # Supabase authentication
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")
    EMAIL = os.getenv("EMAIL")
    PASSWORD = os.getenv("PASSWORD")
    
    # These will be set after authentication
    USER_ID = None
    AUTH_TOKEN = None
    
    # Backend
    CONVERSATION_ORCHESTRATOR_URL = os.getenv("CONVERSATION_ORCHESTRATOR_URL", "ws://localhost:8001/ws")
    
    # ElevenLabs API
    ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
    
    # Wake word detection
    PICOVOICE_ACCESS_KEY = os.getenv("PICOVOICE_ACCESS_KEY")
    WAKE_WORD = os.getenv("WAKE_WORD", "porcupine")
    
    # Audio devices (PipeWire AEC nodes)
    MIC_DEVICE = os.getenv("MIC_DEVICE", "echo_cancel.mic")
    SPEAKER_DEVICE = os.getenv("SPEAKER_DEVICE", "echo_cancel.speaker")
    
    # Audio settings
    SAMPLE_RATE = int(os.getenv("SAMPLE_RATE", "16000"))  # Configurable sample rate
    CHANNELS = 1
    CHUNK_SIZE = 512  # ~32ms frames for low latency
    
    # OpenTelemetry
    OTEL_ENABLED = os.getenv("OTEL_ENABLED", "true").lower() == "true"
    OTEL_EXPORTER_ENDPOINT = os.getenv("OTEL_EXPORTER_ENDPOINT", "http://localhost:4318")
    ENV = os.getenv("ENV", "production")
    
    # OpenTelemetry
    OTEL_ENABLED = os.getenv("OTEL_ENABLED", "true").lower() == "true"
    OTEL_EXPORTER_ENDPOINT = os.getenv("OTEL_EXPORTER_ENDPOINT", "http://localhost:4318")
    ENV = os.getenv("ENV", "production")
    
    @classmethod
    def validate(cls):
        """Validate required configuration"""
        missing = []
        
        if not cls.DEVICE_ID:
            missing.append("DEVICE_ID")
        if not cls.SUPABASE_URL:
            missing.append("SUPABASE_URL")
        if not cls.SUPABASE_ANON_KEY:
            missing.append("SUPABASE_ANON_KEY")
        if not cls.EMAIL:
            missing.append("EMAIL")
        if not cls.PASSWORD:
            missing.append("PASSWORD")
        if not cls.ELEVENLABS_API_KEY:
            missing.append("ELEVENLABS_API_KEY")
        if not cls.PICOVOICE_ACCESS_KEY:
            missing.append("PICOVOICE_ACCESS_KEY")
            
        if missing:
            print(f"✗ Missing required environment variables: {', '.join(missing)}")
            print("Create a .env file with required credentials")
            sys.exit(1)


# =============================================================================
# AUTHENTICATION
# =============================================================================

def authenticate_with_supabase():
    """
    Authenticate with Supabase and fetch auth token and user ID.
    Sets Config.AUTH_TOKEN and Config.USER_ID on success.
    """
    logger = Config.LOGGER
    print("\n🔐 Authenticating with Supabase...")
    
    try:
        # Create Supabase client
        supabase: Client = create_client(Config.SUPABASE_URL, Config.SUPABASE_ANON_KEY)
        
        # Sign in with email and password
        response = supabase.auth.sign_in_with_password({
            "email": Config.EMAIL,
            "password": Config.PASSWORD
        })
        
        # Extract auth token and user ID
        if response.user and response.session:
            Config.AUTH_TOKEN = response.session.access_token
            Config.USER_ID = response.user.id
            print(f"✓ Successfully authenticated")
            print(f"   User ID: {Config.USER_ID}")
            if logger:
                logger.info(
                    "supabase_authenticated",
                    extra={
                        "user_id": Config.USER_ID,
                        "device_id": Config.DEVICE_ID
                    }
                )
            return True
        else:
            print("✗ Authentication failed: No user or session returned")
            if logger:
                logger.error("supabase_authentication_failed", extra={"reason": "no_user_or_session"})
            return False
            
    except Exception as e:
        print(f"✗ Authentication error: {e}")
        if logger:
            logger.error(
                "supabase_authentication_error",
                extra={"error": str(e)},
                exc_info=True
            )
        return False


# =============================================================================
# AUDIO SETUP (from old-raspberry-pi-client)
# =============================================================================

def setup_audio_routing():
    """
    Configure PipeWire audio routing for echo cancellation.
    Sets default source/sink to AEC nodes if available.
    """
    logger = Config.LOGGER
    print("\n🔊 Setting up audio routing...")
    
    # Check if AEC nodes exist
    result = subprocess.run(
        ["pactl", "list", "short", "sources"],
        capture_output=True,
        text=True
    )
    
    has_aec_mic = "echo_cancel" in result.stdout
    
    if has_aec_mic:
        print(f"✓ Echo cancellation available")
        print(f"  Mic: {Config.MIC_DEVICE}")
        print(f"  Speaker: {Config.SPEAKER_DEVICE}")
        
        if logger:
            logger.info(
                "echo_cancellation_available",
                extra={
                    "mic_device": Config.MIC_DEVICE,
                    "speaker_device": Config.SPEAKER_DEVICE
                }
            )
        
        # Set as default devices
        subprocess.run(["pactl", "set-default-source", Config.MIC_DEVICE], 
                      capture_output=True)
        subprocess.run(["pactl", "set-default-sink", Config.SPEAKER_DEVICE], 
                      capture_output=True)
    else:
        print("⚠ Echo cancellation not available")
        print("  Using default audio devices")
        print("  Note: Barge-in may not work properly")
        
        if logger:
            logger.warning(
                "echo_cancellation_unavailable",
                extra={"user_id": Config.USER_ID}
            )


def get_audio_device_index(device_name: str, kind: str = "input") -> int:
    """
    Get sounddevice index for a specific device name.
    
    Args:
        device_name: Device name (e.g., "echo_cancel.mic")
        kind: "input" or "output"
    
    Returns:
        Device index or None if not found
    """
    devices = sd.query_devices()
    
    for idx, device in enumerate(devices):
        # Check if device name matches and has the right channels
        if device_name.lower() in device['name'].lower():
            if kind == "input" and device['max_input_channels'] > 0:
                return idx
            elif kind == "output" and device['max_output_channels'] > 0:
                return idx
    
    # Fallback to default device
    if kind == "input":
        return sd.default.device[0]
    else:
        return sd.default.device[1]


# =============================================================================
# WAKE WORD DETECTION (from old-raspberry-pi-client)
# =============================================================================

class WakeWordDetector:
    """Porcupine-based wake word detection"""
    
    def __init__(self):
        self.porcupine = None
        self.audio_stream = None
        self.detected = False
        self.running = False
        self.logger = Config.LOGGER
        
    def start(self):
        """Initialize Porcupine and start listening"""
        if self.running:
            return

        # Reset detection flag each time we enter listening mode
        self.detected = False

        print(f"\n🎤 Initializing wake word detection...")
        print(f"   Wake word: '{Config.WAKE_WORD}'")
        
        try:
            # Initialize Porcupine with built-in keyword
            self.porcupine = pvporcupine.create(
                access_key=Config.PICOVOICE_ACCESS_KEY,
                keywords=[Config.WAKE_WORD],
                sensitivities=[0.7]  # 0.0 (least sensitive) to 1.0 (most sensitive)
            )
            
            # Get audio device
            mic_index = get_audio_device_index(Config.MIC_DEVICE, "input")
            
            # Start audio stream for wake word detection
            self.audio_stream = sd.InputStream(
                device=mic_index,
                channels=Config.CHANNELS,
                samplerate=self.porcupine.sample_rate,
                blocksize=self.porcupine.frame_length,
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
                        "user_id": Config.USER_ID
                    }
                )
        except Exception as e:
            # Ensure partially-initialized resources are cleaned up
            if self.logger:
                self.logger.error(
                    "wake_word_detection_start_failed",
                    extra={"error": str(e)},
                    exc_info=True
                )
            self.stop()
            raise
        
    def _audio_callback(self, indata, frames, time_info, status):
        """Process audio frames for wake word detection"""
        if status:
            print(f"⚠ Audio status: {status}")
        
        # Convert to the format Porcupine expects
        audio_frame = indata[:, 0].astype(np.int16)
        
        # Process with Porcupine
        keyword_index = self.porcupine.process(audio_frame)
        
        if keyword_index >= 0:
            print(f"\n🎯 Wake word '{Config.WAKE_WORD}' detected!")
            self.detected = True
            if self.logger:
                self.logger.info(
                    "wake_word_detected",
                    extra={
                        "wake_word": Config.WAKE_WORD,
                        "user_id": Config.USER_ID
                    }
                )
    
    def stop(self):
        """Stop wake word detection"""
        if self.audio_stream:
            self.audio_stream.stop()
            self.audio_stream.close()
            self.audio_stream = None
        
        if self.porcupine:
            self.porcupine.delete()
            self.porcupine = None
        
        self.running = False
    
    def cleanup(self):
        """Clean up resources"""
        self.stop()


# =============================================================================
# CONVERSATION ORCHESTRATOR CLIENT
# =============================================================================

class OrchestratorClient:
    """WebSocket client for conversation-orchestrator"""
    
    def __init__(self):
        self.websocket = None
        self.connected = False
        self.running = False
        self.logger = Config.LOGGER
        
    async def connect(self):
        """Connect to conversation-orchestrator"""
        logger = self.logger
        print(f"\n🔌 Connecting to conversation-orchestrator...")
        print(f"   URL: {Config.CONVERSATION_ORCHESTRATOR_URL}")
        
        try:
            # Create SSL context if using wss://
            ssl_context = None
            if Config.CONVERSATION_ORCHESTRATOR_URL.startswith("wss://"):
                ssl_context = ssl.create_default_context(cafile=certifi.where())
            
            # Connect to WebSocket
            self.websocket = await websockets.connect(
                Config.CONVERSATION_ORCHESTRATOR_URL,
                ssl=ssl_context
            )
            
            # Wait for connection acceptance (FastAPI accepts first)
            # Then send authentication
            await self.websocket.send(json.dumps({
                "type": "auth",
                "token": Config.AUTH_TOKEN,
                "device_id": Config.DEVICE_ID,
                "user_id": Config.USER_ID,
            }))
            
            # Wait for connection confirmation
            response = await self.websocket.recv()
            data = json.loads(response)
            
            if data.get("type") == "connected":
                self.connected = True
                self.running = True
                print("✓ Connected to conversation-orchestrator")
                if logger:
                    logger.info(
                        "conversation_orchestrator_connected",
                        extra={
                            "url": Config.CONVERSATION_ORCHESTRATOR_URL,
                            "user_id": Config.USER_ID
                        }
                    )
                return True
            else:
                print(f"✗ Connection failed: {data}")
                if logger:
                    logger.error("conversation_orchestrator_connection_failed", extra={"response": str(data)})
                return False
                
        except Exception as e:
            print(f"✗ Connection error: {e}")
            if logger:
                logger.error(
                    "conversation_orchestrator_connection_error",
                    extra={"error": str(e)},
                    exc_info=True
                )
            return False
    
    async def send_reactive(self):
        """Send reactive conversation request with trace context"""
        if not self.connected:
            return
        
        message = {
            "type": "reactive",
            "user_id": Config.USER_ID,
            "device_id": Config.DEVICE_ID,
        }
        
        # Inject trace context for propagation
        if TELEMETRY_AVAILABLE:
            inject_trace_context(message)
        
        await self.websocket.send(json.dumps(message))
        print("✓ Sent reactive request")
        if self.logger:
            self.logger.info(
                "reactive_request_sent",
                extra={"user_id": Config.USER_ID}
            )
    
    async def send_conversation_start(
        self, conversation_id: str, elevenlabs_conversation_id: str, agent_id: str
    ):
        """Send conversation start notification with trace context"""
        if not self.connected:
            return
        
        message = {
            "type": "conversation_start",
            "conversation_id": conversation_id,
            "elevenlabs_conversation_id": elevenlabs_conversation_id,
            "agent_id": agent_id,
            "device_id": Config.DEVICE_ID,
            "user_id": Config.USER_ID,
            "start_time": datetime.now(timezone.utc).isoformat(),
        }
        
        # Inject trace context for propagation
        if TELEMETRY_AVAILABLE:
            inject_trace_context(message)
        
        await self.websocket.send(json.dumps(message))
        print("✓ Sent conversation_start notification")
        if self.logger:
            self.logger.info(
                "conversation_start_notification_sent",
                extra={
                    "conversation_id": conversation_id,
                    "elevenlabs_conversation_id": elevenlabs_conversation_id,
                    "agent_id": agent_id,
                    "user_id": Config.USER_ID
                }
            )
    
    async def send_conversation_end(
        self, conversation_id: str, elevenlabs_conversation_id: str, 
        agent_id: str, end_reason: str
    ):
        """Send conversation end notification with trace context"""
        if not self.connected:
            return
        
        message = {
            "type": "conversation_end",
            "conversation_id": conversation_id,
            "elevenlabs_conversation_id": elevenlabs_conversation_id,
            "agent_id": agent_id,
            "device_id": Config.DEVICE_ID,
            "user_id": Config.USER_ID,
            "end_time": datetime.now(timezone.utc).isoformat(),
            "end_reason": end_reason,
        }
        
        # Inject trace context for propagation
        if TELEMETRY_AVAILABLE:
            inject_trace_context(message)
        
        await self.websocket.send(json.dumps(message))
        print("✓ Sent conversation_end notification")
        if self.logger:
            self.logger.info(
                "conversation_end_notification_sent",
                extra={
                    "conversation_id": conversation_id,
                    "elevenlabs_conversation_id": elevenlabs_conversation_id,
                    "agent_id": agent_id,
                    "end_reason": end_reason,
                    "user_id": Config.USER_ID
                }
            )
    
    async def receive_message(self):
        """Receive and return a message from orchestrator"""
        if not self.connected:
            return None
        
        try:
            message = await asyncio.wait_for(self.websocket.recv(), timeout=0.1)
            return json.loads(message)
        except asyncio.TimeoutError:
            return None
        except Exception as e:
            print(f"✗ Receive error: {e}")
            return None
    
    async def disconnect(self):
        """Disconnect from orchestrator"""
        self.connected = False
        self.running = False
        if self.websocket:
            await self.websocket.close()
        if self.logger:
            self.logger.info(
                "conversation_orchestrator_disconnected",
                extra={"user_id": Config.USER_ID}
            )


# =============================================================================
# ELEVENLABS CONVERSATION CLIENT
# =============================================================================

class ElevenLabsConversationClient:
    """WebSocket-based conversation client for ElevenLabs"""
    
    def __init__(self, web_socket_url: str, agent_id: str, user_terminate_flag=None):
        self.web_socket_url = web_socket_url
        self.agent_id = agent_id
        self.websocket = None
        self.audio_stream = None
        self.running = False
        self.conversation_id = str(uuid.uuid4())
        self.elevenlabs_conversation_id = None
        self.end_reason = "normal"
        self.silence_timeout = 30.0  # seconds
        self.last_audio_time = None
        self.user_terminate_flag = user_terminate_flag  # Reference to shared flag
        self.logger = Config.LOGGER
        
    async def start(self, orchestrator_client: OrchestratorClient):
        """Start a conversation session"""
        print(f"\n💬 Starting conversation...")
        print(f"   Agent ID: {self.agent_id}")
        print(f"   Conversation ID: {self.conversation_id}")
        
        # Add API key to WebSocket URL
        ws_url = f"{self.web_socket_url}&api_key={Config.ELEVENLABS_API_KEY}"
        
        # Get audio devices
        mic_index = get_audio_device_index(Config.MIC_DEVICE, "input")
        speaker_index = get_audio_device_index(Config.SPEAKER_DEVICE, "output")
        
        # Create SSL context
        ssl_context = ssl.create_default_context(cafile=certifi.where())
        
        try:
            # Open audio stream for conversation
            self.audio_stream = sd.Stream(
                device=(mic_index, speaker_index),
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
                if self.logger:
                    self.logger.info(
                        "elevenlabs_connected",
                        extra={
                            "conversation_id": self.conversation_id,
                            "agent_id": self.agent_id,
                            "user_id": Config.USER_ID
                        }
                    )
                
                # Send conversation initiation
                await websocket.send(json.dumps({
                    "type": "conversation_initiation_client_data"
                }))
                
                print("✓ Conversation started - speak now!")
                if self.logger:
                    self.logger.info(
                        "conversation_started",
                        extra={
                            "conversation_id": self.conversation_id,
                            "agent_id": self.agent_id,
                            "user_id": Config.USER_ID
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
            if self.logger:
                self.logger.error(
                    "conversation_error",
                    extra={
                        "conversation_id": self.conversation_id,
                        "agent_id": self.agent_id,
                        "user_id": Config.USER_ID,
                        "error": str(e)
                    },
                    exc_info=True
                )
            self.end_reason = "network_failure"
        finally:
            await self.stop(orchestrator_client)
    
    async def _send_audio(self):
        """Send microphone audio to WebSocket"""
        try:
            while self.running:
                # Read audio from microphone
                audio_data, _ = self.audio_stream.read(Config.CHUNK_SIZE)
                
                # Check for audio activity (simple energy detection)
                audio_energy = np.abs(audio_data).mean()
                if audio_energy > 100:  # Threshold for detecting speech
                    self.last_audio_time = time.time()
                
                # Encode as base64 and send
                audio_b64 = base64.b64encode(audio_data.tobytes()).decode('utf-8')
                message = {"user_audio_chunk": audio_b64}
                
                await self.websocket.send(json.dumps(message))
                
                # Check for silence timeout
                if self.last_audio_time and (time.time() - self.last_audio_time) > self.silence_timeout:
                    print("\n⏱️  Silence timeout - ending conversation")
                    if self.logger:
                        self.logger.info(
                            "conversation_silence_timeout",
                            extra={
                                "conversation_id": self.conversation_id,
                                "user_id": Config.USER_ID
                            }
                        )
                    self.end_reason = "silence"
                    self.running = False
                    break
                
                # Check for user termination
                if self.user_terminate_flag and self.user_terminate_flag[0]:
                    print("\n🛑 User termination - ending conversation")
                    if self.logger:
                        self.logger.info(
                            "conversation_user_terminated",
                            extra={
                                "conversation_id": self.conversation_id,
                                "user_id": Config.USER_ID
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
            self.end_reason = "network_failure"
            self.running = False
    
    async def _receive_messages(self, orchestrator_client: OrchestratorClient):
        """Receive and process messages from WebSocket"""
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
            if self.logger:
                self.logger.info(
                    "conversation_connection_closed",
                    extra={
                        "conversation_id": self.conversation_id,
                        "end_reason": self.end_reason,
                        "user_id": Config.USER_ID
                    }
                )
            self.running = False
        except asyncio.CancelledError:
            self.end_reason = "user_terminated"
            self.running = False
        except Exception as e:
            print(f"✗ Receive error: {e}")
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
                    "user_id": Config.USER_ID
                }
            )
        
        if self.websocket:
            await self.websocket.close()


# =============================================================================
# MAIN APPLICATION
# =============================================================================

class KinClient:
    """Main application controller"""
    
    def __init__(self):
        self.wake_detector = WakeWordDetector()
        self.orchestrator_client = OrchestratorClient()
        self.running = True
        self.conversation_active = False
        self.awaiting_agent_details = False
        self.user_terminate = [False]  # Use list for mutable reference
        self.shutdown_requested = False
        self.conversation_start_time = None
        self.conversation_trace_context = None
        
        # Setup telemetry if available
        self.metrics = None
        if TELEMETRY_AVAILABLE and Config.OTEL_ENABLED:
            try:
                setup_telemetry(
                    device_id=Config.DEVICE_ID,
                    endpoint=Config.OTEL_EXPORTER_ENDPOINT
                )
                self.metrics = create_client_metrics()
                
                # Initialize logger with device_id
                Config.LOGGER = get_otel_logger(__name__, device_id=Config.DEVICE_ID)
                self.logger = Config.LOGGER
                
                print("✓ OpenTelemetry initialized")
            except Exception as e:
                print(f"⚠️  Failed to initialize telemetry: {e}")
                Config.LOGGER = None
                self.logger = None
        else:
            # Fallback logger without telemetry
            import logging
            Config.LOGGER = logging.getLogger(__name__)
            self.logger = Config.LOGGER
        
        # Setup signal handlers
        signal.signal(signal.SIGUSR1, self._handle_terminate_signal)
        signal.signal(signal.SIGINT, self._handle_interrupt_signal)
        signal.signal(signal.SIGTERM, self._handle_interrupt_signal)
    
    def _handle_terminate_signal(self, sig, frame):
        """Handle user-initiated termination signal"""
        print("\n🛑 User termination signal received")
        if self.logger:
            self.logger.info(
                "terminate_signal_received",
                extra={
                    "signal": "SIGUSR1",
                    "user_id": Config.USER_ID
                }
            )
        self.user_terminate[0] = True
    
    def _handle_interrupt_signal(self, sig, frame):
        """Handle interrupt/termination signals (Ctrl+C, SIGTERM)."""
        signal_name = getattr(signal, "Signals", lambda s: s)(sig)
        print(f"\n🛑 Received {signal_name} - ", end="")
        if self.conversation_active:
            print("ending current conversation...")
            self.user_terminate[0] = True
        else:
            print("shutting down...")
            self.shutdown_requested = True
            self.running = False
    
    def _resume_wake_word_detection(self):
        """Start wake word detector again if it is not already running."""
        if self.wake_detector.running:
            return
        
        self.wake_detector.start()
        print(f"\n✓ Listening for '{Config.WAKE_WORD}' again...")
    
    async def run(self):
        """Main application loop"""
        print("\n" + "="*60)
        print("🎙️  Kin AI Raspberry Pi Client")
        print("="*60)
        
        # Validate configuration
        Config.validate()
        
        # Authenticate with Supabase
        if not authenticate_with_supabase():
            print("✗ Failed to authenticate with Supabase")
            return
        
        # Setup audio routing
        setup_audio_routing()
        
        # Connect to conversation-orchestrator
        connected = await self.orchestrator_client.connect()
        if not connected:
            print("✗ Failed to connect to conversation-orchestrator")
            # Record connection failure
            if self.metrics:
                self.metrics["connection_status"].add(-1, {
                    "device_id": Config.DEVICE_ID
                })
            return
        
        # Record successful connection
        if self.metrics:
            self.metrics["connection_status"].add(1, {
                "device_id": Config.DEVICE_ID
            })
            if TELEMETRY_AVAILABLE:
                add_span_event("orchestrator_connected",
                             device_id=Config.DEVICE_ID)
        
        # Start wake word detection
        self.wake_detector.start()
        
        print("\n" + "="*60)
        print(f"✓ Ready! Say '{Config.WAKE_WORD}' to start a conversation")
        print("  Press Ctrl+C to exit")
        print("="*60 + "\n")
        
        # Main loop
        try:
            while self.running:
                # Check if wake word was detected
                if self.wake_detector.detected and not self.conversation_active:
                    self.wake_detector.detected = False
                    
                    # Record wake word detection telemetry
                    if self.metrics:
                        self.metrics["wake_word_detections"].add(1, {
                            "device_id": Config.DEVICE_ID,
                            "wake_word": Config.WAKE_WORD
                        })
                    
                    # Stop wake word detection during conversation
                    self.wake_detector.stop()
                    
                    # Handle conversation (creates trace inside)
                    await self._handle_conversation()
                
                # Check for messages from orchestrator
                message = await self.orchestrator_client.receive_message()
                if message:
                    await self._handle_orchestrator_message(message)
                
                # Handle proactive conversations (start_conversation message)
                # This is handled in _handle_orchestrator_message
                
                # Small sleep to prevent busy loop
                await asyncio.sleep(0.1)
                
        except KeyboardInterrupt:
            print("\n\n🛑 Shutting down...")
        finally:
            await self.cleanup()
    
    async def _handle_conversation(self):
        """Handle a single conversation session with conversation-level tracing"""
        # Create a new conversation trace (root trace for this conversation)
        if TELEMETRY_AVAILABLE:
            conversation_trace = create_conversation_trace(
                "conversation",
                conversation_type="reactive",
                device_id=Config.DEVICE_ID,
                user_id=Config.USER_ID,
                wake_word=Config.WAKE_WORD
            )
            self.conversation_trace_context = conversation_trace
        else:
            self.conversation_trace_context = None
        
        # Use the conversation trace as context for all operations
        async def _handle_with_trace():
            self.conversation_active = True
            self.awaiting_agent_details = True
            self.user_terminate[0] = False
            
            # Send reactive request (will inject trace context)
            await self.orchestrator_client.send_reactive()
            
            # Wait for agent_details message (with timeout)
            timeout = 10.0  # seconds
            start_time = time.time()
            
            while self.conversation_active and (time.time() - start_time) < timeout:
                # Check for messages from orchestrator
                message = await self.orchestrator_client.receive_message()
                if message:
                    await self._handle_orchestrator_message(message)
                    if not self.conversation_active:
                        break
                
                # Small sleep to prevent busy loop
                await asyncio.sleep(0.1)
            
            if self.awaiting_agent_details:
                print("✗ Timeout waiting for agent details")
                if self.logger:
                    self.logger.error(
                        "agent_details_timeout",
                        extra={"user_id": Config.USER_ID}
                    )
                self.conversation_active = False
                self.awaiting_agent_details = False
                self._resume_wake_word_detection()
        
        # Execute with trace context if available
        if self.conversation_trace_context:
            with self.conversation_trace_context:
                await _handle_with_trace()
        else:
            await _handle_with_trace()
        
    async def _handle_orchestrator_message(self, message: dict):
        """Handle messages from conversation-orchestrator"""
        message_type = message.get("type")
        
        if message_type == "agent_details" or message_type == "start_conversation":
            # Check if conversation is already active
            if self.conversation_active and not self.awaiting_agent_details:
                print("⚠ Conversation already active, ignoring new request")
                return
            
            # Clear pending flag once we have agent details
            self.awaiting_agent_details = False
            
            # Extract agent details
            agent_id = message.get("agent_id")
            web_socket_url = message.get("web_socket_url")
            
            if not agent_id or not web_socket_url:
                print("✗ Invalid agent details received")
                if self.logger:
                    self.logger.error("invalid_agent_details", extra={"message": message})
                return
            
            print(f"✓ Received agent details: {agent_id}")
            if self.logger:
                self.logger.info(
                    "agent_details_received",
                    extra={
                        "agent_id": agent_id,
                        "message_type": message_type,
                        "user_id": Config.USER_ID
                    }
                )
            
            # For proactive conversations (start_conversation), extract and use trace context
            # For reactive conversations (agent_details), trace context is already set from _handle_conversation
            context_token = None
            if message_type == "start_conversation" and TELEMETRY_AVAILABLE:
                # Extract trace context from the proactive conversation message
                context_token = extract_trace_context(message)
                if self.logger:
                    self.logger.info(
                        "proactive_conversation_trace_extracted",
                        extra={
                            "has_traceparent": "traceparent" in message,
                            "user_id": Config.USER_ID
                        }
                    )
            
            # Execute conversation handling with proper trace context
            async def _handle_conversation_with_context():
                # Mark conversation as active
                self.conversation_active = True
                self.user_terminate[0] = False
                self.conversation_start_time = time.time()
                
                # Record conversation start telemetry
                if self.metrics:
                    self.metrics["conversations_started"].add(1, {
                        "device_id": Config.DEVICE_ID,
                        "user_id": Config.USER_ID,
                        "agent_id": agent_id
                    })
                
                # Stop wake word detection during conversation
                self.wake_detector.stop()
                
                # Start ElevenLabs conversation
                client = ElevenLabsConversationClient(
                    web_socket_url, 
                    agent_id,
                    user_terminate_flag=self.user_terminate
                )
                await client.start(self.orchestrator_client)
            
                # Check if user terminated
                if self.user_terminate[0]:
                    print("✓ User terminated conversation")
                
                # Record conversation completion telemetry
                if self.metrics and self.conversation_start_time:
                    duration = time.time() - self.conversation_start_time
                    self.metrics["conversations_completed"].add(1, {
                        "device_id": Config.DEVICE_ID,
                        "user_id": Config.USER_ID,
                        "agent_id": agent_id
                    })
                    self.metrics["conversation_duration"].record(duration, {
                        "device_id": Config.DEVICE_ID,
                        "user_id": Config.USER_ID
                    })
                    self.conversation_start_time = None
                
                # Resume wake word detection
                self._resume_wake_word_detection()
                
                if self.logger:
                    self.logger.info(
                        "wake_word_detection_resumed",
                        extra={"user_id": Config.USER_ID}
                    )
                
                self.conversation_active = False
                self.user_terminate[0] = False
            
            # Execute with trace context if we have one (proactive)
            if context_token is not None:
                try:
                    await _handle_conversation_with_context()
                finally:
                    # Detach the context
                    from opentelemetry import context
                    context.detach(context_token)
            else:
                # For reactive conversations, we're already in the trace context from _handle_conversation
                await _handle_conversation_with_context()
            
        elif message_type == "error":
            error_msg = message.get("message", "Unknown error")
            print(f"✗ Orchestrator error: {error_msg}")
            
            if self.logger:
                self.logger.error(
                    "orchestrator_error_received",
                    extra={
                        "error_message": error_msg,
                        "user_id": Config.USER_ID
                    }
                )
            
            # Record error telemetry
            if self.metrics:
                self.metrics["errors"].add(1, {
                    "device_id": Config.DEVICE_ID,
                    "error_type": "orchestrator_error",
                    "error_message": error_msg
                })
            
            self.conversation_active = False
            self.awaiting_agent_details = False
            
            # If we were waiting on a conversation that failed, resume wake word detection
            self._resume_wake_word_detection()
    
    async def cleanup(self):
        """Clean up resources"""
        self.running = False
        self.wake_detector.cleanup()
        await self.orchestrator_client.disconnect()
        
        # Record disconnection
        if self.metrics:
            self.metrics["connection_status"].add(-1, {
                "device_id": Config.DEVICE_ID
            })
            if TELEMETRY_AVAILABLE:
                add_span_event("orchestrator_disconnected",
                             device_id=Config.DEVICE_ID)
        
        print("✓ Cleanup complete")
        print("👋 Goodbye!\n")


# =============================================================================
# ENTRY POINT
# =============================================================================

def main():
    """Application entry point"""
    # Run the client
    client = KinClient()
    asyncio.run(client.run())


if __name__ == "__main__":
    main()
