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
from datetime import datetime, timezone
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

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


# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    """Application configuration from environment variables"""
    
    # Device credentials
    DEVICE_ID = os.getenv("DEVICE_ID")
    
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
    
    # Audio settings (matches ElevenLabs requirements)
    SAMPLE_RATE = 16000
    CHANNELS = 1
    CHUNK_SIZE = 512  # ~32ms frames for low latency
    
    # Heartbeat interval
    HEARTBEAT_INTERVAL = 10  # seconds
    
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
            return True
        else:
            print("✗ Authentication failed: No user or session returned")
            return False
            
    except Exception as e:
        print(f"✗ Authentication error: {e}")
        return False


# =============================================================================
# AUDIO SETUP (from old-raspberry-pi-client)
# =============================================================================

def setup_audio_routing():
    """
    Configure PipeWire audio routing for echo cancellation.
    Sets default source/sink to AEC nodes if available.
    """
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
        
        # Set as default devices
        subprocess.run(["pactl", "set-default-source", Config.MIC_DEVICE], 
                      capture_output=True)
        subprocess.run(["pactl", "set-default-sink", Config.SPEAKER_DEVICE], 
                      capture_output=True)
    else:
        print("⚠ Echo cancellation not available")
        print("  Using default audio devices")
        print("  Note: Barge-in may not work properly")


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
        except Exception:
            # Ensure partially-initialized resources are cleaned up
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
        
    async def connect(self):
        """Connect to conversation-orchestrator"""
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
                return True
            else:
                print(f"✗ Connection failed: {data}")
                return False
                
        except Exception as e:
            print(f"✗ Connection error: {e}")
            return False
    
    async def send_reactive(self):
        """Send reactive conversation request"""
        if not self.connected:
            return
        
        message = {
            "type": "reactive",
            "user_id": Config.USER_ID,
            "device_id": Config.DEVICE_ID,
        }
        
        await self.websocket.send(json.dumps(message))
        print("✓ Sent reactive request")
    
    async def send_heartbeat(self, device_status: str = "online"):
        """Send heartbeat message"""
        if not self.connected:
            return
        
        message = {
            "type": "heartbeat",
            "user_id": Config.USER_ID,
            "device_id": Config.DEVICE_ID,
            "device_status": device_status,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        
        await self.websocket.send(json.dumps(message))
    
    async def send_conversation_start(
        self, conversation_id: str, elevenlabs_conversation_id: str, agent_id: str
    ):
        """Send conversation start notification"""
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
        
        await self.websocket.send(json.dumps(message))
        print("✓ Sent conversation_start notification")
    
    async def send_conversation_end(
        self, conversation_id: str, elevenlabs_conversation_id: str, 
        agent_id: str, end_reason: str
    ):
        """Send conversation end notification"""
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
        
        await self.websocket.send(json.dumps(message))
        print("✓ Sent conversation_end notification")
    
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
                
                # Send conversation initiation
                await websocket.send(json.dumps({
                    "type": "conversation_initiation_client_data"
                }))
                
                print("✓ Conversation started - speak now!")
                
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
                    self.end_reason = "silence"
                    self.running = False
                    break
                
                # Check for user termination
                if self.user_terminate_flag and self.user_terminate_flag[0]:
                    print("\n🛑 User termination - ending conversation")
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
        
        # Setup signal handlers
        signal.signal(signal.SIGUSR1, self._handle_terminate_signal)
        signal.signal(signal.SIGINT, self._handle_interrupt_signal)
        signal.signal(signal.SIGTERM, self._handle_interrupt_signal)
    
    def _handle_terminate_signal(self, sig, frame):
        """Handle user-initiated termination signal"""
        print("\n🛑 User termination signal received")
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
            return
        
        # Start wake word detection
        self.wake_detector.start()
        
        print("\n" + "="*60)
        print(f"✓ Ready! Say '{Config.WAKE_WORD}' to start a conversation")
        print("  Press Ctrl+C to exit")
        print("="*60 + "\n")
        
        # Start heartbeat task
        heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        
        # Main loop
        try:
            while self.running:
                # Check if wake word was detected
                if self.wake_detector.detected and not self.conversation_active:
                    self.wake_detector.detected = False
                    
                    # Stop wake word detection during conversation
                    self.wake_detector.stop()
                    
                    # Handle conversation
                    await self._handle_conversation()
                
                # Check for messages from orchestrator
                message = await self.orchestrator_client.receive_message()
                if message:
                    await self._handle_orchestrator_message(message)
                
                # Handle trigger-initiated conversations (start_conversation message)
                # This is handled in _handle_orchestrator_message
                
                # Small sleep to prevent busy loop
                await asyncio.sleep(0.1)
                
        except KeyboardInterrupt:
            print("\n\n🛑 Shutting down...")
        finally:
            await self.cleanup()
            heartbeat_task.cancel()
            try:
                await heartbeat_task
            except asyncio.CancelledError:
                pass
    
    async def _heartbeat_loop(self):
        """Send heartbeat messages periodically"""
        while self.running:
            try:
                await asyncio.sleep(Config.HEARTBEAT_INTERVAL)
                if self.running:
                    await self.orchestrator_client.send_heartbeat("online")
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"✗ Heartbeat error: {e}")
    
    async def _handle_conversation(self):
        """Handle a single conversation session"""
        self.conversation_active = True
        self.awaiting_agent_details = True
        self.user_terminate[0] = False
        
        # Send reactive request
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
            self.conversation_active = False
            self.awaiting_agent_details = False
            self._resume_wake_word_detection()
        
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
                return
            
            print(f"✓ Received agent details: {agent_id}")
            
            # Mark conversation as active
            self.conversation_active = True
            self.user_terminate[0] = False
            
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
            
            # Resume wake word detection
            self._resume_wake_word_detection()
            
            self.conversation_active = False
            self.user_terminate[0] = False
            
        elif message_type == "error":
            error_msg = message.get("message", "Unknown error")
            print(f"✗ Orchestrator error: {error_msg}")
            self.conversation_active = False
            self.awaiting_agent_details = False
            
            # If we were waiting on a conversation that failed, resume wake word detection
            self._resume_wake_word_detection()
    
    async def cleanup(self):
        """Clean up resources"""
        self.running = False
        self.wake_detector.cleanup()
        await self.orchestrator_client.disconnect()
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
