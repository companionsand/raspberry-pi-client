"""Conversation orchestrator WebSocket client with telemetry"""

import json
import asyncio
import ssl
import websockets
import certifi
from datetime import datetime, timezone
from lib.config import Config

# Import telemetry (will check if available)
try:
    from lib.telemetry import inject_trace_context
    TELEMETRY_AVAILABLE = True
except ImportError:
    TELEMETRY_AVAILABLE = False


class OrchestratorClient:
    """WebSocket client for conversation-orchestrator with full telemetry support"""
    
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
                            "user_id": Config.USER_ID,
                            "device_id": Config.DEVICE_ID
                        }
                    )
                return True
            else:
                print(f"✗ Connection failed: {data}")
                if logger:
                    logger.error(
                        "conversation_orchestrator_connection_failed",
                        extra={
                            "response": str(data),
                            "user_id": Config.USER_ID
                        }
                    )
                return False
                
        except Exception as e:
            print(f"✗ Connection error: {e}")
            if logger:
                logger.error(
                    "conversation_orchestrator_connection_error",
                    extra={
                        "error": str(e),
                        "user_id": Config.USER_ID
                    },
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
                extra={
                    "user_id": Config.USER_ID,
                    "device_id": Config.DEVICE_ID
                }
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
                    "user_id": Config.USER_ID,
                    "device_id": Config.DEVICE_ID
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
                    "user_id": Config.USER_ID,
                    "device_id": Config.DEVICE_ID
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
        except websockets.exceptions.ConnectionClosed as e:
            # Normal closure (1000/1001) is expected, don't spam logs
            if e.code in (1000, 1001):
                # Clean disconnect, mark as disconnected
                self.connected = False
                return None
            else:
                # Abnormal closure, log it
                print(f"✗ Connection closed unexpectedly: {e.code} - {e.reason}")
                if self.logger:
                    self.logger.error(
                        "orchestrator_connection_closed_abnormally",
                        extra={
                            "code": e.code,
                            "reason": e.reason,
                            "user_id": Config.USER_ID
                        }
                    )
                self.connected = False
                return None
        except Exception as e:
            # Only log unexpected errors
            print(f"✗ Receive error: {e}")
            if self.logger:
                self.logger.error(
                    "orchestrator_receive_error",
                    extra={
                        "error": str(e),
                        "user_id": Config.USER_ID
                    }
                )
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
                extra={
                    "user_id": Config.USER_ID,
                    "device_id": Config.DEVICE_ID
                }
            )

