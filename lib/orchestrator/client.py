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
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 10
        self.reconnect_base_delay = 1.0  # Base delay for exponential backoff
        self.last_health_check = 0
        self.health_check_interval = 30  # Check connection health every 30 seconds
        
    def is_connection_alive(self):
        """Check if the WebSocket connection is actually alive"""
        if not self.connected:
            return False
        if not self.websocket:
            self.connected = False
            return False
        # Check if websocket is open and not closed
        try:
            # Check the state attribute which is more reliable than 'open'
            # State 1 = OPEN, State 2 = CLOSING, State 3 = CLOSED
            if hasattr(self.websocket, 'state'):
                from websockets.protocol import State
                if self.websocket.state != State.OPEN:
                    self.connected = False
                    return False
            # Fallback to 'open' attribute if state not available
            elif hasattr(self.websocket, 'open'):
                if not self.websocket.open:
                    self.connected = False
                    return False
            # If neither attribute exists, assume connection is alive
            # (will fail on actual send/recv if not)
        except (AttributeError, ImportError):
            # If we can't check the state, rely on connected flag
            # Actual send/recv operations will catch connection failures
            pass
        return True
    
    async def connect(self, is_reconnect=False):
        """Connect to conversation-orchestrator
        
        Args:
            is_reconnect: True if this is a reconnection attempt
        """
        logger = self.logger
        
        if is_reconnect:
            print(f"\n🔄 Reconnecting to conversation-orchestrator (attempt {self.reconnect_attempts + 1}/{self.max_reconnect_attempts})...")
        else:
            print(f"\n🔌 Connecting to conversation-orchestrator...")
            print(f"   URL: {Config.ORCHESTRATOR_URL}")
        
        try:
            # Create SSL context if using wss://
            ssl_context = None
            if Config.ORCHESTRATOR_URL.startswith("wss://"):
                ssl_context = ssl.create_default_context(cafile=certifi.where())
            
            # Connect to WebSocket
            self.websocket = await websockets.connect(
                Config.ORCHESTRATOR_URL,
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
                self.reconnect_attempts = 0  # Reset on successful connection
                self.last_health_check = 0  # Reset health check timer to allow grace period
                
                if is_reconnect:
                    print("✓ Reconnected to conversation-orchestrator")
                else:
                    print("✓ Connected to conversation-orchestrator")
                    
                if logger:
                    logger.info(
                        "conversation_orchestrator_connected",
                        extra={
                            "url": Config.ORCHESTRATOR_URL,
                            "user_id": Config.USER_ID,
                            "device_id": Config.DEVICE_ID,
                            "is_reconnect": is_reconnect
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
                            "user_id": Config.USER_ID,
                            "is_reconnect": is_reconnect
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
                        "user_id": Config.USER_ID,
                        "is_reconnect": is_reconnect
                    },
                    exc_info=True
                )
            return False
    
    async def reconnect(self):
        """Attempt to reconnect with exponential backoff"""
        if self.reconnect_attempts >= self.max_reconnect_attempts:
            print(f"✗ Max reconnection attempts ({self.max_reconnect_attempts}) reached")
            if self.logger:
                self.logger.error(
                    "max_reconnection_attempts_reached",
                    extra={
                        "attempts": self.reconnect_attempts,
                        "user_id": Config.USER_ID,
                        "device_id": Config.DEVICE_ID
                    }
                )
            return False
        
        # Calculate exponential backoff delay
        delay = self.reconnect_base_delay * (2 ** self.reconnect_attempts)
        delay = min(delay, 60)  # Cap at 60 seconds
        
        print(f"⏳ Waiting {delay:.1f}s before reconnection attempt...")
        if self.logger:
            self.logger.info(
                "reconnection_scheduled",
                extra={
                    "delay_seconds": delay,
                    "attempt": self.reconnect_attempts + 1,
                    "user_id": Config.USER_ID,
                    "device_id": Config.DEVICE_ID
                }
            )
        
        await asyncio.sleep(delay)
        self.reconnect_attempts += 1
        
        return await self.connect(is_reconnect=True)
    
    async def send_reactive(self):
        """Send reactive conversation request with trace context"""
        if not self.is_connection_alive():
            print("✗ Cannot send reactive request: connection not alive")
            if self.logger:
                self.logger.warning(
                    "send_reactive_failed_disconnected",
                    extra={
                        "user_id": Config.USER_ID,
                        "device_id": Config.DEVICE_ID
                    }
                )
            return False
        
        try:
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
            return True
        except Exception as e:
            print(f"✗ Failed to send reactive request: {e}")
            if self.logger:
                self.logger.error(
                    "send_reactive_error",
                    extra={
                        "error": str(e),
                        "user_id": Config.USER_ID,
                        "device_id": Config.DEVICE_ID
                    }
                )
            self.connected = False
            return False
    
    async def send_conversation_start(
        self, conversation_id: str, elevenlabs_conversation_id: str, agent_id: str
    ):
        """Send conversation start notification with trace context"""
        if not self.is_connection_alive():
            print("✗ Cannot send conversation_start: connection not alive")
            if self.logger:
                self.logger.warning(
                    "send_conversation_start_failed_disconnected",
                    extra={
                        "conversation_id": conversation_id,
                        "user_id": Config.USER_ID,
                        "device_id": Config.DEVICE_ID
                    }
                )
            return False
        
        try:
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
            return True
        except Exception as e:
            print(f"✗ Failed to send conversation_start: {e}")
            if self.logger:
                self.logger.error(
                    "send_conversation_start_error",
                    extra={
                        "error": str(e),
                        "conversation_id": conversation_id,
                        "user_id": Config.USER_ID,
                        "device_id": Config.DEVICE_ID
                    }
                )
            self.connected = False
            return False
    
    async def send_conversation_end(
        self, conversation_id: str, elevenlabs_conversation_id: str, 
        agent_id: str, end_reason: str
    ):
        """Send conversation end notification with trace context"""
        if not self.is_connection_alive():
            print("✗ Cannot send conversation_end: connection not alive")
            if self.logger:
                self.logger.warning(
                    "send_conversation_end_failed_disconnected",
                    extra={
                        "conversation_id": conversation_id,
                        "user_id": Config.USER_ID,
                        "device_id": Config.DEVICE_ID
                    }
                )
            return False
        
        try:
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
            return True
        except Exception as e:
            print(f"✗ Failed to send conversation_end: {e}")
            if self.logger:
                self.logger.error(
                    "send_conversation_end_error",
                    extra={
                        "error": str(e),
                        "conversation_id": conversation_id,
                        "user_id": Config.USER_ID,
                        "device_id": Config.DEVICE_ID
                    }
                )
            self.connected = False
            return False
    
    async def check_connection_health(self):
        """Periodically check if connection is still healthy (silent check)"""
        import time
        current_time = time.time()
        
        # Only check at intervals to avoid overhead
        if current_time - self.last_health_check < self.health_check_interval:
            return True
        
        self.last_health_check = current_time
        
        # Check if connection is alive (silently - no logging on success)
        if not self.is_connection_alive():
            # Only log when health check actually fails
            print("⚠️  Connection health check failed: connection not alive")
            if self.logger:
                self.logger.warning(
                    "connection_health_check_failed",
                    extra={
                        "user_id": Config.USER_ID,
                        "device_id": Config.DEVICE_ID
                    }
                )
            return False
        
        # Health check passed - no logging needed
        return True
    
    async def receive_message(self):
        """Receive and return a message from orchestrator"""
        if not self.is_connection_alive():
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
                print("ℹ️  Connection closed normally")
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
                            "user_id": Config.USER_ID,
                            "device_id": Config.DEVICE_ID
                        }
                    )
                self.connected = False
                return None
        except Exception as e:
            # Mark as disconnected and log error
            print(f"✗ Receive error: {e}")
            if self.logger:
                self.logger.error(
                    "orchestrator_receive_error",
                    extra={
                        "error": str(e),
                        "user_id": Config.USER_ID,
                        "device_id": Config.DEVICE_ID
                    }
                )
            self.connected = False
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

