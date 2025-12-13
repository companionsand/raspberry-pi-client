"""Conversation orchestrator WebSocket client with telemetry"""

import json
import asyncio
import ssl
from typing import Optional
import websockets
import certifi
import requests
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
    
    def __init__(self, context_manager=None):
        self.websocket = None
        self.connected = False
        self.running = False
        self.logger = Config.LOGGER
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 10
        self.reconnect_base_delay = 1.0  # Base delay for exponential backoff
        self.context_manager = context_manager
        self.token_refresh_task = None  # Background task handle
        self.current_wake_word_detection_id = None  # NEW: Track wake word detection ID
        
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
    
    async def connect(self, is_reconnect=False) -> tuple:
        """Connect to conversation-orchestrator
        
        Args:
            is_reconnect: True if this is a reconnection attempt
            
        Returns:
            (success, auth_failed) tuple where:
            - success: True if connected successfully
            - auth_failed: True if failure was due to authentication (token invalid/expired)
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
            # Then send authentication with device metadata
            auth_message = {
                "type": "auth",
                "token": Config.AUTH_TOKEN,
                "device_id": Config.DEVICE_ID,
                "user_id": Config.USER_ID,
            }
            
            # Include location if available from context manager
            if self.context_manager and self.context_manager.has_location_data:
                location_string = self.context_manager.get_location_string()
                if location_string:
                    auth_message['current_location'] = location_string
            
            await self.websocket.send(json.dumps(auth_message))
            
            # Wait for connection confirmation
            response = await self.websocket.recv()
            data = json.loads(response)
            
            if data.get("type") == "connected":
                self.connected = True
                self.running = True
                self.reconnect_attempts = 0  # Reset on successful connection
                
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
                return (True, False)  # Success, no auth failure
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
                return (False, False)  # Failed, but not necessarily auth
                
        except websockets.exceptions.ConnectionClosed as e:
            # Check if this is an auth failure (1008 policy violation)
            is_auth_failure = e.code == 1008 and (
                "Invalid token" in str(e.reason) or 
                "policy violation" in str(e.reason)
            )
            
            if is_auth_failure:
                print(f"✗ Authentication failed: {e.reason}")
            else:
                print(f"✗ Connection closed: {e.code} - {e.reason}")
                
            if logger:
                logger.error(
                    "conversation_orchestrator_connection_closed",
                    extra={
                        "error": str(e),
                        "code": e.code,
                        "reason": e.reason,
                        "is_auth_failure": is_auth_failure,
                        "user_id": Config.USER_ID,
                        "is_reconnect": is_reconnect
                    },
                    exc_info=True
                )
            return (False, is_auth_failure)
            
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
            return (False, False)  # Failed, but not necessarily auth
    
    async def reconnect(self):
        """Attempt to reconnect with exponential backoff and token refresh on auth failures"""
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
        
        # Try to connect
        success, auth_failed = await self.connect(is_reconnect=True)
        
        if success:
            return True
        
        # If auth failed, try to refresh token (REACTIVE FALLBACK)
        if auth_failed:
            print("🔄 Token invalid/expired, refreshing authentication...")
            if self.logger:
                self.logger.info(
                    "reactive_token_refresh_triggered",
                    extra={
                        "trigger": "auth_failure_during_reconnect",
                        "user_id": Config.USER_ID,
                        "device_id": Config.DEVICE_ID
                    }
                )
            
            # Attempt to refresh token
            refresh_success = await self.refresh_auth_token()
            
            if refresh_success:
                print("✓ Token refreshed, retrying connection...")
                # Reset reconnection attempts - fresh start with new token
                self.reconnect_attempts = 0
                # Try connecting again immediately with new token
                success, _ = await self.connect(is_reconnect=True)
                return success
            else:
                print("✗ Failed to refresh token")
                if self.logger:
                    self.logger.error(
                        "reactive_token_refresh_failed",
                        extra={
                            "user_id": Config.USER_ID,
                            "device_id": Config.DEVICE_ID
                        }
                    )
                return False
        
        # Network or other error - will retry on next attempt
        return False
    
    async def refresh_auth_token(self) -> bool:
        """
        Refresh the authentication token proactively.
        
        Returns:
            True if token refreshed successfully, False otherwise
        """
        import time
        from lib.auth import authenticate
        
        logger = self.logger
        
        if logger:
            logger.info(
                "token_refresh_starting",
                extra={
                    "user_id": Config.USER_ID,
                    "device_id": Config.DEVICE_ID
                }
            )
        
        print("🔄 Refreshing authentication token...")
        
        try:
            # Re-authenticate to get fresh token
            auth_result = authenticate()
            
            if auth_result and auth_result.get("success"):
                print("✓ Token refreshed successfully")
                
                if logger:
                    logger.info(
                        "token_refreshed",
                        extra={
                            "user_id": Config.USER_ID,
                            "device_id": Config.DEVICE_ID,
                            "new_expiry": Config.AUTH_TOKEN_EXPIRES_AT
                        }
                    )
                return True
            else:
                print(f"✗ Token refresh failed: {auth_result.get('reason') if auth_result else 'unknown'}")
                
                if logger:
                    logger.error(
                        "token_refresh_failed",
                        extra={
                            "reason": auth_result.get("reason") if auth_result else "unknown",
                            "user_id": Config.USER_ID,
                            "device_id": Config.DEVICE_ID
                        }
                    )
                return False
                
        except Exception as e:
            print(f"✗ Token refresh error: {e}")
            
            if logger:
                logger.error(
                    "token_refresh_error",
                    extra={
                        "error": str(e),
                        "user_id": Config.USER_ID,
                        "device_id": Config.DEVICE_ID
                    },
                    exc_info=True
                )
            return False
    
    async def start_token_refresh_monitor(self):
        """
        Background task to proactively refresh token before expiry.
        
        Monitors token expiry and refreshes at 75% of lifetime.
        Runs continuously until client is stopped.
        """
        logger = self.logger
        
        if logger:
            logger.info(
                "token_refresh_monitor_started",
                extra={
                    "user_id": Config.USER_ID,
                    "device_id": Config.DEVICE_ID
                }
            )
        
        # Check every 30 seconds
        check_interval = 30
        
        # Refresh when 25% of lifetime remains (minimum 10 minutes buffer)
        min_buffer_seconds = 600  # 10 minutes minimum
        
        while self.running:
            try:
                await asyncio.sleep(check_interval)
                
                # Check if we have a token and expiry time
                if not Config.AUTH_TOKEN or not Config.AUTH_TOKEN_EXPIRES_AT:
                    continue
                
                # Calculate time until expiry
                import time
                time_until_expiry = Config.AUTH_TOKEN_EXPIRES_AT - time.time()
                
                # Calculate 25% of original token lifetime for refresh buffer
                # Assuming standard 1-hour token, this would be 15 minutes
                # But we'll use at least 10 minutes as minimum buffer
                refresh_buffer = max(min_buffer_seconds, 900)  # 15 minutes or 10 minutes minimum
                
                # If token expires soon, refresh it
                if time_until_expiry < refresh_buffer:
                    if logger:
                        logger.info(
                            "token_expiry_approaching",
                            extra={
                                "seconds_until_expiry": int(time_until_expiry),
                                "user_id": Config.USER_ID,
                                "device_id": Config.DEVICE_ID
                            }
                        )
                    
                    print(f"⏰ Token expires in {int(time_until_expiry/60)} minutes, refreshing...")
                    
                    # Refresh the token
                    success = await self.refresh_auth_token()
                    
                    if not success:
                        # If refresh fails, we'll try again on next check
                        if logger:
                            logger.warning(
                                "token_refresh_failed_will_retry",
                                extra={
                                    "retry_in_seconds": check_interval,
                                    "seconds_until_expiry": int(time_until_expiry),
                                    "user_id": Config.USER_ID,
                                    "device_id": Config.DEVICE_ID
                                }
                            )
                
            except Exception as e:
                if logger:
                    logger.error(
                        "token_refresh_monitor_error",
                        extra={
                            "error": str(e),
                            "user_id": Config.USER_ID,
                            "device_id": Config.DEVICE_ID
                        },
                        exc_info=True
                    )
                # Continue monitoring even if there's an error
                await asyncio.sleep(check_interval)
        
        if logger:
            logger.info(
                "token_refresh_monitor_stopped",
                extra={
                    "user_id": Config.USER_ID,
                    "device_id": Config.DEVICE_ID
                }
            )
    
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
                "wake_word_detection_id": self.current_wake_word_detection_id,  # NEW
            }
            
            # Clear the detection_id after sending
            self.current_wake_word_detection_id = None  # NEW
            
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
    
    async def send_wake_word_detection(
        self,
        wake_word: str,
        wake_word_detector_result: bool,
        asr_result: Optional[bool],
        audio_data: bytes,
        timestamp: str,
        detected_at: str,  # NEW: ISO timestamp from device
        asr_error: Optional[str] = None,
        transcript: Optional[str] = None,
        confidence_score: Optional[float] = None,
        audio_duration_ms: Optional[int] = None,
        retry_attempts: int = 3
    ) -> bool:
        """Send wake word detection data with audio to orchestrator (async, non-blocking).
        
        Args:
            wake_word: Expected wake word
            wake_word_detector_result: Picovoice result (true/false)
            asr_result: Scribe v2 result (true/false/null)
            audio_data: Raw audio bytes (WAV format)
            timestamp: Timestamp in YYYYMMDDHHmmss format
            detected_at: ISO timestamp when wake word was detected
            asr_error: Error message if Scribe failed
            transcript: Actual transcript from Scribe
            confidence_score: Confidence score (optional)
            audio_duration_ms: Audio duration in milliseconds
            retry_attempts: Number of retry attempts
            
        Returns:
            True if sent successfully, False otherwise
        """
        if not self.is_connection_alive():
            if self.logger:
                self.logger.warning(
                    "send_wake_word_detection_failed_disconnected",
                    extra={
                        "user_id": Config.USER_ID,
                        "device_id": Config.DEVICE_ID
                    }
                )
            return False
        
        import base64
        import time
        
        # Encode audio to base64
        audio_base64 = base64.b64encode(audio_data).decode('utf-8')
        
        message = {
            "type": "wake_word_detection",
            "wake_word": wake_word,
            "wake_word_detector_result": wake_word_detector_result,
            "asr_result": asr_result,
            "audio_data": audio_base64,
            "timestamp": timestamp,
            "detected_at": detected_at,  # NEW
            "asr_error": asr_error,
            "transcript": transcript,
            "confidence_score": confidence_score,
            "audio_duration_ms": audio_duration_ms,
        }
        
        # Retry with exponential backoff
        for attempt in range(retry_attempts):
            try:
                await self.websocket.send(json.dumps(message))
                
                if self.logger:
                    self.logger.info(
                        "wake_word_detection_sent",
                        extra={
                            "wake_word": wake_word,
                            "detector_result": wake_word_detector_result,
                            "asr_result": asr_result,
                            "audio_size_bytes": len(audio_data),
                            "attempt": attempt + 1,
                            "user_id": Config.USER_ID,
                            "device_id": Config.DEVICE_ID
                        }
                    )
                
                # Wait for response with detection_id (with timeout)
                # Loop and retry receive_message() until we get the response or timeout
                start_time = time.time()
                timeout_secs = 5.0
                response_received = False
                
                while time.time() - start_time < timeout_secs:
                    try:
                        response = await self.receive_message()
                        if response and response.get("type") == "wake_word_detection_created":
                            self.current_wake_word_detection_id = response.get("wake_word_detection_id")
                            if self.logger:
                                self.logger.info(
                                    "wake_word_detection_id_received",
                                    extra={
                                        "detection_id": self.current_wake_word_detection_id,
                                        "user_id": Config.USER_ID,
                                        "device_id": Config.DEVICE_ID
                                    }
                                )
                            response_received = True
                            break
                        # If we got a different message type, keep waiting
                        await asyncio.sleep(0.05)  # Small delay before retry
                    except Exception as e:
                        if self.logger:
                            self.logger.debug(
                                "wake_word_detection_id_receive_error",
                                extra={
                                    "error": str(e),
                                    "user_id": Config.USER_ID,
                                    "device_id": Config.DEVICE_ID
                                }
                            )
                        await asyncio.sleep(0.05)
                
                if not response_received and self.logger:
                    self.logger.warning(
                        "wake_word_detection_id_timeout",
                        extra={
                            "user_id": Config.USER_ID,
                            "device_id": Config.DEVICE_ID
                        }
                    )
                
                return True
                
            except Exception as e:
                if self.logger:
                    self.logger.warning(
                        "send_wake_word_detection_failed",
                        extra={
                            "error": str(e),
                            "attempt": attempt + 1,
                            "user_id": Config.USER_ID,
                            "device_id": Config.DEVICE_ID
                        }
                    )
                
                if attempt < retry_attempts - 1:
                    # Exponential backoff: 0.5s, 1s, 2s
                    await asyncio.sleep(0.5 * (2 ** attempt))
                    continue
                
                # All retries failed
                self.connected = False
                if self.logger:
                    self.logger.error(
                        "send_wake_word_detection_exhausted_retries",
                        extra={
                            "attempts": retry_attempts,
                            "user_id": Config.USER_ID,
                            "device_id": Config.DEVICE_ID
                        }
                    )
                return False
        
        return False
    
    def send_wake_word_detection_sync(
        self,
        wake_word: str,
        wake_word_detector_result: bool,
        asr_result: Optional[bool],
        audio_data: bytes,
        timestamp: str,
        detected_at: str,
        asr_error: Optional[str] = None,
        transcript: Optional[str] = None,
        confidence_score: Optional[float] = None,
        audio_duration_ms: Optional[int] = None,
        retry_attempts: int = 3
    ) -> Optional[str]:
        """Send wake word detection data via HTTP POST (synchronous, thread-safe).
        
        This is the "nuclear option" - completely synchronous HTTP request that can be
        called from any thread without asyncio complexity.
        
        Args:
            wake_word: Expected wake word
            wake_word_detector_result: Picovoice result (true/false)
            asr_result: Scribe v2 result (true/false/null)
            audio_data: Raw audio bytes (WAV format)
            timestamp: Timestamp in YYYYMMDDHHmmss format
            detected_at: ISO timestamp when wake word was detected
            asr_error: Error message if Scribe failed
            transcript: Actual transcript from Scribe
            confidence_score: Confidence score (optional)
            audio_duration_ms: Audio duration in milliseconds
            retry_attempts: Number of retry attempts
            
        Returns:
            wake_word_detection_id if successful, None otherwise
        """
        import base64
        import time
        
        # Build HTTP endpoint URL from WebSocket URL
        # ws://host:port/ws -> http://host:port/wake-word-detection
        # wss://host:port/ws -> https://host:port/wake-word-detection
        ws_url = Config.ORCHESTRATOR_URL
        if ws_url.startswith("ws://"):
            http_url = ws_url.replace("ws://", "http://").replace("/ws", "/wake-word-detection")
        elif ws_url.startswith("wss://"):
            http_url = ws_url.replace("wss://", "https://").replace("/ws", "/wake-word-detection")
        else:
            print(f"✗ Invalid WebSocket URL format: {ws_url}")
            return None
        
        # Encode audio to base64
        audio_base64 = base64.b64encode(audio_data).decode('utf-8')
        
        payload = {
            "wake_word": wake_word,
            "wake_word_detector_result": wake_word_detector_result,
            "asr_result": asr_result,
            "audio_data": audio_base64,
            "timestamp": timestamp,
            "detected_at": detected_at,
            "asr_error": asr_error,
            "transcript": transcript,
            "confidence_score": confidence_score,
            "audio_duration_ms": audio_duration_ms,
        }
        
        headers = {
            "Authorization": f"Bearer {Config.AUTH_TOKEN}",
            "Content-Type": "application/json"
        }
        
        # Retry with exponential backoff
        for attempt in range(retry_attempts):
            try:
                response = requests.post(
                    http_url,
                    json=payload,
                    headers=headers,
                    timeout=10.0  # 10 second timeout
                )
                
                if response.status_code == 200:
                    result = response.json()
                    detection_id = result.get("wake_word_detection_id")
                    
                    # Store the ID for conversation_start linking
                    self.current_wake_word_detection_id = detection_id
                    
                    if self.logger:
                        self.logger.info(
                            "wake_word_detection_sent_http",
                            extra={
                                "wake_word": wake_word,
                                "detector_result": wake_word_detector_result,
                                "asr_result": asr_result,
                                "audio_size_bytes": len(audio_data),
                                "detection_id": detection_id,
                                "attempt": attempt + 1,
                                "user_id": Config.USER_ID,
                                "device_id": Config.DEVICE_ID
                            }
                        )
                    
                    return detection_id
                else:
                    error_msg = f"HTTP {response.status_code}: {response.text}"
                    if self.logger:
                        self.logger.warning(
                            "wake_word_detection_http_error",
                            extra={
                                "status_code": response.status_code,
                                "error": error_msg,
                                "attempt": attempt + 1,
                                "user_id": Config.USER_ID,
                                "device_id": Config.DEVICE_ID
                            }
                        )
                    
                    if attempt < retry_attempts - 1:
                        time.sleep(0.5 * (2 ** attempt))  # Exponential backoff
                    
            except requests.exceptions.Timeout:
                if self.logger:
                    self.logger.warning(
                        "wake_word_detection_http_timeout",
                        extra={
                            "attempt": attempt + 1,
                            "user_id": Config.USER_ID,
                            "device_id": Config.DEVICE_ID
                        }
                    )
                if attempt < retry_attempts - 1:
                    time.sleep(0.5 * (2 ** attempt))
                    
            except Exception as e:
                if self.logger:
                    self.logger.warning(
                        "wake_word_detection_http_failed",
                        extra={
                            "error": str(e),
                            "error_type": type(e).__name__,
                            "attempt": attempt + 1,
                            "user_id": Config.USER_ID,
                            "device_id": Config.DEVICE_ID
                        }
                    )
                if attempt < retry_attempts - 1:
                    time.sleep(0.5 * (2 ** attempt))
        
        # All retries failed
        if self.logger:
            self.logger.error(
                "wake_word_detection_http_all_retries_failed",
                extra={
                    "user_id": Config.USER_ID,
                    "device_id": Config.DEVICE_ID
                }
            )
        return None
    
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
