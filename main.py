#!/usr/bin/env python3
"""
Kin AI Raspberry Pi Client (v2)
================================
Modular, refactored client with ALSA-only audio, full telemetry, and LED feedback.

Features:
- Wake word detection using Porcupine
- Real-time conversation via ElevenLabs WebSocket API
- ReSpeaker hardware AEC (with fallback to default devices)
- Communication with conversation-orchestrator via WebSocket
- Device authentication via provisioned Ed25519 credentials
- OpenTelemetry observability (traces, spans, logs - no metrics)
- LED visual feedback for device states
- WiFi setup mode for devices without internet connectivity

Usage:
    python main.py

Requirements:
    - Raspberry Pi OS with ALSA
    - Audio device (ReSpeaker recommended, fallback to any mic/speaker)
    - Environment variables: see Config class
"""

import os
import sys
import signal
import time
import asyncio
import logging

# Configure logging BEFORE any other imports (critical for telemetry)
# This ensures OTEL handler captures console output
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    force=True,
)

# Import local modules
from lib.config import Config
from lib.auth import authenticate
from lib.audio import get_audio_devices, LEDController
from lib.wake_word import WakeWordDetector
from lib.orchestrator import OrchestratorClient
from lib.elevenlabs import ElevenLabsConversationClient
from lib.local_storage import ContextManager

# Import WiFi setup module (optional - graceful degradation)
try:
    from lib.wifi_setup import WiFiSetupManager
    from lib.wifi_setup.connectivity import ConnectivityChecker
    WIFI_SETUP_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  WiFi setup module not available: {e}")
    WIFI_SETUP_AVAILABLE = False

# Import telemetry (optional - graceful degradation)
try:
    from lib.telemetry import (
        setup_telemetry,
        get_logger as get_otel_logger,
        add_span_event,
        create_conversation_trace,
        inject_trace_context,
        extract_trace_context,
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
# MAIN APPLICATION
# =============================================================================

class KinClient:
    """Main application controller with full telemetry and LED feedback"""
    
    def __init__(self):
        self.wake_detector = None
        self.context_manager = ContextManager()
        self.orchestrator_client = OrchestratorClient(context_manager=self.context_manager)
        self.led_controller = None
        self.running = True
        self.conversation_active = False
        self.awaiting_agent_details = False
        self.user_terminate = [False]  # Use list for mutable reference
        self.shutdown_requested = False
        self.conversation_start_time = None
        self.conversation_trace_context = None
        
        # Audio device indices (will be set after detection)
        self.mic_device_index = None
        self.speaker_device_index = None
        self.has_hardware_aec = False
        
        # Activity tracking for wrapper idle monitoring
        self.activity_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".last_activity")
        
        # Setup telemetry if available
        if TELEMETRY_AVAILABLE and Config.OTEL_ENABLED:
            try:
                setup_telemetry(
                    device_id=Config.DEVICE_ID,
                    endpoint=Config.OTEL_EXPORTER_ENDPOINT
                )
                
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
            Config.LOGGER = logging.getLogger(__name__)
            self.logger = Config.LOGGER
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._handle_interrupt_signal)
        signal.signal(signal.SIGTERM, self._handle_interrupt_signal)
    
    def _update_activity(self):
        """Update activity timestamp for wrapper idle monitoring"""
        try:
            # Touch the activity file to update its modification time
            os.utime(self.activity_file, None)
        except Exception:
            # If file doesn't exist or can't be touched, silently ignore
            pass
    
    def _handle_interrupt_signal(self, sig, frame):
        """Handle interrupt/termination signals for full shutdown (Ctrl+C, SIGTERM)"""
        signal_name = signal.Signals(sig).name if hasattr(signal, "Signals") else str(sig)
        print(f"\n🛑 Received {signal_name} - ", end="")
        
        if self.conversation_active:
            print("ending current conversation...")
            self.user_terminate[0] = True
        else:
            print("shutting down...")
            self.shutdown_requested = True
            self.running = False
        
        if self.logger:
            self.logger.info(
                "interrupt_signal_received",
                extra={
                    "signal": signal_name,
                    "conversation_active": self.conversation_active,
                    "user_id": Config.USER_ID,
                    "device_id": Config.DEVICE_ID
                }
            )
    
    def _resume_wake_word_detection(self):
        """Start wake word detector again if it is not already running."""
        if self.wake_detector and self.wake_detector.running:
            return
        
        if self.wake_detector:
            self.wake_detector.start()
            print(f"\n✓ Listening for '{Config.WAKE_WORD}' again...")
    
    async def run(self):
        """Main application loop"""
        print("\n" + "="*60)
        print("🎙️  Kin AI Raspberry Pi Client (v2)")
        print("="*60)
        
        # Validate configuration
        Config.validate()
        
        # Initialize LED controller early (before WiFi setup)
        # This allows WiFi setup to show LED feedback
        self.led_controller = LEDController(enabled=Config.LED_ENABLED)
        self.led_controller.set_state(LEDController.STATE_BOOT)
        
        # Check if WiFi setup should be skipped (default: no)
        skip_wifi_setup = os.getenv('SKIP_WIFI_SETUP', 'false').lower() == 'true'
        pairing_code = None  # Will be set by WiFi setup if needed
        
        # WiFi Setup Mode - only if enabled and available
        if not skip_wifi_setup and WIFI_SETUP_AVAILABLE:
            print("\n📡 Checking connectivity...")
            
            connectivity_checker = ConnectivityChecker()
            has_internet, orchestrator_reachable = await connectivity_checker.check_full_connectivity(
                orchestrator_retries=3
            )
            
            if not has_internet or not orchestrator_reachable:
                if not has_internet:
                    print("✗ No internet connection detected")
                else:
                    print("✗ Internet connected but orchestrator unreachable")
                
                # WiFi setup + pairing loop - retry if pairing fails
                max_setup_attempts = 3
                setup_attempt = 0
                authenticated = False
                
                while setup_attempt < max_setup_attempts and not authenticated:
                    setup_attempt += 1
                    print(f"\n🔧 Entering WiFi setup mode (attempt {setup_attempt}/{max_setup_attempts})...")
                    print("="*60)
                    
                    # Start WiFi setup manager with LED controller
                    wifi_manager = WiFiSetupManager(led_controller=self.led_controller)
                    pairing_code, success = await wifi_manager.start_setup_mode()
                    
                    if success and pairing_code:
                        print(f"\n✓ WiFi connected!")
                        print(f"  Pairing code received: {pairing_code}")
                        print("="*60)
                        
                        # Note: AP is now stopped, user's device has disconnected
                        # User won't see authentication status on web page
                        # If auth fails, we'll restart AP so they can see error message
                        
                        print("\n🔐 Authenticating with pairing code...")
                        
                        if authenticate(pairing_code=pairing_code):
                            authenticated = True
                            print("✓ Authentication and pairing successful!")
                            print("  Device is now paired and starting...")
                            
                            # Clean up (HTTP server already stopped during WiFi switch)
                            try:
                                await wifi_manager.http_server.stop()
                            except:
                                pass
                        else:
                            print("\n✗ Authentication or pairing failed")
                            if setup_attempt < max_setup_attempts:
                                print("  Possible reasons:")
                                print("    - Incorrect pairing code")
                                print("    - Pairing code expired or already used")
                                print("    - Device not registered in admin portal")
                                print("    - Backend service temporarily unavailable")
                                print(f"\n  Cleaning up and restarting setup mode...")
                                
                                # CRITICAL: Must delete WiFi connection, otherwise device gets stuck
                                # If WiFi stays connected, main loop won't re-enter setup mode
                                # But device isn't authenticated, so can't start normal operation
                                # Result: Deadlock!
                                if wifi_manager._wifi_credentials:
                                    failed_ssid = wifi_manager._wifi_credentials[0]
                                    print(f"  Deleting WiFi connection: {failed_ssid}")
                                    try:
                                        import subprocess
                                        subprocess.run(['sudo', 'nmcli', 'connection', 'delete', failed_ssid], 
                                                     capture_output=True, timeout=5)
                                    except Exception as e:
                                        print(f"  Warning: Could not delete connection: {e}")
                                
                                # Clear credentials from manager so setup starts fresh
                                wifi_manager._wifi_credentials = None
                                wifi_manager._pairing_code = None
                                
                                print(f"  Please reconnect to Kin_Setup and try again")
                                await asyncio.sleep(3)
                            else:
                                print(f"  Max attempts ({max_setup_attempts}) reached")
                                await asyncio.sleep(3)
                    else:
                        print("\n✗ WiFi setup failed")
                        # Clean up HTTP server if it's still running
                        try:
                            await wifi_manager.http_server.stop()
                        except:
                            pass
                        
                        if setup_attempt < max_setup_attempts:
                            print(f"  Retrying... (attempt {setup_attempt + 1}/{max_setup_attempts})")
                            await asyncio.sleep(3)
                
                if not authenticated:
                    print("\n✗ Setup and authentication failed after all attempts")
                    print("  Device will retry on next boot")
                    print("="*60)
                    return
            else:
                print("✓ Connectivity confirmed")
                
                # Authenticate without pairing code (device already paired)
                if not authenticate():
                    print("✗ Failed to authenticate device")
                    return
        else:
            # WiFi setup skipped, authenticate normally
            if not authenticate():
                print("✗ Failed to authenticate device")
                return
        
        # LED controller already initialized before WiFi setup
        # Keep boot state while initializing remaining components
        self.led_controller.set_state(LEDController.STATE_BOOT)
        
        # Initialize context manager with logger
        self.context_manager.set_logger(self.logger)
        
        # Start context manager (performs initial fetch with timeout)
        print("📍 Fetching location data...")
        await self.context_manager.start()
        
        if self.context_manager.has_location_data:
            print("✓ Location data loaded")
        else:
            print("⚠️  Location data unavailable - continuing without location")
        
        # Detect audio devices
        self.mic_device_index, self.speaker_device_index, self.has_hardware_aec = get_audio_devices()
        
        # Initialize wake word detector with detected microphone
        self.wake_detector = WakeWordDetector(
            mic_device_index=self.mic_device_index,
            orchestrator_client=self.orchestrator_client
        )
        
        # Connect to conversation-orchestrator
        connected = await self.orchestrator_client.connect()
        if not connected:
            self.led_controller.set_state(LEDController.STATE_ERROR)
            print("✗ Failed to connect to conversation-orchestrator")
            return
        
        # Record successful connection
        if TELEMETRY_AVAILABLE:
            add_span_event("orchestrator_connected", device_id=Config.DEVICE_ID)
        
        # Start wake word detection
        self.wake_detector.start()
        
        # System ready - show idle state (soft breathing, ready for wake word)
        self.led_controller.set_state(LEDController.STATE_IDLE)
        
        print("\n" + "="*60)
        print(f"✓ Ready! Say '{Config.WAKE_WORD}' to start a conversation")
        print("  Press Ctrl+C to exit")
        print("="*60 + "\n")
        
        # Main loop
        try:
            while self.running:
                # Check for shutdown request (from Ctrl+C or other signals)
                if self.shutdown_requested:
                    print("🛑 Shutdown requested, exiting main loop...")
                    break
                
                # Check if wake word was detected
                if self.wake_detector.detected and not self.conversation_active:
                    self.wake_detector.detected = False
                    
                    # Update activity - wake word detected
                    self._update_activity()
                    
                    # Verify connection before starting conversation
                    if not self.orchestrator_client.is_connection_alive():
                        print("✗ Cannot start conversation: not connected to orchestrator")
                        self.led_controller.set_state(LEDController.STATE_ERROR)
                        await asyncio.sleep(2)
                        self.led_controller.set_state(LEDController.STATE_IDLE)
                        self._resume_wake_word_detection()
                        continue
                    
                    # Show wake word detected state (color burst)
                    self.led_controller.set_state(LEDController.STATE_WAKE_WORD_DETECTED)
                    
                    # Stop wake word detection during conversation
                    self.wake_detector.stop()
                    
                    # Handle conversation
                    await self._handle_conversation()
                
                # Check for messages from orchestrator
                message = await self.orchestrator_client.receive_message()
                if message:
                    # Update activity - message received
                    self._update_activity()
                    await self._handle_orchestrator_message(message)
                elif not self.orchestrator_client.connected and not self.conversation_active:
                    # Connection lost (detected by receive_message setting connected=False)
                    # Attempt to reconnect only if not in conversation
                    print("⚠️  Connection lost, attempting to reconnect...")
                    self.led_controller.set_state(LEDController.STATE_ERROR)
                    
                    reconnected = await self.orchestrator_client.reconnect()
                    if reconnected:
                        self.led_controller.set_state(LEDController.STATE_IDLE)
                        # Refresh location/weather data on reconnection
                        asyncio.create_task(self.context_manager.force_refresh())
                    else:
                        # Failed to reconnect, wait and try again
                        await asyncio.sleep(5)
                        continue
                
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
            self.user_terminate[0] = False
            
            # Check if we have cached agent details (fast path)
            if Config.DEFAULT_REACTIVE_AGENT_ID and Config.DEFAULT_REACTIVE_WEB_SOCKET_URL:
                print("✓ Using cached default reactive agent (fast wake word response)")
                if self.logger:
                    self.logger.info(
                        "using_cached_agent_details",
                        extra={
                            "agent_id": Config.DEFAULT_REACTIVE_AGENT_ID,
                            "user_id": Config.USER_ID,
                            "device_id": Config.DEVICE_ID
                        }
                    )
                
                # Use cached agent details directly
                self.awaiting_agent_details = False
                
                # Start conversation with cached agent details
                await self._start_elevenlabs_conversation(
                    agent_id=Config.DEFAULT_REACTIVE_AGENT_ID,
                    web_socket_url=Config.DEFAULT_REACTIVE_WEB_SOCKET_URL,
                    trace_context=None  # Already in trace context
                )
            else:
                # No cached agent - use traditional flow (send reactive request and wait)
                print("⚠ No cached agent details - requesting from orchestrator")
                if self.logger:
                    self.logger.info(
                        "requesting_agent_details_from_orchestrator",
                        extra={
                            "user_id": Config.USER_ID,
                            "device_id": Config.DEVICE_ID
                        }
                    )
                
                self.awaiting_agent_details = True
                
                # Send reactive request (will inject trace context)
                send_success = await self.orchestrator_client.send_reactive()
                
                if not send_success:
                    print("✗ Failed to send reactive request")
                    if self.logger:
                        self.logger.error(
                            "reactive_request_send_failed",
                            extra={
                                "user_id": Config.USER_ID,
                                "device_id": Config.DEVICE_ID
                            }
                        )
                    self.conversation_active = False
                    self.awaiting_agent_details = False
                    self.led_controller.set_state(LEDController.STATE_ERROR)
                    await asyncio.sleep(2)  # Brief pause to show error
                    self._resume_wake_word_detection()
                    self.led_controller.set_state(LEDController.STATE_IDLE)
                    return
                
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
                    
                    # Check if connection died while waiting
                    if not self.orchestrator_client.is_connection_alive():
                        print("✗ Connection lost while waiting for agent details")
                        if self.logger:
                            self.logger.error(
                                "connection_lost_during_agent_wait",
                                extra={
                                    "user_id": Config.USER_ID,
                                    "device_id": Config.DEVICE_ID
                                }
                            )
                        break
                    
                    # Small sleep to prevent busy loop
                    await asyncio.sleep(0.1)
                
                if self.awaiting_agent_details:
                    print("✗ Timeout waiting for agent details")
                    if self.logger:
                        self.logger.error(
                            "agent_details_timeout",
                            extra={
                                "user_id": Config.USER_ID,
                                "device_id": Config.DEVICE_ID
                            }
                        )
                    self.conversation_active = False
                    self.awaiting_agent_details = False
                    self.led_controller.set_state(LEDController.STATE_ERROR)
                    await asyncio.sleep(2)  # Brief pause to show error
                    self._resume_wake_word_detection()
                    self.led_controller.set_state(LEDController.STATE_IDLE)
        
        # Execute with trace context if available
        if self.conversation_trace_context:
            with self.conversation_trace_context:
                await _handle_with_trace()
        else:
            await _handle_with_trace()
    
    async def _start_elevenlabs_conversation(self, agent_id: str, web_socket_url: str, trace_context=None):
        """
        Start an ElevenLabs conversation with the given agent details.
        
        Args:
            agent_id: Agent ID
            web_socket_url: ElevenLabs WebSocket URL
            trace_context: Optional trace context token for proactive conversations
        """
        async def _execute_conversation():
            # Mark conversation as active
            self.conversation_active = True
            self.user_terminate[0] = False
            self.conversation_start_time = time.time()
            
            # Update activity - conversation starting
            self._update_activity()
            
            # Stop wake word detection during conversation
            self.wake_detector.stop()
            
            # Show conversation state (pulsating green - active conversation)
            self.led_controller.set_state(LEDController.STATE_CONVERSATION)
            
            # Start ElevenLabs conversation
            client = ElevenLabsConversationClient(
                web_socket_url, 
                agent_id,
                mic_device_index=self.mic_device_index,
                speaker_device_index=self.speaker_device_index,
                user_terminate_flag=self.user_terminate,
                led_controller=self.led_controller  # Pass LED controller for audio-reactive feedback
            )
            await client.start(self.orchestrator_client)
        
            # Update activity - conversation ended
            self._update_activity()
            
            # Check if user terminated
            if self.user_terminate[0]:
                print("✓ User terminated conversation")
            
            # Resume wake word detection
            self._resume_wake_word_detection()
            
            # Back to idle state (soft breathing, ready for wake word)
            self.led_controller.set_state(LEDController.STATE_IDLE)
            
            if self.logger:
                self.logger.info(
                    "wake_word_detection_resumed",
                    extra={
                        "user_id": Config.USER_ID,
                        "device_id": Config.DEVICE_ID
                    }
                )
            
            self.conversation_active = False
            self.user_terminate[0] = False
            self.conversation_start_time = None
        
        # Execute with trace context if we have one (proactive)
        if trace_context is not None:
            try:
                await _execute_conversation()
            finally:
                # Detach the context
                from opentelemetry import context
                context.detach(trace_context)
        else:
            # For reactive conversations, we're already in the trace context from _handle_conversation
            await _execute_conversation()
    
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
                    self.logger.error(
                        "invalid_agent_details",
                        extra={
                            "message": message,
                            "user_id": Config.USER_ID
                        }
                    )
                return
            
            print(f"✓ Received agent details: {agent_id}")
            if self.logger:
                self.logger.info(
                    "agent_details_received",
                    extra={
                        "agent_id": agent_id,
                        "message_type": message_type,
                        "user_id": Config.USER_ID,
                        "device_id": Config.DEVICE_ID
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
                            "user_id": Config.USER_ID,
                            "device_id": Config.DEVICE_ID
                        }
                    )
            
            # Start the conversation using the helper method
            await self._start_elevenlabs_conversation(
                agent_id=agent_id,
                web_socket_url=web_socket_url,
                trace_context=context_token
            )
            
        elif message_type == "error":
            error_msg = message.get("message", "Unknown error")
            print(f"✗ Orchestrator error: {error_msg}")
            
            if self.logger:
                self.logger.error(
                    "orchestrator_error_received",
                    extra={
                        "error_message": error_msg,
                        "user_id": Config.USER_ID,
                        "device_id": Config.DEVICE_ID
                    }
                )
            
            self.led_controller.set_state(LEDController.STATE_ERROR)
            await asyncio.sleep(2)  # Brief pause to show error
            
            self.conversation_active = False
            self.awaiting_agent_details = False
            
            # If we were waiting on a conversation that failed, resume wake word detection
            self._resume_wake_word_detection()
            self.led_controller.set_state(LEDController.STATE_IDLE)
    
    async def cleanup(self):
        """Clean up resources gracefully"""
        print("\n🧹 Cleaning up...")
        self.running = False
        
        # Stop context manager
        if self.context_manager:
            await self.context_manager.stop()
        
        # Stop LED animations and turn off
        if self.led_controller:
            self.led_controller.set_state(LEDController.STATE_OFF)
            self.led_controller.cleanup()
        
        # Stop wake word detector
        if self.wake_detector:
            self.wake_detector.cleanup()
        
        # Disconnect from orchestrator
        await self.orchestrator_client.disconnect()
        
        # Record disconnection
        if TELEMETRY_AVAILABLE:
            add_span_event("orchestrator_disconnected", device_id=Config.DEVICE_ID)
        
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
