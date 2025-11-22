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
- Supabase authentication on startup
- OpenTelemetry observability (traces, spans, logs - no metrics)
- LED visual feedback for device states

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
from lib.auth import authenticate_with_supabase
from lib.audio import get_audio_devices, verify_audio_setup, LEDController
from lib.wake_word import WakeWordDetector
from lib.orchestrator import OrchestratorClient
from lib.elevenlabs import ElevenLabsConversationClient

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
        self.orchestrator_client = OrchestratorClient()
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
        signal.signal(signal.SIGUSR1, self._handle_terminate_signal)
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
    
    def _handle_terminate_signal(self, sig, frame):
        """Handle user-initiated conversation termination signal (SIGUSR1)"""
        print("\n🛑 User termination signal received (conversation only)")
        if self.logger:
            self.logger.info(
                "terminate_signal_received",
                extra={
                    "signal": "SIGUSR1",
                    "user_id": Config.USER_ID,
                    "device_id": Config.DEVICE_ID
                }
            )
        self.user_terminate[0] = True
    
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
        
        # Initialize LED controller and show boot state
        self.led_controller = LEDController(enabled=Config.LED_ENABLED)
        self.led_controller.set_state(LEDController.STATE_BOOT)
        
        # Validate configuration
        Config.validate()
        
        # Authenticate with Supabase
        if not authenticate_with_supabase():
            self.led_controller.set_state(LEDController.STATE_ERROR)
            print("✗ Failed to authenticate with Supabase")
            return
        
        # Verify and detect audio devices
        verify_audio_setup()
        self.mic_device_index, self.speaker_device_index, self.has_hardware_aec = get_audio_devices()
        
        # Initialize wake word detector with detected microphone
        self.wake_detector = WakeWordDetector(mic_device_index=self.mic_device_index)
        
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
        print("  Send SIGUSR1 to end conversation only")
        print("="*60 + "\n")
        
        # Main loop
        try:
            while self.running:
                # Check for shutdown request (from Ctrl+C or other signals)
                if self.shutdown_requested:
                    print("🛑 Shutdown requested, exiting main loop...")
                    break
                
                # Periodic connection health check
                if not await self.orchestrator_client.check_connection_health():
                    # Connection is unhealthy, attempt to reconnect
                    print("⚠️  Connection unhealthy, attempting to reconnect...")
                    self.led_controller.set_state(LEDController.STATE_ERROR)
                    
                    # Attempt reconnection
                    reconnected = await self.orchestrator_client.reconnect()
                    if reconnected:
                        self.led_controller.set_state(LEDController.STATE_IDLE)
                    else:
                        # Failed to reconnect, continue loop and try again later
                        await asyncio.sleep(5)
                        continue
                
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
                elif not self.orchestrator_client.is_connection_alive() and not self.conversation_active:
                    # Connection lost, attempt to reconnect (only if not in conversation)
                    print("⚠️  Connection lost, attempting to reconnect...")
                    self.led_controller.set_state(LEDController.STATE_ERROR)
                    
                    reconnected = await self.orchestrator_client.reconnect()
                    if reconnected:
                        self.led_controller.set_state(LEDController.STATE_IDLE)
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
            self.awaiting_agent_details = True
            self.user_terminate[0] = False
            
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
            
            # Execute conversation handling with proper trace context
            async def _handle_conversation_with_context():
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
                    user_terminate_flag=self.user_terminate
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

