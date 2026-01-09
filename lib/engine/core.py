"""
KinEngine - Core engine for the Kin AI Raspberry Pi client.

This module provides the main engine that orchestrates:
- Audio capture and playback via AudioManager
- Wake word detection via WakeWordDetector  
- Human presence detection via HumanPresenceDetector
- Conversations via ElevenLabsConversationClient
- Signal publishing for visualization and logging

The engine can be run from CLI (main.py) or GUI (gui.py) mode.
"""

import asyncio
import logging
import os
import signal
import time
from datetime import datetime
from typing import Optional

from lib.config import Config
from lib.signals import SignalBus, TextSignal, ScalarSignal

# Import audio components
from lib.audio import AudioManager, get_audio_devices, LEDController, VoiceFeedback

# Import detection components
from lib.detection import WakeWordDetector, HumanPresenceDetector

# Import agent components
from lib.agent import OrchestratorClient, ElevenLabsConversationClient, ContextManager

# Import music mode
from lib.music import MusicModeController

from lib.setup import run_startup_sequence


# Import telemetry (optional - graceful degradation)
try:
    from lib.telemetry import (
        setup_telemetry,
        get_logger as get_otel_logger,
        add_span_event,
        create_conversation_trace,
        extract_trace_context,
        setup_stdout_redirect,
        cleanup_stdout_redirect,
    )
    TELEMETRY_AVAILABLE = True
except ImportError:
    TELEMETRY_AVAILABLE = False
    # Provide no-op fallbacks
    def get_otel_logger(name, device_id=None):
        return logging.getLogger(name)
    def setup_stdout_redirect(name=None):
        return False
    def cleanup_stdout_redirect():
        pass
    def add_span_event(*args, **kwargs):
        pass
    def create_conversation_trace(*args, **kwargs):
        return None
    def extract_trace_context(*args, **kwargs):
        return None


class KinEngine:
    """
    Core engine for the Kin AI client.
    
    Orchestrates all components and exposes a signal-based interface
    for consumers (CLI, GUI) to observe and interact with the system.
    
    Usage:
        engine = KinEngine()
        
        # Subscribe to signals
        engine.signal_bus.subscribe(TextSignal, my_handler)
        
        # Run the engine
        await engine.run()
    """
    
    def __init__(self):
        """Initialize the engine."""
        # Signal bus for pub/sub communication
        self.signal_bus = SignalBus()
        
        # Components (initialized during run())
        self.audio_manager: Optional[AudioManager] = None
        self.wake_detector: Optional[WakeWordDetector] = None
        self.presence_detector: Optional[HumanPresenceDetector] = None
        self.context_manager = ContextManager()
        self.orchestrator_client = OrchestratorClient(context_manager=self.context_manager)
        self.led_controller: Optional[LEDController] = None
        self.voice_feedback: Optional[VoiceFeedback] = None
        
        # State
        self.running = True
        self.conversation_active = False
        self.awaiting_agent_details = False
        self.user_terminate = [False]  # Mutable reference for signal handlers
        self.shutdown_requested = False
        self.conversation_start_time: Optional[float] = None
        self.conversation_trace_context = None
        
        # Audio device info
        self.mic_device_index: Optional[int] = None
        self.speaker_device_index: Optional[int] = None
        self.has_hardware_aec = False
        
        # Activity tracking
        self.activity_file = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            ".last_activity"
        )
        
        # Logger (set during initialization)
        self.logger: Optional[logging.Logger] = None
        
        # Background tasks
        self._background_tasks = []
    
    def _publish_text(self, category: str, message: str, level: str = "info") -> None:
        """Convenience method to publish a TextSignal."""
        self.signal_bus.publish(TextSignal(
            timestamp=time.monotonic(),
            source="engine",
            category=category,
            message=message,
            level=level
        ))
    
    def _publish_scalar(self, name: str, value: float) -> None:
        """Convenience method to publish a ScalarSignal."""
        self.signal_bus.publish(ScalarSignal(
            timestamp=time.monotonic(),
            source="engine",
            name=name,
            value=value
        ))
    
    def inject_signal(self, signal_obj) -> bool:
        """
        Inject a signal into the bus (for external triggering).
        
        Useful for GUI buttons that simulate events (e.g., wake word).
        
        Args:
            signal_obj: Signal to inject
            
        Returns:
            True if published successfully
        """
        return self.signal_bus.publish(signal_obj)
    
    def inject_wake_word(self) -> None:
        """Inject a simulated wake word detection."""
        self._publish_text("wake_word", f"Simulated wake word '{Config.WAKE_WORD}'")
        if self.wake_detector:
            self.wake_detector.detected = True
    
    def get_audio_window(self, stream: str, seconds: float):
        """
        Get recent audio from a stream.
        
        Args:
            stream: Stream name ("aec_input", "agent_output", "raw_input")
            seconds: Duration to retrieve
            
        Returns:
            NumPy array of audio samples
        """
        if self.audio_manager:
            return self.audio_manager.get_audio_window(stream, seconds)
        import numpy as np
        return np.array([], dtype=np.int16)
    
    def get_stream_last_write_time(self, stream: str) -> float:
        """
        Get when data was last written to a stream.
        
        Args:
            stream: Stream name
            
        Returns:
            Monotonic timestamp of last write, or 0.0 if none
        """
        if self.audio_manager:
            return self.audio_manager.get_stream_last_write_time(stream)
        return 0.0
    
    def _setup_signal_handlers(self) -> None:
        """Setup OS signal handlers (only works from main thread)."""
        import threading
        if threading.current_thread() is not threading.main_thread():
            # Can't set signal handlers from non-main thread
            self._publish_text("system", "Skipping signal handlers (not main thread)", "debug")
            return
        
        signal.signal(signal.SIGINT, self._handle_interrupt_signal)
        signal.signal(signal.SIGTERM, self._handle_interrupt_signal)
    
    def _handle_interrupt_signal(self, sig, frame):
        """Handle interrupt/termination signals."""
        signal_name = signal.Signals(sig).name if hasattr(signal, "Signals") else str(sig)
        
        if self.conversation_active:
            self._publish_text("system", f"Received {signal_name} - ending conversation...", "warning")
            self.user_terminate[0] = True
        else:
            self._publish_text("system", f"Received {signal_name} - shutting down...", "warning")
            self.shutdown_requested = True
            self.running = False
            self.user_terminate[0] = True
        
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
    
    def _update_activity(self) -> None:
        """Update activity timestamp for wrapper idle monitoring."""
        try:
            os.utime(self.activity_file, None)
        except Exception:
            pass
    
    def _resume_wake_word_detection(self) -> None:
        """Resume wake word detection after conversation ends."""
        if self.wake_detector and self.wake_detector.running:
            return
        
        if self.wake_detector:
            self.wake_detector.start()
            self._publish_text("system", f"Listening for '{Config.WAKE_WORD}' again...")
        
        if self.presence_detector and not self.presence_detector.running:
            self.presence_detector.start()
    
    async def _initialize_telemetry(self) -> None:
        """Initialize telemetry/logging."""
        if Config.MAC_MODE:
            Config.LOGGER = None
            self.logger = None
            return
        
        if TELEMETRY_AVAILABLE and Config.OTEL_ENABLED:
            try:
                setup_telemetry(
                    device_id=Config.DEVICE_ID,
                    endpoint=Config.OTEL_EXPORTER_ENDPOINT
                )
                Config.LOGGER = get_otel_logger(__name__, device_id=Config.DEVICE_ID)
                self.logger = Config.LOGGER
                self._publish_text("system", "OpenTelemetry initialized")
            except Exception as e:
                self._publish_text("system", f"Failed to initialize telemetry: {e}", "warning")
                Config.LOGGER = None
                self.logger = None
        else:
            Config.LOGGER = logging.getLogger(__name__)
            self.logger = Config.LOGGER
            if TELEMETRY_AVAILABLE and Config.OTEL_ENABLED and not Config.DEVICE_ID:
                self._publish_text("system", "Telemetry skipped (development mode)", "warning")
    
    async def _initialize_components(self) -> bool:
        """
        Initialize all engine components.
        
        Returns:
            True if successful, False if initialization failed
        """
        # Initialize voice feedback early
        self.voice_feedback = VoiceFeedback(speaker_device_index=None)
        
        # Validate configuration
        Config.validate()
        
        if Config.DEV_MODE:
            self._publish_text("system", "DEVELOPMENT MODE - Missing credentials", "error")
            return False
        
        # Initialize LED controller
        if Config.MAC_MODE:
            led_enabled = False
        else:
            led_enabled = Config.LED_ENABLED if Config.LED_ENABLED is not None else True
        self.led_controller = LEDController(enabled=led_enabled)
        self.led_controller.set_state(LEDController.STATE_BOOT)
        
        # Run startup sequence
        startup_result = await run_startup_sequence(
            led_controller=self.led_controller,
            voice_feedback=self.voice_feedback,
            user_terminate_flag=self.user_terminate,
            logger=self.logger
        )
        
        if not startup_result.success:
            return False
    
        # Keep boot state while initializing
        self.led_controller.set_state(LEDController.STATE_BOOT)
        
        # Initialize context manager
        self.context_manager.set_logger(self.logger)
        
        if not Config.MAC_MODE:
            self._publish_text("system", "Fetching location data...")
            await self.context_manager.start()
            
            if self.context_manager.has_location_data:
                self._publish_text("system", "Location data loaded")
            else:
                self._publish_text("system", "Location data unavailable", "warning")
        
        # Initialize ReSpeaker tuning
        if not Config.MAC_MODE and Config.RESPEAKER_CONFIG:
            self._publish_text("system", "Checking for ReSpeaker hardware...")
            from lib.audio.respeaker import ReSpeakerController
            
            respeaker = ReSpeakerController(
                config=Config.RESPEAKER_CONFIG,
                logger=self.logger
            )
            
            if respeaker.is_available():
                if respeaker.initialize():
                    self._publish_text("system", "ReSpeaker initialized successfully")
                else:
                    self._publish_text("system", "Some ReSpeaker parameters failed", "warning")
        
        # Detect audio devices
        self.mic_device_index, self.speaker_device_index, self.has_hardware_aec = get_audio_devices()
        
        # Update voice feedback
        if self.voice_feedback:
            self.voice_feedback.speaker_device_index = self.speaker_device_index
        
        # Play startup message (respecting quiet hours)
        if self.voice_feedback:
            should_skip = False
            if Config.VOICE_FEEDBACK_QUIET_HOURS_ENABLED:
                current_hour = datetime.now().hour
                is_quiet_hours = current_hour >= 20 or current_hour < 10
                if is_quiet_hours:
                    should_skip = True
            
            if not should_skip:
                self.voice_feedback.play("startup")
        
        # Initialize AudioManager with signal bus
        self._publish_text("system", "Initializing AudioManager...")
        self.audio_manager = AudioManager(
            use_webrtc_aec=Config.USE_WEBRTC_AEC,
            signal_bus=self.signal_bus
        )
        
        if not self.audio_manager.start():
            self._publish_text("system", "AudioManager failed to start", "error")
            self.led_controller.set_state(LEDController.STATE_ERROR)
            return False
        
        # Initialize presence detector
        self._publish_text("system", "Initializing presence detector...")
        self.presence_detector = HumanPresenceDetector(
            mic_device_index=self.mic_device_index,
            threshold=Config.HUMAN_PRESENCE_DETECTION_SCORE_THRESHOLD,
            weights=Config.YAMNET_WEIGHTS,
            orchestrator_client=self.orchestrator_client,
            event_loop=asyncio.get_event_loop()
        )
        
        # Initialize wake word detector
        self.wake_detector = WakeWordDetector(
            audio_manager=self.audio_manager,
            orchestrator_client=self.orchestrator_client,
            presence_detector=self.presence_detector
        )
        
        # Connect to orchestrator
        connected, _ = await self.orchestrator_client.connect()
        if not connected:
            self._publish_text("system", "Failed to connect to orchestrator", "error")
            self.led_controller.set_state(LEDController.STATE_ERROR)
            return False
        
        # Start token refresh monitor
        self.orchestrator_client.token_refresh_task = asyncio.create_task(
            self.orchestrator_client.start_token_refresh_monitor()
        )
        
        if TELEMETRY_AVAILABLE:
            add_span_event("orchestrator_connected", device_id=Config.DEVICE_ID)
        
        # Start detection
        self.wake_detector.start()
        if self.presence_detector:
            self.presence_detector.start()
        
        # Ready state
        self.led_controller.set_state(LEDController.STATE_IDLE)
        self._publish_text("system", f"Ready! Say '{Config.WAKE_WORD}' to start a conversation")
        
        return True
    
    async def run(self) -> None:
        """
        Main engine loop.
        
        Call this to start the engine. It will initialize all components
        and run until shutdown is requested.
        """
        # Start signal bus dispatcher
        self.signal_bus.start()
        
        self._publish_text("system", "Kin AI Raspberry Pi Client (v2)")
        
        # Setup signal handlers
        self._setup_signal_handlers()
        
        # Setup stdout redirect for telemetry
        if TELEMETRY_AVAILABLE and not Config.MAC_MODE:
            setup_stdout_redirect("raspberry-pi-client")
        
        # Initialize telemetry
        await self._initialize_telemetry()
        
        # Initialize components
        if not await self._initialize_components():
            await self.cleanup()
            return
        
        # Main loop
        try:
            while self.running:
                if self.shutdown_requested:
                    self._publish_text("system", "Shutdown requested, exiting...")
                    break
                
                # Check for wake word detection
                if self.wake_detector.detected and not self.conversation_active:
                    self.wake_detector.detected = False
                    self._update_activity()
                    
                    if not self.orchestrator_client.is_connection_alive():
                        self._publish_text("system", "Cannot start conversation: not connected", "error")
                        self.led_controller.set_state(LEDController.STATE_ERROR)
                        await asyncio.sleep(2)
                        self.led_controller.set_state(LEDController.STATE_IDLE)
                        self._resume_wake_word_detection()
                        continue
                    
                    self.led_controller.set_state(LEDController.STATE_WAKE_WORD_DETECTED)
                    self.wake_detector.stop()
                    
                    if self.presence_detector:
                        self.presence_detector.stop()
                    
                    await self._handle_conversation()
                
                # Check for orchestrator messages
                message = await self.orchestrator_client.receive_message()
                if message:
                    self._update_activity()
                    await self._handle_orchestrator_message(message)
                elif not self.orchestrator_client.connected and not self.conversation_active:
                    self._publish_text("system", "Connection lost, reconnecting...", "warning")
                    self.led_controller.set_state(LEDController.STATE_ERROR)
                    
                    reconnected = await self.orchestrator_client.reconnect()
                    if reconnected:
                        self.led_controller.set_state(LEDController.STATE_IDLE)
                        asyncio.create_task(self.context_manager.force_refresh())
                    else:
                        await asyncio.sleep(5)
                        continue
                
                await asyncio.sleep(0.1)
                
        except KeyboardInterrupt:
            self._publish_text("system", "Keyboard interrupt received")
        finally:
            await self.cleanup()
    
    async def _handle_conversation(self) -> None:
        """Handle a single conversation session."""
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
        
        async def _handle_with_trace():
            self.conversation_active = True
            self.user_terminate[0] = False
            
            # Check for cached agent details
            if Config.DEFAULT_REACTIVE_AGENT_ID and Config.DEFAULT_REACTIVE_WEB_SOCKET_URL:
                self._publish_text("conversation", "Using cached agent (fast wake word response)")
                
                self.awaiting_agent_details = False
                
                await self._start_elevenlabs_conversation(
                    agent_id=Config.DEFAULT_REACTIVE_AGENT_ID,
                    web_socket_url=Config.DEFAULT_REACTIVE_WEB_SOCKET_URL,
                    trace_context=None
                )
            else:
                self._publish_text("conversation", "Requesting agent from orchestrator...")
                
                self.awaiting_agent_details = True
                send_success = await self.orchestrator_client.send_reactive()
                
                if not send_success:
                    self._publish_text("conversation", "Failed to send reactive request", "error")
                    self.conversation_active = False
                    self.awaiting_agent_details = False
                    self.led_controller.set_state(LEDController.STATE_ERROR)
                    await asyncio.sleep(2)
                    self._resume_wake_word_detection()
                    self.led_controller.set_state(LEDController.STATE_IDLE)
                    return
                
                # Wait for agent details
                timeout = 10.0
                start_time = time.time()
                
                while self.conversation_active and (time.time() - start_time) < timeout:
                    message = await self.orchestrator_client.receive_message()
                    if message:
                        await self._handle_orchestrator_message(message)
                        if not self.conversation_active:
                            break
                    
                    if not self.orchestrator_client.is_connection_alive():
                        self._publish_text("conversation", "Connection lost while waiting", "error")
                        break
                    
                    await asyncio.sleep(0.1)
                
                if self.awaiting_agent_details:
                    self._publish_text("conversation", "Timeout waiting for agent", "error")
                    self.conversation_active = False
                    self.awaiting_agent_details = False
                    self.led_controller.set_state(LEDController.STATE_ERROR)
                    await asyncio.sleep(2)
                    self._resume_wake_word_detection()
                    self.led_controller.set_state(LEDController.STATE_IDLE)
        
        if self.conversation_trace_context:
            with self.conversation_trace_context:
                await _handle_with_trace()
        else:
            await _handle_with_trace()
    
    async def _start_elevenlabs_conversation(
        self,
        agent_id: str,
        web_socket_url: str,
        trace_context=None
    ) -> None:
        """Start an ElevenLabs conversation."""
        async def _execute_conversation():
            self.conversation_active = True
            self.user_terminate[0] = False
            self.conversation_start_time = time.time()
            
            self._update_activity()
            self.wake_detector.stop()
            
            if self.presence_detector:
                self.presence_detector.stop()
            
            self.led_controller.set_state(LEDController.STATE_CONVERSATION)
            
            client = ElevenLabsConversationClient(
                web_socket_url,
                agent_id,
                audio_manager=self.audio_manager,
                user_terminate_flag=self.user_terminate,
                led_controller=self.led_controller,
                signal_bus=self.signal_bus
            )
            
            self._publish_text("conversation", "Starting ElevenLabs conversation...")
            await client.start(self.orchestrator_client)
            
            self._update_activity()
            
            if self.user_terminate[0]:
                self._publish_text("conversation", "User terminated conversation")
            
            # Handle music mode
            if client.end_reason == "music_mode":
                self._publish_text("conversation", "Entering music mode...")
                self.conversation_active = False
                self._requested_music_genre = client.music_genre
                await self._run_music_mode()
            
            self._resume_wake_word_detection()
            self.led_controller.set_state(LEDController.STATE_IDLE)
            
            self.conversation_active = False
            self.user_terminate[0] = False
            self.conversation_start_time = None
        
        if trace_context is not None:
            try:
                await _execute_conversation()
            finally:
                from opentelemetry import context
                context.detach(trace_context)
        else:
            await _execute_conversation()
    
    async def _run_music_mode(self) -> None:
        """Run music playback mode."""
        genre = getattr(self, '_requested_music_genre', None)
        controller = MusicModeController(
            audio_manager=self.audio_manager,
            speaker_device_index=self.speaker_device_index,
            led_controller=self.led_controller,
            logger=self.logger
        )
        await controller.run(genre=genre)
    
    async def _handle_orchestrator_message(self, message: dict) -> None:
        """Handle messages from the orchestrator."""
        message_type = message.get("type")
        
        if message_type in ("agent_details", "start_conversation"):
            if self.conversation_active and not self.awaiting_agent_details:
                self._publish_text("conversation", "Already in conversation, ignoring", "warning")
                return
            
            self.awaiting_agent_details = False
            
            agent_id = message.get("agent_id")
            web_socket_url = message.get("web_socket_url")
            
            if not agent_id or not web_socket_url:
                self._publish_text("conversation", "Invalid agent details received", "error")
                return
            
            self._publish_text("conversation", f"Received agent: {agent_id}")
            
            context_token = None
            if message_type == "start_conversation" and TELEMETRY_AVAILABLE:
                context_token = extract_trace_context(message)
            
            await self._start_elevenlabs_conversation(
                agent_id=agent_id,
                web_socket_url=web_socket_url,
                trace_context=context_token
            )
            
        elif message_type == "error":
            error_msg = message.get("message", "Unknown error")
            self._publish_text("orchestrator", f"Error: {error_msg}", "error")
            
            self.led_controller.set_state(LEDController.STATE_ERROR)
            await asyncio.sleep(2)
            
            self.conversation_active = False
            self.awaiting_agent_details = False
            
            self._resume_wake_word_detection()
            self.led_controller.set_state(LEDController.STATE_IDLE)
    
    async def cleanup(self) -> None:
        """Clean up all resources."""
        self._publish_text("system", "Cleaning up...")
        self.running = False
        
        # Stop token refresh
        if self.orchestrator_client and self.orchestrator_client.token_refresh_task:
            self.orchestrator_client.token_refresh_task.cancel()
            try:
                await self.orchestrator_client.token_refresh_task
            except asyncio.CancelledError:
                pass
        
        # Stop context manager
        if self.context_manager:
            await self.context_manager.stop()
        
        # Stop LED controller
        if self.led_controller:
            self.led_controller.set_state(LEDController.STATE_OFF)
            self.led_controller.cleanup()
        
        # Stop detectors
        if self.wake_detector:
            self.wake_detector.cleanup()
        
        if self.presence_detector:
            self.presence_detector.cleanup()
        
        # Stop audio manager
        if self.audio_manager:
            self.audio_manager.stop()
        
        # Disconnect from orchestrator
        await self.orchestrator_client.disconnect()
        
        if TELEMETRY_AVAILABLE:
            add_span_event("orchestrator_disconnected", device_id=Config.DEVICE_ID)
        
        # Stop signal bus
        self.signal_bus.stop()
        
        self._publish_text("system", "Cleanup complete. Goodbye!")
        
        if TELEMETRY_AVAILABLE:
            cleanup_stdout_redirect()

