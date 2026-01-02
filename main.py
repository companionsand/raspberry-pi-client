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
- Setup mode for device pairing and WiFi configuration when needed

Usage:
    python main.py

Requirements:
    - Raspberry Pi OS with ALSA
    - Audio device (ReSpeaker recommended, fallback to any mic/speaker)
    - Environment variables: see Config class
"""

import os
import signal
import time
import asyncio
import logging
from datetime import datetime

# Configure logging BEFORE any other imports (critical for telemetry)
# This ensures OTEL handler captures console output
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    force=True,
)

# Import local modules
from lib.config import Config
from lib.device_auth import authenticate
from lib.audio import get_audio_devices, LEDController, VoiceFeedback
from lib.detection import WakeWordDetector, HumanPresenceDetector
from lib.agent import OrchestratorClient, ElevenLabsConversationClient, ContextManager

# Import setup module (optional - graceful degradation)
try:
    from lib.setup import SetupManager
    from lib.setup.connectivity import ConnectivityChecker
    SETUP_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Setup module not available: {e}")
    SETUP_AVAILABLE = False

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
    print("⚠️  Telemetry module not available - running without telemetry")
    TELEMETRY_AVAILABLE = False
    # Provide no-op fallbacks
    def get_otel_logger(name, device_id=None):
        import logging
        return logging.getLogger(name)
    def setup_stdout_redirect():
        return False
    def cleanup_stdout_redirect():
        pass

# Setup stdout/stderr redirection EARLY (before any print statements)
# This captures all print() output and sends it to OTEL/BetterStack
if TELEMETRY_AVAILABLE:
    setup_stdout_redirect("raspberry-pi-client")


# =============================================================================
# MAIN APPLICATION
# =============================================================================

class KinClient:
    """Main application controller with full telemetry and LED feedback"""
    
    def __init__(self):
        self.wake_detector = None
        self.presence_detector = None  # Will be initialized after runtime config is loaded
        self.context_manager = ContextManager()
        self.orchestrator_client = OrchestratorClient(context_manager=self.context_manager)
        self.led_controller = None
        self.voice_feedback = None  # Will be initialized after speaker detection
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
        
        # Setup telemetry if available (skip in dev mode when device_id is missing)
        # Skip telemetry initialization if device_id is None to avoid connection errors
        if TELEMETRY_AVAILABLE and Config.OTEL_ENABLED and Config.DEVICE_ID:
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
            # Fallback logger without telemetry (dev mode or telemetry disabled)
            Config.LOGGER = logging.getLogger(__name__)
            self.logger = Config.LOGGER
            if TELEMETRY_AVAILABLE and Config.OTEL_ENABLED and not Config.DEVICE_ID:
                print("⚠️  Telemetry skipped (development mode - no device_id)")
        
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
            # Also set user_terminate so setup can detect shutdown
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
    
    def _resume_wake_word_detection(self):
        """Start wake word detector again if it is not already running."""
        if self.wake_detector and self.wake_detector.running:
            return
        
        if self.wake_detector:
            self.wake_detector.start()
            print(f"\n✓ Listening for '{Config.WAKE_WORD}' again...")
        
        # Resume presence detection
        if self.presence_detector and not self.presence_detector.running:
            self.presence_detector.start()
    
    async def run(self):
        """Main application loop"""
        print("\n" + "="*60)
        print("🎙️  Kin AI Raspberry Pi Client (v2)")
        print("="*60)
        
        # Initialize voice feedback early (speaker device will be set after detection)
        self.voice_feedback = VoiceFeedback(speaker_device_index=None)
        
        # Validate configuration
        config_valid = Config.validate()
        
        # Check if running in development mode (missing credentials)
        if Config.DEV_MODE:
            print("\n" + "="*60)
            print("⚠️  DEVELOPMENT MODE - Missing Required Configuration")
            print("="*60)
            print("\nThe following services are DISABLED due to missing credentials:")
            print("  ✗ Device Authentication (requires DEVICE_ID and DEVICE_PRIVATE_KEY)")
            print("  ✗ Orchestrator Connection (requires authentication)")
            print("  ✗ WiFi Setup (requires authentication)")
            print("  ✗ Wake Word Detection (requires PICOVOICE_ACCESS_KEY from backend)")
            print("  ✗ Presence Detection (requires authentication)")
            print("  ✗ Voice Conversations (requires ELEVENLABS_API_KEY from backend)")
            print("\nTo enable full functionality:")
            print("  1. Provision device through admin portal")
            print("  2. Set DEVICE_ID and DEVICE_PRIVATE_KEY environment variables")
            print("  3. Or create a .env file with these credentials")
            print("\n" + "="*60)
            print("Exiting - this application requires device credentials to run.")
            print("="*60 + "\n")
            return
        
        # Initialize LED controller early (before setup)
        # This allows setup to show LED feedback
        # Default to True if not set yet (will be set after authentication)
        led_enabled = Config.LED_ENABLED if Config.LED_ENABLED is not None else True
        self.led_controller = LEDController(enabled=led_enabled)
        self.led_controller.set_state(LEDController.STATE_BOOT)
        
        # Determine if setup should be skipped
        # Priority: env var (for testing) → cached config → default (false = allow setup)
        skip_wifi_setup = False  # Default: allow WiFi setup
        config_source = "default"  # Track where the value came from
        
        # Check env var override first (for testing/debugging)
        env_skip_wifi = os.getenv('SKIP_WIFI_SETUP')
        if env_skip_wifi:
            skip_wifi_setup = env_skip_wifi.lower() == 'true'
            config_source = "env_var"
            print(f"ℹ️  Using SKIP_WIFI_SETUP from environment: {skip_wifi_setup}")
        else:
            # Check cached config from last successful authentication
            cached_config = Config.load_device_config_cache()
            if cached_config and cached_config.get("device"):
                cached_skip = cached_config["device"].get("skip_wifi_setup", "false")
                skip_wifi_setup = cached_skip.lower() == 'true' if isinstance(cached_skip, str) else bool(cached_skip)
                config_source = "cached_config"
                print(f"ℹ️  Using skip_wifi_setup from cached config: {skip_wifi_setup}")
            else:
                print(f"ℹ️  No cached config found, using default: skip_wifi_setup={skip_wifi_setup}")
        
        # Log final skip_wifi_setup value for monitoring
        print("=" * 60)
        print(f"⚙️  SKIP_WIFI_SETUP: {skip_wifi_setup}")
        print(f"   Setup Mode: {'DISABLED' if skip_wifi_setup else 'ENABLED'}")
        print(f"   Config Source: {config_source}")
        print("=" * 60)
        
        # Structured log for monitoring/alerting
        if self.logger:
            self.logger.info(
                "skip_wifi_setup_config",
                extra={
                    "skip_wifi_setup": skip_wifi_setup,
                    "wifi_setup_mode": "disabled" if skip_wifi_setup else "enabled",
                    "config_source": config_source
                }
            )
        
        pairing_code = None  # Will be set by setup if needed
        
        # Setup Mode - only if enabled and available
        if not skip_wifi_setup and SETUP_AVAILABLE:
            print("\n📡 Checking connectivity...")
            
            connectivity_checker = ConnectivityChecker()
            has_internet, orchestrator_reachable = await connectivity_checker.check_full_connectivity(
                orchestrator_retries=3
            )
            
            if not has_internet:
                print("✗ No internet connection detected")
                
                # Play voice feedback
                if self.voice_feedback:
                    self.voice_feedback.play("no_internet")
                
                # Setup + pairing loop - retry if pairing fails
                # Note: Setup interface collects both WiFi credentials (for connectivity)
                # and pairing code (for device-to-user linking). These are conceptually separate.
                max_setup_attempts = 3
                setup_attempt = 0
                authenticated = False
                
                while setup_attempt < max_setup_attempts and not authenticated:
                    setup_attempt += 1
                    print(f"\n🔧 Entering setup mode (attempt {setup_attempt}/{max_setup_attempts})...")
                    print("="*60)
                    print("ℹ️  Note: Setup requires sudo privileges for network management.")
                    print("   If prompted for a password, enter your Raspberry Pi user password.")
                    print("="*60)
                    
                    # Start setup manager with LED controller and voice feedback
                    # Pass shutdown flag so setup can respond to CTRL+C
                    setup_manager = SetupManager(
                        led_controller=self.led_controller,
                        voice_feedback=self.voice_feedback,
                        shutdown_flag=self.user_terminate  # Use same flag for shutdown
                    )
                    pairing_code, success = await setup_manager.start_setup_mode()
                    
                    if success and pairing_code:
                        print(f"\n✓ WiFi connected!")
                        print(f"  Pairing code received: {pairing_code}")
                        print("="*60)
                        
                        # Note: AP is now stopped, user's device has disconnected
                        # User won't see authentication status on web page
                        # If auth fails, we'll restart AP so they can see error message
                        
                        print("\n🔐 Authenticating with pairing code...")
                        
                        auth_result = authenticate(pairing_code=pairing_code)
                        if auth_result and auth_result.get("success"):
                            authenticated = True
                            print("✓ Authentication and pairing successful!")
                            print("  Device is now paired and starting...")
                            
                            # Update LED controller with backend config
                            if self.led_controller:
                                self.led_controller.enabled = Config.LED_ENABLED
                            
                            # Clean up (HTTP server already stopped during WiFi switch)
                            try:
                                await setup_manager.http_server.stop()
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
                                if setup_manager._wifi_credentials:
                                    failed_ssid = setup_manager._wifi_credentials[0]
                                    print(f"  Deleting WiFi connection: {failed_ssid}")
                                    try:
                                        import subprocess
                                        subprocess.run(['sudo', 'nmcli', 'connection', 'delete', failed_ssid], 
                                                     capture_output=True, timeout=5)
                                    except Exception as e:
                                        print(f"  Warning: Could not delete connection: {e}")
                                
                                # Clear credentials from manager so setup starts fresh
                                setup_manager._wifi_credentials = None
                                setup_manager._pairing_code = None
                                
                                print(f"  Please reconnect to Kin_Setup and try again")
                                await asyncio.sleep(3)
                            else:
                                print(f"  Max attempts ({max_setup_attempts}) reached")
                                await asyncio.sleep(3)
                    else:
                        print("\n✗ Setup failed")
                        # Clean up HTTP server if it's still running
                        try:
                            await setup_manager.http_server.stop()
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
                print("✓ Internet connection confirmed")
                if not orchestrator_reachable:
                    print("⚠ Warning: Orchestrator is unreachable")
                    print("  This may be due to:")
                    print("    - Orchestrator service is down")
                    print("    - Network/firewall issues")
                    print("    - Incorrect orchestrator URL")
                    print("  Device will continue but may have limited functionality")
                
                # Try to authenticate without pairing code (device might already be paired)
                auth_result = authenticate()
                
                if not auth_result or not auth_result.get("success"):
                    # Authentication failed - check the reason
                    reason = auth_result.get("reason") if auth_result else "unknown"
                    
                    if reason == "unpaired":
                        # Device is authenticated but not paired - enter setup mode to collect pairing code
                        # Note: Device already has internet, so WiFi setup is not needed for connectivity.
                        # Setup manager will detect internet and skip WiFi/sudo setup automatically.
                        print("⚠️  Device is not paired with a user")
                        print("   Entering pairing mode to collect pairing code...")
                        print("   Note: Device already has internet - no WiFi setup needed (no sudo required)")
                        
                        # Play voice feedback
                        if self.voice_feedback:
                            self.voice_feedback.play("device_not_paired")
                        
                        # Setup + pairing loop
                        max_setup_attempts = 3
                        setup_attempt = 0
                        authenticated = False
                        
                        while setup_attempt < max_setup_attempts and not authenticated:
                            setup_attempt += 1
                            print(f"\n🔧 Entering pairing mode (attempt {setup_attempt}/{max_setup_attempts})...")
                            print("="*60)
                            print("ℹ️  Note: Pairing mode uses existing network connection.")
                            print("   No sudo password required - device already has internet.")
                            print("="*60)
                            
                            # Start setup manager with LED controller and voice feedback
                            # Pass shutdown flag so setup can respond to CTRL+C
                            setup_manager = SetupManager(
                                led_controller=self.led_controller,
                                voice_feedback=self.voice_feedback,
                                shutdown_flag=self.user_terminate  # Use same flag for shutdown
                            )
                            pairing_code, success = await setup_manager.start_setup_mode()
                            
                            if success and pairing_code:
                                print(f"\n✓ Setup complete!")
                                print(f"  Pairing code received: {pairing_code}")
                                print("="*60)
                                
                                print("\n🔐 Authenticating with pairing code...")
                                
                                pair_result = authenticate(pairing_code=pairing_code)
                                if pair_result and pair_result.get("success"):
                                    authenticated = True
                                    print("✓ Authentication and pairing successful!")
                                    print("  Device is now paired and starting...")
                                    
                                    # Update LED controller with backend config
                                    if self.led_controller:
                                        self.led_controller.enabled = Config.LED_ENABLED
                                    
                                    # Clean up
                                    try:
                                        await setup_manager.http_server.stop()
                                    except:
                                        pass
                                else:
                                    print("✗ Authentication failed with pairing code")
                                    print("  Please verify the pairing code is correct")
                                    
                                    # Clean up before retry
                                    try:
                                        await setup_manager.http_server.stop()
                                    except:
                                        pass
                                    
                                    if setup_attempt < max_setup_attempts:
                                        print("  Possible reasons:")
                                        print("    - Incorrect pairing code")
                                        print("    - Pairing code expired or already used")
                                        print("    - Device not registered in admin portal")
                                        print(f"\n  Cleaning up and restarting setup mode...")
                                        
                                        # Delete WiFi connection to prevent device from getting stuck
                                        if setup_manager._wifi_credentials:
                                            failed_ssid = setup_manager._wifi_credentials[0]
                                            print(f"  Deleting WiFi connection: {failed_ssid}")
                                            try:
                                                import subprocess
                                                subprocess.run(['sudo', 'nmcli', 'connection', 'delete', failed_ssid], 
                                                             capture_output=True, timeout=5)
                                            except Exception as e:
                                                print(f"  Warning: Could not delete connection: {e}")
                                        
                                        # Clear credentials from manager so setup starts fresh
                                        setup_manager._wifi_credentials = None
                                        setup_manager._pairing_code = None
                                        
                                        print(f"  Please reconnect to Kin_Setup and try again")
                                        await asyncio.sleep(3)
                            else:
                                print("✗ Setup failed or cancelled")
                                
                                # Clean up
                                try:
                                    await setup_manager.http_server.stop()
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
                        # Other authentication error (not unpaired) - exit with error
                        print(f"✗ Failed to authenticate device (reason: {reason})")
                        return
                else:
                    # Authentication successful, device is already paired
                    print("✓ Device authenticated and paired")
                    
                    # Update LED controller with backend config
                    if self.led_controller:
                        self.led_controller.enabled = Config.LED_ENABLED
        else:
            # Setup skipped, authenticate normally
            auth_result = authenticate()
            if not auth_result or not auth_result.get("success"):
                reason = auth_result.get("reason") if auth_result else "unknown"
                print(f"✗ Failed to authenticate device (reason: {reason})")
                return
            
            # Update LED controller with backend config
            if self.led_controller:
                self.led_controller.enabled = Config.LED_ENABLED
        
        # LED controller already initialized before setup
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
        
        # Initialize ReSpeaker hardware tuning if configuration is available
        print("🔧 Checking for ReSpeaker hardware...")
        if Config.RESPEAKER_CONFIG:
            from lib.audio.respeaker import ReSpeakerController
            
            respeaker = ReSpeakerController(
                config=Config.RESPEAKER_CONFIG,
                logger=self.logger
            )
            
            if respeaker.is_available():
                print("✓ ReSpeaker detected, applying tuning parameters...")
                if respeaker.initialize():
                    print("✓ ReSpeaker initialized successfully")
                else:
                    print("⚠️  Some ReSpeaker parameters failed to apply - check logs")
            else:
                print("⚠️  ReSpeaker tuning tools not available (this should not happen)")
        else:
            print("⚠️  No ReSpeaker configuration available from backend")
        
        # Detect audio devices
        self.mic_device_index, self.speaker_device_index, self.has_hardware_aec = get_audio_devices()
        
        # Update voice feedback with detected speaker device
        if self.voice_feedback:
            self.voice_feedback.speaker_device_index = self.speaker_device_index
        
        # Play startup message now that speaker device is configured
        # Skip during quiet hours (8pm-10am) to avoid noise
        if self.voice_feedback:
            current_hour = datetime.now().hour
            is_quiet_hours = current_hour >= 20 or current_hour < 10
            
            if is_quiet_hours:
                if self.logger:
                    self.logger.info("[Main] Skipping startup voice feedback (quiet hours: 8pm-10am)")
            else:
                self.voice_feedback.play("startup")
        
        # Initialize human presence detector first (so we can pass it to wake word detector)
        print("\n📊 Initializing human presence detector...")
        self.presence_detector = HumanPresenceDetector(
            mic_device_index=self.mic_device_index,
            threshold=Config.HUMAN_PRESENCE_DETECTION_SCORE_THRESHOLD,
            weights=Config.YAMNET_WEIGHTS,
            orchestrator_client=self.orchestrator_client,
            event_loop=asyncio.get_event_loop()  # Pass main event loop for async operations
        )
        
        # Initialize wake word detector with detected microphone
        # Pass presence_detector so they can share the audio stream
        self.wake_detector = WakeWordDetector(
            mic_device_index=self.mic_device_index,
            orchestrator_client=self.orchestrator_client,
            presence_detector=self.presence_detector
        )
        
        # Connect to conversation-orchestrator
        connected, _ = await self.orchestrator_client.connect()
        if not connected:
            self.led_controller.set_state(LEDController.STATE_ERROR)
            print("✗ Failed to connect to conversation-orchestrator")
            return
        
        # Start background token refresh monitor
        print("✓ Starting token refresh monitor...")
        self.orchestrator_client.token_refresh_task = asyncio.create_task(
            self.orchestrator_client.start_token_refresh_monitor()
        )
        
        # Record successful connection
        if TELEMETRY_AVAILABLE:
            add_span_event("orchestrator_connected", device_id=Config.DEVICE_ID)
        
        # Start wake word detection
        self.wake_detector.start()
        
        # Start presence detection
        if self.presence_detector:
            self.presence_detector.start()
        
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
                    
                    # Stop presence detection during conversation
                    if self.presence_detector:
                        self.presence_detector.stop()
                    
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
            
            # Stop presence detection during conversation
            if self.presence_detector:
                self.presence_detector.stop()
            
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
        
        # Stop token refresh monitor
        if self.orchestrator_client and self.orchestrator_client.token_refresh_task:
            self.orchestrator_client.token_refresh_task.cancel()
            try:
                await self.orchestrator_client.token_refresh_task
            except asyncio.CancelledError:
                pass
        
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
        
        # Stop presence detector
        if self.presence_detector:
            self.presence_detector.cleanup()
        
        # Disconnect from orchestrator
        await self.orchestrator_client.disconnect()
        
        # Record disconnection
        if TELEMETRY_AVAILABLE:
            add_span_event("orchestrator_disconnected", device_id=Config.DEVICE_ID)
        
        print("✓ Cleanup complete")
        print("👋 Goodbye!\n")
        
        # Cleanup stdout/stderr redirection (restore original streams)
        if TELEMETRY_AVAILABLE:
            cleanup_stdout_redirect()


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
