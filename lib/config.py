"""Configuration management from environment variables"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config:
    """Application configuration from environment variables"""
    
    # =========================================================================
    # PROVISIONED CREDENTIALS (stored in .env during provisioning)
    # =========================================================================
    DEVICE_ID = os.getenv("DEVICE_ID")
    DEVICE_PRIVATE_KEY = os.getenv("DEVICE_PRIVATE_KEY")
    
    # Logger (will be set after device_id is validated and telemetry is initialized)
    LOGGER = None
    
    # =========================================================================
    # HARDCODED CONFIGURATION
    # =========================================================================
    # Backend (hardcoded - can be changed via software update)
    ORCHESTRATOR_URL = "wss://conversation-orchestrator.onrender.com/ws"
    
    # Audio settings (ALSA-only, single ReSpeaker device for both capture and playback)
    SAMPLE_RATE = 16000  # 16kHz for both capture and playback (hardcoded)
    CHANNELS = 1  # Mono (ReSpeaker AEC expects mono reference)
    CHUNK_SIZE = 512  # ~32ms frames for low latency
    
    # =========================================================================
    # TURN TRACKER SETTINGS
    # =========================================================================
    # Skip turn tracking entirely (no tracking, no logs, no reports)
    SKIP_TURN_TRACKING = os.getenv("SKIP_TURN_TRACKING", "true").lower() == "true"
    
    # Speaker monitor mode: "loopback" for ALSA loopback, "" to disable
    SPEAKER_MONITOR_MODE = os.getenv("SPEAKER_MONITOR_MODE", "")
    SPEAKER_MONITOR_LOOPBACK_DEVICE = os.getenv("SPEAKER_MONITOR_LOOPBACK_DEVICE", "speaker_monitor")
    
    # Turn tracker VAD settings
    TURN_TRACKER_VAD_THRESHOLD = float(os.getenv("TURN_TRACKER_VAD_THRESHOLD", "0.5"))
    
    # User (mic) settings - allow short utterances like "Yep"
    TURN_TRACKER_USER_SILENCE_TIMEOUT = float(os.getenv("TURN_TRACKER_USER_SILENCE_TIMEOUT", "2.5"))
    TURN_TRACKER_USER_MIN_TURN_DURATION = float(os.getenv("TURN_TRACKER_USER_MIN_TURN_DURATION", "0.15"))
    TURN_TRACKER_USER_MIN_SPEECH_ONSET = float(os.getenv("TURN_TRACKER_USER_MIN_SPEECH_ONSET", "0.08"))
    
    # Agent (speaker) settings
    TURN_TRACKER_AGENT_SILENCE_TIMEOUT = float(os.getenv("TURN_TRACKER_AGENT_SILENCE_TIMEOUT", "2.5"))
    TURN_TRACKER_AGENT_MIN_TURN_DURATION = float(os.getenv("TURN_TRACKER_AGENT_MIN_TURN_DURATION", "0.2"))
    TURN_TRACKER_AGENT_MIN_SPEECH_ONSET = float(os.getenv("TURN_TRACKER_AGENT_MIN_SPEECH_ONSET", "0.08"))
    
    TURN_TRACKER_DEBOUNCE_WINDOW = float(os.getenv("TURN_TRACKER_DEBOUNCE_WINDOW", "0.5"))
    
    # =========================================================================
    # RUNTIME CONFIGURATION (fetched from backend after authentication)
    # =========================================================================
    # These will be populated after device authentication
    USER_ID = None
    AUTH_TOKEN = None  # JWT token for WebSocket authentication
    
    # API Keys (fetched from backend)
    ELEVENLABS_API_KEY = None
    PICOVOICE_ACCESS_KEY = None
    
    # System settings (fetched from backend)
    WAKE_WORD = None
    LED_ENABLED = None
    LED_BRIGHTNESS = 60  # Default, may be overridden
    
    # Default reactive agent (cached from backend for faster wake word response)
    DEFAULT_REACTIVE_AGENT_ID = None
    DEFAULT_REACTIVE_WEB_SOCKET_URL = None
    
    # OTEL defaults come from env so telemetry can start before runtime config arrives
    OTEL_ENABLED = os.getenv("OTEL_ENABLED", "true").lower() == "true"
    OTEL_EXPORTER_ENDPOINT = os.getenv("OTEL_EXPORTER_ENDPOINT", "http://localhost:4318")
    
    # Environment
    ENV = os.getenv("ENV", "production")
    
    @classmethod
    def validate(cls):
        """Validate required configuration"""
        missing = []
        
        # Only DEVICE_ID and DEVICE_PRIVATE_KEY are required
        if not cls.DEVICE_ID:
            missing.append("DEVICE_ID")
        if not cls.DEVICE_PRIVATE_KEY:
            missing.append("DEVICE_PRIVATE_KEY")
            
        if missing:
            print(f"✗ Missing required environment variables: {', '.join(missing)}")
            print("   This device needs to be provisioned through the admin portal.")
            print("   Download the installer package and run ./install.sh")
            sys.exit(1)
    
    @classmethod
    def load_runtime_config(cls, config_data: dict):
        """
        Load runtime configuration from backend.
        
        Args:
            config_data: Configuration dict from /auth/device/config endpoint
        """
        device_config = config_data.get("device", {})
        system_config = config_data.get("system", {})
        api_keys = config_data.get("api_keys", {})
        default_reactive_agent = config_data.get("default_reactive_agent")
        
        # Set device info
        cls.USER_ID = device_config.get("user_id")
        
        # Set API keys
        cls.PICOVOICE_ACCESS_KEY = api_keys.get("picovoice_access_key") or device_config.get("picovoice_access_key")
        cls.ELEVENLABS_API_KEY = system_config.get("elevenlabs_api_key")
        
        # Set system settings
        cls.WAKE_WORD = system_config.get("wake_word", "porcupine")
        cls.LED_ENABLED = system_config.get("led_enabled", "true").lower() == "true"
        cls.OTEL_ENABLED = system_config.get("otel_enabled", "true").lower() == "true"
        
        # Set default reactive agent (for fast wake word response)
        if default_reactive_agent:
            cls.DEFAULT_REACTIVE_AGENT_ID = default_reactive_agent.get("agent_id")
            cls.DEFAULT_REACTIVE_WEB_SOCKET_URL = default_reactive_agent.get("web_socket_url")
        else:
            cls.DEFAULT_REACTIVE_AGENT_ID = None
            cls.DEFAULT_REACTIVE_WEB_SOCKET_URL = None
        
        # OTEL endpoint is ALWAYS the local collector on the device
        # The local collector forwards to the central endpoint (configured by wrapper)
        cls.OTEL_EXPORTER_ENDPOINT = "http://localhost:4318"
        
        # Note: SAMPLE_RATE is hardcoded (16000 Hz) - not configurable
        
        print(f"✓ Runtime configuration loaded")
        print(f"   Wake Word: {cls.WAKE_WORD}")
        print(f"   Sample Rate: {cls.SAMPLE_RATE} Hz")
        print(f"   OTEL Enabled: {cls.OTEL_ENABLED}")
        print(f"   LED Enabled: {cls.LED_ENABLED}")
        if cls.DEFAULT_REACTIVE_AGENT_ID:
            print(f"   Default Reactive Agent: Cached (fast wake word response)")
        else:
            print(f"   Default Reactive Agent: Not configured (will fetch on wake word)")
