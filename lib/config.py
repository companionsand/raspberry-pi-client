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
    CHANNELS = 1  # Mono output (ReSpeaker AEC expects mono reference)
    CHUNK_SIZE = 512  # ~32ms frames for low latency
    
    # ReSpeaker 4-Mic Array channel configuration
    # The ReSpeaker outputs 6 channels: Ch0 = AEC-processed, Ch1-4 = raw mics, Ch5 = reference
    # We open all 6 channels and extract Ch0 (AEC-processed) in Python code
    RESPEAKER_CHANNELS = 6  # ReSpeaker 4-Mic Array outputs 6 channels
    RESPEAKER_AEC_CHANNEL = 0  # Channel 0 is AEC-processed (echo-cancelled)
    
    # =========================================================================
    # RUNTIME CONFIGURATION (fetched from backend after authentication)
    # =========================================================================
    # These will be populated after device authentication
    USER_ID = None
    AUTH_TOKEN = None  # JWT token for WebSocket authentication
    AUTH_TOKEN_EXPIRES_AT = None  # Unix timestamp when token expires (decoded from JWT)
    
    # =========================================================================
    # LOGGING CONFIGURATION (fetched from device_settings)
    # =========================================================================
    # These control which log categories are shown
    SHOW_AEC_DEBUG_LOGS = True
    SHOW_ELEVENLABS_AUDIO_CHUNK_LOGS = True
    SHOW_CONVERSATION_STATUS_LOGS = True
    SHOW_AGENT_TURN_LOGS = True
    SHOW_LED_STATE_LOGS = True
    SHOW_VAD_DIAGNOSTIC_LOGS = True
    SHOW_WAKE_WORD_DEBUG_LOGS = True
    
    # API Keys (fetched from backend)
    ELEVENLABS_API_KEY = None
    PICOVOICE_ACCESS_KEY = None
    GOOGLE_API_KEY = None
    
    # System settings (fetched from backend)
    WAKE_WORD = None
    LED_ENABLED = None
    LED_BRIGHTNESS = 60  # Default, may be overridden
    WAKE_WORD_ASR_SIMILARITY_THRESHOLD = 0.6  # Default similarity threshold for wake word matching
    SPEAKER_VOLUME_PERCENT = 50  # Default speaker volume (0-100)
    
    # Default reactive agent (cached from backend for faster wake word response)
    DEFAULT_REACTIVE_AGENT_ID = None
    DEFAULT_REACTIVE_WEB_SOCKET_URL = None
    
    # OTEL defaults come from env so telemetry can start before runtime config arrives
    OTEL_ENABLED = os.getenv("OTEL_ENABLED", "true").lower() == "true"
    OTEL_EXPORTER_ENDPOINT = os.getenv("OTEL_EXPORTER_ENDPOINT", "http://localhost:4318")
    
    # ReSpeaker tuning configuration (fetched from backend, saved to file for wrapper)
    RESPEAKER_CONFIG = None
    
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
        logging_settings = config_data.get("logging_settings", {})
        
        # Set device info
        cls.USER_ID = device_config.get("user_id")
        
        # Set API keys
        cls.PICOVOICE_ACCESS_KEY = api_keys.get("picovoice_access_key") or device_config.get("picovoice_access_key")
        cls.ELEVENLABS_API_KEY = system_config.get("elevenlabs_api_key")
        cls.GOOGLE_API_KEY = system_config.get("google_api_key")
        
        # Set system settings
        cls.WAKE_WORD = system_config.get("wake_word", "porcupine")
        cls.LED_ENABLED = system_config.get("led_enabled", "true").lower() == "true"
        cls.OTEL_ENABLED = system_config.get("otel_enabled", "true").lower() == "true"
        cls.WAKE_WORD_ASR_SIMILARITY_THRESHOLD = float(system_config.get("wake_word_asr_similarity_threshold", "0.6"))
        cls.SPEAKER_VOLUME_PERCENT = int(system_config.get("speaker_volume_percent", "50"))
        
        # Set logging settings (default to True if not provided)
        cls.SHOW_AEC_DEBUG_LOGS = str(logging_settings.get("show_aec_debug_logs", "true")).lower() == "true"
        cls.SHOW_ELEVENLABS_AUDIO_CHUNK_LOGS = str(logging_settings.get("show_elevenlabs_audio_chunk_logs", "true")).lower() == "true"
        cls.SHOW_CONVERSATION_STATUS_LOGS = str(logging_settings.get("show_conversation_status_logs", "true")).lower() == "true"
        cls.SHOW_AGENT_TURN_LOGS = str(logging_settings.get("show_agent_turn_logs", "true")).lower() == "true"
        cls.SHOW_LED_STATE_LOGS = str(logging_settings.get("show_led_state_logs", "true")).lower() == "true"
        cls.SHOW_VAD_DIAGNOSTIC_LOGS = str(logging_settings.get("show_vad_diagnostic_logs", "true")).lower() == "true"
        cls.SHOW_WAKE_WORD_DEBUG_LOGS = str(logging_settings.get("show_wake_word_debug_logs", "true")).lower() == "true"
        
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
        
        # Load ReSpeaker configuration (if provided by backend)
        # agc_gain: 1.0 = minimum (no amplification), higher values amplify echo
        # agc_on_off: 0 = AGC disabled (prevents echo amplification)
        cls.RESPEAKER_CONFIG = system_config.get("respeaker_config", {
            "agc_gain": 1.0,
            "agc_on_off": 0,
            "aec_freeze_on_off": 0,
            "echo_on_off": 1,
            "hpf_on_off": 1,
            "stat_noise_on_off": 1,
            "gamma_e": 2.0,
            "gamma_enl": 3.0,
            "gamma_etail": 2.0
        })
        
        # Note: ReSpeaker config is now applied by the Python client on startup (main.py)
        # We no longer need to save to file for the wrapper to read on next boot
        # Keeping _save_respeaker_config() method for backward compatibility / debugging
        # cls._save_respeaker_config()
        
        # Note: SAMPLE_RATE is hardcoded (16000 Hz) - not configurable
        
        print(f"✓ Runtime configuration loaded")
        print(f"   Wake Word: {cls.WAKE_WORD}")
        print(f"   Sample Rate: {cls.SAMPLE_RATE} Hz")
        print(f"   OTEL Enabled: {cls.OTEL_ENABLED}")
        print(f"   LED Enabled: {cls.LED_ENABLED}")
        print(f"   Wake Word ASR Similarity Threshold: {cls.WAKE_WORD_ASR_SIMILARITY_THRESHOLD}")
        if cls.DEFAULT_REACTIVE_AGENT_ID:
            print(f"   Default Reactive Agent: Cached (fast wake word response)")
        else:
            print(f"   Default Reactive Agent: Not configured (will fetch on wake word)")
    
    @classmethod
    def _save_respeaker_config(cls):
        """
        Save ReSpeaker configuration to file for wrapper to read on next boot.
        
        The wrapper's respeaker-init.sh script will read this file and apply
        the tuning parameters before starting the Python client.
        """
        import json
        
        config_file = os.path.expanduser("~/.respeaker_config.json")
        
        try:
            with open(config_file, 'w') as f:
                json.dump(cls.RESPEAKER_CONFIG, f, indent=2)
            print(f"✓ ReSpeaker config saved to {config_file} (will apply on next restart)")
        except Exception as e:
            print(f"⚠️  Could not save ReSpeaker config: {e}")
