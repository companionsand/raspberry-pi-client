"""Configuration management from environment variables"""

import os
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
    CHUNK_SIZE = 480  # ~30ms frames (multiple of 160 for WebRTC AEC, balance between latency and overflow prevention)
    
    # ReSpeaker 4-Mic Array channel configuration
    # The ReSpeaker outputs 6 channels: Ch0 = AEC-processed, Ch1-4 = raw mics, Ch5 = reference
    # We open all 6 channels and extract Ch0 (AEC-processed) in Python code
    RESPEAKER_CHANNELS = 6  # ReSpeaker 4-Mic Array outputs 6 channels
    RESPEAKER_AEC_CHANNEL = 0  # Channel 0 is AEC-processed (echo-cancelled)
    RESPEAKER_REFERENCE_CHANNEL = 5  # Channel 5 is playback loopback (for WebRTC AEC)
    
    # WebRTC AEC Configuration (Software AEC on top of hardware beamforming)
    # Enable this to use WebRTC AEC3 for superior echo cancellation beyond hardware AEC
    USE_WEBRTC_AEC = os.getenv("USE_WEBRTC_AEC", "false").lower() == "true"  # Default: False (opt-in)
    WEBRTC_AEC_STREAM_DELAY_MS = int(os.getenv("WEBRTC_AEC_STREAM_DELAY_MS", "100"))  # USB audio delay (50-200ms typical)
    WEBRTC_AEC_NS_LEVEL = int(os.getenv("WEBRTC_AEC_NS_LEVEL", "1"))  # Noise suppression: 0-3 (0=off, 1=moderate, 3=max)
    WEBRTC_AEC_AGC_MODE = int(os.getenv("WEBRTC_AEC_AGC_MODE", "2"))  # AGC mode: 1=adaptive, 2=fixed digital
    
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
    SHOW_PRESENCE_DETECTION_DEBUG_LOGS = False
    
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
    
    # Presence detection settings (fetched from backend)
    YAMNET_WEIGHTS = {}  # Dict mapping event names to weights (0.0-1.0)
    HUMAN_PRESENCE_DETECTION_SCORE_THRESHOLD = 0.3  # Threshold for positive detection
    
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
    
    # Voice feedback quiet hours (8pm-10am) - set to "false" to always play startup message
    VOICE_FEEDBACK_QUIET_HOURS_ENABLED = os.getenv("VOICE_FEEDBACK_QUIET_HOURS_ENABLED", "true").lower() == "true"
    
    # =========================================================================
    # MAC DEVELOPMENT MODE
    # =========================================================================
    # When enabled, skips Raspberry Pi-specific features:
    # - OpenTelemetry setup (OTEL)
    # - WiFi setup mode
    # - LED controller
    # - Location fetching (WiFi triangulation)
    # - ReSpeaker tuning
    MAC_MODE = os.getenv("MAC_MODE", "false").lower() == "true"
    
    # =========================================================================
    # WEB DASHBOARD CONFIGURATION
    # =========================================================================
    # Hostname for mDNS discovery (e.g., "kin" -> "kin.local")
    WEB_HOSTNAME = os.getenv("WEB_HOSTNAME", "kin")
    # Port for the web dashboard server
    WEB_PORT = int(os.getenv("WEB_PORT", "8080"))
    
    @classmethod
    def validate(cls):
        """Validate required configuration.
        
        Returns:
            bool: True if configuration is valid, False if missing required credentials.
                 When False, DEV_MODE is set to True.
        """
        missing = []
        
        # Only DEVICE_ID and DEVICE_PRIVATE_KEY are required
        if not cls.DEVICE_ID:
            missing.append("DEVICE_ID")
        if not cls.DEVICE_PRIVATE_KEY:
            missing.append("DEVICE_PRIVATE_KEY")
            
        if missing:
            cls.DEV_MODE = True
            return False
        
        cls.DEV_MODE = False
        return True
    
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
        
        # Set presence detection settings
        cls.YAMNET_WEIGHTS = system_config.get("yamnet_weights", {})
        cls.HUMAN_PRESENCE_DETECTION_SCORE_THRESHOLD = float(system_config.get("human_presence_detection_score_threshold", "0.3"))
        
        # Set logging settings (default to True if not provided, except presence debug logs)
        cls.SHOW_AEC_DEBUG_LOGS = str(logging_settings.get("show_aec_debug_logs", "true")).lower() == "true"
        cls.SHOW_ELEVENLABS_AUDIO_CHUNK_LOGS = str(logging_settings.get("show_elevenlabs_audio_chunk_logs", "true")).lower() == "true"
        cls.SHOW_CONVERSATION_STATUS_LOGS = str(logging_settings.get("show_conversation_status_logs", "true")).lower() == "true"
        cls.SHOW_AGENT_TURN_LOGS = str(logging_settings.get("show_agent_turn_logs", "true")).lower() == "true"
        cls.SHOW_LED_STATE_LOGS = str(logging_settings.get("show_led_state_logs", "true")).lower() == "true"
        cls.SHOW_VAD_DIAGNOSTIC_LOGS = str(logging_settings.get("show_vad_diagnostic_logs", "true")).lower() == "true"
        cls.SHOW_WAKE_WORD_DEBUG_LOGS = str(logging_settings.get("show_wake_word_debug_logs", "true")).lower() == "true"
        cls.SHOW_PRESENCE_DETECTION_DEBUG_LOGS = str(logging_settings.get("show_presence_detection_debug_logs", "false")).lower() == "true"
        
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
        # Optimized settings based on December 2025 testing
        cls.RESPEAKER_CONFIG = system_config.get("respeaker_config", {
            # AGC Settings - Unity gain prevents artifacts
            "agc_gain": 1.0,  # 0 dB - strictly unity gain
            "agc_on_off": 0,  # Keep AGC disabled
            
            # AEC Settings - Prevent filter saturation
            "aec_freeze_on_off": 0,  # Keep AEC adaptive
            "aecnorm": 16.0,  # Essential to prevent filter saturation
            
            # Echo Suppression - Safety margin against echo leakage
            "echo_on_off": 1,  # Enable echo suppression
            "gamma_e": 2.5,  # Increased from 2.0 for safety margin
            "gamma_enl": 3.0,  # Retain baseline
            "gamma_etail": 2.0,  # Handle internal reverberation
            
            # Noise Suppression - Retain baseline
            "stat_noise_on_off": 1,  # Enable stationary noise suppression
            "gamma_ns": 1.0,  # Sufficient suppression without distortion
            "min_ns": 0.15,  # -16 dB floor
            
            # Other Filters
            "hpf_on_off": 1,  # Enable high-pass filter
            "transientonoff": 1,  # Help with sudden noises
            
            # CRITICAL: NLAEC must remain disabled to prevent device failure
            "nlaec_mode": 0  # MUST be OFF (0) - enabling causes device bricking
        })
        
        # Note: ReSpeaker config is now applied by the Python client on startup (main.py)
        # We no longer need to save to file for the wrapper to read on next boot
        # Keeping _save_respeaker_config() method for backward compatibility / debugging
        # cls._save_respeaker_config()
        
        # Note: SAMPLE_RATE is hardcoded (16000 Hz) - not configurable
        
        # Save radio stations to local cache (for offline use and music playback)
        radio_stations = config_data.get("radio_stations")
        if radio_stations:
            cls._save_radio_cache(radio_stations)
        
        print(f"✓ Runtime configuration loaded")
        print(f"   Wake Word: {cls.WAKE_WORD}")
        print(f"   Sample Rate: {cls.SAMPLE_RATE} Hz")
        print(f"   OTEL Enabled: {cls.OTEL_ENABLED}")
        print(f"   LED Enabled: {cls.LED_ENABLED}")
        print(f"   Speaker Volume: {cls.SPEAKER_VOLUME_PERCENT}%")
        print(f"   Wake Word ASR Similarity Threshold: {cls.WAKE_WORD_ASR_SIMILARITY_THRESHOLD}")
        print(f"   Presence Detection Threshold: {cls.HUMAN_PRESENCE_DETECTION_SCORE_THRESHOLD}")
        print(f"   YAMNet Weights Loaded: {len(cls.YAMNET_WEIGHTS)} events")
        if radio_stations:
            total_stations = sum(len(v) for v in radio_stations.values())
            print(f"   Radio Stations: {total_stations} stations cached")
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
    
    @classmethod
    def _save_radio_cache(cls, stations: dict):
        """
        Save radio stations to local cache file for offline use.
        
        The music player (StationRegistry) reads from this file.
        Stations are fetched by the server and included in device config.
        
        Args:
            stations: Dict of stations by genre/category
        """
        import json
        
        cache_file = os.path.expanduser("~/.kin_radio_cache.json")
        
        try:
            with open(cache_file, 'w') as f:
                json.dump(stations, f, indent=2)
            # Silent save - main print statement shows station count
        except Exception as e:
            print(f"⚠️  Could not save radio cache: {e}")
    
    @classmethod
    def save_device_config_cache(cls, config_data: dict):
        """
        Save full device configuration to cache file.
        
        This allows the device to use cached config on boot if internet is unavailable.
        Cache includes skip_wifi_setup and other critical settings.
        
        Args:
            config_data: Full config dict from /auth/device/config endpoint
        """
        import json
        
        config_file = os.path.expanduser("~/.kin_config.json")
        
        try:
            with open(config_file, 'w') as f:
                json.dump(config_data, f, indent=2)
            print(f"✓ Device config cached to {config_file}")
        except Exception as e:
            print(f"⚠️  Could not save device config cache: {e}")
    
    @classmethod
    def load_device_config_cache(cls):
        """
        Load cached device configuration from file.
        
        Returns:
            dict: Cached config data, or None if cache doesn't exist or is invalid
        """
        import json
        
        config_file = os.path.expanduser("~/.kin_config.json")
        
        if not os.path.exists(config_file):
            return None
        
        try:
            with open(config_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️  Could not load device config cache: {e}")
            return None