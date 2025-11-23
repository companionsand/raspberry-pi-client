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
    ORCHESTRATOR_URL = os.getenv("CONVERSATION_ORCHESTRATOR_URL", "ws://localhost:8001/ws")
    
    # Audio settings (ALSA-only, single ReSpeaker device for both capture and playback)
    SAMPLE_RATE = 16000  # 16kHz for both capture and playback (may be overridden by backend)
    CHANNELS = 1  # Mono (ReSpeaker AEC expects mono reference)
    CHUNK_SIZE = 512  # ~32ms frames for low latency
    
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
    OTEL_ENABLED = None
    OTEL_EXPORTER_ENDPOINT = None
    
    # Environment
    ENV = os.getenv("ENV", "production")
    
    # =========================================================================
    # LEGACY SUPPORT (for backward compatibility with old .env files)
    # =========================================================================
    # If these are set in .env, use them (for testing/development)
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")
    EMAIL = os.getenv("EMAIL")
    PASSWORD = os.getenv("PASSWORD")
    
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
        
        # Set device info
        cls.USER_ID = device_config.get("user_id")
        
        # Set API keys
        cls.PICOVOICE_ACCESS_KEY = api_keys.get("picovoice_access_key") or device_config.get("picovoice_access_key")
        cls.ELEVENLABS_API_KEY = system_config.get("elevenlabs_api_key")
        
        # Set system settings
        cls.WAKE_WORD = system_config.get("wake_word", "porcupine")
        cls.LED_ENABLED = system_config.get("led_enabled", "true").lower() == "true"
        cls.OTEL_ENABLED = system_config.get("otel_enabled", "true").lower() == "true"
        cls.OTEL_EXPORTER_ENDPOINT = system_config.get("otel_endpoint", "http://localhost:4318")
        
        # Override sample rate if provided
        sample_rate_str = system_config.get("sample_rate")
        if sample_rate_str:
            try:
                cls.SAMPLE_RATE = int(sample_rate_str)
            except (ValueError, TypeError):
                pass
        
        print(f"✓ Runtime configuration loaded")
        print(f"   Wake Word: {cls.WAKE_WORD}")
        print(f"   OTEL Enabled: {cls.OTEL_ENABLED}")
        print(f"   LED Enabled: {cls.LED_ENABLED}")

