"""Configuration management from environment variables"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config:
    """Application configuration from environment variables"""
    
    # Device credentials
    DEVICE_ID = os.getenv("DEVICE_ID")
    
    # Logger (will be set after device_id is validated and telemetry is initialized)
    LOGGER = None
    
    # Supabase authentication
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")
    EMAIL = os.getenv("EMAIL")
    PASSWORD = os.getenv("PASSWORD")
    
    # These will be set after authentication
    USER_ID = None
    AUTH_TOKEN = None
    
    # Backend
    CONVERSATION_ORCHESTRATOR_URL = os.getenv("CONVERSATION_ORCHESTRATOR_URL", "ws://localhost:8001/ws")
    
    # ElevenLabs API
    ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
    
    # Wake word detection
    PICOVOICE_ACCESS_KEY = os.getenv("PICOVOICE_ACCESS_KEY")
    WAKE_WORD = os.getenv("WAKE_WORD", "porcupine")
    
    # Audio settings (ALSA-only, single ReSpeaker device for both capture and playback)
    SAMPLE_RATE = 16000  # 16kHz for both capture and playback
    CHANNELS = 1  # Mono (ReSpeaker AEC expects mono reference)
    CHUNK_SIZE = 512  # ~32ms frames for low latency
    
    # LED settings (optional - for ReSpeaker visual feedback)
    LED_ENABLED = os.getenv("LED_ENABLED", "true").lower() == "true"
    LED_BRIGHTNESS = int(os.getenv("LED_BRIGHTNESS", "60"))  # 0-100, default 60% for subtlety
    
    # OpenTelemetry
    OTEL_ENABLED = os.getenv("OTEL_ENABLED", "true").lower() == "true"
    OTEL_EXPORTER_ENDPOINT = os.getenv("OTEL_EXPORTER_ENDPOINT", "http://localhost:4318")
    ENV = os.getenv("ENV", "production")
    
    @classmethod
    def validate(cls):
        """Validate required configuration"""
        missing = []
        
        if not cls.DEVICE_ID:
            missing.append("DEVICE_ID")
        if not cls.SUPABASE_URL:
            missing.append("SUPABASE_URL")
        if not cls.SUPABASE_ANON_KEY:
            missing.append("SUPABASE_ANON_KEY")
        if not cls.EMAIL:
            missing.append("EMAIL")
        if not cls.PASSWORD:
            missing.append("PASSWORD")
        if not cls.ELEVENLABS_API_KEY:
            missing.append("ELEVENLABS_API_KEY")
        if not cls.PICOVOICE_ACCESS_KEY:
            missing.append("PICOVOICE_ACCESS_KEY")
            
        if missing:
            print(f"✗ Missing required environment variables: {', '.join(missing)}")
            print("Create a .env file with required credentials")
            sys.exit(1)

