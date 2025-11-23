"""Authentication - supports both device auth and legacy Supabase auth"""

import os
from typing import Optional
from lib.config import Config
from lib.device_auth import authenticate_device


def authenticate() -> bool:
    """
    Authenticate device using the new device auth system.
    Falls back to legacy Supabase auth if configured.
    
    Sets Config.AUTH_TOKEN, Config.USER_ID, and runtime configuration on success.
    
    Returns:
        bool: True if authentication successful, False otherwise
    """
    logger = Config.LOGGER
    
    # Try device authentication first
    if Config.DEVICE_PRIVATE_KEY:
        auth_result = authenticate_device()
        
        if auth_result:
            # Set auth token and user ID
            Config.AUTH_TOKEN = auth_result["jwt_token"]
            Config.USER_ID = auth_result["device"]["user_id"]
            
            # Load runtime configuration
            Config.load_runtime_config(auth_result["config"])
            
            return True
        else:
            return False
    
    # Fall back to legacy Supabase authentication (for backward compatibility)
    elif Config.SUPABASE_URL and Config.EMAIL and Config.PASSWORD:
        print("\n⚠️  Using legacy Supabase authentication")
        print("   Consider reprovisioning this device for enhanced security")
        return authenticate_with_supabase_legacy()
    
    else:
        print("✗ No authentication method available")
        print("   Device needs to be provisioned with DEVICE_PRIVATE_KEY")
        return False


def authenticate_with_supabase_legacy():
    """
    Legacy Supabase authentication (for backward compatibility).
    
    Returns:
        bool: True if authentication successful, False otherwise
    """
    try:
        from supabase import create_client, Client
    except ImportError:
        print("✗ Supabase package not installed")
        return False
    
    logger = Config.LOGGER
    print("\n🔐 Authenticating with Supabase...")
    
    try:
        # Create Supabase client
        supabase: Client = create_client(Config.SUPABASE_URL, Config.SUPABASE_ANON_KEY)
        
        # Sign in with email and password
        response = supabase.auth.sign_in_with_password({
            "email": Config.EMAIL,
            "password": Config.PASSWORD
        })
        
        # Extract auth token and user ID
        if response.user and response.session:
            Config.AUTH_TOKEN = response.session.access_token
            Config.USER_ID = response.user.id
            
            # Need to set runtime config from env vars for legacy mode
            Config.ELEVENLABS_API_KEY = Config.ELEVENLABS_API_KEY or os.getenv("ELEVENLABS_API_KEY")
            Config.PICOVOICE_ACCESS_KEY = Config.PICOVOICE_ACCESS_KEY or os.getenv("PICOVOICE_ACCESS_KEY")
            Config.WAKE_WORD = Config.WAKE_WORD or os.getenv("WAKE_WORD", "porcupine")
            Config.LED_ENABLED = os.getenv("LED_ENABLED", "true").lower() == "true"
            Config.OTEL_ENABLED = os.getenv("OTEL_ENABLED", "true").lower() == "true"
            Config.OTEL_EXPORTER_ENDPOINT = os.getenv("OTEL_EXPORTER_ENDPOINT", "http://localhost:4318")
            
            print(f"✓ Successfully authenticated")
            print(f"   User ID: {Config.USER_ID}")
            if logger:
                logger.info(
                    "supabase_authenticated",
                    extra={
                        "user_id": Config.USER_ID,
                        "device_id": Config.DEVICE_ID
                    }
                )
            return True
        else:
            print("✗ Authentication failed: No user or session returned")
            if logger:
                logger.error("supabase_authentication_failed", extra={"reason": "no_user_or_session"})
            return False
            
    except Exception as e:
        print(f"✗ Authentication error: {e}")
        if logger:
            logger.error(
                "supabase_authentication_error",
                extra={"error": str(e)},
                exc_info=True
            )
        return False


# For backward compatibility
authenticate_with_supabase = authenticate_with_supabase_legacy

