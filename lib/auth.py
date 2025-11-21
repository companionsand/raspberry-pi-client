"""Supabase authentication"""

from supabase import create_client, Client
from lib.config import Config


def authenticate_with_supabase():
    """
    Authenticate with Supabase and fetch auth token and user ID.
    Sets Config.AUTH_TOKEN and Config.USER_ID on success.
    
    Returns:
        bool: True if authentication successful, False otherwise
    """
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

