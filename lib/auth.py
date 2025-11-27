"""Device authentication using Ed25519 public key cryptography"""

from typing import Optional
from lib.config import Config
from lib.device_auth import authenticate_device


def authenticate(pairing_code: Optional[str] = None) -> bool:
    """
    Authenticate device using device auth system.
    
    Args:
        pairing_code: Optional pairing code from WiFi setup (kept in memory only)
    
    Sets Config.AUTH_TOKEN, Config.USER_ID, and runtime configuration on success.
    
    Returns:
        bool: True if authentication successful, False otherwise
    """
    if not Config.DEVICE_PRIVATE_KEY:
        print("✗ No DEVICE_PRIVATE_KEY found")
        print("   Device needs to be provisioned with credentials")
        print("   Download the installer package from admin portal and run ./install.sh")
        return False
    
    auth_result = authenticate_device(pairing_code=pairing_code)
    
    if auth_result:
        # Set auth token and user ID
        Config.AUTH_TOKEN = auth_result["jwt_token"]
        Config.USER_ID = auth_result["device"]["user_id"]
        
        # Load runtime configuration
        Config.load_runtime_config(auth_result["config"])
        
        return True
    else:
        return False

