"""Device authentication using Ed25519 public key cryptography"""

from lib.config import Config
from lib.device_auth import authenticate_device


def authenticate() -> bool:
    """
    Authenticate device using device auth system.
    
    Sets Config.AUTH_TOKEN, Config.USER_ID, and runtime configuration on success.
    
    Returns:
        bool: True if authentication successful, False otherwise
    """
    if not Config.DEVICE_PRIVATE_KEY:
        print("✗ No DEVICE_PRIVATE_KEY found")
        print("   Device needs to be provisioned with credentials")
        print("   Download the installer package from admin portal and run ./install.sh")
        return False
    
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

