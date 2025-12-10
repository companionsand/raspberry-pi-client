"""Device authentication using Ed25519 public key cryptography"""

from typing import Optional, Dict, Any
from lib.config import Config
from lib.device_auth import authenticate_device


def authenticate(pairing_code: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Authenticate device using device auth system.
    
    Args:
        pairing_code: Optional pairing code from WiFi setup (kept in memory only)
    
    Sets Config.AUTH_TOKEN, Config.USER_ID, and runtime configuration on success.
    
    Returns:
        Dict with authentication status:
        - success: True if fully authenticated and paired
        - jwt_token, device, config: Present if success=True
        - reason: Reason for failure if success=False
          - "unpaired": Device authenticated but not paired with user
          - "not_found": Device ID not found in system
          - "network_error": Network/connection error
          - "auth_error": Authentication failed
        
        Returns None if no credentials found
    """
    if not Config.DEVICE_PRIVATE_KEY:
        print("✗ No DEVICE_PRIVATE_KEY found")
        print("   Device needs to be provisioned with credentials")
        print("   Download the installer package from admin portal and run ./install.sh")
        return {"success": False, "reason": "no_credentials"}
    
    auth_result = authenticate_device(pairing_code=pairing_code)
    
    if auth_result and auth_result.get("success"):
        # Set auth token and user ID
        Config.AUTH_TOKEN = auth_result["jwt_token"]
        Config.USER_ID = auth_result["device"]["user_id"]
        
        # Load runtime configuration
        Config.load_runtime_config(auth_result["config"])
    
    return auth_result

