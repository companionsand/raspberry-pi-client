"""Device authentication using Ed25519 public key cryptography"""

import time
import jwt
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
        
        # Decode JWT to get expiry time (don't verify signature - we trust our own auth)
        try:
            decoded = jwt.decode(
                auth_result["jwt_token"],
                options={"verify_signature": False}  # No need to verify - we just got it from our server
            )
            Config.AUTH_TOKEN_EXPIRES_AT = decoded.get("exp")
            
            if Config.AUTH_TOKEN_EXPIRES_AT:
                time_until_expiry = Config.AUTH_TOKEN_EXPIRES_AT - time.time()
                print(f"   Token expires in {int(time_until_expiry/60)} minutes")
        except Exception as e:
            print(f"⚠️  Could not decode token expiry: {e}")
            # Fallback to 1 hour assumption if decode fails
            Config.AUTH_TOKEN_EXPIRES_AT = time.time() + 3600
        
        # Load runtime configuration
        Config.load_runtime_config(auth_result["config"])
    
    return auth_result

