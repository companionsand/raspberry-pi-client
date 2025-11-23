"""Device authentication using Ed25519 public key cryptography"""

import base64
import requests
from typing import Optional, Dict, Any
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519

from lib.config import Config


def load_private_key(private_key_base64: str) -> ed25519.Ed25519PrivateKey:
    """
    Load Ed25519 private key from base64 string.
    
    Args:
        private_key_base64: Base64-encoded private key
        
    Returns:
        Ed25519PrivateKey object
    """
    private_key_bytes = base64.b64decode(private_key_base64)
    return ed25519.Ed25519PrivateKey.from_private_bytes(private_key_bytes)


def authenticate_device() -> Optional[Dict[str, Any]]:
    """
    Authenticate device using challenge-response mechanism.
    
    Returns:
        Dict with authentication info:
        - jwt_token: JWT token for API access
        - device: Device information
        - config: Runtime configuration (if paired)
        
        Returns None if authentication fails
    """
    logger = Config.LOGGER
    print("\n🔐 Authenticating device...")
    
    try:
        orchestrator_base_url = Config.ORCHESTRATOR_URL.replace("ws://", "http://").replace("wss://", "https://").replace("/ws", "")
        
        # Step 1: Request challenge
        print(f"   Requesting challenge for device {Config.DEVICE_ID}...")
        challenge_response = requests.post(
            f"{orchestrator_base_url}/auth/device/challenge",
            json={"device_id": Config.DEVICE_ID},
            timeout=10
        )
        
        if challenge_response.status_code == 404:
            print(f"✗ Device not found: {Config.DEVICE_ID}")
            print("   Please provision this device through the admin portal.")
            return None
            
        challenge_response.raise_for_status()
        challenge_data = challenge_response.json()
        
        challenge = challenge_data["challenge"]
        timestamp = challenge_data["timestamp"]
        
        print(f"   ✓ Challenge received")
        
        # Step 2: Sign challenge with private key
        print("   Signing challenge...")
        private_key = load_private_key(Config.DEVICE_PRIVATE_KEY)
        
        # Message is: challenge:timestamp
        message = f"{challenge}:{timestamp}".encode()
        signature = private_key.sign(message)
        signature_b64 = base64.b64encode(signature).decode()
        
        # Step 3: Verify signature and get JWT
        print("   Verifying signature...")
        verify_response = requests.post(
            f"{orchestrator_base_url}/auth/device/verify",
            json={
                "device_id": Config.DEVICE_ID,
                "challenge": challenge,
                "signature": signature_b64
            },
            timeout=10
        )
        
        verify_response.raise_for_status()
        verify_data = verify_response.json()
        
        jwt_token = verify_data["jwt_token"]
        device_info = verify_data["device"]
        
        print(f"✓ Device authenticated successfully")
        print(f"   Device: {device_info.get('device_name', 'Unknown')}")
        print(f"   User ID: {device_info.get('user_id', 'Not paired')}")
        
        if logger:
            logger.info(
                "device_authenticated",
                extra={
                    "device_id": Config.DEVICE_ID,
                    "user_id": device_info.get("user_id"),
                    "device_name": device_info.get("device_name")
                }
            )
        
        # Step 4: Fetch runtime configuration
        print("   Fetching runtime configuration...")
        config_response = requests.get(
            f"{orchestrator_base_url}/auth/device/config",
            headers={"Authorization": f"Bearer {jwt_token}"},
            timeout=10
        )
        
        config_response.raise_for_status()
        config_data = config_response.json()
        
        # Check if device is paired
        if not config_data.get("paired"):
            print(f"\n⚠️  {config_data.get('error_message', 'Device not paired with a user')}")
            if config_data.get("admin_portal_url"):
                print(f"   Admin Portal: {config_data['admin_portal_url']}")
            return None
        
        print(f"✓ Configuration loaded")
        
        return {
            "jwt_token": jwt_token,
            "device": device_info,
            "config": config_data
        }
        
    except requests.exceptions.RequestException as e:
        print(f"✗ Authentication failed: {e}")
        if logger:
            logger.error(
                "device_authentication_failed",
                extra={"error": str(e)},
                exc_info=True
            )
        return None
    except Exception as e:
        print(f"✗ Authentication error: {e}")
        if logger:
            logger.error(
                "device_authentication_error",
                extra={"error": str(e)},
                exc_info=True
            )
        return None


def get_serial_number() -> Optional[str]:
    """
    Get Raspberry Pi serial number from /proc/cpuinfo.
    
    Returns:
        Serial number string or None if not found
    """
    try:
        with open('/proc/cpuinfo', 'r') as f:
            for line in f:
                if line.startswith('Serial'):
                    return line.split(':')[1].strip()
    except Exception:
        pass
    return None

