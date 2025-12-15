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


def authenticate_device(pairing_code: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Authenticate device using challenge-response mechanism.
    
    Args:
        pairing_code: Optional pairing code from WiFi setup (in memory only)
    
    Returns:
        Dict with authentication info:
        - success: True if fully authenticated and paired
        - jwt_token: JWT token for API access (if success=True)
        - device: Device information (if success=True)
        - config: Runtime configuration (if success=True and paired)
        - reason: Reason for failure (if success=False)
          - "unpaired": Device authenticated but not paired with user
          - "not_found": Device ID not found in system
          - "network_error": Network/connection error
          - "auth_error": Authentication failed (wrong keys, etc.)
        
        Returns None for backward compatibility with old error cases
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
            return {"success": False, "reason": "not_found"}
            
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
            
            # Try to pair using pairing code if provided
            if pairing_code:
                print(f"   Found pairing code, attempting to pair device...")
                serial_number = get_serial_number()
                if serial_number:
                    if _pair_device(pairing_code, serial_number, orchestrator_base_url):
                        print("✓ Device paired successfully, retrying authentication...")
                        # Retry authentication after pairing (no need to pass pairing_code again)
                        return authenticate_device()
                    else:
                        print("✗ Failed to pair device with pairing code")
                else:
                    print("✗ Could not read device serial number for pairing")
            else:
                print("   No pairing code found. Device needs to be paired through WiFi setup.")
            
            if config_data.get("admin_portal_url"):
                print(f"   Admin Portal: {config_data['admin_portal_url']}")
            return {"success": False, "reason": "unpaired"}
        
        print(f"✓ Configuration loaded")
        
        # Cache the configuration for use on next boot (in case of no internet)
        Config.save_device_config_cache(config_data)
        
        return {
            "success": True,
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
        return {"success": False, "reason": "network_error", "error": str(e)}
    except Exception as e:
        print(f"✗ Authentication error: {e}")
        if logger:
            logger.error(
                "device_authentication_error",
                extra={"error": str(e)},
                exc_info=True
            )
        return {"success": False, "reason": "auth_error", "error": str(e)}


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




def _pair_device(pairing_code: str, serial_number: str, orchestrator_base_url: str) -> bool:
    """
    Pair device with user using pairing code and serial number.
    
    Args:
        pairing_code: 4-digit pairing code
        serial_number: Device serial number
        orchestrator_base_url: Base URL of conversation orchestrator
        
    Returns:
        True if pairing successful, False otherwise
    """
    logger = Config.LOGGER
    try:
        print(f"   Pairing device with code {pairing_code}...")
        response = requests.post(
            f"{orchestrator_base_url}/pairing/pair",
            json={
                "pairing_code": pairing_code,
                "serial_number": serial_number
            },
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            if result.get("success"):
                print(f"✓ Device paired successfully")
                # Remove pairing code file after successful pairing
                try:
                    import os
                    os.remove('/tmp/kin_pairing_code')
                except:
                    pass
                
                if logger:
                    logger.info(
                        "device_paired",
                        extra={
                            "device_id": Config.DEVICE_ID,
                            "serial_number": serial_number,
                            "pairing_code": pairing_code
                        }
                    )
                return True
            else:
                print(f"✗ Pairing failed: {result.get('message', 'Unknown error')}")
        else:
            error_data = response.json() if response.content else {}
            error_msg = error_data.get("detail", f"HTTP {response.status_code}")
            print(f"✗ Pairing failed: {error_msg}")
            
    except requests.exceptions.RequestException as e:
        print(f"✗ Pairing request failed: {e}")
        if logger:
            logger.error(
                "device_pairing_failed",
                extra={"error": str(e), "pairing_code": pairing_code},
                exc_info=True
            )
    except Exception as e:
        print(f"✗ Pairing error: {e}")
        if logger:
            logger.error(
                "device_pairing_error",
                extra={"error": str(e), "pairing_code": pairing_code},
                exc_info=True
            )
    
    return False

