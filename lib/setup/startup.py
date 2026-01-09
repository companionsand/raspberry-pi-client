"""
Startup Orchestration Module

Handles the device startup sequence including:
- Skip WiFi setup decision logic
- Connectivity checking
- Setup mode loop with retries
- Authentication orchestration
- "Unpaired" flow handling

This module consolidates the startup logic that was previously in main.py,
providing a clean interface for the main application to use.
"""

import os
import asyncio
import subprocess
from dataclasses import dataclass
from enum import Enum
from typing import Optional, List

from lib.config import Config
from lib.device_auth import authenticate
from .manager import SetupManager
from .connectivity import ConnectivityChecker


class StartupFailureReason(Enum):
    """Reasons why startup might fail."""
    NONE = "none"
    SETUP_UNAVAILABLE = "setup_unavailable"
    NO_INTERNET = "no_internet"
    AUTH_FAILED = "auth_failed"
    UNPAIRED = "unpaired"
    MAX_RETRIES = "max_retries"


@dataclass
class StartupResult:
    """Result of the startup sequence."""
    success: bool
    reason: StartupFailureReason = StartupFailureReason.NONE
    error_message: Optional[str] = None


def _determine_skip_wifi_setup(logger=None) -> tuple[bool, str]:
    """
    Determine if WiFi setup should be skipped.
    
    Priority: MAC_MODE → env var → cached config → default (false)
    
    Returns:
        Tuple of (skip_wifi_setup, config_source)
    """
    # MAC_MODE: Force skip WiFi setup (NetworkManager not available on Mac)
    if Config.MAC_MODE:
        print("ℹ️  MAC_MODE: Skipping WiFi setup")
        return True, "mac_mode"
    
    # Check env var override first (for testing/debugging)
    if os.getenv('SKIP_WIFI_SETUP'):
        env_skip_wifi = os.getenv('SKIP_WIFI_SETUP')
        skip_wifi_setup = env_skip_wifi.lower() == 'true'
        print(f"ℹ️  Using SKIP_WIFI_SETUP from environment: {skip_wifi_setup}")
        return skip_wifi_setup, "env_var"
    
    # Check cached config from last successful authentication
    cached_config = Config.load_device_config_cache()
    if cached_config and cached_config.get("device"):
        cached_skip = cached_config["device"].get("skip_wifi_setup", "false")
        skip_wifi_setup = cached_skip.lower() == 'true' if isinstance(cached_skip, str) else bool(cached_skip)
        print(f"ℹ️  Using skip_wifi_setup from cached config: {skip_wifi_setup}")
        return skip_wifi_setup, "cached_config"
    
    # Default: allow WiFi setup
    print("ℹ️  No cached config found, using default: skip_wifi_setup=False")
    return False, "default"


def _delete_wifi_connection(ssid: str) -> None:
    """
    Delete a WiFi connection to prevent device from getting stuck.
    
    CRITICAL: Must delete WiFi connection on auth failure, otherwise device gets stuck.
    If WiFi stays connected, main loop won't re-enter setup mode.
    But device isn't authenticated, so can't start normal operation.
    Result: Deadlock!
    """
    print(f"  Deleting WiFi connection: {ssid}")
    try:
        subprocess.run(
            ['sudo', 'nmcli', 'connection', 'delete', ssid],
            capture_output=True,
            timeout=5
        )
    except Exception as e:
        print(f"  Warning: Could not delete connection: {e}")


async def _cleanup_setup_manager(setup_manager: SetupManager) -> None:
    """Clean up setup manager HTTP server."""
    try:
        await setup_manager.http_server.stop()
    except Exception:
        pass


async def _run_setup_loop(
    led_controller,
    voice_feedback,
    user_terminate_flag: List[bool],
    max_attempts: int = 3,
    mode: str = "setup",  # "setup" or "pairing"
    logger=None
) -> StartupResult:
    """
    Run the setup/pairing loop with retries.
    
    Args:
        led_controller: LED controller for visual feedback
        voice_feedback: Voice feedback controller
        user_terminate_flag: Mutable flag for shutdown detection
        max_attempts: Maximum number of retry attempts
        mode: "setup" for full WiFi setup, "pairing" for pairing-only mode
        logger: Optional logger
        
    Returns:
        StartupResult indicating success/failure
    """
    is_pairing_only = mode == "pairing"
    
    for attempt in range(1, max_attempts + 1):
        mode_label = "pairing mode" if is_pairing_only else "setup mode"
        print(f"\n🔧 Entering {mode_label} (attempt {attempt}/{max_attempts})...")
        print("=" * 60)
        
        if is_pairing_only:
            print("ℹ️  Note: Pairing mode uses existing network connection.")
            print("   No sudo password required - device already has internet.")
        else:
            print("ℹ️  Note: Setup requires sudo privileges for network management.")
            print("   If prompted for a password, enter your Raspberry Pi user password.")
        print("=" * 60)
        
        # Start setup manager
        setup_manager = SetupManager(
            led_controller=led_controller,
            voice_feedback=voice_feedback
        )
        
        pairing_code, success = await setup_manager.start_setup_mode()
        
        if success and pairing_code:
            print(f"\n✓ {'Setup' if not is_pairing_only else 'Pairing'} complete!")
            print(f"  Pairing code received: {pairing_code}")
            print("=" * 60)
            
            print("\n🔐 Authenticating with pairing code...")
            
            auth_result = authenticate(pairing_code=pairing_code)
            if auth_result and auth_result.get("success"):
                print("✓ Authentication and pairing successful!")
                print("  Device is now paired and starting...")
                
                # Update LED controller with backend config
                if led_controller:
                    led_controller.enabled = Config.LED_ENABLED
                
                await _cleanup_setup_manager(setup_manager)
                return StartupResult(success=True)
            
            # Authentication failed
            print("\n✗ Authentication or pairing failed")
            
            # Clean up before potential retry
            await _cleanup_setup_manager(setup_manager)
            
            if attempt < max_attempts:
                print("  Possible reasons:")
                print("    - Incorrect pairing code")
                print("    - Pairing code expired or already used")
                print("    - Device not registered in admin portal")
                print("    - Backend service temporarily unavailable")
                print(f"\n  Cleaning up and restarting {mode_label}...")
                
                # Delete WiFi connection to prevent deadlock
                if setup_manager._wifi_credentials:
                    _delete_wifi_connection(setup_manager._wifi_credentials[0])
                
                # Clear credentials for fresh retry
                setup_manager._wifi_credentials = None
                setup_manager._pairing_code = None
                
                print("  Please reconnect to Kin_Setup and try again")
                await asyncio.sleep(3)
            else:
                print(f"  Max attempts ({max_attempts}) reached")
                await asyncio.sleep(3)
        else:
            # Setup itself failed (not auth)
            label = "Setup" if not is_pairing_only else "Pairing"
            print(f"\n✗ {label} failed or cancelled")
            
            await _cleanup_setup_manager(setup_manager)
            
            if attempt < max_attempts:
                print(f"  Retrying... (attempt {attempt + 1}/{max_attempts})")
                await asyncio.sleep(3)
    
    # All attempts exhausted
    print("\n✗ Setup and authentication failed after all attempts")
    print("  Device will retry on next boot")
    print("=" * 60)
    return StartupResult(
        success=False,
        reason=StartupFailureReason.MAX_RETRIES,
        error_message="Setup and authentication failed after all attempts"
    )


async def run_startup_sequence(
    led_controller,
    voice_feedback,
    user_terminate_flag: List[bool],
    logger=None
) -> StartupResult:
    """
    Run the complete startup sequence including connectivity check, setup, and authentication.
    
    This is the main entry point for the startup orchestration. It handles:
    1. Determining whether to skip WiFi setup
    2. Checking connectivity
    3. Running setup mode if needed (no internet or unpaired)
    4. Authenticating the device
    
    Args:
        led_controller: LED controller for visual feedback
        voice_feedback: Voice feedback controller
        user_terminate_flag: Mutable flag (list) for shutdown detection
        logger: Optional logger for structured logging
        
    Returns:
        StartupResult indicating success/failure and reason
    """
    # Determine skip_wifi_setup setting
    skip_wifi_setup, config_source = _determine_skip_wifi_setup(logger)
    
    # Log final skip_wifi_setup value for monitoring
    print("=" * 60)
    print(f"⚙️  SKIP_WIFI_SETUP: {skip_wifi_setup}")
    print(f"   Setup Mode: {'DISABLED' if skip_wifi_setup else 'ENABLED'}")
    print(f"   Config Source: {config_source}")
    print("=" * 60)
    
    if logger:
        logger.info(
            "skip_wifi_setup_config",
            extra={
                "skip_wifi_setup": skip_wifi_setup,
                "wifi_setup_mode": "disabled" if skip_wifi_setup else "enabled",
                "config_source": config_source
            }
        )
    
    # Setup Mode - only if enabled and available
    if not skip_wifi_setup:
        print("\n📡 Checking connectivity...")
        
        connectivity_checker = ConnectivityChecker()
        has_internet, orchestrator_reachable = await connectivity_checker.check_full_connectivity(
            orchestrator_retries=3
        )
        
        if not has_internet:
            print("✗ No internet connection detected")
            
            # Play voice feedback
            if voice_feedback:
                voice_feedback.play("no_internet")
            
            # Run full setup loop (WiFi + pairing)
            return await _run_setup_loop(
                led_controller=led_controller,
                voice_feedback=voice_feedback,
                user_terminate_flag=user_terminate_flag,
                mode="setup",
                logger=logger
            )
        
        # Has internet
        print("✓ Internet connection confirmed")
        if not orchestrator_reachable:
            print("⚠ Warning: Orchestrator is unreachable")
            print("  This may be due to:")
            print("    - Orchestrator service is down")
            print("    - Network/firewall issues")
            print("    - Incorrect orchestrator URL")
            print("  Device will continue but may have limited functionality")
        
        # Try to authenticate without pairing code (device might already be paired)
        auth_result = authenticate()
        
        if auth_result and auth_result.get("success"):
            # Already paired and authenticated
            print("✓ Device authenticated and paired")
            if led_controller:
                led_controller.enabled = Config.LED_ENABLED
            return StartupResult(success=True)
        
        # Authentication failed - check the reason
        reason = auth_result.get("reason") if auth_result else "unknown"
        
        if reason == "unpaired":
            # Device is authenticated but not paired - enter pairing mode
            print("⚠️  Device is not paired with a user")
            print("   Entering pairing mode to collect pairing code...")
            print("   Note: Device already has internet - no WiFi setup needed (no sudo required)")
            
            if voice_feedback:
                voice_feedback.play("device_not_paired")
            
            # Run pairing-only loop
            return await _run_setup_loop(
                led_controller=led_controller,
                voice_feedback=voice_feedback,
                user_terminate_flag=user_terminate_flag,
                mode="pairing",
                logger=logger
            )
        
        # Other authentication error - exit with error
        print(f"✗ Failed to authenticate device (reason: {reason})")
        return StartupResult(
            success=False,
            reason=StartupFailureReason.AUTH_FAILED,
            error_message=f"Authentication failed: {reason}"
        )
    
    # Setup skipped or unavailable - authenticate directly
    auth_result = authenticate()
    if not auth_result or not auth_result.get("success"):
        reason = auth_result.get("reason") if auth_result else "unknown"
        print(f"✗ Failed to authenticate device (reason: {reason})")
        return StartupResult(
            success=False,
            reason=StartupFailureReason.AUTH_FAILED,
            error_message=f"Authentication failed: {reason}"
        )
    
    # Update LED controller with backend config
    if led_controller:
        led_controller.enabled = Config.LED_ENABLED
    
    return StartupResult(success=True)

