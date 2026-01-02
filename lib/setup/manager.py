"""
Setup Manager

Orchestrates the device setup process including:
- Creating a WiFi access point
- Running HTTP server for configuration
- Handling WiFi connection attempts
- Validating internet connectivity
- Collecting device pairing code

The setup interface handles both WiFi setup and device pairing:
- WiFi Setup: Provides network connectivity (only needed when device has no internet)
- Device Pairing: Links device to user account (always needed if device is unpaired)
"""

import asyncio
import logging
from typing import Optional, Tuple, Callable

from .access_point import AccessPoint
from .http_server import SetupHTTPServer
from .network_connector import NetworkConnector
from .connectivity import ConnectivityChecker
from .constants import (
    CONFIG_TIMEOUT,
    USER_READ_DELAY,
    VOICE_FEEDBACK_DELAY,
    ERROR_DISPLAY_DELAY,
    RETRY_DELAY,
    LOG_INTERVAL,
    STATUS_CONNECTING,
    STATUS_ERROR,
)

logger = logging.getLogger(__name__)


class SetupManager:
    """Manages the complete device setup flow (WiFi setup and device pairing)"""
    
    def __init__(
        self,
        ap_ssid: str = "Kin_Setup",
        ap_password: str = "kinsetup123",
        ap_interface: str = "wlan0",
        http_port: int = 8080,
        max_retries: int = 5,
        led_controller = None,
        voice_feedback = None,
        shutdown_flag = None
    ):
        self.ap_ssid = ap_ssid
        self.ap_password = ap_password
        self.ap_interface = ap_interface
        self.http_port = http_port
        self.max_retries = max_retries
        self.led_controller = led_controller
        self.voice_feedback = voice_feedback
        self.shutdown_flag = shutdown_flag  # Reference to shutdown flag from main client
        
        self.access_point = AccessPoint(ap_ssid, ap_interface, ap_password)
        self.http_server = SetupHTTPServer(http_port, ap_interface, ap_ssid, ap_password)
        self.network_connector = NetworkConnector(ap_interface)
        self.connectivity_checker = ConnectivityChecker()
        
        self._pairing_code: Optional[str] = None
        self._wifi_credentials: Optional[Tuple[str, str]] = None
        
        # Import LEDController once for reuse
        self._LEDController = None
        try:
            from ..audio import LEDController
            self._LEDController = LEDController
        except ImportError:
            pass
    
    async def start_setup_mode(self) -> Tuple[str, bool]:
        """
        Start setup mode and wait for configuration.
        
        If device already has internet, skips WiFi setup (no sudo required) and only
        collects pairing code via existing network. Otherwise, creates access point
        for full WiFi setup.
        
        Returns:
            Tuple of (pairing_code, success)
        """
        logger.info("Starting setup mode...")
        
        # Clean up any old pairing code files and processes from previous runs
        await self._cleanup_old_state()
        
        # Check if device already has internet connectivity
        logger.info("Checking internet connectivity...")
        has_internet = await self.connectivity_checker.check_internet()
        
        if has_internet:
            logger.info("✓ Device already has internet connection")
            logger.info("Skipping WiFi setup - only collecting pairing code")
            return await self._start_pairing_only_mode()
        
        logger.info("✗ No internet connection - full WiFi setup required")
        return await self._start_full_setup_mode()
    
    def _set_led_state(self, state: str):
        """Set LED controller state if available."""
        if self.led_controller and self._LEDController:
            self.led_controller.set_state(state)
    
    def _play_voice_feedback(self, message: str):
        """Play voice feedback if available."""
        if self.voice_feedback:
            self.voice_feedback.play(message)
    
    def _check_shutdown(self) -> bool:
        """Check if shutdown has been requested."""
        return self.shutdown_flag and self.shutdown_flag[0]
    
    async def _run_with_retries(
        self,
        operation: Callable,
        operation_name: str,
        check_success: Optional[Callable] = None
    ) -> Tuple[Optional[str], bool]:
        """
        Run an operation with retry logic.
        
        Args:
            operation: Async function to run that returns (pairing_code, success)
            operation_name: Name of operation for logging
            check_success: Optional function to check if result indicates success
            
        Returns:
            Tuple of (pairing_code, success)
        """
        retry_count = 0
        
        while retry_count < self.max_retries:
            if self._check_shutdown():
                logger.info("Shutdown requested, stopping setup...")
                await self._cleanup()
                return None, False
            
            try:
                await self._cleanup()
                result = await operation()
                
                # Check if operation was successful
                if check_success:
                    is_success = check_success(result)
                else:
                    # Default: check if result is a tuple with success=True and pairing_code
                    if isinstance(result, tuple) and len(result) == 2:
                        pairing_code, success = result
                        is_success = success and pairing_code is not None
                    else:
                        is_success = False
                
                if is_success:
                    # Return the result tuple
                    if isinstance(result, tuple) and len(result) == 2:
                        return result
                    return result, True
                
                retry_count += 1
                if retry_count < self.max_retries:
                    logger.info(f"Retrying {operation_name}... ({retry_count}/{self.max_retries})")
                    await asyncio.sleep(RETRY_DELAY)
                
            except Exception as e:
                logger.error(f"Error during {operation_name}: {e}", exc_info=True)
                retry_count += 1
                if retry_count < self.max_retries:
                    await asyncio.sleep(RETRY_DELAY)
        
        logger.error(f"{operation_name} failed after {self.max_retries} attempts")
        await self._cleanup()
        return None, False
    
    async def _start_pairing_only_mode(self) -> Tuple[str, bool]:
        """
        Start pairing-only mode (device has internet, no WiFi setup needed).
        Uses existing network connection - no sudo required.
        
        Returns:
            Tuple of (pairing_code, success)
        """
        logger.info("Starting pairing-only mode...")
        
        async def pairing_operation():
            # Get device IP address on existing network
            device_ip = await self.network_connector.get_device_ip()
            if not device_ip:
                logger.error("Could not determine device IP address")
                return None, False
            
            # Start HTTP server for pairing code collection (no AP needed)
            logger.info(f"Starting HTTP server on port {self.http_port} (no access point needed)")
            await self.http_server.start(self._handle_pairing_only_config, pairing_only=True, device_ip=device_ip)
            
            # Set LED state to WIFI_SETUP (ready for configuration)
            if self._LEDController:
                self._set_led_state(self._LEDController.STATE_WIFI_SETUP)
            
            logger.info(f"Pairing mode active. Open http://{device_ip}:{self.http_port} in your browser")
            logger.info("Note: No WiFi setup needed - device already has internet connection")
            
            # Don't play WiFi setup message in pairing-only mode (no WiFi setup needed)
            # Voice feedback already played earlier (device_not_paired)
            
            # Wait for user to submit pairing code
            success = await self._wait_for_pairing_code()
            
            if success and self._pairing_code:
                logger.info("✓ Pairing code received successfully")
                return self._pairing_code, True
            else:
                logger.warning("No pairing code received")
                return None, False
        
        return await self._run_with_retries(
            pairing_operation,
            "pairing code collection",
            lambda result: result is not None and result[0] is not None and result[1]
        )
    
    async def _start_full_setup_mode(self) -> Tuple[str, bool]:
        """
        Start full setup mode (no internet - WiFi setup required).
        Creates access point and requires sudo for network management.
        
        Returns:
            Tuple of (pairing_code, success)
        """
        logger.info("Starting full WiFi setup mode...")
        
        async def full_setup_operation():
            # Create access point (requires sudo)
            logger.info(f"Creating access point: {self.ap_ssid}")
            await self.access_point.start()
            
            # Start HTTP server for configuration
            logger.info(f"Starting HTTP server on port {self.http_port}")
            await self.http_server.start(self._handle_wifi_config)
            
            # Set LED state to WIFI_SETUP (ready for configuration)
            if self._LEDController:
                self._set_led_state(self._LEDController.STATE_WIFI_SETUP)
            
            logger.info(f"Setup mode active. Connect to '{self.ap_ssid}' (password: {self.ap_password}) and go to http://192.168.4.1:{self.http_port}")
            
            # Play voice feedback to guide user
            self._play_voice_feedback("wifi_setup_ready")
            
            # Wait for user to submit configuration
            success = await self._wait_for_configuration()
            
            if success:
                return await self._handle_wifi_connection()
            
            return None, False
        
        return await self._run_with_retries(
            full_setup_operation,
            "WiFi setup",
            lambda result: result is not None and result[0] is not None and result[1]
        )
    
    async def _handle_pairing_only_config(self, ssid: str, password: str, pairing_code: str) -> bool:
        """
        Callback for pairing-only mode (device has internet, no WiFi setup needed).
        Only pairing code is required; WiFi credentials are ignored.
        
        Args:
            ssid: Ignored (not needed when device has internet)
            password: Ignored (not needed when device has internet)
            pairing_code: Device pairing code (4 digits) for user account pairing
            
        Returns:
            True if pairing code was accepted
        """
        logger.info("=" * 60)
        logger.info("[Setup] ✓ Pairing code received from web interface")
        logger.info(f"[Setup]   Pairing code: {pairing_code}")
        logger.info("[Setup]   Note: WiFi credentials ignored (device already has internet)")
        logger.info("=" * 60)
        
        # Only store pairing code, ignore WiFi credentials
        self._pairing_code = pairing_code
        return True
    
    async def _wait_for_input(
        self,
        check_ready: Callable[[], bool],
        input_name: str,
        timeout: int = CONFIG_TIMEOUT
    ) -> bool:
        """
        Wait for user input through web interface.
        
        Args:
            check_ready: Function that returns True when input is ready
            input_name: Name of input being waited for (for logging)
            timeout: Maximum time to wait in seconds
            
        Returns:
            True if input was received
        """
        start_time = asyncio.get_event_loop().time()
        last_log_time = start_time
        
        logger.info(f"[Setup] Waiting for {input_name} (timeout: {timeout}s)")
        
        while True:
            if self._check_shutdown():
                logger.info(f"[Setup] Shutdown requested, stopping {input_name} wait...")
                return False
            
            if check_ready():
                elapsed = asyncio.get_event_loop().time() - start_time
                logger.info(f"[Setup] {input_name.capitalize()} received after {elapsed:.0f}s")
                return True
            
            elapsed = asyncio.get_event_loop().time() - start_time
            
            # Log progress periodically
            if elapsed - (last_log_time - start_time) >= LOG_INTERVAL:
                remaining = timeout - elapsed
                logger.info(f"[Setup] Still waiting for {input_name}... ({remaining:.0f}s remaining)")
                last_log_time = asyncio.get_event_loop().time()
            
            if elapsed >= timeout:
                logger.warning(f"[Setup] {input_name.capitalize()} timeout reached after {timeout}s")
                return False
            
            await asyncio.sleep(1)
    
    async def _wait_for_pairing_code(self, timeout: int = CONFIG_TIMEOUT) -> bool:
        """Wait for user to submit pairing code."""
        return await self._wait_for_input(
            lambda: self._pairing_code is not None,
            "pairing code",
            timeout
        )
    
    async def _wait_for_configuration(self, timeout: int = CONFIG_TIMEOUT) -> bool:
        """Wait for user to submit full configuration (WiFi + pairing code)."""
        return await self._wait_for_input(
            lambda: self._wifi_credentials is not None and self._pairing_code is not None,
            "configuration",
            timeout
        )
    
    async def _handle_wifi_config(self, ssid: str, password: str, pairing_code: str) -> bool:
        """
        Callback for when user submits configuration via web interface.
        
        Note: This collects both WiFi credentials AND pairing code, but they serve
        different purposes:
        - WiFi credentials: For network connectivity (only needed if no internet)
        - Pairing code: For linking device to user account (always needed if unpaired)
        
        Args:
            ssid: WiFi network name (SSID)
            password: WiFi password
            pairing_code: Device pairing code (4 digits) for user account pairing
            
        Returns:
            True if configuration was accepted
        """
        logger.info("=" * 60)
        logger.info("[Setup] ✓ Configuration received from web interface")
        logger.info(f"[Setup]   Target SSID: {ssid}")
        logger.info(f"[Setup]   Password length: {len(password)} chars")
        logger.info(f"[Setup]   Pairing code: {pairing_code}")
        logger.info("=" * 60)
        
        self._wifi_credentials = (ssid, password)
        self._pairing_code = pairing_code
        return True
    
    async def _wait_for_configuration(self, timeout: int = 300) -> bool:
        """
        Wait for user to submit configuration through web interface.
        
        Args:
            timeout: Maximum time to wait in seconds (default 5 minutes)
            
        Returns:
            True if configuration was received
        """
        start_time = asyncio.get_event_loop().time()
        last_log_time = start_time
        log_interval = 30  # Log every 30 seconds
        
        logger.info(f"[Setup] Waiting for configuration (timeout: {timeout}s)")
        
        while True:
            # Check for shutdown request
            if self.shutdown_flag and self.shutdown_flag[0]:
                logger.info("[Setup] Shutdown requested, stopping configuration wait...")
                return False
            
            if self._wifi_credentials and self._pairing_code:
                elapsed = asyncio.get_event_loop().time() - start_time
                logger.info(f"[Setup] Configuration received after {elapsed:.0f}s")
                return True
            
            elapsed = asyncio.get_event_loop().time() - start_time
            
            # Log progress periodically
            if elapsed - (last_log_time - start_time) >= log_interval:
                remaining = timeout - elapsed
                logger.info(f"[Setup] Still waiting... ({remaining:.0f}s remaining)")
                last_log_time = asyncio.get_event_loop().time()
            
            if elapsed >= timeout:
                logger.warning(f"[Setup] Configuration timeout reached after {timeout}s")
                return False
            
            await asyncio.sleep(1)
    
    async def _handle_wifi_connection(self) -> Tuple[str, bool]:
        """
        Handle WiFi connection flow after credentials are received.
        
        Returns:
            Tuple of (pairing_code, success)
        """
        # Set LED state to ATTEMPTING_CONNECTION
        if self._LEDController:
            self._set_led_state(self._LEDController.STATE_ATTEMPTING_CONNECTION)
        
        # Update status: inform user they'll lose connection
        self.http_server.set_status(
            STATUS_CONNECTING,
            "✓ Credentials received!\n\nDevice will now connect to your WiFi network.\n\nYou can disconnect from Kin_Setup."
        )
        
        # Play voice feedback
        self._play_voice_feedback("connecting_to_wifi")
        
        await asyncio.sleep(USER_READ_DELAY)  # Give user time to read before AP stops
        
        # Try to connect to configured WiFi (this will stop the AP)
        if await self._connect_to_wifi():
            return await self._verify_wifi_connection()
        else:
            await self._handle_connection_error(
                "Failed to connect to WiFi",
                "Please check your WiFi password and try again"
            )
            return None, False
    
    async def _verify_wifi_connection(self) -> Tuple[str, bool]:
        """
        Verify WiFi connection and internet access.
        
        Returns:
            Tuple of (pairing_code, success)
        """
        self.http_server.set_status(STATUS_CONNECTING, "Verifying internet connection...")
        
        if await self.connectivity_checker.check_internet():
            self.http_server.set_status(
                STATUS_CONNECTING,
                "WiFi connected! Now ready for authentication..."
            )
            
            self._play_voice_feedback("wifi_connected")
            await asyncio.sleep(VOICE_FEEDBACK_DELAY)
            
            # Keep AP and HTTP server running so user can see auth status
            # They will be stopped by main.py after showing final status
            
            logger.info("Setup completed successfully! Returning to main for authentication...")
            return self._pairing_code, True
        else:
            await self._handle_connection_error(
                "Connected to WiFi but no internet access",
                "Please check your router's internet connection and try again"
            )
            return None, False
    
    async def _handle_connection_error(self, message: str, error_details: str):
        """Handle connection errors with user feedback."""
        logger.warning(message)
        self._play_voice_feedback("wifi_not_connected")
        self.http_server.set_status(STATUS_ERROR, message, error_details)
        await asyncio.sleep(ERROR_DISPLAY_DELAY)
    
    async def _connect_to_wifi(self) -> bool:
        """
        Connect to the configured WiFi network.
        
        Returns:
            True if connection successful
        """
        if not self._wifi_credentials:
            logger.error("[Setup] No WiFi credentials available!")
            return False
        
        ssid, password = self._wifi_credentials
        logger.info("=" * 60)
        logger.info(f"[Setup] Attempting to connect to WiFi network: {ssid}")
        logger.info(f"[Setup] This will stop the access point...")
        logger.info("=" * 60)
        
        result = await self.network_connector.connect(ssid, password)
        
        if result:
            logger.info(f"[Setup] ✓ Successfully connected to {ssid}")
        else:
            logger.error(f"[Setup] ✗ Failed to connect to {ssid}")
        
        return result
    
    async def _cleanup_old_state(self):
        """Clean up old pairing codes and processes from previous runs"""
        logger.info("Cleaning up old state from previous runs...")
        
        # Remove old pairing code file if it exists (from bash script era)
        try:
            import os
            if os.path.exists('/tmp/kin_pairing_code'):
                os.remove('/tmp/kin_pairing_code')
                logger.info("Removed old pairing code file")
        except Exception as e:
            logger.debug(f"Could not remove old pairing code: {e}")
        
        # Clear environment variable if set
        try:
            import os
            if 'DEVICE_PAIRING_CODE' in os.environ:
                del os.environ['DEVICE_PAIRING_CODE']
                logger.info("Cleared DEVICE_PAIRING_CODE from environment")
        except Exception as e:
            logger.debug(f"Could not clear environment variable: {e}")
        
        # Kill any old HTTP server processes on our port
        try:
            # Try to find processes listening on our port
            result = await asyncio.create_subprocess_exec(
                'lsof', '-ti', f':{self.http_port}',
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL
            )
            stdout, _ = await result.communicate()
            
            if stdout:
                pids = stdout.decode().strip().split('\n')
                for pid in pids:
                    if pid.strip():
                        try:
                            await asyncio.create_subprocess_exec('kill', pid.strip())
                            logger.info(f"Killed old HTTP server process (PID: {pid.strip()})")
                        except:
                            pass
        except FileNotFoundError:
            # lsof not available, try alternative
            logger.debug("lsof not available, skipping process cleanup")
        except Exception as e:
            logger.debug(f"Could not check for old HTTP processes: {e}")
    
    async def _cleanup(self):
        """Clean up all resources and reset state"""
        logger.info("Cleaning up setup resources...")
        
        try:
            await self.http_server.stop()
        except Exception as e:
            logger.debug(f"Error stopping HTTP server: {e}")
        
        try:
            await self.access_point.stop()
        except Exception as e:
            logger.debug(f"Error stopping access point: {e}")
        
        # Reset state
        self._wifi_credentials = None
        # Don't reset pairing code as we might need it for retry

