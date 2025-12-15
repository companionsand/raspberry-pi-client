"""
WiFi Setup Manager

Orchestrates the WiFi setup process including:
- Creating a WiFi access point
- Running HTTP server for configuration
- Handling WiFi connection attempts
- Validating internet connectivity
"""

import asyncio
import logging
from typing import Optional, Tuple

from .access_point import AccessPoint
from .http_server import SetupHTTPServer
from .network_connector import NetworkConnector
from .connectivity import ConnectivityChecker

logger = logging.getLogger(__name__)


class WiFiSetupManager:
    """Manages the complete WiFi setup flow"""
    
    def __init__(
        self,
        ap_ssid: str = "Kin_Setup",
        ap_password: str = "kinsetup123",
        ap_interface: str = "wlan0",
        http_port: int = 8080,
        max_retries: int = 5,
        led_controller = None,
        voice_feedback = None
    ):
        self.ap_ssid = ap_ssid
        self.ap_password = ap_password
        self.ap_interface = ap_interface
        self.http_port = http_port
        self.max_retries = max_retries
        self.led_controller = led_controller
        self.voice_feedback = voice_feedback
        
        self.access_point = AccessPoint(ap_ssid, ap_interface, ap_password)
        self.http_server = SetupHTTPServer(http_port, ap_interface, ap_ssid, ap_password)
        self.network_connector = NetworkConnector(ap_interface)
        self.connectivity_checker = ConnectivityChecker()
        
        self._pairing_code: Optional[str] = None
        self._wifi_credentials: Optional[Tuple[str, str]] = None
    
    async def start_setup_mode(self) -> Tuple[str, bool]:
        """
        Start WiFi setup mode and wait for configuration.
        
        Returns:
            Tuple of (pairing_code, success)
        """
        logger.info("Starting WiFi setup mode...")
        
        # Clean up any old pairing code files and processes from previous runs
        await self._cleanup_old_state()
        
        retry_count = 0
        
        while retry_count < self.max_retries:
            try:
                # Clean up any previous state
                await self._cleanup()
                
                # Create access point
                logger.info(f"Creating access point: {self.ap_ssid}")
                await self.access_point.start()
                
                # Start HTTP server for configuration
                logger.info(f"Starting HTTP server on port {self.http_port}")
                await self.http_server.start(self._handle_wifi_config)
                
                # Set LED state to WIFI_SETUP (ready for configuration)
                if self.led_controller:
                    # Import STATE_WIFI_SETUP constant
                    from ..audio import LEDController
                    self.led_controller.set_state(LEDController.STATE_WIFI_SETUP)
                
                logger.info(f"WiFi setup active. Connect to '{self.ap_ssid}' (password: {self.ap_password}) and go to http://192.168.4.1:{self.http_port}")
                
                # Play voice feedback to guide user
                if self.voice_feedback:
                    self.voice_feedback.play("wifi_setup_ready")
                
                # Wait for user to submit configuration
                success = await self._wait_for_configuration()
                
                if success:
                    # Set LED state to ATTEMPTING_CONNECTION
                    if self.led_controller:
                        from ..audio import LEDController
                        self.led_controller.set_state(LEDController.STATE_ATTEMPTING_CONNECTION)
                    
                    # Update status: inform user they'll lose connection
                    self.http_server.set_status("connecting", "✓ Credentials received!\n\nDevice will now connect to your WiFi network.\n\nYou can disconnect from Kin_Setup.")
                    
                    # Play voice feedback
                    if self.voice_feedback:
                        self.voice_feedback.play("connecting_to_wifi")
                    
                    await asyncio.sleep(5)  # Give user time to read before AP stops
                    
                    # Try to connect to configured WiFi (this will stop the AP)
                    if await self._connect_to_wifi():
                        # Verify internet connectivity
                        self.http_server.set_status("connecting", "Verifying internet connection...")
                        if await self.connectivity_checker.check_internet():
                            self.http_server.set_status("connecting", "WiFi connected! Now ready for authentication...")
                            
                            # Play voice feedback
                            if self.voice_feedback:
                                self.voice_feedback.play("wifi_connected")
                            
                            await asyncio.sleep(3)  # Give voice feedback time to complete
                            
                            # Keep AP and HTTP server running so user can see auth status
                            # They will be stopped by main.py after showing final status
                            
                            logger.info("WiFi setup completed successfully! Returning to main for authentication...")
                            return self._pairing_code, True
                        else:
                            logger.warning("Connected to WiFi but no internet access")
                            
                            # Play voice feedback
                            if self.voice_feedback:
                                self.voice_feedback.play("wifi_not_connected")
                            
                            self.http_server.set_status("error", "Connected to WiFi but no internet access", "Please check your router's internet connection and try again")
                            await asyncio.sleep(8)  # Give user time to read
                    else:
                        logger.warning("Failed to connect to configured WiFi")
                        
                        # Play voice feedback
                        if self.voice_feedback:
                            self.voice_feedback.play("wifi_not_connected")
                        
                        self.http_server.set_status("error", "Failed to connect to WiFi", "Please check your WiFi password and try again")
                        await asyncio.sleep(8)  # Give user time to read
                
                retry_count += 1
                if retry_count < self.max_retries:
                    logger.info(f"Retrying WiFi setup... ({retry_count}/{self.max_retries})")
                    # LED will be set to WIFI_SETUP again when AP/HTTP server restart
                    await asyncio.sleep(2)
                
            except Exception as e:
                logger.error(f"Error during WiFi setup: {e}", exc_info=True)
                retry_count += 1
                await asyncio.sleep(2)
        
        logger.error(f"WiFi setup failed after {self.max_retries} attempts")
        await self._cleanup()
        return None, False
    
    async def _handle_wifi_config(self, ssid: str, password: str, pairing_code: str) -> bool:
        """
        Callback for when user submits WiFi configuration.
        
        Args:
            ssid: WiFi network name
            password: WiFi password
            pairing_code: Device pairing code
            
        Returns:
            True if configuration was accepted
        """
        logger.info("=" * 60)
        logger.info("[WiFi Setup] ✓ Configuration received from web interface")
        logger.info(f"[WiFi Setup]   Target SSID: {ssid}")
        logger.info(f"[WiFi Setup]   Password length: {len(password)} chars")
        logger.info(f"[WiFi Setup]   Pairing code: {pairing_code}")
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
        
        logger.info(f"[WiFi Setup] Waiting for configuration (timeout: {timeout}s)")
        
        while True:
            if self._wifi_credentials and self._pairing_code:
                elapsed = asyncio.get_event_loop().time() - start_time
                logger.info(f"[WiFi Setup] Configuration received after {elapsed:.0f}s")
                return True
            
            elapsed = asyncio.get_event_loop().time() - start_time
            
            # Log progress periodically
            if elapsed - (last_log_time - start_time) >= log_interval:
                remaining = timeout - elapsed
                logger.info(f"[WiFi Setup] Still waiting... ({remaining:.0f}s remaining)")
                last_log_time = asyncio.get_event_loop().time()
            
            if elapsed >= timeout:
                logger.warning(f"[WiFi Setup] Configuration timeout reached after {timeout}s")
                return False
            
            await asyncio.sleep(1)
    
    async def _connect_to_wifi(self) -> bool:
        """
        Connect to the configured WiFi network.
        
        Returns:
            True if connection successful
        """
        if not self._wifi_credentials:
            logger.error("[WiFi Setup] No WiFi credentials available!")
            return False
        
        ssid, password = self._wifi_credentials
        logger.info("=" * 60)
        logger.info(f"[WiFi Setup] Attempting to connect to WiFi network: {ssid}")
        logger.info(f"[WiFi Setup] This will stop the access point...")
        logger.info("=" * 60)
        
        result = await self.network_connector.connect(ssid, password)
        
        if result:
            logger.info(f"[WiFi Setup] ✓ Successfully connected to {ssid}")
        else:
            logger.error(f"[WiFi Setup] ✗ Failed to connect to {ssid}")
        
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
        logger.info("Cleaning up WiFi setup resources...")
        
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

