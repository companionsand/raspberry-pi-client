"""
Access Point Manager

Handles creation and management of WiFi access point using NetworkManager.
"""

import asyncio
import logging
import subprocess
from typing import Optional

logger = logging.getLogger(__name__)


class AccessPoint:
    """Manages WiFi access point creation and teardown"""
    
    def __init__(self, ssid: str, interface: str = "wlan0", password: str = "kinsetup123"):
        self.ssid = ssid
        self.interface = interface
        self.password = password
        self.connection_name = "Kin_Hotspot"
        self._is_running = False
    
    async def start(self):
        """Create and start the WiFi access point"""
        logger.info(f"Starting access point: {self.ssid} on {self.interface}")
        
        try:
            # Clean up any existing hotspot connection
            await self._cleanup_existing()
            
            # Unblock WiFi
            await self._run_sudo_cmd(['rfkill', 'unblock', 'wifi'])
            
            # Ensure the interface is fully down and clean
            logger.debug(f"Ensuring {self.interface} is clean...")
            await self._run_sudo_cmd([
                'ip', 'addr', 'flush', 'dev', self.interface
            ], check=False, suppress_output=True)
            
            # Bring interface down then back up
            await self._run_sudo_cmd([
                'ip', 'link', 'set', self.interface, 'down'
            ], check=False, suppress_output=True)
            
            await asyncio.sleep(1)
            
            await self._run_sudo_cmd([
                'ip', 'link', 'set', self.interface, 'up'
            ], check=False, suppress_output=True)
            
            await asyncio.sleep(1)
            
            # First, try the simple nmcli hotspot command (NetworkManager 1.16+)
            # This is the recommended way and handles all settings automatically
            logger.info("Creating hotspot using nmcli device wifi hotspot...")
            try:
                await self._run_sudo_cmd([
                    'nmcli', 'device', 'wifi', 'hotspot',
                    'ifname', self.interface,
                    'con-name', self.connection_name,
                    'ssid', self.ssid,
                    'password', self.password
                ])
                
                # Wait a moment for it to stabilize
                await asyncio.sleep(3)
                
                # Verify it's active
                result = await self._run_cmd([
                    'nmcli', 'connection', 'show', '--active'
                ], capture_output=True)
                
                if self.connection_name in result.stdout:
                    logger.info("Access point started successfully via hotspot command")
                    # Log connection details for debugging
                    await self._log_connection_details()
                    self._is_running = True
                    return
                else:
                    logger.warning("Hotspot command succeeded but connection not active, trying manual method...")
                    await self._cleanup_existing()
                    
            except Exception as e:
                logger.warning(f"Hotspot command failed ({e}), trying manual method...")
                await self._cleanup_existing()
            
            # Fallback: Manual connection creation (for older NetworkManager versions)
            logger.info("Creating hotspot connection manually...")
            await self._run_sudo_cmd([
                'nmcli', 'connection', 'add',
                'type', 'wifi',
                'ifname', self.interface,
                'con-name', self.connection_name,
                'autoconnect', 'no',
                'ssid', self.ssid
            ])
            
            # Configure hotspot settings with WPA2 security
            logger.info("Configuring hotspot with WPA2-PSK security...")
            await self._run_sudo_cmd([
                'nmcli', 'connection', 'modify', self.connection_name,
                '802-11-wireless.mode', 'ap',
                '802-11-wireless.band', 'bg',
                '802-11-wireless-security.key-mgmt', 'wpa-psk',
                '802-11-wireless-security.proto', 'rsn',
                '802-11-wireless-security.pairwise', 'ccmp',
                '802-11-wireless-security.group', 'ccmp',
                '802-11-wireless-security.psk', self.password,
                'ipv4.method', 'shared',
                'ipv4.address', '192.168.4.1/24'
            ])
            
            # Start the hotspot
            logger.info("Activating hotspot...")
            await self._run_sudo_cmd(['nmcli', 'connection', 'up', self.connection_name])
            
            # Wait a moment for it to stabilize
            await asyncio.sleep(3)
            
            # Verify it's active
            result = await self._run_cmd([
                'nmcli', 'connection', 'show', '--active'
            ], capture_output=True)
            
            if self.connection_name in result.stdout:
                logger.info("Access point started successfully via manual method")
                # Log connection details for debugging
                await self._log_connection_details()
                self._is_running = True
            else:
                raise Exception("Hotspot not active after starting")
                
        except Exception as e:
            logger.error(f"Failed to start access point: {e}")
            await self.stop()  # Clean up on failure
            raise
    
    async def stop(self):
        """Stop the WiFi access point"""
        if not self._is_running:
            return
        
        logger.info("Stopping access point...")
        
        try:
            # Bring down the hotspot
            await self._run_sudo_cmd([
                'nmcli', 'connection', 'down', self.connection_name
            ], check=False)
            
            # Delete the hotspot connection
            await self._run_sudo_cmd([
                'nmcli', 'connection', 'delete', self.connection_name
            ], check=False)
            
            logger.info("Access point stopped")
            self._is_running = False
            
        except Exception as e:
            logger.error(f"Error stopping access point: {e}")
    
    async def _cleanup_existing(self):
        """Clean up any existing hotspot connections and processes"""
        try:
            logger.debug("Cleaning up existing hotspot connections...")
            
            # Disconnect and remove any existing Kin hotspot
            # Suppress errors as connection may not exist
            await self._run_sudo_cmd([
                'nmcli', 'connection', 'down', self.connection_name
            ], check=False, suppress_output=True)
            
            await self._run_sudo_cmd([
                'nmcli', 'connection', 'delete', self.connection_name
            ], check=False, suppress_output=True)
            
            # Wait a moment for NetworkManager to clean up dnsmasq processes
            await asyncio.sleep(2)
            
            # Kill any lingering dnsmasq processes bound to our IP
            try:
                logger.debug("Checking for lingering dnsmasq processes...")
                # Find and kill dnsmasq processes for our interface
                await self._run_sudo_cmd([
                    'pkill', '-f', f'dnsmasq.*{self.interface}'
                ], check=False, suppress_output=True)
                
                # Give processes time to die
                await asyncio.sleep(1)
            except Exception as e:
                logger.debug(f"No dnsmasq processes to clean up: {e}")
            
        except Exception as e:
            logger.debug(f"No existing hotspot to clean up: {e}")
    
    async def _log_connection_details(self):
        """Log connection details for debugging"""
        try:
            result = await self._run_cmd([
                'nmcli', 'connection', 'show', self.connection_name
            ], capture_output=True)
            
            # Extract security settings
            for line in result.stdout.split('\n'):
                if '802-11-wireless' in line or 'ipv4' in line:
                    logger.debug(f"  {line.strip()}")
                    
        except Exception as e:
            logger.debug(f"Could not log connection details: {e}")
    
    async def _run_cmd(self, cmd: list, check: bool = True, capture_output: bool = False, suppress_output: bool = False) -> Optional[subprocess.CompletedProcess]:
        """Run a command asynchronously"""
        try:
            if capture_output:
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                stdout, stderr = await proc.communicate()
                result = subprocess.CompletedProcess(
                    cmd, proc.returncode,
                    stdout=stdout.decode() if stdout else "",
                    stderr=stderr.decode() if stderr else ""
                )
                if check and result.returncode != 0:
                    raise subprocess.CalledProcessError(
                        result.returncode, cmd, result.stdout, result.stderr
                    )
                return result
            elif suppress_output:
                # Suppress both stdout and stderr
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL
                )
                returncode = await proc.wait()
                if check and returncode != 0:
                    raise subprocess.CalledProcessError(returncode, cmd)
                return None
            else:
                proc = await asyncio.create_subprocess_exec(*cmd)
                returncode = await proc.wait()
                if check and returncode != 0:
                    raise subprocess.CalledProcessError(returncode, cmd)
                return None
        except subprocess.CalledProcessError as e:
            if check:
                raise
            logger.debug(f"Command failed (ignored): {cmd}, error: {e}")
            return None
    
    async def _run_sudo_cmd(self, cmd: list, check: bool = True, suppress_output: bool = False) -> Optional[subprocess.CompletedProcess]:
        """Run a command with sudo"""
        return await self._run_cmd(['sudo'] + cmd, check=check, suppress_output=suppress_output)

