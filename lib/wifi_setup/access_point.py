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
    
    def __init__(self, ssid: str, interface: str = "wlan0"):
        self.ssid = ssid
        self.interface = interface
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
            
            # Delete existing connection if it exists (suppress errors as it may not exist)
            await self._run_cmd(['nmcli', 'connection', 'delete', self.connection_name], check=False, suppress_output=True)
            
            # Create new hotspot connection
            logger.info("Creating hotspot connection...")
            await self._run_sudo_cmd([
                'nmcli', 'connection', 'add',
                'type', 'wifi',
                'ifname', self.interface,
                'con-name', self.connection_name,
                'autoconnect', 'no',
                'ssid', self.ssid
            ])
            
            # Configure hotspot settings
            logger.info("Configuring hotspot...")
            await self._run_sudo_cmd([
                'nmcli', 'connection', 'modify', self.connection_name,
                '802-11-wireless.mode', 'ap',
                '802-11-wireless.band', 'bg',
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
                logger.info("Access point started successfully")
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
        """Clean up any existing hotspot connections"""
        try:
            # Disconnect and remove any existing Kin hotspot
            # Suppress errors as connection may not exist
            await self._run_sudo_cmd([
                'nmcli', 'connection', 'down', self.connection_name
            ], check=False, suppress_output=True)
            
            await self._run_sudo_cmd([
                'nmcli', 'connection', 'delete', self.connection_name
            ], check=False, suppress_output=True)
            
        except Exception as e:
            logger.debug(f"No existing hotspot to clean up: {e}")
    
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

