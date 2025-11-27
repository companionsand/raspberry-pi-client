"""
Network Connector

Handles WiFi network connection using NetworkManager.
"""

import asyncio
import logging
import subprocess

logger = logging.getLogger(__name__)


class NetworkConnector:
    """Manages WiFi network connections"""
    
    def __init__(self, interface: str = "wlan0"):
        self.interface = interface
    
    async def connect(self, ssid: str, password: str = "", timeout: int = 30) -> bool:
        """
        Connect to a WiFi network.
        
        Args:
            ssid: Network SSID
            password: Network password (empty for open networks)
            timeout: Connection timeout in seconds
            
        Returns:
            True if connection successful
        """
        logger.info(f"Connecting to WiFi network: {ssid}")
        
        try:
            # Rescan to ensure network is visible
            await self._run_sudo_cmd(['nmcli', 'device', 'wifi', 'rescan'], check=False)
            await asyncio.sleep(3)
            
            # Verify network exists
            result = await self._run_cmd([
                'nmcli', '-t', '-f', 'SSID', 'device', 'wifi', 'list'
            ], capture_output=True)
            
            available_networks = [line.strip() for line in result.stdout.split('\n') if line.strip()]
            
            if ssid not in available_networks:
                logger.error(f"Network '{ssid}' not found. Available: {available_networks[:5]}")
                return False
            
            # Disconnect from current network if any
            current_connection = await self._get_current_connection()
            if current_connection:
                logger.info(f"Disconnecting from current network: {current_connection}")
                await self._run_sudo_cmd(['nmcli', 'connection', 'down', current_connection], check=False)
                await asyncio.sleep(2)
            
            # Connect to new network
            cmd = ['nmcli', 'device', 'wifi', 'connect', ssid, 'ifname', self.interface]
            if password:
                cmd.extend(['password', password])
            
            logger.info("Attempting connection...")
            result = await self._run_sudo_cmd(cmd, capture_output=True, timeout=timeout)
            
            if result and result.returncode == 0:
                logger.info("Connection command succeeded")
                
                # Wait for connection to stabilize
                await asyncio.sleep(5)
                
                # Verify connection
                if await self._verify_connection(ssid):
                    logger.info(f"Successfully connected to {ssid}")
                    return True
                else:
                    logger.warning("Connection command succeeded but device not connected")
                    return False
            else:
                error_msg = result.stderr if result else "Unknown error"
                logger.error(f"Connection failed: {error_msg}")
                return False
                
        except asyncio.TimeoutError:
            logger.error(f"Connection attempt timed out after {timeout}s")
            return False
        except Exception as e:
            logger.error(f"Error connecting to WiFi: {e}", exc_info=True)
            return False
    
    async def _get_current_connection(self) -> str:
        """Get the currently active WiFi connection on our interface"""
        try:
            result = await self._run_cmd([
                'nmcli', '-t', '-f', 'NAME,DEVICE', 'connection', 'show', '--active'
            ], capture_output=True)
            
            for line in result.stdout.split('\n'):
                if self.interface in line:
                    return line.split(':')[0]
            
            return ""
        except Exception as e:
            logger.debug(f"Error getting current connection: {e}")
            return ""
    
    async def _verify_connection(self, ssid: str) -> bool:
        """Verify that we're connected to the specified network"""
        try:
            # Check device status
            result = await self._run_cmd([
                'nmcli', '-t', '-f', 'GENERAL.STATE', 'device', 'show', self.interface
            ], capture_output=True)
            
            if 'connected' not in result.stdout.lower():
                return False
            
            # Check connection name contains SSID
            result = await self._run_cmd([
                'nmcli', '-t', '-f', 'GENERAL.CONNECTION', 'device', 'show', self.interface
            ], capture_output=True)
            
            connection = result.stdout.strip().split(':')[-1] if ':' in result.stdout else result.stdout.strip()
            
            # Connection name often matches SSID
            return ssid in connection or connection in ssid
            
        except Exception as e:
            logger.debug(f"Error verifying connection: {e}")
            return False
    
    async def disconnect(self):
        """Disconnect from current WiFi network"""
        try:
            current = await self._get_current_connection()
            if current:
                logger.info(f"Disconnecting from: {current}")
                await self._run_sudo_cmd(['nmcli', 'connection', 'down', current], check=False)
        except Exception as e:
            logger.debug(f"Error disconnecting: {e}")
    
    async def _run_cmd(self, cmd: list, check: bool = True, capture_output: bool = False, timeout: int = 30):
        """Run a command asynchronously"""
        try:
            if capture_output:
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
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
            else:
                proc = await asyncio.create_subprocess_exec(*cmd)
                returncode = await asyncio.wait_for(proc.wait(), timeout=timeout)
                if check and returncode != 0:
                    raise subprocess.CalledProcessError(returncode, cmd)
                return None
        except asyncio.TimeoutError:
            raise
        except subprocess.CalledProcessError:
            if check:
                raise
            return None
    
    async def _run_sudo_cmd(self, cmd: list, check: bool = True, capture_output: bool = False, timeout: int = 30):
        """Run a command with sudo"""
        return await self._run_cmd(['sudo'] + cmd, check=check, capture_output=capture_output, timeout=timeout)

