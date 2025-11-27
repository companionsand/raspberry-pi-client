"""
Connectivity Checker

Checks internet connectivity and orchestrator reachability.
"""

import asyncio
import logging
from typing import Optional

import aiohttp

logger = logging.getLogger(__name__)


class ConnectivityChecker:
    """Checks network and service connectivity"""
    
    def __init__(self):
        # Import Config here to avoid circular imports
        from lib.config import Config
        
        # Use hardcoded orchestrator URL from Config
        # Allow override via environment variable for testing
        self.orchestrator_url = Config.ORCHESTRATOR_URL
    
    async def check_internet(self, timeout: int = 5) -> bool:
        """
        Check if device has internet connectivity.
        
        Args:
            timeout: Timeout in seconds
            
        Returns:
            True if internet is accessible
        """
        try:
            # Try to ping Google DNS
            proc = await asyncio.create_subprocess_exec(
                'ping', '-c', '1', '-W', str(timeout), '8.8.8.8',
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL
            )
            returncode = await asyncio.wait_for(proc.wait(), timeout=timeout + 1)
            
            if returncode == 0:
                logger.info("Internet connectivity confirmed")
                return True
            else:
                logger.warning("Internet ping failed")
                return False
                
        except asyncio.TimeoutError:
            logger.warning("Internet check timed out")
            return False
        except Exception as e:
            logger.warning(f"Internet check error: {e}")
            return False
    
    async def check_orchestrator(self, retries: int = 3, retry_delay: int = 5) -> bool:
        """
        Check if conversation orchestrator is reachable.
        
        Args:
            retries: Number of retry attempts
            retry_delay: Delay between retries in seconds
            
        Returns:
            True if orchestrator is reachable
        """
        if not self.orchestrator_url:
            logger.error("Orchestrator URL not configured")
            return False
        
        # Convert WebSocket URL to HTTP for health check
        # wss://... -> https://..., ws://... -> http://...
        http_url = self.orchestrator_url.replace("wss://", "https://").replace("ws://", "http://")
        # Remove /ws suffix if present
        http_url = http_url.replace("/ws", "")
        health_url = f"{http_url.rstrip('/')}/health"
        
        for attempt in range(retries):
            try:
                logger.info(f"Checking orchestrator connectivity (attempt {attempt + 1}/{retries})...")
                
                async with aiohttp.ClientSession() as session:
                    async with session.get(health_url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                        if response.status == 200:
                            logger.info("Orchestrator is reachable")
                            return True
                        else:
                            logger.warning(f"Orchestrator returned status {response.status}")
                
            except asyncio.TimeoutError:
                logger.warning(f"Orchestrator check timed out (attempt {attempt + 1})")
            except aiohttp.ClientError as e:
                logger.warning(f"Orchestrator connection error (attempt {attempt + 1}): {e}")
            except Exception as e:
                logger.error(f"Unexpected error checking orchestrator (attempt {attempt + 1}): {e}")
            
            if attempt < retries - 1:
                logger.info(f"Retrying in {retry_delay}s...")
                await asyncio.sleep(retry_delay)
        
        logger.error(f"Orchestrator unreachable after {retries} attempts")
        return False
    
    async def check_full_connectivity(self, orchestrator_retries: int = 3) -> tuple[bool, bool]:
        """
        Check both internet and orchestrator connectivity.
        
        Args:
            orchestrator_retries: Number of retries for orchestrator check
            
        Returns:
            Tuple of (has_internet, orchestrator_reachable)
        """
        has_internet = await self.check_internet()
        
        if not has_internet:
            return False, False
        
        orchestrator_reachable = await self.check_orchestrator(retries=orchestrator_retries)
        
        return has_internet, orchestrator_reachable

