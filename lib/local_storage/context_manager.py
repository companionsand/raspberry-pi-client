"""Context manager singleton for location data"""

import asyncio
from datetime import datetime
from typing import Optional, Dict
from lib.location import fetch_location
from lib.config import Config


class ContextManager:
    """
    Singleton manager for location context data.
    Fetches location once at startup using WiFi triangulation.
    """
    
    _instance = None
    _lock = asyncio.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        # Only initialize once
        if self._initialized:
            return
            
        self._initialized = True
        self._location_data: Optional[Dict] = None
        self._logger = None
        self._fetch_timeout = 10.0  # 10 seconds for WiFi scan + geolocation
    
    def set_logger(self, logger):
        """Set logger for context manager"""
        self._logger = logger
    
    async def start(self):
        """Start the context manager and fetch location once"""
        if self._logger:
            self._logger.info("context_manager_starting", extra={"action": "initial_fetch"})
        
        # Fetch location once at startup
        await self._fetch_location()
        
        if self._logger:
            self._logger.info(
                "context_manager_started",
                extra={
                    "has_location": self._location_data is not None,
                    "latitude": self._location_data.get('latitude') if self._location_data else None,
                    "longitude": self._location_data.get('longitude') if self._location_data else None,
                    "city": self._location_data.get('city') if self._location_data else None,
                }
            )
    
    async def stop(self):
        """Stop the context manager"""
        if self._logger:
            self._logger.info("context_manager_stopped")
    
    async def force_refresh(self):
        """Force refresh location data (e.g., after reconnection)"""
        if self._logger:
            self._logger.info("context_manager_force_refresh", extra={"action": "manual_refresh"})
        await self._fetch_location()
    
    async def _fetch_location(self):
        """Fetch location data using WiFi triangulation"""
        try:
            # Check if Google API key is available
            google_api_key = Config.GOOGLE_API_KEY
            
            if not google_api_key:
                if self._logger:
                    self._logger.info(
                        "location_fetch_skipped",
                        extra={"reason": "no_google_api_key"}
                    )
                return
            
            # Fetch location via WiFi triangulation
            location = await fetch_location(
                google_api_key=google_api_key,
                timeout=self._fetch_timeout
            )
            
            if location:
                self._location_data = location
                
                if self._logger:
                    self._logger.info(
                        "location_fetched",
                        extra={
                            "latitude": location.get('latitude'),
                            "longitude": location.get('longitude'),
                            "accuracy": location.get('accuracy'),
                            "city": location.get('city'),
                            "state": location.get('state'),
                            "country": location.get('country'),
                        }
                    )
            else:
                if self._logger:
                    self._logger.warning("location_fetch_failed")
                    
        except Exception as e:
            if self._logger:
                self._logger.error(
                    "location_fetch_error",
                    extra={"error": str(e)},
                    exc_info=True
                )
    
    def get_dynamic_variables(self) -> Dict[str, str]:
        """
        Get dynamic variables for ElevenLabs conversation initiation.
        
        Returns:
            Dictionary of dynamic variables. Always includes date/time variables.
            Location variables (lat/lon/city/state/country) are included if available.
        """
        variables = {}
        
        # Always include date/time variables
        now = datetime.now()
        
        # Format date as "26th November 2025"
        day = now.day
        suffix = 'th' if 11 <= day <= 13 else {1: 'st', 2: 'nd', 3: 'rd'}.get(day % 10, 'th')
        formatted_date = now.strftime(f"%d{suffix} %B %Y").lstrip('0')
        
        variables['current_date'] = formatted_date
        variables['current_day_of_week'] = now.strftime("%A")
        variables['current_time'] = now.strftime("%I:%M %p").lstrip('0')
        
        # Add timezone in IANA format (e.g., "Asia/Kolkata")
        try:
            # Try to read from /etc/timezone (Linux/Raspberry Pi)
            with open('/etc/timezone', 'r') as f:
                variables['timezone'] = f.read().strip()
        except (FileNotFoundError, IOError):
            # Fallback: Try to get timezone from Python (works on Mac/Windows)
            try:
                import subprocess
                # On Mac, use systemsetup or read from /etc/localtime symlink
                result = subprocess.run(
                    ['readlink', '/etc/localtime'],
                    capture_output=True,
                    text=True,
                    timeout=2
                )
                if result.returncode == 0:
                    # Output is like /var/db/timezone/zoneinfo/Asia/Kolkata
                    tz_path = result.stdout.strip()
                    # Extract timezone from path (last two components for region/city)
                    parts = tz_path.split('/')
                    if len(parts) >= 2:
                        variables['timezone'] = '/'.join(parts[-2:])
                    else:
                        variables['timezone'] = 'UTC'
                else:
                    variables['timezone'] = 'UTC'
            except Exception:
                variables['timezone'] = 'UTC'
        
        # Always include location variables (required by ElevenLabs agent tools)
        # Use actual values if available, otherwise empty strings
        if self._location_data:
            variables['latitude'] = str(self._location_data.get('latitude', ''))
            variables['longitude'] = str(self._location_data.get('longitude', ''))
            variables['city'] = self._location_data.get('city', '')
            variables['state'] = self._location_data.get('state', '')
            variables['country'] = self._location_data.get('country', '')
        else:
            # Provide empty values to satisfy required dynamic variables
            variables['latitude'] = ''
            variables['longitude'] = ''
            variables['city'] = ''
            variables['state'] = ''
            variables['country'] = ''
        
        return variables
    
    def get_location_string(self) -> str:
        """
        Get location as a formatted string for websocket (city, state, country).
        
        Returns:
            Location string like "Noida, Uttar Pradesh, India" or empty string
        """
        if not self._location_data:
            return ""
        
        city = self._location_data.get('city', '')
        state = self._location_data.get('state', '')
        country = self._location_data.get('country', '')
        
        # Build location string from available parts
        parts = [p for p in [city, state, country] if p]
        return ', '.join(parts)
    
    @property
    def has_location_data(self) -> bool:
        """Check if location data is available"""
        return self._location_data is not None
