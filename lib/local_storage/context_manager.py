"""Context manager singleton for location and weather data"""

import asyncio
from datetime import datetime
from typing import Optional, Dict
from lib.location import fetch_location
from lib.weather import fetch_weather


class ContextManager:
    """
    Singleton manager for location and weather context data.
    Periodically fetches and caches data for use throughout the application.
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
        self._weather_data: Optional[Dict] = None
        self._update_task: Optional[asyncio.Task] = None
        self._running = False
        self._logger = None
        self._update_interval = 3600  # 1 hour in seconds
        self._fetch_timeout = 3.0  # 3 seconds
    
    def set_logger(self, logger):
        """Set logger for context manager"""
        self._logger = logger
    
    async def start(self):
        """Start the context manager and begin periodic updates"""
        if self._running:
            return
        
        self._running = True
        
        # Perform initial fetch (blocking with timeout)
        if self._logger:
            self._logger.info("context_manager_starting", extra={"action": "initial_fetch"})
        
        await self._update_data()
        
        # Start background update task
        self._update_task = asyncio.create_task(self._periodic_update())
        
        if self._logger:
            self._logger.info(
                "context_manager_started",
                extra={
                    "has_location": self._location_data is not None,
                    "has_weather": self._weather_data is not None
                }
            )
    
    async def stop(self):
        """Stop the context manager"""
        self._running = False
        
        if self._update_task:
            self._update_task.cancel()
            try:
                await self._update_task
            except asyncio.CancelledError:
                pass
        
        if self._logger:
            self._logger.info("context_manager_stopped")
    
    async def force_refresh(self):
        """Force an immediate refresh of location and weather data"""
        if self._logger:
            self._logger.info("context_manager_force_refresh")
        
        await self._update_data()
    
    async def _update_data(self):
        """Fetch and update location and weather data"""
        try:
            # Fetch location
            location = await fetch_location(timeout=self._fetch_timeout)
            
            if location:
                self._location_data = location
                
                if self._logger:
                    self._logger.info(
                        "location_fetched",
                        extra={
                            "city": location.get('city'),
                            "country": location.get('country')
                        }
                    )
                
                # Fetch weather using location coordinates
                weather = await fetch_weather(
                    latitude=location['latitude'],
                    longitude=location['longitude'],
                    timezone=location['timezone'],
                    timeout=self._fetch_timeout
                )
                
                if weather:
                    self._weather_data = weather
                    
                    if self._logger:
                        self._logger.info(
                            "weather_fetched",
                            extra={
                                "current_temp": weather['current']['temp'],
                                "units": weather['units']['temp']
                            }
                        )
                else:
                    if self._logger:
                        self._logger.warning("weather_fetch_failed")
            else:
                if self._logger:
                    self._logger.warning("location_fetch_failed")
                    
        except Exception as e:
            if self._logger:
                self._logger.error(
                    "context_update_error",
                    extra={"error": str(e)},
                    exc_info=True
                )
    
    async def _periodic_update(self):
        """Periodically update location and weather data"""
        while self._running:
            try:
                await asyncio.sleep(self._update_interval)
                
                if self._running:
                    await self._update_data()
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                if self._logger:
                    self._logger.error(
                        "periodic_update_error",
                        extra={"error": str(e)},
                        exc_info=True
                    )
    
    def get_dynamic_variables(self) -> Dict[str, str]:
        """
        Get dynamic variables for ElevenLabs conversation initiation.
        
        Returns:
            Dictionary of dynamic variables. Always includes date/time variables.
            Location and weather variables are only included if data is available.
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
        
        # Add timezone if we have location data, otherwise use system timezone
        if self._location_data:
            variables['timezone'] = self._location_data.get('timezone', 'UTC')
        else:
            # Try to get system timezone
            import time
            variables['timezone'] = time.tzname[time.daylight]
        
        # Only add location/weather variables if data is available
        if self._location_data and self._weather_data:
            # Location
            city = self._location_data.get('city', '')
            region = self._location_data.get('region', '')
            country = self._location_data.get('country', '')
            
            # Format location string
            location_parts = [p for p in [city, region, country] if p]
            variables['current_location'] = ', '.join(location_parts)
            
            # Current weather
            current = self._weather_data['current']
            variables['current_temp'] = str(round(current['temp']))
            variables['current_precipitation'] = str(round(current['precipitation'], 1))
            variables['current_wind_speed'] = str(round(current['wind_speed']))
            
            # Units
            units = self._weather_data['units']
            variables['temp_unit'] = units['temp']
            variables['precipitation_unit'] = units['precipitation']
            variables['wind_speed_unit'] = units['wind_speed']
            
            # 12-hour forecast
            hourly = self._weather_data['hourly']
            variables['twelve_hour_forecast_temp'] = ','.join(
                str(round(t)) for t in hourly['temp']
            )
            variables['twelve_hour_forecast_precipitation'] = ','.join(
                str(round(p, 1)) for p in hourly['precipitation']
            )
            variables['twelve_hour_forecast_wind_speed'] = ','.join(
                str(round(w)) for w in hourly['wind_speed']
            )
            
            # 7-day forecast
            daily = self._weather_data['daily']
            variables['seven_day_forecast_max_temp'] = ','.join(
                str(round(t)) for t in daily['max_temp']
            )
            variables['seven_day_forecast_min_temp'] = ','.join(
                str(round(t)) for t in daily['min_temp']
            )
            variables['seven_day_forecast_precipitation'] = ','.join(
                str(round(p, 1)) for p in daily['precipitation']
            )
            variables['seven_day_forecast_wind_speed'] = ','.join(
                str(round(w)) for w in daily['wind_speed']
            )
        
        return variables
    
    @property
    def has_location_data(self) -> bool:
        """Check if location data is available"""
        return self._location_data is not None
    
    @property
    def has_weather_data(self) -> bool:
        """Check if weather data is available"""
        return self._weather_data is not None

