"""Weather fetcher using Open-Meteo API"""

import asyncio
import aiohttp
from typing import Optional, Dict, List
from datetime import datetime


async def fetch_weather(latitude: float, longitude: float, timezone: str = "auto", timeout: float = 3.0) -> Optional[Dict]:
    """
    Fetch current weather and forecast from Open-Meteo API
    
    Args:
        latitude: Location latitude
        longitude: Location longitude
        timezone: Timezone identifier (e.g., 'Asia/Kolkata')
        timeout: Request timeout in seconds (default: 3.0)
    
    Returns:
        Dictionary with weather data or None if fetch fails:
        {
            'current': {
                'temp': 20.5,
                'precipitation': 0.0,
                'wind_speed': 15.3
            },
            'hourly': {
                'temp': [20, 21, 22, ...],  # next 12 hours
                'precipitation': [0, 0, 0, ...],
                'wind_speed': [15, 16, 17, ...]
            },
            'daily': {
                'max_temp': [25, 26, 24, ...],  # next 7 days
                'min_temp': [15, 16, 14, ...],
                'precipitation': [0, 0, 5, ...],
                'wind_speed': [15, 16, 17, ...]
            },
            'units': {
                'temp': '°C',
                'precipitation': 'mm',
                'wind_speed': 'km/h'
            }
        }
    """
    try:
        # Build API URL with required parameters
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            'latitude': latitude,
            'longitude': longitude,
            'current': 'temperature_2m,precipitation,wind_speed_10m',
            'hourly': 'temperature_2m,precipitation,wind_speed_10m',
            'daily': 'temperature_2m_max,temperature_2m_min,precipitation_sum,wind_speed_10m_max',
            'timezone': timezone,
            'forecast_days': 7
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(
                url,
                params=params,
                timeout=aiohttp.ClientTimeout(total=timeout)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    # Extract current weather
                    current_data = data.get('current', {})
                    current = {
                        'temp': current_data.get('temperature_2m', 0),
                        'precipitation': current_data.get('precipitation', 0),
                        'wind_speed': current_data.get('wind_speed_10m', 0)
                    }
                    
                    # Extract hourly forecast (next 12 hours from current time)
                    hourly_data = data.get('hourly', {})
                    hourly_times = hourly_data.get('time', [])
                    
                    # Find current hour index
                    current_time = datetime.fromisoformat(current_data.get('time', ''))
                    current_hour_idx = 0
                    for i, time_str in enumerate(hourly_times):
                        hour_time = datetime.fromisoformat(time_str)
                        if hour_time >= current_time:
                            current_hour_idx = i
                            break
                    
                    # Get next 12 hours
                    end_idx = min(current_hour_idx + 12, len(hourly_times))
                    hourly = {
                        'temp': hourly_data.get('temperature_2m', [])[current_hour_idx:end_idx],
                        'precipitation': hourly_data.get('precipitation', [])[current_hour_idx:end_idx],
                        'wind_speed': hourly_data.get('wind_speed_10m', [])[current_hour_idx:end_idx]
                    }
                    
                    # Extract daily forecast (7 days)
                    daily_data = data.get('daily', {})
                    daily = {
                        'max_temp': daily_data.get('temperature_2m_max', [])[:7],
                        'min_temp': daily_data.get('temperature_2m_min', [])[:7],
                        'precipitation': daily_data.get('precipitation_sum', [])[:7],
                        'wind_speed': daily_data.get('wind_speed_10m_max', [])[:7]
                    }
                    
                    # Extract units
                    current_units = data.get('current_units', {})
                    units = {
                        'temp': current_units.get('temperature_2m', '°C'),
                        'precipitation': current_units.get('precipitation', 'mm'),
                        'wind_speed': current_units.get('wind_speed_10m', 'km/h')
                    }
                    
                    return {
                        'current': current,
                        'hourly': hourly,
                        'daily': daily,
                        'units': units
                    }
                else:
                    return None
                    
    except asyncio.TimeoutError:
        return None
    except Exception:
        return None

