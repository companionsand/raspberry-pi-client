"""Location fetcher using ipwho.is API"""

import asyncio
import aiohttp
from typing import Optional, Dict


async def fetch_location(timeout: float = 3.0) -> Optional[Dict[str, str]]:
    """
    Fetch current location using IP-based geolocation from ipwho.is
    
    Args:
        timeout: Request timeout in seconds (default: 3.0)
    
    Returns:
        Dictionary with location data or None if fetch fails:
        {
            'city': 'Noida',
            'region': 'Uttar Pradesh', 
            'country': 'India',
            'latitude': 28.5355,
            'longitude': 77.3910,
            'timezone': 'Asia/Kolkata'
        }
    """
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                'https://ipwho.is',
                timeout=aiohttp.ClientTimeout(total=timeout)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    # Check if request was successful
                    if not data.get('success', False):
                        return None
                    
                    # Extract relevant location data
                    return {
                        'city': data.get('city', ''),
                        'region': data.get('region', ''),
                        'country': data.get('country', ''),
                        'latitude': data.get('latitude', 0.0),
                        'longitude': data.get('longitude', 0.0),
                        'timezone': data.get('timezone', {}).get('id', 'UTC')
                    }
                else:
                    return None
                    
    except asyncio.TimeoutError:
        return None
    except Exception:
        return None

