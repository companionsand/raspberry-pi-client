"""WiFi-based geolocation for Raspberry Pi using Google Geolocation API"""

import subprocess
import re
import json
import asyncio
import aiohttp
from typing import Optional, Dict, List


class WiFiScanner:
    """Scan nearby WiFi access points on Raspberry Pi"""
    
    def __init__(self, interface: str = "wlan0"):
        self.interface = interface
    
    def _scan_with_iw(self) -> List[Dict]:
        """Scan WiFi using iw (modern nl80211 tool)"""
        try:
            output = subprocess.check_output(
                ["sudo", "iw", "dev", self.interface, "scan"],
                stderr=subprocess.DEVNULL,
                timeout=10
            ).decode('utf-8', errors='ignore')
            
            bss_sections = output.split('BSS ')
            aps = []
            
            for section in bss_sections[1:]:
                # Extract BSSID
                bssid_match = re.match(r'([0-9a-f:]{17})', section)
                if not bssid_match:
                    continue
                
                # Extract signal strength
                signal_match = re.search(r'signal: (-?\d+\.\d+) dBm', section)
                if not signal_match:
                    continue
                
                ap = {
                    "macAddress": bssid_match.group(1).upper(),
                    "signalStrength": int(float(signal_match.group(1))),
                }
                
                # Extract SSID (optional)
                ssid_match = re.search(r'SSID: (.+)', section)
                if ssid_match:
                    ap["ssid"] = ssid_match.group(1).strip()
                
                aps.append(ap)
            
            return aps
            
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
            return []
    
    def _scan_with_nmcli(self) -> List[Dict]:
        """Scan WiFi using nmcli (NetworkManager)"""
        try:
            # Rescan
            subprocess.run(
                ["nmcli", "dev", "wifi", "rescan"],
                stderr=subprocess.DEVNULL,
                timeout=5
            )
            
            # Get list
            output = subprocess.check_output(
                ["nmcli", "-f", "BSSID,SIGNAL", "dev", "wifi", "list"],
                stderr=subprocess.DEVNULL,
                timeout=10
            ).decode('utf-8', errors='ignore')
            
            lines = output.strip().split('\n')[1:]  # Skip header
            aps = []
            
            for line in lines:
                parts = line.split()
                if len(parts) >= 2:
                    bssid = parts[0]
                    try:
                        signal_pct = int(parts[1])
                        # Convert percentage to dBm (approximation)
                        signal_dbm = int(-100 + (signal_pct * 0.5))
                    except ValueError:
                        continue
                    
                    aps.append({
                        "macAddress": bssid.upper(),
                        "signalStrength": signal_dbm,
                    })
            
            return aps
            
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
            return []
    
    def _scan_with_iwlist(self) -> List[Dict]:
        """Scan WiFi using iwlist (traditional method)"""
        try:
            output = subprocess.check_output(
                ["sudo", "iwlist", self.interface, "scan"],
                stderr=subprocess.DEVNULL,
                timeout=10
            ).decode('utf-8', errors='ignore')
            
            cells = output.split("Cell ")
            aps = []
            
            for cell in cells[1:]:
                # Extract MAC address
                mac_match = re.search(r"Address: ([0-9A-Fa-f:]{17})", cell)
                if not mac_match:
                    continue
                
                # Extract signal strength
                signal_match = re.search(r"Signal level=(-?\d+) dBm", cell)
                if not signal_match:
                    continue
                
                aps.append({
                    "macAddress": mac_match.group(1).upper(),
                    "signalStrength": int(signal_match.group(1)),
                })
            
            return aps
            
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
            return []
    
    def scan(self) -> List[Dict]:
        """Scan WiFi using best available method"""
        # Try iw first (most modern)
        aps = self._scan_with_iw()
        if aps:
            return aps
        
        # Try nmcli (doesn't require sudo)
        aps = self._scan_with_nmcli()
        if aps:
            return aps
        
        # Fall back to iwlist
        return self._scan_with_iwlist()


async def fetch_location_from_wifi(
    google_api_key: str,
    interface: str = "wlan0",
    timeout: float = 10.0
) -> Optional[Dict]:
    """
    Fetch location using WiFi triangulation via Google Geolocation API
    
    Args:
        google_api_key: Google Maps Geolocation API key
        interface: WiFi interface name (default: wlan0)
        timeout: Request timeout in seconds
    
    Returns:
        Dictionary with location data or None if fetch fails:
        {
            'latitude': 28.5355,
            'longitude': 77.3910,
            'accuracy': 25.0,
            'city': 'Noida',
            'state': 'Uttar Pradesh',
            'country': 'India'
        }
    """
    # Scan WiFi access points
    scanner = WiFiScanner(interface=interface)
    access_points = scanner.scan()
    
    if not access_points:
        return None
    
    # Query Google Geolocation API
    url = f"https://www.googleapis.com/geolocation/v1/geolocate?key={google_api_key}"
    
    payload = {
        "considerIp": False,
        "wifiAccessPoints": [
            {
                "macAddress": ap["macAddress"],
                "signalStrength": ap["signalStrength"],
            }
            for ap in access_points[:20]  # Google accepts up to 20
        ]
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            # Get lat/lon from geolocation
            async with session.post(
                url,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=timeout)
            ) as response:
                if response.status != 200:
                    return None
                
                data = await response.json()
                
                latitude = data["location"]["lat"]
                longitude = data["location"]["lng"]
                accuracy = data.get("accuracy", 0)
                
                # Reverse geocode to get city, state, country
                reverse_url = f"https://maps.googleapis.com/maps/api/geocode/json"
                params = {
                    "latlng": f"{latitude},{longitude}",
                    "key": google_api_key
                }
                
                async with session.get(
                    reverse_url,
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=timeout)
                ) as geo_response:
                    city = None
                    state = None
                    country = None
                    
                    if geo_response.status == 200:
                        geo_data = await geo_response.json()
                        
                        if geo_data.get("status") == "OK" and geo_data.get("results"):
                            # Extract city, state, country from address components
                            for result in geo_data["results"]:
                                components = result.get("address_components", [])
                                
                                for component in components:
                                    types = component.get("types", [])
                                    
                                    if "locality" in types and not city:
                                        city = component.get("long_name")
                                    elif "administrative_area_level_1" in types and not state:
                                        state = component.get("long_name")
                                    elif "country" in types and not country:
                                        country = component.get("long_name")
                                
                                # Break if we have all three
                                if city and state and country:
                                    break
                    
                    return {
                        'latitude': latitude,
                        'longitude': longitude,
                        'accuracy': accuracy,
                        'city': city or '',
                        'state': state or '',
                        'country': country or ''
                    }
                    
    except asyncio.TimeoutError:
        return None
    except Exception:
        return None

