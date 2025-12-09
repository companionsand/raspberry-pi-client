# WiFi Setup Enhanced Logging Guide

## Overview

Comprehensive debug logging has been added to all WiFi setup components to help diagnose connection issues. All logs are prefixed with component identifiers for easy filtering.

## Log Prefixes

- `[HTTP]` - HTTP server events (requests, responses, configuration)
- `[WiFi Setup]` - WiFi setup manager flow
- `[NetConnect]` - Network connection attempts
- `[AP]` - Access point creation/management (from access_point.py)

## What You'll See in the Logs

### 1. Access Point Startup

```
Starting access point: Kin_Setup on wlan0
Creating hotspot using nmcli device wifi hotspot...
Access point started successfully via hotspot command
```

Or if hotspot command fails:

```
Hotspot command failed, trying manual method...
Configuring hotspot with WPA2-PSK security...
Access point started successfully via manual method
```

### 2. HTTP Server Startup

```
[HTTP] Starting HTTP server on 0.0.0.0:8080
[HTTP] AP SSID: Kin_Setup, Password: kinsetup123
[HTTP] WiFi Interface: wlan0
[HTTP] ✓ HTTP server listening on port 8080
[HTTP] Access the setup page at: http://192.168.4.1:8080
```

### 3. User Accessing Web Interface

```
[HTTP] GET / from 192.168.4.2
[HTTP] Serving setup page to 192.168.4.2
```

### 4. User Requesting AP Info

```
[HTTP] GET /ap-info from 192.168.4.2
[HTTP] Serving AP info to 192.168.4.2
[HTTP] Sending AP info: SSID=Kin_Setup
```

### 5. User Scanning for Networks

```
[HTTP] GET /networks from 192.168.4.2
[HTTP] Network scan requested by 192.168.4.2
[HTTP] Starting WiFi network scan...
[HTTP] Running nmcli wifi rescan...
[HTTP] Fetching WiFi network list...
[HTTP] Found 12 WiFi networks
[HTTP] Networks: ['MyHomeWiFi', 'Neighbor_Network', 'CoffeeShop', ...]...
```

### 6. User Submitting Configuration

```
[HTTP] POST /configure from 192.168.4.2
[HTTP] Configuration submission from 192.168.4.2
[HTTP] Processing configuration submission...
[HTTP] Receiving 87 bytes of configuration data
[HTTP] Configuration received:
[HTTP]   SSID: MyHomeWiFi
[HTTP]   Password: ************
[HTTP]   Pairing Code: 1234
[HTTP] Calling configuration callback...
============================================================
[WiFi Setup] ✓ Configuration received from web interface
[WiFi Setup]   Target SSID: MyHomeWiFi
[WiFi Setup]   Password length: 12 chars
[WiFi Setup]   Pairing code: 1234
============================================================
[HTTP] Configuration accepted, sending success response
```

### 7. Attempting WiFi Connection

```
[HTTP] Status update: connecting - ✓ Credentials received!

Device will now connect to your WiFi network.

You can disconnect from Kin_Setup.
The device will reconnect to you if setup fails.

============================================================
[WiFi Setup] Attempting to connect to WiFi network: MyHomeWiFi
[WiFi Setup] This will stop the access point...
============================================================
[NetConnect] Connecting to WiFi network: MyHomeWiFi
[NetConnect] Rescanning WiFi networks...
[NetConnect] Verifying MyHomeWiFi is visible...
[NetConnect] ✓ Network 'MyHomeWiFi' found in scan
[NetConnect] Disconnecting from current network: Kin_Hotspot
[NetConnect] Connecting with password (12 chars)
[NetConnect] Executing: nmcli device wifi connect MyHomeWiFi ...
[NetConnect] ✓ Connection command succeeded
[NetConnect] Output: Device 'wlan0' successfully activated with '...'
[NetConnect] Waiting for connection to stabilize (5s)...
[NetConnect] Verifying connection to MyHomeWiFi...
[NetConnect] ✓ Successfully connected to MyHomeWiFi
[WiFi Setup] ✓ Successfully connected to MyHomeWiFi
```

### 8. Verifying Internet

```
[HTTP] Status update: connecting - Verifying internet connection...
Internet connectivity confirmed
[HTTP] Status update: connecting - WiFi connected! Now ready for authentication...
WiFi setup completed successfully! Returning to main for authentication...
```

### 9. Status Polling (every 2 seconds while connecting)

```
[HTTP] GET /status from 192.168.4.2
[HTTP] Status check from 192.168.4.2
[HTTP] Status check: connecting - Verifying internet connection...
```

### 10. Common Errors

**Network not found:**

```
[NetConnect] ✗ Network 'WrongSSID' not found!
[NetConnect] Available networks: ['Network1', 'Network2', ...]
```

**Connection failed:**

```
[NetConnect] ✗ Connection failed: Secrets were required, but not provided
[WiFi Setup] ✗ Failed to connect to MyHomeWiFi
[HTTP] Status update: error - Failed to connect to WiFi (Error: Please check your WiFi password and try again)
```

**Connection timeout:**

```
[NetConnect] ✗ Connection attempt timed out after 30s
```

**Port already in use:**

```
[HTTP] ✗ Port 8080 is already in use!
[HTTP] Try: sudo lsof -ti :8080 | xargs kill
```

## Filtering Logs

### View only HTTP server logs:

```bash
sudo journalctl -u kin-client -f | grep "\[HTTP\]"
```

### View only connection attempts:

```bash
sudo journalctl -u kin-client -f | grep "\[NetConnect\]"
```

### View only WiFi setup flow:

```bash
sudo journalctl -u kin-client -f | grep "\[WiFi Setup\]"
```

### View all WiFi-related logs:

```bash
sudo journalctl -u kin-client -f | grep -E "\[HTTP\]|\[WiFi Setup\]|\[NetConnect\]|\[AP\]"
```

### View errors only:

```bash
sudo journalctl -u kin-client -f | grep "✗"
```

### View successes only:

```bash
sudo journalctl -u kin-client -f | grep "✓"
```

## Manual Testing Mode

If running manually (not as a service), you'll see the logs directly in your terminal:

```bash
cd raspberry-pi-client
python main.py
```

The logging is configured at INFO level by default, which shows all important events. For even more detail, you can temporarily change the log level:

```python
# In main.py or lib/config.py
logging.basicConfig(level=logging.DEBUG)  # Shows everything
```

## Typical Successful Flow (Log Summary)

```
1. [AP] Access point created: Kin_Setup
2. [HTTP] HTTP server started on port 8080
3. [WiFi Setup] Waiting for configuration (timeout: 300s)
4. [HTTP] GET / - User opens web page
5. [HTTP] GET /networks - User scans networks
6. [HTTP] POST /configure - User submits config
7. [WiFi Setup] Configuration received
8. [NetConnect] Connecting to target network
9. [NetConnect] ✓ Connected successfully
10. [WiFi Setup] ✓ Setup complete
```

## Troubleshooting with Logs

### Issue: "Network doesn't appear"

Look for:

- `Access point started successfully` - AP is running
- Check device's WiFi scan can see Kin_Setup

### Issue: "Can't connect to Kin_Setup"

Look for:

- `Configuring hotspot with WPA2-PSK security` - Security is enabled
- Check password in logs: `Password: kinsetup123`

### Issue: "Web page doesn't load"

Look for:

- `[HTTP] ✓ HTTP server listening on port 8080`
- `[HTTP] GET / from 192.168.4.x` - Device is reaching server
- If no GET requests, check device IP (should be 192.168.4.x)

### Issue: "Configuration submission fails"

Look for:

- `[HTTP] POST /configure` - Request reached server
- `[HTTP] Configuration received` - Server parsed it
- Check validation errors (SSID missing, invalid pairing code, etc.)

### Issue: "Can't connect to home WiFi"

Look for:

- `[NetConnect] ✗ Network 'SSID' not found!` - Network not visible
- `[NetConnect] ✗ Connection failed: Secrets were required` - Wrong password
- `[NetConnect] ✗ Connection attempt timed out` - Network unreachable

### Issue: "No internet after connecting"

Look for:

- `[NetConnect] ✓ Successfully connected` - WiFi works
- `Internet connectivity confirmed` - Internet works
- If WiFi works but no internet: router/ISP issue

## Log Storage

Logs are stored by systemd:

- **Live logs:** `sudo journalctl -u kin-client -f`
- **Recent logs:** `sudo journalctl -u kin-client -n 100`
- **Since boot:** `sudo journalctl -u kin-client -b`
- **Time range:** `sudo journalctl -u kin-client --since "1 hour ago"`

Logs persist across reboots and can be searched later for debugging.

## Performance Impact

The enhanced logging has minimal performance impact:

- INFO level (default): ~100 log lines during setup
- DEBUG level: ~300 log lines during setup
- Each log line: <1KB
- Total overhead: <1MB for full setup flow

The logging is designed to be helpful without overwhelming the system.
