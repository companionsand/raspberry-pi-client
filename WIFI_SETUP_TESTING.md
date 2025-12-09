# WiFi Setup Testing Guide

## Quick Test Procedure

### 1. Enable WiFi Setup Mode

Edit your `.env` file and set:

```bash
SKIP_WIFI_SETUP=false
```

Or simply comment out/remove the line if it says `SKIP_WIFI_SETUP=true`.

### 2. Start the Client

```bash
python main.py
```

### 3. Check the Logs

You should see output like:

```
🔧 Entering WiFi setup mode...
Starting access point: Kin_Setup on wlan0
Creating hotspot connection...
Configuring hotspot...
Activating hotspot...
Access point started successfully
Starting HTTP server on port 8080
WiFi setup active. Connect to 'Kin_Setup' (password: kinsetup123) and go to http://192.168.4.1:8080
```

### 4. Connect from Your Laptop/Phone

**On macOS:**

1. Click WiFi icon in menu bar
2. Look for `Kin_Setup` network (should have a lock icon 🔒)
3. Click to connect
4. Enter password: `kinsetup123`
5. You should connect successfully and stay connected

**On Windows:**

1. Click network icon in system tray
2. Look for `Kin_Setup` network (should show as secured)
3. Click Connect
4. Enter network security key: `kinsetup123`
5. Click Next - you should connect successfully

**On iOS/Android:**

1. Go to Settings → WiFi
2. Look for `Kin_Setup` network
3. Tap it and enter password: `kinsetup123`
4. You should connect successfully

### 5. Access the Web Interface

1. Open a web browser
2. Go to: `http://192.168.4.1:8080`
3. You should see:
   - Kin logo and "Device Setup" heading
   - A gray info box showing:
     ```
     Connection Info:
     Network: Kin_Setup
     Password: kinsetup123
     ```
   - WiFi Network dropdown (with "Scanning..." initially)
   - WiFi Password field
   - Pairing Code field
   - Connect button

### 6. Complete Setup

1. Wait for networks to load in the dropdown
2. Select your home WiFi network
3. Enter your WiFi password
4. Enter a 4-digit pairing code (e.g., `1234`)
5. Click "Connect"
6. You should see:
   - "Configuration received. Connecting to WiFi..." message
   - Device will switch from AP mode to connect to your WiFi
   - You'll lose connection to Kin_Setup (this is expected)
   - Device should connect to your home WiFi

### 7. Expected Behavior Changes

**Before the fix:**

- ❌ `Kin_Setup` appears but connection fails
- ❌ Network disappears from list quickly
- ❌ "Unable to connect" errors on laptop

**After the fix:**

- ✅ `Kin_Setup` appears with lock icon
- ✅ Connection succeeds immediately with password
- ✅ Connection remains stable
- ✅ Web interface loads without issues
- ✅ Setup process completes successfully

## Troubleshooting

### "I still can't see Kin_Setup"

- Check that the client logs show "Access point started successfully"
- Try running: `sudo nmcli connection show --active` on the Pi
- Make sure your laptop's WiFi is enabled and scanning

### "Connection fails even with the password"

- Make sure you're using exactly: `kinsetup123` (all lowercase, no spaces)
- Check the client logs for any errors
- Try restarting the client
- Verify NetworkManager is installed on the Pi: `sudo apt install network-manager`

### "I can connect but can't access 192.168.4.1:8080"

- Check your laptop got an IP address (should be 192.168.4.x)
- Try pinging: `ping 192.168.4.1`
- Check the client logs for HTTP server errors
- Make sure no firewall is blocking port 8080

### "The password shown on the web page is wrong"

- The page fetches it dynamically from `/ap-info` endpoint
- Check browser console (F12) for JavaScript errors
- If blank, refresh the page

## Advanced: Changing the Default Password

If you want to use a different password, you can modify the code:

**Option 1: In manager.py (around line 28)**

```python
def __init__(
    self,
    ap_ssid: str = "Kin_Setup",
    ap_password: str = "YOUR_PASSWORD_HERE",  # Change this
    ap_interface: str = "wlan0",
    ...
```

**Option 2: Add environment variable support**
This would require adding code to read from `.env` file, e.g.:

```python
ap_password: str = os.getenv("WIFI_SETUP_PASSWORD", "kinsetup123")
```

## Success Criteria

✅ `Kin_Setup` network appears with security indicator
✅ Can connect using password `kinsetup123`
✅ Connection is stable (doesn't disconnect)
✅ Web interface loads and shows connection info
✅ Can select WiFi network and submit credentials
✅ Device successfully switches to home WiFi

If all of these work, the fix is successful! 🎉
