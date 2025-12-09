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
- Check if NetworkManager is running: `systemctl status NetworkManager`

### "Connection fails even with the password"

This is the most common issue. Here's how to diagnose:

**Step 1: Check the logs on the Pi**
Look for these specific log lines:

```
Creating hotspot using nmcli device wifi hotspot...
Access point started successfully via hotspot command
```

OR

```
Hotspot command failed, trying manual method...
Configuring hotspot with WPA2-PSK security...
Access point started successfully via manual method
```

**Step 2: Verify the connection on the Pi**
SSH into your Pi and run:

```bash
# Check if connection is active
sudo nmcli connection show --active

# Check connection details (look for security settings)
sudo nmcli connection show Kin_Hotspot | grep -i security

# Check WiFi interface status
sudo nmcli device status
```

You should see output like:

```
802-11-wireless-security.key-mgmt:      wpa-psk
802-11-wireless-security.proto:         rsn
802-11-wireless-security.pairwise:      ccmp
802-11-wireless-security.group:         ccmp
802-11-wireless-security.psk:           kinsetup123
```

**Step 3: Check NetworkManager version**

```bash
nmcli --version
```

The hotspot command requires NetworkManager 1.16+. If you have an older version, it will fall back to manual configuration.

**Step 4: Manual debugging**
Try creating the hotspot manually to see the exact error:

```bash
# Stop any existing hotspot
sudo nmcli connection down Kin_Hotspot
sudo nmcli connection delete Kin_Hotspot

# Create new hotspot
sudo nmcli device wifi hotspot ifname wlan0 con-name Kin_Hotspot ssid Kin_Setup password kinsetup123

# Check if it's active
sudo nmcli connection show --active
```

**Step 5: Common issues and fixes**

- **"Error: No suitable device found"**: Your wlan0 interface might be busy or named differently
  - Check: `nmcli device status`
  - Try: `sudo nmcli device disconnect wlan0` first
- **"Error: Connection activation failed"**: NetworkManager might have issues
  - Try: `sudo systemctl restart NetworkManager`
  - Check logs: `sudo journalctl -u NetworkManager -n 50`
- **Password rejected on client**: The password might not be set correctly
  - Verify: `sudo nmcli connection show Kin_Hotspot | grep psk`
  - Should show: `802-11-wireless-security.psk: kinsetup123`

**Step 6: Try a different password**
If `kinsetup123` doesn't work, try a more WPA2-friendly password:

```bash
# Use a stronger password with special characters
sudo nmcli connection modify Kin_Hotspot 802-11-wireless-security.psk "KinSetup@2024"
sudo nmcli connection up Kin_Hotspot
```

**Step 7: iOS-specific issues**
iOS has strict WiFi security policies:

- It labels WPA2-PSK as "Weak Security" (this is normal, WPA3 is preferred)
- It may refuse to connect if the password is too simple
- Try using Settings → WiFi → Kin_Setup → "Join Anyway" if you see a warning

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
