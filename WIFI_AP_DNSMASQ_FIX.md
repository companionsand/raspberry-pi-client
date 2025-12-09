# WiFi AP dnsmasq "Address Already in Use" Fix

## ✅ Automatic Fix Available!

**This issue is now automatically resolved by `launch.sh` on every boot and restart.**

If you're using the `raspberry-pi-client-wrapper` with the systemd service, you don't need to do anything - the fix runs automatically.

See `../raspberry-pi-client-wrapper/WIFI_AP_AUTO_FIX.md` for details.

---

This document is kept for reference and manual troubleshooting.

---

## Problem

The WiFi access point configuration is correct (WPA2 security is properly set up), but the hotspot fails to activate with this error:

```
dnsmasq: failed to create listening socket for 192.168.4.1: Address already in use
device (wlan0): state change: activated -> failed (reason 'ip-config-unavailable')
Activation: failed for connection 'Kin_Hotspot'
```

## Root Cause

When NetworkManager creates a WiFi hotspot, it automatically starts `dnsmasq` to provide DHCP and DNS services to connected clients. The error occurs when:

1. **System-wide dnsmasq service is already running** - Most common cause! A system dnsmasq service (PID 888 in your case) is already running and conflicts with NetworkManager's dnsmasq
2. A previous hotspot attempt didn't clean up properly
3. The 192.168.4.1 IP address is still assigned to the interface
4. dnsmasq processes are still running from the previous attempt
5. Port 53 (DNS) is still bound to that address

### How to Check

Run the diagnostic script to see if you have this issue:

```bash
bash debug_wifi_ap.sh > debug.txt
cat debug.txt | grep "dnsmasq processes:"
```

If you see a dnsmasq process that's NOT related to NetworkManager (doesn't have `--interface=wlan0` or similar), you have a conflict.

## Solution

The code has been updated to perform thorough cleanup before starting the hotspot:

1. **Kill dnsmasq processes** - Remove any lingering dnsmasq instances
2. **Flush IP addresses** - Clear any IP addresses from the interface
3. **Reset interface** - Bring the interface down and back up
4. **Wait between operations** - Give NetworkManager time to fully clean up

## Automated Fix (Recommended)

We've created a script that detects and fixes the conflict automatically:

```bash
# On your Raspberry Pi, in the raspberry-pi-client directory:
sudo bash fix_dnsmasq_conflict.sh
```

This script will:

1. Detect if a system dnsmasq service is running
2. Either disable it (if not needed) OR configure it to exclude wlan0
3. Clean up all NetworkManager dnsmasq processes
4. Reset the wlan0 interface
5. Restart NetworkManager
6. Test hotspot creation
7. Report results

## Manual Fix (If Automated Fix Fails)

If the automated fix doesn't work, run these commands manually on the Pi:

### Step 1: Stop the service

```bash
sudo systemctl stop kin-client
```

### Step 2: Clean up everything

```bash
# Take down any existing hotspot
sudo nmcli connection down Kin_Hotspot 2>/dev/null
sudo nmcli connection delete Kin_Hotspot 2>/dev/null

# Kill all dnsmasq processes
sudo pkill -9 dnsmasq

# Flush IP addresses from wlan0
sudo ip addr flush dev wlan0

# Reset the interface
sudo ip link set wlan0 down
sleep 2
sudo ip link set wlan0 up
sleep 2

# Verify it's clean
ip addr show wlan0
# Should NOT show 192.168.4.1
```

### Step 3: Check for port conflicts

```bash
# Check if anything is listening on port 53
sudo netstat -tulpn | grep :53

# Check if anything is using 192.168.4.1
sudo netstat -tulpn | grep 192.168.4.1
```

If you see any processes, kill them:

```bash
sudo kill -9 <PID>
```

### Step 4: Restart NetworkManager

```bash
sudo systemctl restart NetworkManager
sleep 5
```

### Step 5: Try creating hotspot manually

```bash
sudo nmcli device wifi hotspot ifname wlan0 con-name Kin_Hotspot ssid Kin_Setup password kinsetup123

# Check status
sudo nmcli connection show --active | grep Kin_Hotspot

# Check interface
ip addr show wlan0
# Should now show 192.168.4.1

# Check dnsmasq is running
ps aux | grep dnsmasq
```

### Step 6: If manual hotspot works, restart the service

```bash
# Clean up the manual test
sudo nmcli connection down Kin_Hotspot

# Restart service
sudo systemctl start kin-client

# Watch logs
sudo journalctl -u kin-client -f
```

## Prevention

The updated code now automatically:

- Waits 2 seconds after disconnecting connections (gives NetworkManager time to stop dnsmasq)
- Kills lingering dnsmasq processes
- Flushes IP addresses from the interface
- Resets the interface state

This should prevent the issue from happening in the future.

## System dnsmasq Conflict (Most Common Issue)

### What is it?

You have a system-wide dnsmasq service that was installed (possibly by another package or for local DNS caching). This conflicts with NetworkManager's dnsmasq for the hotspot.

### How to identify:

```bash
ps aux | grep dnsmasq
# Look for a process that does NOT have "NetworkManager" in its path
```

### Solution:

**Disable system dnsmasq (recommended)**

The simplest and most reliable solution is to disable the system dnsmasq service. NetworkManager will handle DNS/DHCP for the WiFi hotspot.

```bash
sudo systemctl stop dnsmasq
sudo systemctl disable dnsmasq
```

Most Raspberry Pi setups don't need a system-wide dnsmasq service. If you do need it for other purposes, you'll need to configure it carefully to not conflict with NetworkManager.

## Common Scenarios

### Scenario 1: Multiple Restart Attempts

**Problem:** Restarting the service multiple times quickly causes conflicts

**Solution:** Always wait at least 5 seconds between restarts:

```bash
sudo systemctl stop kin-client
sleep 5
sudo systemctl start kin-client
```

### Scenario 2: Manual Testing Interference

**Problem:** Testing nmcli commands manually while service is running

**Solution:** Always stop the service first:

```bash
sudo systemctl stop kin-client
# ... do manual testing ...
sudo systemctl start kin-client
```

### Scenario 3: Systemd Doesn't Wait Long Enough

**Problem:** Systemd restarts service too quickly on failure

**Solution:** Check/update the service file to include proper restart delays:

```bash
sudo nano /etc/systemd/system/kin-client.service

# Add or update:
[Service]
RestartSec=10  # Wait 10 seconds before restart
```

Then reload:

```bash
sudo systemctl daemon-reload
sudo systemctl restart kin-client
```

## Verification

After applying the fix, you should see in the logs:

```
[AP] Cleaning up existing hotspot connections...
[AP] Ensuring wlan0 is clean...
[AP] Creating hotspot using nmcli device wifi hotspot...
[AP] Access point started successfully via hotspot command
```

And NetworkManager logs should show:

```
device (wlan0): Activation: (wifi) Stage 2 of 5 (Device Configure) successful. Started Wi-Fi Hotspot "Kin_Setup"
dnsmasq-manager: starting dnsmasq...
device (wlan0): state change: secondaries -> activated
device (wlan0): Activation: successful, device activated.
```

**NOT:**

```
dnsmasq: failed to create listening socket for 192.168.4.1: Address already in use  # ← BAD
```

## Additional Debugging

### Check what's using port 53:

```bash
sudo lsof -i :53
sudo ss -tulpn | grep :53
```

### Check what's using 192.168.4.1:

```bash
sudo lsof -i @192.168.4.1
ip addr | grep 192.168.4.1
```

### Monitor dnsmasq in real-time:

```bash
# Terminal 1: Watch dnsmasq processes
watch -n 0.5 'ps aux | grep dnsmasq'

# Terminal 2: Start the service
sudo systemctl restart kin-client
```

### See NetworkManager's dnsmasq config:

```bash
sudo ls -la /var/run/NetworkManager/
sudo cat /var/run/NetworkManager/dnsmasq-*.conf
```

## Success Indicators

Your hotspot is working when:

1. ✅ `sudo nmcli connection show --active` shows `Kin_Hotspot`
2. ✅ `ip addr show wlan0` shows `inet 192.168.4.1/24`
3. ✅ `ps aux | grep dnsmasq` shows dnsmasq running with `--interface=wlan0`
4. ✅ Your phone/laptop can see `Kin_Setup` network
5. ✅ You can connect with password `kinsetup123`
6. ✅ Your device gets an IP like `192.168.4.2` (check with `ip addr` after connecting)
7. ✅ You can browse to `http://192.168.4.1:8080`

If all these work, your hotspot is fully functional! 🎉
