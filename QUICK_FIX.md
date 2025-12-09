# Quick Fix for WiFi Hotspot Issue

## ⚠️ NOTE: This is Now Automatic!

**As of the latest update, `launch.sh` automatically fixes dnsmasq conflicts on every boot and restart.**

You only need this manual fix if you're troubleshooting or not using the wrapper's `launch.sh`.

See `WIFI_AP_AUTO_FIX.md` in the wrapper directory for details on the automatic fix.

---

## Problem

Your diagnostic shows a **system dnsmasq service** (PID 888) is conflicting with NetworkManager's dnsmasq for the WiFi hotspot.

## Quick Solution

### Option 1: Use the Automated Fix Script (Recommended)

```bash
# Copy the script to your Pi
# Then run:
cd raspberry-pi-client
sudo bash fix_dnsmasq_conflict.sh
```

This will automatically detect and fix the conflict.

### Option 2: Manual Fix (Fast & Reliable)

**Simple approach - just disable system dnsmasq:**

```bash
# Stop the client service
sudo systemctl stop kin-client

# Stop and disable system dnsmasq (NetworkManager will handle DNS/DHCP)
sudo systemctl stop dnsmasq
sudo systemctl disable dnsmasq

# Clean up
sudo pkill -9 dnsmasq
sudo nmcli connection down Kin_Hotspot 2>/dev/null
sudo nmcli connection delete Kin_Hotspot 2>/dev/null
sudo ip addr flush dev wlan0
sudo ip link set wlan0 down && sleep 2 && sudo ip link set wlan0 up
sleep 2

# Restart NetworkManager
sudo systemctl restart NetworkManager
sleep 5

# Test hotspot (should work first time now!)
sudo nmcli device wifi hotspot ifname wlan0 con-name Kin_Hotspot ssid Kin_Setup password kinsetup123

# Check it works
sudo nmcli connection show --active | grep Kin_Hotspot
# Should show: Kin_Hotspot  <uuid>  wifi  wlan0

ip addr show wlan0 | grep inet
# May show: inet 10.42.0.1/24 (NetworkManager default)
# The Python code will change this to 192.168.4.1/24 for consistency

# Optional: Force IP to 192.168.4.1 now for testing web interface
sudo nmcli connection modify Kin_Hotspot ipv4.addresses '192.168.4.1/24'
sudo nmcli connection down Kin_Hotspot
sudo nmcli connection up Kin_Hotspot
ip addr show wlan0 | grep inet
# Should now show: inet 192.168.4.1/24

# If working, clean up test and start service
sudo nmcli connection down Kin_Hotspot
sudo systemctl start kin-client
sudo journalctl -u kin-client -f
```

## Verification

After applying the fix, you should see:

```bash
# Hotspot should be active
sudo nmcli connection show --active | grep Kin_Hotspot
# Should show: Kin_Hotspot  <uuid>  wifi  wlan0

# IP should be assigned
ip addr show wlan0 | grep inet
# Should show: inet 192.168.4.1/24 ...

# NetworkManager's dnsmasq should be running
ps aux | grep dnsmasq | grep wlan0
# Should show a dnsmasq process with --interface=wlan0

# Your phone should see "Kin_Setup" network
```

## What Was Wrong?

Your system has a dnsmasq service (PID 888) that was installed separately. When NetworkManager tries to start its own dnsmasq for the hotspot, they conflict because they can't both bind to the same ports/interfaces.

The fix either:

- **Disables** the system dnsmasq (if you don't need it), OR
- **Configures** it to exclude wlan0 (so NetworkManager can use it)

## Next Steps

Once the fix is applied and the hotspot is working:

1. Try connecting from your phone/laptop to `Kin_Setup` with password `kinsetup123`
2. You should get an IP like `192.168.4.2`
3. Browse to `http://192.168.4.1:8080`
4. You should see the web interface

If it works, you're done! 🎉

If you still have issues, check:

- Is your device's WiFi actually trying to connect?
- Can you see "Kin_Setup" in the WiFi list?
- Does it show as "Secured" with a lock icon?
