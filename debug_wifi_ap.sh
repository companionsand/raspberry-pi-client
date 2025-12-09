#!/bin/bash
# WiFi Access Point Diagnostic Script
# Run this on the Raspberry Pi to diagnose WiFi AP issues

echo "================================"
echo "WiFi Access Point Diagnostics"
echo "================================"
echo ""

# Check if running as root
if [ "$EUID" -eq 0 ]; then
    SUDO=""
else
    SUDO="sudo"
fi

echo "1. NetworkManager Version"
echo "-------------------------"
nmcli --version
echo ""

echo "2. NetworkManager Status"
echo "-------------------------"
systemctl status NetworkManager --no-pager | head -5
echo ""

echo "3. WiFi Interface Status"
echo "-------------------------"
$SUDO nmcli device status | grep -E "DEVICE|wifi"
echo ""

echo "4. Active Connections"
echo "-------------------------"
$SUDO nmcli connection show --active
echo ""

echo "5. Kin_Hotspot Connection (if exists)"
echo "-------------------------"
if $SUDO nmcli connection show Kin_Hotspot &>/dev/null; then
    echo "✓ Kin_Hotspot connection exists"
    echo ""
    echo "Security Settings:"
    $SUDO nmcli connection show Kin_Hotspot | grep -E "802-11-wireless-security|802-11-wireless\.mode|802-11-wireless\.ssid|ipv4\.method|ipv4\.address"
else
    echo "✗ Kin_Hotspot connection not found"
fi
echo ""

echo "6. WiFi Interface Details"
echo "-------------------------"
$SUDO iw dev wlan0 info 2>/dev/null || echo "wlan0 interface not found or iw not installed"
echo ""

echo "7. IP Address on wlan0"
echo "-------------------------"
ip addr show wlan0 2>/dev/null || echo "wlan0 interface not found"
echo ""

echo "8. Recent NetworkManager Logs"
echo "-------------------------"
$SUDO journalctl -u NetworkManager -n 20 --no-pager
echo ""

echo "9. Check for Port/Address Conflicts"
echo "-------------------------"
echo "Processes using port 53 (DNS):"
$SUDO lsof -i :53 2>/dev/null || echo "  (none)"
echo ""
echo "Processes using 192.168.4.1:"
$SUDO lsof -i @192.168.4.1 2>/dev/null || echo "  (none)"
echo ""
echo "dnsmasq processes:"
ps aux | grep -v grep | grep dnsmasq || echo "  (none running)"
echo ""

echo "================================"
echo "Manual Test"
echo "================================"
echo ""
echo "To manually test the hotspot, run:"
echo ""
echo "  # Clean up"
echo "  $SUDO nmcli connection down Kin_Hotspot 2>/dev/null"
echo "  $SUDO nmcli connection delete Kin_Hotspot 2>/dev/null"
echo ""
echo "  # Create hotspot"
echo "  $SUDO nmcli device wifi hotspot ifname wlan0 con-name Kin_Hotspot ssid Kin_Setup password kinsetup123"
echo ""
echo "  # Verify"
echo "  $SUDO nmcli connection show --active | grep Kin_Hotspot"
echo ""
echo "If the hotspot command fails, you may need to update NetworkManager:"
echo "  $SUDO apt update && $SUDO apt install network-manager"
echo ""
