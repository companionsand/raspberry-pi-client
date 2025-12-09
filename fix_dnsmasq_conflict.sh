#!/bin/bash
# Fix dnsmasq conflict for WiFi hotspot
# Run this on the Raspberry Pi as root or with sudo

set -e

echo "================================"
echo "Fixing dnsmasq Conflict"
echo "================================"
echo ""

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo "Please run as root or with sudo"
    exit 1
fi

echo "Step 1: Check for system dnsmasq service"
echo "----------------------------------------"
if systemctl is-active --quiet dnsmasq; then
    echo "✓ System dnsmasq service is running"
    echo ""
    echo "Checking if dnsmasq is needed..."
    
    # Check if anything depends on it
    DEPS=$(systemctl list-dependencies --reverse dnsmasq.service | grep -v dnsmasq.service | wc -l)
    
    if [ "$DEPS" -gt 1 ]; then
        echo "⚠️  Other services depend on dnsmasq"
        echo "    Will configure it to not conflict instead"
        STOP_DNSMASQ=false
    else
        echo "✓ No critical dependencies found"
        echo "    Will disable system dnsmasq"
        STOP_DNSMASQ=true
    fi
else
    echo "✓ System dnsmasq service is not running"
    STOP_DNSMASQ=false
fi
echo ""

echo "Step 2: Stop any Kin client services"
echo "----------------------------------------"
systemctl stop kin-client 2>/dev/null || true
echo "✓ Services stopped"
echo ""

echo "Step 3: Clean up existing hotspot"
echo "----------------------------------------"
nmcli connection down Kin_Hotspot 2>/dev/null || true
nmcli connection delete Kin_Hotspot 2>/dev/null || true
echo "✓ Hotspot cleaned up"
echo ""

echo "Step 4: Kill all NetworkManager dnsmasq processes"
echo "----------------------------------------"
pkill -9 -f "dnsmasq.*NetworkManager" 2>/dev/null || true
sleep 2
echo "✓ NetworkManager dnsmasq processes killed"
echo ""

if [ "$STOP_DNSMASQ" = true ]; then
    echo "Step 5: Disable system dnsmasq service"
    echo "----------------------------------------"
    systemctl stop dnsmasq
    systemctl disable dnsmasq
    echo "✓ System dnsmasq disabled"
    echo ""
else
    echo "Step 5: Configure system dnsmasq to not bind to all interfaces"
    echo "----------------------------------------"
    
    # Create a config file to bind only to specific interfaces (not wlan0)
    cat > /etc/dnsmasq.d/99-no-wlan0.conf <<EOF
# Don't bind to wlan0 - let NetworkManager handle it
except-interface=wlan0
EOF
    
    echo "✓ Created /etc/dnsmasq.d/99-no-wlan0.conf"
    
    # Restart dnsmasq to apply changes
    systemctl restart dnsmasq
    echo "✓ System dnsmasq restarted with new config"
    echo ""
fi

echo "Step 6: Reset wlan0 interface"
echo "----------------------------------------"
ip addr flush dev wlan0 2>/dev/null || true
ip link set wlan0 down
sleep 2
ip link set wlan0 up
sleep 2
echo "✓ wlan0 interface reset"
echo ""

echo "Step 7: Restart NetworkManager"
echo "----------------------------------------"
systemctl restart NetworkManager
sleep 5
echo "✓ NetworkManager restarted"
echo ""

echo "Step 8: Test hotspot creation"
echo "----------------------------------------"
echo "Creating test hotspot..."
if nmcli device wifi hotspot ifname wlan0 con-name Kin_Hotspot ssid Kin_Setup password kinsetup123; then
    echo "✓ Hotspot created successfully!"
    echo ""
    
    sleep 3
    
    echo "Checking status..."
    if nmcli connection show --active | grep -q Kin_Hotspot; then
        echo "✅ Hotspot is ACTIVE"
        
        # Check IP address
        IP=$(ip addr show wlan0 | grep "inet " | awk '{print $2}')
        if [ -n "$IP" ]; then
            echo "✅ IP address assigned: $IP"
        else
            echo "⚠️  No IP address on wlan0"
        fi
        
        # Check dnsmasq
        if ps aux | grep -v grep | grep -q "dnsmasq.*wlan0"; then
            echo "✅ dnsmasq is running for wlan0"
        else
            echo "⚠️  dnsmasq not running for wlan0"
        fi
    else
        echo "❌ Hotspot not active"
    fi
else
    echo "❌ Failed to create hotspot"
    echo ""
    echo "Check NetworkManager logs:"
    echo "  sudo journalctl -u NetworkManager -n 50"
fi
echo ""

echo "Step 9: Clean up test and restore service"
echo "----------------------------------------"
nmcli connection down Kin_Hotspot 2>/dev/null || true
sleep 2
echo "✓ Test hotspot stopped"
echo ""

echo "================================"
echo "Fix Complete!"
echo "================================"
echo ""
echo "You can now start the kin-client service:"
echo "  sudo systemctl start kin-client"
echo "  sudo journalctl -u kin-client -f"
echo ""

if [ "$STOP_DNSMASQ" = true ]; then
    echo "NOTE: System dnsmasq service has been DISABLED."
    echo "If you need it for other purposes, you can re-enable it with:"
    echo "  sudo systemctl enable dnsmasq"
    echo "  sudo systemctl start dnsmasq"
    echo "Then use the configuration approach instead."
fi
