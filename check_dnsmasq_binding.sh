#!/bin/bash
# Check what dnsmasq is actually binding to

echo "System dnsmasq process:"
ps aux | grep -v grep | grep "^dnsmasq.*888"

echo ""
echo "What ports is system dnsmasq listening on?"
sudo lsof -p 1511 -a -i 2>/dev/null || echo "Process not found or no network bindings"

echo ""
echo "Check dnsmasq config:"
cat /etc/dnsmasq.d/99-no-wlan0.conf 2>/dev/null || echo "Config file doesn't exist"

echo ""
echo "Is system dnsmasq actually respecting the config?"
sudo systemctl status dnsmasq | head -20
