#!/usr/bin/env python3
"""
Radar Sensor Diagnostic Tool

Helps diagnose why the radar sensor isn't being detected.
Checks USB devices, serial ports, and attempts connection.

Usage:
    python scripts/diagnose_radar.py
"""

import sys
import os
import glob
import subprocess

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("="*60)
print("  MR60FDA1 Radar Sensor Diagnostic")
print("="*60)
print()

# 1. Check USB devices
print("1. Checking USB devices...")
try:
    result = subprocess.run(['lsusb'], capture_output=True, text=True, timeout=5)
    if result.returncode == 0:
        usb_output = result.stdout
        print(usb_output)
        
        # Look for CH340 (common USB-Serial chip)
        if '1a86' in usb_output.lower() or '7523' in usb_output.lower():
            print("   ✓ Found CH340 USB-Serial adapter (common with MR60FDA1)")
        else:
            print("   ⚠️  No CH340 adapter found in lsusb")
            print("      The radar may use a different USB-Serial chip")
    else:
        print("   ✗ Could not run lsusb")
except Exception as e:
    print(f"   ✗ Error checking USB: {e}")

print()

# 2. Check serial ports
print("2. Checking available serial ports...")
tty_usb = glob.glob('/dev/ttyUSB*')
tty_acm = glob.glob('/dev/ttyACM*')
tty_radar = glob.glob('/dev/radar*')

all_ports = sorted(tty_usb + tty_acm + tty_radar)

if all_ports:
    print(f"   Found {len(all_ports)} port(s):")
    for port in all_ports:
        print(f"     - {port}")
        
        # Check permissions
        if os.access(port, os.R_OK | os.W_OK):
            print(f"       ✓ Read/write access OK")
        else:
            print(f"       ✗ Permission denied (try: sudo chmod 666 {port})")
else:
    print("   ✗ No serial ports found!")
    print("      Make sure the radar is connected via USB")

print()

# 3. Try to detect radar on each port
if all_ports:
    print("3. Testing ports for radar heartbeat...")
    import serial
    import time
    
    HEADER = bytes([0x53, 0x59])
    
    for port in all_ports:
        print(f"\n   Testing {port}...")
        try:
            ser = serial.Serial(port, 115200, timeout=1.0)
            time.sleep(0.5)  # Wait for data
            
            if ser.in_waiting:
                data = ser.read(ser.in_waiting)
                if HEADER in data:
                    print(f"      ✓ FOUND! Radar detected on {port}")
                    print(f"         Data sample: {data[:20].hex()}")
                else:
                    print(f"      ⚠️  Port active but no radar header found")
                    print(f"         Data sample: {data[:20].hex()}")
            else:
                print(f"      ⚠️  Port opened but no data received")
                print(f"         (Radar may need time to send heartbeat)")
            
            ser.close()
        except serial.SerialException as e:
            print(f"      ✗ Could not open: {e}")
        except PermissionError:
            print(f"      ✗ Permission denied (try: sudo chmod 666 {port})")
        except Exception as e:
            print(f"      ✗ Error: {e}")
else:
    print("3. Skipping port tests (no ports found)")

print()

# 4. Check udev rules
print("4. Checking udev rules...")
udev_rules = '/etc/udev/rules.d/99-mr60fda1.rules'
if os.path.exists(udev_rules):
    print(f"   ✓ udev rule exists: {udev_rules}")
    with open(udev_rules, 'r') as f:
        content = f.read()
        if 'radar0' in content:
            print("   ✓ Rule contains radar0 symlink")
        else:
            print("   ⚠️  Rule doesn't create radar0 symlink")
else:
    print(f"   ⚠️  udev rule not found: {udev_rules}")
    print("      Install with: sudo cp config/99-mr60fda1.rules /etc/udev/rules.d/")

print()

# 5. Check if radar0 symlink exists
print("5. Checking for /dev/radar0 symlink...")
if os.path.exists('/dev/radar0'):
    print("   ✓ /dev/radar0 exists")
    if os.path.islink('/dev/radar0'):
        target = os.readlink('/dev/radar0')
        print(f"      → Points to: {target}")
else:
    print("   ⚠️  /dev/radar0 not found (udev rule may not be active)")

print()

# 6. Recommendations
print("="*60)
print("  Recommendations:")
print("="*60)
print()

if not all_ports:
    print("1. ✗ NO SERIAL PORTS FOUND")
    print("   → Check USB connection (try different port)")
    print("   → Check if radar is powered (LED should be on)")
    print("   → Try: dmesg | tail -20 (look for USB device messages)")
    print()
elif len(all_ports) == 1:
    print(f"1. Try manual port specification:")
    print(f"   python scripts/test_radar.py --port {all_ports[0]}")
    print()
else:
    print("1. Multiple ports found - try each manually:")
    for port in all_ports:
        print(f"   python scripts/test_radar.py --port {port}")
    print()

print("2. If permission errors:")
print("   sudo chmod 666 /dev/ttyUSB*")
print("   OR add user to dialout group: sudo usermod -aG dialout $USER")
print()

print("3. If radar still not detected:")
print("   → Check radar power LED is on")
print("   → Try different USB cable/port")
print("   → Check dmesg for USB errors: dmesg | grep -i usb")
print()

print("="*60)

