# Raspberry Pi Production Reliability Guide

> Production-grade settings for Kin AI devices running 24/7 in seniors' homes — no manual reboots, no SD card corruption, no WiFi drops.

---

## Quick Reference

| Priority | Setting | Why |
|----------|---------|-----|
| 🔴 CRITICAL | WiFi power off | Prevents WebSocket disconnects |
| 🔴 CRITICAL | USB autosuspend off | Prevents ReSpeaker disappearing |
| 🔴 CRITICAL | OverlayFS enabled | Prevents SD card corruption on power loss |
| 🔴 CRITICAL | Hardware watchdog | Auto-reboots on OS freeze |
| 🟡 URGENT | ZRAM enabled | Prevents OOM crashes |
| 🟡 URGENT | NTP time sync | Prevents SSL cert failures |
| 🟡 URGENT | Power button disabled | Prevents accidental shutdowns |

---

## 1. CRITICAL: Connectivity & Power

### 1.1 Disable WiFi Power Management

```bash
# Disable immediately
sudo iwconfig wlan0 power off

# Make persistent via NetworkManager
CONN_NAME=$(nmcli -t -f NAME connection show --active | head -1)
sudo nmcli connection modify "$CONN_NAME" 802-11-wireless.powersave 2

# Verify
iwconfig wlan0 | grep Power
# Expected: Power Management:off
```

**Why:** WiFi power saving causes the radio to sleep periodically, killing long-running WebSocket connections to the orchestrator and ElevenLabs.

---

### 1.2 Disable USB Autosuspend (ReSpeaker)

```bash
# Create udev rules for ReSpeaker (VID:PID 2886:0018)
cat <<'EOF' | sudo tee /etc/udev/rules.d/99-respeaker-power.rules
# ReSpeaker 4-Mic Array - prevent USB autosuspend
SUBSYSTEM=="usb", ATTR{idVendor}=="2886", ATTR{idProduct}=="0018", ATTR{power/control}="on"
SUBSYSTEM=="usb", ATTR{idVendor}=="2886", ATTR{idProduct}=="0018", ATTR{power/autosuspend}="-1"
EOF

# Reload udev
sudo udevadm control --reload-rules
sudo udevadm trigger

# Verify
cat /sys/bus/usb/devices/*/idVendor | grep -l 2886 | xargs -I {} dirname {} | xargs -I {} cat {}/power/control
# Expected: on
```

**Why:** USB autosuspend causes the ReSpeaker to "disappear" after periods of silence, causing wake-word detection failures.

---

### 1.3 CPU Performance Governor

```bash
# Set immediately
echo 'performance' | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# Make persistent
sudo apt-get install -y cpufrequtils
echo 'GOVERNOR="performance"' | sudo tee /etc/default/cpufrequtils
sudo systemctl enable cpufrequtils

# Verify
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor
# Expected: performance
```

**Why:** Variable CPU clock speeds cause audio glitching and stuttering during wake-word detection. Performance mode ensures consistent low-latency audio processing.

---

### 1.4 Hardware Watchdog

```bash
# Install watchdog daemon
sudo apt-get install -y watchdog

# Configure watchdog
cat <<'EOF' | sudo tee /etc/watchdog.conf
# Hardware watchdog device
watchdog-device = /dev/watchdog
watchdog-timeout = 15

# Reboot triggers
max-load-1 = 24
min-memory = 1

# Check interval
interval = 10
EOF

# Enable kernel watchdog module (Pi 5)
echo 'dtparam=watchdog=on' | sudo tee -a /boot/firmware/config.txt

# Enable and start
sudo systemctl enable watchdog
sudo systemctl start watchdog

# Verify
sudo systemctl status watchdog
# Expected: active (running)
```

**Why:** Hard-reboots the device if the OS freezes. Essential for devices in seniors' homes where manual reboots aren't an option.

---

## 2. CRITICAL: Filesystem Protection

### 2.1 Enable Read-Only Overlay (OverlayFS)

**⚠️ CRITICAL FOR PRODUCTION:** This is the #1 protection against SD card corruption.

```bash
# Enable via raspi-config (interactive)
sudo raspi-config
# Navigate to: Performance Options → Overlay File System → Enable

# Or enable non-interactively
sudo raspi-config nonint do_overlayfs 0

# Reboot to activate
sudo reboot
```

**To temporarily disable for updates:**
```bash
# Disable overlay (requires reboot)
sudo raspi-config nonint do_overlayfs 1
sudo reboot

# Make changes...

# Re-enable overlay
sudo raspi-config nonint do_overlayfs 0
sudo reboot
```

**Why:** Seniors will unplug the device to "turn it off." Without OverlayFS, power loss during write operations corrupts the SD card, bricking the device permanently.

---

### 2.2 Log Limiting (Prevent SD Card Fill)

```bash
# Configure journald limits
cat <<'EOF' | sudo tee /etc/systemd/journald.conf.d/size-limit.conf
[Journal]
SystemMaxUse=50M
SystemMaxFileSize=10M
MaxRetentionSec=7day
Compress=yes
EOF

# Apply immediately
sudo systemctl restart systemd-journald

# Verify
journalctl --disk-usage
# Should show < 50M
```

**Why:** Default log rotation isn't aggressive enough. A runaway error loop (e.g., audio device disconnecting) can fill the SD card in minutes, crashing the OS.

---

## 3. CRITICAL: Memory Management

### 3.1 Enable ZRAM (Compressed Swap)

```bash
# Install zram-tools
sudo apt-get install -y zram-tools

# Configure ZRAM (50% of RAM, compressed)
cat <<'EOF' | sudo tee /etc/default/zramswap
ALGO=zstd
PERCENT=50
PRIORITY=100
EOF

# Enable and start
sudo systemctl enable zramswap
sudo systemctl start zramswap

# Verify
swapon --show
# Should show /dev/zram0 with ~2GB (on 4GB Pi)

# Check compression ratio
cat /sys/block/zram0/comp_algorithm
# Expected: zstd
```

**Why:** Pi 5 has limited RAM for simultaneous LLM streaming + wake word detection + audio processing. ZRAM compresses memory to prevent OOM crashes without wearing out the SD card (unlike swap files).

---

### 3.2 OOM Protection for Kin Service

```bash
# Protect Kin service from OOM killer
sudo mkdir -p /etc/systemd/system/kin-client.service.d
cat <<'EOF' | sudo tee /etc/systemd/system/kin-client.service.d/oom.conf
[Service]
OOMScoreAdjust=-900
EOF

sudo systemctl daemon-reload
```

**Why:** When memory is low, Linux kills processes to free RAM. OOMScoreAdjust=-900 ensures Kin is one of the last processes killed.

---

## 4. CRITICAL: Time Synchronization

### 4.1 NTP Time Sync Check

**Add to application startup (Python):**
```python
import time
import subprocess
import logging

def wait_for_valid_time(timeout: int = 120) -> bool:
    """
    Block application start until system time is valid.
    Pi 5 has no battery-backed RTC - boots with wrong date (1970).
    Invalid time causes SSL cert validation failures.
    """
    start = time.time()
    while time.time() - start < timeout:
        # Check if year is reasonable (after 2024)
        if time.localtime().tm_year >= 2024:
            logging.info("✓ System time is valid")
            return True
        
        # Force NTP sync
        subprocess.run(['sudo', 'timedatectl', 'set-ntp', 'true'], 
                      capture_output=True)
        
        logging.info("⏳ Waiting for NTP time sync...")
        time.sleep(5)
    
    logging.error("✗ Time sync failed - SSL connections will fail")
    return False

# Call at application startup, before any HTTPS/WSS connections
if not wait_for_valid_time():
    # Either exit or continue with warning
    pass
```

**System-level fix (fallback):**
```bash
# Ensure timesyncd is enabled
sudo timedatectl set-ntp true

# Add fallback NTP servers
cat <<'EOF' | sudo tee /etc/systemd/timesyncd.conf.d/fallback.conf
[Time]
NTP=time.google.com time.cloudflare.com
FallbackNTP=pool.ntp.org
EOF

sudo systemctl restart systemd-timesyncd
```

**Why:** Pi 5 has no battery-backed RTC. If it boots with the wrong date (1970), all SSL certificate validations fail, and backend connections are rejected.

---

## 5. CRITICAL: Physical Device Protection

### 5.1 Disable Physical Power Button

```bash
# Method 1: Via device tree overlay (preferred)
echo 'dtoverlay=gpio-poweroff,gpiopin=4,active_low=0' | sudo tee -a /boot/firmware/config.txt

# Method 2: Via logind.conf (disable power key handling)
sudo mkdir -p /etc/systemd/logind.conf.d
cat <<'EOF' | sudo tee /etc/systemd/logind.conf.d/disable-power-button.conf
[Login]
HandlePowerKey=ignore
HandlePowerKeyLongPress=ignore
EOF

sudo systemctl restart systemd-logind

# Verify
loginctl show-session | grep HandlePowerKey
# Expected: HandlePowerKey=ignore
```

**Why:** Seniors may accidentally press the power button while handling the device. Ignoring it prevents unexpected shutdowns.

---

### 5.2 Disable Unused Interfaces (Power Saving)

```bash
# Disable Bluetooth (if not used)
echo 'dtoverlay=disable-bt' | sudo tee -a /boot/firmware/config.txt

# Disable HDMI (if no display)
# Add to /boot/firmware/config.txt:
# hdmi_blanking=2

# Disable onboard LEDs (optional - reduces visual distraction)
echo 'dtparam=act_led_trigger=none' | sudo tee -a /boot/firmware/config.txt
echo 'dtparam=pwr_led_trigger=none' | sudo tee -a /boot/firmware/config.txt
```

**Why:** Reduces power consumption and potential interference. Bluetooth can interfere with WiFi on 2.4GHz.

---

## 6. IMPORTANT: Network Stability

### 6.1 Use DHCP (NOT Static IP)

```bash
# Ensure NetworkManager uses DHCP (default)
CONN_NAME=$(nmcli -t -f NAME connection show --active | head -1)
sudo nmcli connection modify "$CONN_NAME" ipv4.method auto

# Remove any static IP configuration
sudo nmcli connection modify "$CONN_NAME" ipv4.addresses ""
sudo nmcli connection modify "$CONN_NAME" ipv4.gateway ""

# Verify
nmcli connection show "$CONN_NAME" | grep ipv4.method
# Expected: ipv4.method: auto
```

**Why:** Hardcoding static IPs (e.g., 192.168.1.100) bricks connectivity when the device enters a senior's home with a different router subnet (e.g., 10.0.0.1/24).

---

### 6.2 Aggressive Auto-Reconnect

```bash
CONN_NAME=$(nmcli -t -f NAME connection show --active | head -1)

# Enable aggressive auto-reconnect
sudo nmcli connection modify "$CONN_NAME" \
  connection.autoconnect yes \
  connection.autoconnect-priority 100 \
  connection.autoconnect-retries 0

# 0 = infinite retries
```

**Why:** Ensures device automatically reconnects to WiFi after router reboots or temporary outages.

---

### 6.3 TCP Keepalives for WebSockets

```bash
cat <<'EOF' | sudo tee /etc/sysctl.d/99-tcp-keepalive.conf
# Aggressive keepalives for WebSocket connections
net.ipv4.tcp_keepalive_time = 60
net.ipv4.tcp_keepalive_intvl = 10
net.ipv4.tcp_keepalive_probes = 6
EOF

sudo sysctl -p /etc/sysctl.d/99-tcp-keepalive.conf
```

**Why:** Detects dead WebSocket connections faster, allowing the application to reconnect.

---

## 7. IMPORTANT: Audio Configuration (Pure ALSA)

### 7.1 Disable PipeWire and Configure ALSA

**Kin AI uses pure ALSA for lowest latency and direct hardware control.**

```bash
# 1. Completely remove PipeWire (Pi 5 Bookworm uses it by default)
systemctl --user stop pipewire pipewire-pulse wireplumber
systemctl --user disable pipewire pipewire-pulse wireplumber
systemctl --user mask pipewire pipewire-pulse wireplumber
sudo apt-get remove --purge pipewire pipewire-pulse wireplumber

# 2. Configure ALSA for ReSpeaker (card 2)
cat <<'EOF' | sudo tee /etc/asound.conf
pcm.!default {
    type asym
    playback.pcm "plughw:2,0"
    capture.pcm  "plughw:2,0"
}
ctl.!default {
    type hw
    card 2
}
EOF

# 3. Verify ReSpeaker is detected
arecord -l
aplay -l
# Should show card 2: ReSpeaker

# 4. Test audio
arecord -d 3 -f S16_LE -r 16000 -c 1 test.wav
aplay test.wav
```

**Why:** Pure ALSA provides lowest latency for wake-word detection and eliminates audio routing complexity. PipeWire adds unnecessary overhead for our use case.

---

## 8. IMPORTANT: Service Configuration

### 8.1 Systemd Service Hardening

```bash
cat <<'EOF' | sudo tee /etc/systemd/system/kin-client.service
[Unit]
Description=Kin AI Voice Assistant
After=network-online.target sound.target
Wants=network-online.target
StartLimitIntervalSec=300
StartLimitBurst=5

[Service]
Type=simple
User=kin
WorkingDirectory=/home/kin/kin-ai-prototype
Environment="PATH=/home/kin/kin-ai-env/bin"
ExecStart=/home/kin/kin-ai-env/bin/python main.py

# Restart behavior
Restart=always
RestartSec=10

# Resource limits
OOMScoreAdjust=-900
LimitNOFILE=65536

# Security hardening
NoNewPrivileges=true
ProtectSystem=strict
ProtectHome=read-only
ReadWritePaths=/home/kin/kin-ai-prototype/data

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable kin-client
```

---

## 9. PRODUCTION CHECKLIST ✅

### Pre-Deployment (One-Time Setup)

```bash
#!/bin/bash
# production-setup.sh - Run this before shipping device

set -e
echo "🏭 Kin AI Production Setup"

# 1. WiFi Power Management
echo "📡 Disabling WiFi power management..."
sudo iwconfig wlan0 power off
CONN_NAME=$(nmcli -t -f NAME connection show --active | head -1)
sudo nmcli connection modify "$CONN_NAME" 802-11-wireless.powersave 2

# 2. USB Autosuspend
echo "🔌 Disabling USB autosuspend..."
cat <<'EOF' | sudo tee /etc/udev/rules.d/99-respeaker-power.rules
SUBSYSTEM=="usb", ATTR{idVendor}=="2886", ATTR{idProduct}=="0018", ATTR{power/control}="on"
SUBSYSTEM=="usb", ATTR{idVendor}=="2886", ATTR{idProduct}=="0018", ATTR{power/autosuspend}="-1"
EOF
sudo udevadm control --reload-rules

# 3. CPU Performance
echo "⚡ Setting CPU performance mode..."
sudo apt-get install -y cpufrequtils
echo 'GOVERNOR="performance"' | sudo tee /etc/default/cpufrequtils
sudo systemctl enable cpufrequtils

# 4. Watchdog
echo "🐕 Enabling hardware watchdog..."
sudo apt-get install -y watchdog
cat <<'EOF' | sudo tee /etc/watchdog.conf
watchdog-device = /dev/watchdog
watchdog-timeout = 15
max-load-1 = 24
min-memory = 1
interval = 10
EOF
echo 'dtparam=watchdog=on' | sudo tee -a /boot/firmware/config.txt
sudo systemctl enable watchdog

# 5. ZRAM
echo "💾 Enabling ZRAM..."
sudo apt-get install -y zram-tools
cat <<'EOF' | sudo tee /etc/default/zramswap
ALGO=zstd
PERCENT=50
PRIORITY=100
EOF
sudo systemctl enable zramswap

# 6. Log Limiting
echo "📝 Configuring log limits..."
sudo mkdir -p /etc/systemd/journald.conf.d
cat <<'EOF' | sudo tee /etc/systemd/journald.conf.d/size-limit.conf
[Journal]
SystemMaxUse=50M
SystemMaxFileSize=10M
MaxRetentionSec=7day
Compress=yes
EOF

# 7. Power Button
echo "🔘 Disabling power button..."
sudo mkdir -p /etc/systemd/logind.conf.d
cat <<'EOF' | sudo tee /etc/systemd/logind.conf.d/disable-power-button.conf
[Login]
HandlePowerKey=ignore
HandlePowerKeyLongPress=ignore
EOF

# 8. TCP Keepalives
echo "🌐 Configuring TCP keepalives..."
cat <<'EOF' | sudo tee /etc/sysctl.d/99-tcp-keepalive.conf
net.ipv4.tcp_keepalive_time = 60
net.ipv4.tcp_keepalive_intvl = 10
net.ipv4.tcp_keepalive_probes = 6
EOF

# 9. Disable Bluetooth
echo "📵 Disabling Bluetooth..."
echo 'dtoverlay=disable-bt' | sudo tee -a /boot/firmware/config.txt

# 10. Time sync
echo "⏰ Configuring NTP..."
sudo timedatectl set-ntp true

echo ""
echo "✅ Base configuration complete!"
echo ""
echo "⚠️  FINAL STEP: Enable OverlayFS before shipping!"
echo "    Run: sudo raspi-config → Performance → Overlay File System → Enable"
echo ""
echo "🔄 Reboot required: sudo reboot"
```

### Final Pre-Ship Verification

```bash
#!/bin/bash
# verify-production.sh - Run before shipping

echo "🔍 Production Verification Checklist"
echo ""

# WiFi power
WIFI_POWER=$(iwconfig wlan0 2>/dev/null | grep -o "Power Management:.*")
echo "WiFi Power: $WIFI_POWER"
[[ "$WIFI_POWER" == *"off"* ]] && echo "  ✅ OK" || echo "  ❌ FAIL"

# CPU governor
CPU_GOV=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor)
echo "CPU Governor: $CPU_GOV"
[[ "$CPU_GOV" == "performance" ]] && echo "  ✅ OK" || echo "  ❌ FAIL"

# Watchdog
WD_STATUS=$(systemctl is-active watchdog)
echo "Watchdog: $WD_STATUS"
[[ "$WD_STATUS" == "active" ]] && echo "  ✅ OK" || echo "  ❌ FAIL"

# ZRAM
ZRAM=$(swapon --show | grep zram)
echo "ZRAM: ${ZRAM:-not found}"
[[ -n "$ZRAM" ]] && echo "  ✅ OK" || echo "  ❌ FAIL"

# OverlayFS
OVERLAY=$(mount | grep "overlay on / ")
echo "OverlayFS: ${OVERLAY:-not enabled}"
[[ -n "$OVERLAY" ]] && echo "  ✅ OK" || echo "  ❌ FAIL - ENABLE BEFORE SHIPPING!"

# Time
YEAR=$(date +%Y)
echo "System Year: $YEAR"
[[ "$YEAR" -ge "2024" ]] && echo "  ✅ OK" || echo "  ❌ FAIL"

# Power button
PB=$(cat /etc/systemd/logind.conf.d/disable-power-button.conf 2>/dev/null | grep HandlePowerKey)
echo "Power Button: ${PB:-not configured}"
[[ "$PB" == *"ignore"* ]] && echo "  ✅ OK" || echo "  ❌ FAIL"

echo ""
echo "🏁 Verification complete"
```

---

## 10. ADDITIONAL CONSIDERATIONS

### What About Firmware Updates?

```bash
# Check Pi 5 EEPROM version
sudo rpi-eeprom-update

# Update if needed (do before enabling OverlayFS)
sudo rpi-eeprom-update -a
```

### What About OTA Updates?

With OverlayFS enabled, the filesystem is read-only. For OTA updates:
1. Remotely disable OverlayFS
2. Reboot into writable mode
3. Apply updates
4. Re-enable OverlayFS
5. Reboot

Consider a dedicated update partition or A/B partitioning for safer OTAs.

### What About Offline Operation?

Current implementation requires internet. For true Alexa-like reliability, consider:
- Local wake-word detection (already done via Porcupine)
- Cached "I'm having trouble connecting" audio response
- LED indication for connectivity status
- Automatic retry with exponential backoff

### What About Remote Monitoring?

For a fleet of devices, consider:
- Heartbeat endpoint (device pings server every N minutes)
- Error log aggregation (OpenTelemetry already configured)
- Remote reboot capability via backend command
- Firmware version reporting

---

**Last Updated:** December 2024  
**Target:** Raspberry Pi 5 + Bookworm + ReSpeaker 4-Mic  
**Use Case:** Always-on voice assistant in seniors' homes
