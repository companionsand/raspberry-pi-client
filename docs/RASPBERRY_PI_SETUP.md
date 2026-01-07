# Raspberry Pi 5 + ReSpeaker Quick Setup

> Run the Kin AI Pi client end to end in ~15 minutes.

---

## 1. Hardware Checklist

- Raspberry Pi 5 (2GB min, 4GB recommended) + 5V/3A USB‑C PSU
- ReSpeaker 4 Mic Array v2.0 (USB) + powered speaker on the ReSpeaker 3.5 mm jack
- 32 GB micro‑SD (Raspberry Pi OS Lite 64‑bit)
- Ethernet or strong Wi‑Fi, heatsink/fan recommended

---

## 2. Base System

```bash
# flash OS with Pi Imager (enable SSH, set hostname kin-ai, user kin)
ssh kin@kin-ai.local        # default password you set

# keep SSH alive
cat <<'EOF' >> ~/.ssh/config
Host kin-ai.local
    ServerAliveInterval 60
    ServerAliveCountMax 3
    TCPKeepAlive yes
EOF

# basic packages
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3-venv python3-pip alsa-utils ffmpeg git
```

Disable PipeWire/PulseAudio (we rely on ALSA + ReSpeaker hardware AEC):

```bash
systemctl --user stop pipewire pipewire-pulse wireplumber
systemctl --user disable pipewire pipewire-pulse wireplumber
systemctl --user mask pipewire pipewire-pulse wireplumber
```

---

## 3. Audio (ALSA‑only, ReSpeaker as default)

**Note:** Audio configuration is now **automatic**! The raspberry-pi-client service will:

- Auto-detect ReSpeaker and its ALSA card number (may be card 1, 2, 3, etc.)
- Configure `/etc/asound.conf` with the correct card number, AEC channel routing, and software volume control
- Set initial volume from backend configuration (`speaker_volume_percent`, default 50%)

Verify ReSpeaker is detected:

```bash
arecord -l   # confirm ReSpeaker (usually card 2)
aplay -l
```

**Manual configuration (optional):** If you need to manually configure before running the service:

First, find your ReSpeaker card number:

```bash
aplay -l  # Look for ReSpeaker/ArrayUAC10 - note the card number
```

Then create the config (replace `2` with your card number if different):

```bash
cat <<'EOF' | sudo tee /etc/asound.conf
# ALSA Configuration - ReSpeaker with Software Volume
pcm.respeaker_aec {
    type route
    slave {
        pcm "hw:2,0"
        channels 6
    }
    ttable.0.0 1
}

pcm.respeaker_mono {
    type plug
    slave.pcm "respeaker_aec"
}

pcm.respeaker_out_raw {
    type plug
    slave.pcm "hw:2,0"
}

pcm.respeaker_out {
    type softvol
    slave.pcm "respeaker_out_raw"
    control {
        name "Softvol"
        card 2
    }
    min_dB -30.0
    max_dB 0.0
}

pcm.!default {
    type asym
    playback.pcm "respeaker_out"
    capture.pcm "respeaker_mono"
}

ctl.!default {
    type hw
    card 2
}
EOF
```

Test audio:

```bash
arecord -d 3 -f S16_LE -r 16000 -c 1 test.wav
aplay test.wav
```

You should hear yourself from the speaker plugged into the ReSpeaker.

**Volume Control:**

```bash
# Find your ReSpeaker card number first:
aplay -l  # Look for ReSpeaker card number (e.g., card 2)

# Temporary adjustment (resets on next service start):
alsamixer -c <card_number>  # Adjust 'Softvol' control (0-100)

# Or via command line:
amixer -c <card_number> set Softvol 75%

# Example for card 2:
alsamixer -c 2
amixer -c 2 set Softvol 75%

# Persistent (survives restarts):
# Update speaker_volume_percent in device settings via admin portal
# Volume will be applied automatically on next service start
```

---

## 4. App Install

```bash
# one‑time
python3 -m venv ~/kin-ai-env
source ~/kin-ai-env/bin/activate
mkdir -p ~/kin-ai-prototype
cd ~/kin-ai-prototype

# copy code from laptop
scp pi-client/new_main.py kin@kin-ai.local:~/kin-ai-prototype/main.py
scp pi-client/pyproject.toml kin@kin-ai.local:~/kin-ai-prototype/pyproject.toml

# Install dependencies using uv
uv sync
```

`.env` template:

```bash
cat <<'EOF' > ~/kin-ai-prototype/.env
# Required device credentials (from admin portal)
DEVICE_ID=your-device-uuid-here
DEVICE_PRIVATE_KEY=your-base64-encoded-private-key-here

# Optional configuration
SKIP_WIFI_SETUP=true
OTEL_ENABLED=true
ENV=production
EOF
chmod 600 ~/kin-ai-prototype/.env
```

**Important:** Device credentials must be provisioned through the admin portal. The .env file is included in the installer package downloaded from the portal. Most configuration (API keys, wake word, LED settings) is fetched from backend after authentication.

Run it:

```bash
source ~/kin-ai-env/bin/activate
cd ~/kin-ai-prototype
python main.py
```

Expected logs: Device auth success → ALSA verification → “Ready! Say ‘porcupine’…”.

---

## 5. Configure Sudoers (Required for Auto-Configuration)

The service needs passwordless sudo for audio configuration (ALSA, ReSpeaker tuning).

```bash
# Run the setup script
sudo ./scripts/setup_sudoers.sh kin  # or 'aushim' for development

# Verify it worked
sudo -l -U kin | grep tee  # Should show /usr/bin/tee /etc/asound.conf
```

This allows the service to:

- Auto-configure `/etc/asound.conf`
- Apply ReSpeaker tuning parameters
- Set audio volume controls

---

## 6. Deploy as Service (optional)

```bash
cat <<'EOF' | sudo tee /etc/systemd/system/kin-client.service
[Unit]
Description=Kin AI Client
After=network-online.target sound.target

[Service]
User=kin
WorkingDirectory=/home/kin/kin-ai-prototype
Environment="PATH=/home/kin/kin-ai-env/bin"
ExecStart=/home/kin/kin-ai-env/bin/python main.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable --now kin-client
```

Useful commands:

```bash
sudo systemctl status kin-client
sudo journalctl -u kin-client -f
```

---

## 6. WiFi Setup Mode (Optional - for devices without initial network access)

If your Raspberry Pi doesn't have network connectivity, you can use WiFi setup mode:

1. Set `SKIP_WIFI_SETUP=false` in your `.env` file (or omit it, as false is the default)
2. When the device boots, it will create a WiFi access point:
   - **Network Name:** `Kin_Setup`
   - **Password:** `kinsetup123`
3. Connect your laptop/phone to the `Kin_Setup` network
4. Open a web browser and go to: `http://192.168.4.1:8080`
5. Select your home WiFi network, enter the password, and provide a 4-digit pairing code
6. The device will connect to your WiFi and complete authentication

**Note:** The web interface will display the connection info at the top in case you get disconnected.

---

## 7. Troubleshooting Cheat Sheet

| Symptom                                | Fix                                                                                             |
| -------------------------------------- | ----------------------------------------------------------------------------------------------- |
| `arecord` silent / -91 dB              | Re-check `/etc/asound.conf`, run `arecord -D plughw:2,0 ...`, ensure ReSpeaker is card 2.       |
| `python -c "import sounddevice"` hangs | PipeWire still running → re-run the disable commands above.                                     |
| Wake word ignored                      | Run `arecord -d 3 test.wav`; if silent, check mic gain (`amixer -c 2`).                         |
| No speaker audio                       | Ensure speaker is on ReSpeaker jack, run `aplay test.wav`, check `amixer -c 2 set Speaker 80%`. |
| LEDs off                               | `pip install pixel-ring`, or set `LED_ENABLED=false`.                                           |
| Ctrl+C not working                     | Latest `main.py` handles SIGINT; ensure you redeployed it.                                      |
| LED "Permission denied" error          | Setup udev rules for USB HID access - see below.                                                |

### LED Permission Fix

If you see "Permission denied" when initializing LEDs, you need udev rules for USB HID access:

```bash
# Create udev rules for ReSpeaker LED control
sudo tee /etc/udev/rules.d/99-respeaker.rules > /dev/null <<'EOF'
# ReSpeaker 4-Mic Array (USB VID:PID 2886:0018)
SUBSYSTEM=="usb", ATTR{idVendor}=="2886", ATTR{idProduct}=="0018", MODE="0666", GROUP="plugdev"
SUBSYSTEM=="hidraw", ATTRS{idVendor}=="2886", ATTRS{idProduct}=="0018", MODE="0666", GROUP="plugdev"
EOF

# Reload rules and add user to plugdev group
sudo udevadm control --reload-rules
sudo udevadm trigger
sudo usermod -a -G plugdev $USER

# Reboot or re-plug the ReSpeaker USB to apply
```

---

## 8. Quick Reference

```bash
# Audio devices
arecord -l
aplay -l

# Temp / throttling
vcgencmd measure_temp
vcgencmd get_throttled

# Logs (manual run)
python main.py

# Logs (service)
sudo journalctl -u kin-client -f

# Restart service
sudo systemctl restart kin-client
```

That’s it—once `python main.py` shows “Ready! Say ‘porcupine’…”, you’re good to test wake word + conversation.\*\*\*
