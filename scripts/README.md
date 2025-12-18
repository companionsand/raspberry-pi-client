# Scripts Directory

Utility scripts for Raspberry Pi client development and deployment.

## Scripts

### `reset_audio_config.sh`

**Purpose:** Reset ALSA audio configuration to test auto-configuration from scratch

**Usage:**

```bash
# Complete removal (test from absolute scratch)
sudo ./scripts/reset_audio_config.sh

# Keep minimal config (test with baseline)
sudo ./scripts/reset_audio_config.sh --keep-minimal
```

**What it does:**

- Stops kin-client service
- Backs up `/etc/asound.conf` and ALSA state
- Removes softvol configuration and controls
- Reloads ALSA

**Documentation:** See [AUDIO_RESET_GUIDE.md](./AUDIO_RESET_GUIDE.md) for detailed guide

**Use cases:**

- Testing auto-configuration logic changes
- Debugging audio issues
- Clean slate for audio troubleshooting

---

### `setup_sudoers.sh`

**Purpose:** Configure passwordless sudo for audio configuration commands

**Usage:**

```bash
# For production (kin user)
sudo ./scripts/setup_sudoers.sh kin

# For development (aushim user)
sudo ./scripts/setup_sudoers.sh aushim

# Auto-detect (tries kin, falls back to aushim)
sudo ./scripts/setup_sudoers.sh
```

**What it does:**

- Creates `/etc/sudoers.d/raspberry-pi-client`
- Allows service user to run audio commands without password:
  - ReSpeaker tuning (python3 tuning.py)
  - ALSA config (tee, cp, amixer, alsactl)
  - Audio diagnostics (aplay, arecord)
  - Time sync (timedatectl)

**Use cases:**

- Required for auto-configuration of `/etc/asound.conf`
- Required for ReSpeaker tuning on startup
- One-time setup during device provisioning

---

### `generate_voice_messages.py`

**Purpose:** Generate voice message WAV files for voice feedback system

**Usage:**

```bash
python scripts/generate_voice_messages.py
```

**What it does:**

- Generates pre-recorded voice messages using TTS
- Creates WAV files in correct format (16kHz, mono, 16-bit)
- Saves to `lib/voice_feedback/voice_messages/`

**Messages generated:**

- `startup.wav` - "Starting up"
- `no_internet.wav` - "No internet connection"
- `wifi_setup.wav` - "WiFi setup mode"
- And others...

**Use cases:**

- Creating voice messages for new device
- Updating voice messages with better TTS
- Regenerating after message text changes

---

## Development Workflow

### Testing Audio Auto-Configuration

1. Make changes to `lib/audio/device_detection.py`
2. Reset audio config: `sudo ./scripts/reset_audio_config.sh`
3. Deploy code to Pi
4. Start service and verify: `sudo systemctl start kin-client`
5. Check logs: `sudo journalctl -u kin-client -f`

### Updating Voice Messages

1. Edit message text in `generate_voice_messages.py`
2. Run: `python scripts/generate_voice_messages.py`
3. Copy generated WAV files to Pi
4. Test: `aplay lib/voice_feedback/voice_messages/startup.wav`

## Quick Reference

```bash
# Setup sudoers (one-time, required for auto-config)
sudo ./scripts/setup_sudoers.sh

# Audio reset (full)
sudo ./scripts/reset_audio_config.sh

# Audio reset (keep minimal config)
sudo ./scripts/reset_audio_config.sh --keep-minimal

# Generate voice messages
python scripts/generate_voice_messages.py

# View backups
ls -lh /root/audio_config_backups/
```

## First-Time Setup Checklist

1. ✅ Install dependencies (`sudo apt install alsa-utils python3-venv`)
2. ✅ Configure sudoers: `sudo ./scripts/setup_sudoers.sh`
3. ✅ Deploy code and create virtualenv
4. ✅ Configure service (see RASPBERRY_PI_SETUP.md)
5. ✅ Start service: `sudo systemctl start kin-client`
