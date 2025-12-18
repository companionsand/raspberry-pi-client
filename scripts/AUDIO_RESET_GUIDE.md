# Audio Configuration Reset Guide

## Quick Start

```bash
# On Raspberry Pi (via SSH):
sudo ./scripts/reset_audio_config.sh

# Or keep minimal config instead of deleting:
sudo ./scripts/reset_audio_config.sh --keep-minimal
```

## What Gets Reset

### 1. `/etc/asound.conf` (ALSA Configuration File)

**What it is:**

- System-wide ALSA configuration
- Defines PCM devices, routing, and controls (like softvol)

**What the script does:**

- Backs up existing file to `/root/audio_config_backups/`
- Either deletes it OR replaces with minimal config (no softvol)

### 2. ALSA State File (`/var/lib/alsa/asound.state`)

**What it is:**

- Stores current state of all ALSA controls (volume levels, mute states, etc.)
- Created and managed by `alsactl`
- **Persists across reboots** (saved/restored automatically)

**Why it matters:**

- Even if you delete `/etc/asound.conf`, the Softvol control remains in the state file
- ALSA will continue to show the Softvol control until state is cleared

**What the script does:**

- Backs up the state file
- Removes Softvol entries from the state
- Reloads ALSA with `alsactl restore` or `alsactl init`

### 3. Running Processes

**What the script does:**

- Stops `kin-client` service
- Kills any processes using audio devices
- Ensures clean reset

## Reset Modes

### Mode 1: Complete Removal (Default)

```bash
sudo ./scripts/reset_audio_config.sh
```

**Result:**

- `/etc/asound.conf`: DELETED
- ALSA state: Softvol removed
- System uses ALSA defaults (usually first available device)

**Use case:** Testing auto-configuration from absolute scratch

### Mode 2: Minimal Config

```bash
sudo ./scripts/reset_audio_config.sh --keep-minimal
```

**Result:**

- `/etc/asound.conf`: Minimal config (direct hardware access, no softvol)
- ALSA state: Softvol removed
- System uses ReSpeaker hardware directly (no AEC routing, no volume control)

**Use case:** Testing with a baseline config, comparing before/after auto-config

## Why Two Files?

| File                         | Purpose                                            | Persistence              |
| ---------------------------- | -------------------------------------------------- | ------------------------ |
| `/etc/asound.conf`           | **Config template** - Defines devices and controls | Manual (you create/edit) |
| `/var/lib/alsa/asound.state` | **Runtime state** - Stores control values          | Automatic (ALSA manages) |

**Example:**

1. You create `/etc/asound.conf` with softvol definition
2. First time PCM is opened, ALSA creates Softvol control
3. Softvol state is saved to `/var/lib/alsa/asound.state`
4. Even if you delete `/etc/asound.conf`, Softvol persists in state file!

## Testing Auto-Configuration

### Step 1: Reset

```bash
sudo ./scripts/reset_audio_config.sh
# Verify: sudo amixer -c 2 sget Softvol  # Should error (no control)
```

### Step 2: Test Auto-Config

```bash
# Start service (will auto-configure)
sudo systemctl start kin-client

# Watch logs
sudo journalctl -u kin-client -f
```

**Expected logs:**

```
⚠ /etc/asound.conf: Not found, creating for card 2 with softvol...
✓ /etc/asound.conf: Created for card 2 with softvol control
  Setting softvol to 50%...
  ✓ Softvol set to 50%
```

### Step 3: Verify

```bash
# Check config was created
cat /etc/asound.conf

# Check Softvol control exists
sudo amixer -c 2 sget Softvol

# Check volume was set
# Should show 50% (or whatever backend configured)
```

## Troubleshooting

### Softvol Still Exists After Reset

**Cause:** ALSA state not fully cleared

**Solution:**

```bash
# Nuclear option: Delete state file and reboot
sudo rm /var/lib/alsa/asound.state
sudo reboot
```

### Wrong Card Number

**Problem:** Script detects wrong card for ReSpeaker

**Solution:**

```bash
# Find correct card manually
aplay -l | grep -i respeaker

# Edit minimal config manually if needed
sudo nano /etc/asound.conf
```

### Service Fails to Start After Reset

**Possible causes:**

- Config syntax error
- Permissions issue
- Device in use

**Debug:**

```bash
# Check service logs
sudo journalctl -u kin-client -n 50

# Test ALSA directly
aplay -l
arecord -l
speaker-test -c 2 -t wav
```

## Manual Reset (Alternative)

If the script doesn't work, reset manually:

```bash
# 1. Stop service
sudo systemctl stop kin-client

# 2. Backup and remove config
sudo cp /etc/asound.conf /etc/asound.conf.backup
sudo rm /etc/asound.conf

# 3. Clear ALSA state
sudo alsactl kill rescan
sudo rm /var/lib/alsa/asound.state
sudo alsactl init

# 4. Reboot (recommended)
sudo reboot
```

## Backups

All backups are saved to: `/root/audio_config_backups/`

**Restore from backup:**

```bash
# List backups
ls -lh /root/audio_config_backups/

# Restore specific backup
sudo cp /root/audio_config_backups/asound.conf.backup.20250118_143022 /etc/asound.conf
sudo systemctl restart kin-client
```

## Quick Reference Commands

```bash
# Check current config
cat /etc/asound.conf

# Check ALSA controls
amixer -c 2 scontrols  # List all controls
amixer -c 2 sget Softvol  # Check Softvol specifically

# Check ALSA state file
cat /var/lib/alsa/asound.state

# Find ReSpeaker card
aplay -l | grep -i respeaker

# Reload ALSA without reboot
sudo alsactl kill rescan
sudo alsactl restore

# Test audio
speaker-test -c 1 -t wav
```

## Development Workflow

**Typical testing cycle:**

1. Make code changes to auto-config logic
2. Run reset script: `sudo ./scripts/reset_audio_config.sh`
3. Deploy updated code to Pi
4. Start service: `sudo systemctl start kin-client`
5. Verify auto-config worked: Check logs + test audio
6. Repeat as needed

**Pro tip:** Keep terminal with logs open:

```bash
sudo journalctl -u kin-client -f
```
