#!/bin/bash
# =============================================================================
# Setup Sudoers for Raspberry Pi Client
# =============================================================================
# This script configures passwordless sudo for commands needed by the
# raspberry-pi-client service (ReSpeaker tuning, ALSA config, etc.)
#
# Usage:
#   sudo ./scripts/setup_sudoers.sh [username]
#
# Default username: kin (or aushim for development)

set -e

# Detect username (default to kin, fallback to aushim for dev)
USERNAME="${1:-kin}"

# Check if user exists
if ! id "$USERNAME" &>/dev/null; then
    # If kin doesn't exist, try aushim (development)
    if id "aushim" &>/dev/null; then
        USERNAME="aushim"
    else
        echo "✗ Error: User '$USERNAME' not found and 'aushim' not found"
        echo "  Usage: sudo ./scripts/setup_sudoers.sh <username>"
        exit 1
    fi
fi

echo "Setting up sudoers for user: $USERNAME"

# Create sudoers file
SUDOERS_FILE="/etc/sudoers.d/raspberry-pi-client"

cat > "$SUDOERS_FILE" << EOF
# Raspberry Pi Client - Passwordless sudo for audio and ReSpeaker commands
# This allows the service to configure ALSA and ReSpeaker without password prompts

# ReSpeaker tuning (via Python script)
$USERNAME ALL=(ALL) NOPASSWD: /usr/bin/python3 /home/$USERNAME/*/usb_4_mic_array/tuning.py *
$USERNAME ALL=(ALL) NOPASSWD: /usr/bin/python3 */usb_4_mic_array/tuning.py *

# ALSA configuration
$USERNAME ALL=(ALL) NOPASSWD: /usr/bin/tee /etc/asound.conf
$USERNAME ALL=(ALL) NOPASSWD: /usr/bin/cp /etc/asound.conf /etc/asound.conf.backup.*
$USERNAME ALL=(ALL) NOPASSWD: /usr/bin/amixer *
$USERNAME ALL=(ALL) NOPASSWD: /usr/sbin/alsactl *

# Audio diagnostics
$USERNAME ALL=(ALL) NOPASSWD: /usr/bin/aplay *
$USERNAME ALL=(ALL) NOPASSWD: /usr/bin/arecord *

# Time sync (for network issues)
$USERNAME ALL=(ALL) NOPASSWD: /usr/bin/timedatectl set-ntp *
EOF

# Set correct permissions (sudoers files must be 0440)
chmod 0440 "$SUDOERS_FILE"

# Verify syntax
if ! visudo -cf "$SUDOERS_FILE"; then
    echo "✗ Error: Sudoers file has syntax errors"
    rm "$SUDOERS_FILE"
    exit 1
fi

echo "✓ Sudoers configuration created: $SUDOERS_FILE"
echo "✓ User '$USERNAME' can now run audio commands with sudo (no password)"
echo ""
echo "Commands allowed:"
echo "  - ReSpeaker tuning (python3 tuning.py)"
echo "  - ALSA config (tee, cp, amixer, alsactl)"
echo "  - Audio diagnostics (aplay, arecord)"
echo "  - Time sync (timedatectl)"
