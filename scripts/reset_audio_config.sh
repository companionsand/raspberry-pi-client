#!/bin/bash
# =============================================================================
# Reset Audio Configuration Script
# =============================================================================
# This script completely removes Softvol and ALSA configuration, allowing you
# to test auto-configuration from scratch.
#
# What it does:
# 1. Stops the kin-client service (if running)
# 2. Backs up /etc/asound.conf
# 3. Removes /etc/asound.conf (or restores to minimal config)
# 4. Clears ALSA state file to remove Softvol control
# 5. Reloads ALSA
#
# Usage:
#   sudo ./scripts/reset_audio_config.sh [--keep-minimal]
#
# Options:
#   --keep-minimal    Keep a minimal ALSA config instead of deleting entirely

set -e  # Exit on error

# =============================================================================
# Configuration
# =============================================================================
ASOUND_CONF="/etc/asound.conf"
ALSA_STATE="/var/lib/alsa/asound.state"
BACKUP_DIR="/root/audio_config_backups"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

KEEP_MINIMAL=false
if [[ "$1" == "--keep-minimal" ]]; then
    KEEP_MINIMAL=true
fi

# =============================================================================
# Helper Functions
# =============================================================================
print_header() {
    echo ""
    echo "═══════════════════════════════════════════════════════════════"
    echo "  $1"
    echo "═══════════════════════════════════════════════════════════════"
    echo ""
}

print_step() {
    echo "→ $1"
}

print_success() {
    echo "✓ $1"
}

print_warning() {
    echo "⚠ $1"
}

print_error() {
    echo "✗ $1"
}

# =============================================================================
# Main Script
# =============================================================================
print_header "Audio Configuration Reset Script"

# Check if running as root
if [[ $EUID -ne 0 ]]; then
   print_error "This script must be run as root (use sudo)"
   exit 1
fi

# Step 1: Stop kin-client service if running
print_step "Checking kin-client service..."
if systemctl is-active --quiet kin-client; then
    print_step "Stopping kin-client service..."
    systemctl stop kin-client
    print_success "Service stopped"
else
    print_warning "Service not running"
fi

# Step 2: Kill any processes using audio devices
print_step "Checking for processes using audio devices..."
AUDIO_PIDS=$(lsof /dev/snd/* 2>/dev/null | awk 'NR>1 {print $2}' | sort -u || true)
if [[ -n "$AUDIO_PIDS" ]]; then
    print_warning "Found processes using audio devices: $AUDIO_PIDS"
    print_step "Killing audio processes..."
    echo "$AUDIO_PIDS" | xargs -r kill -9 || true
    sleep 1
    print_success "Audio processes killed"
else
    print_success "No processes using audio devices"
fi

# Step 3: Create backup directory
print_step "Creating backup directory..."
mkdir -p "$BACKUP_DIR"
print_success "Backup directory: $BACKUP_DIR"

# Step 4: Backup /etc/asound.conf if it exists
if [[ -f "$ASOUND_CONF" ]]; then
    BACKUP_FILE="$BACKUP_DIR/asound.conf.backup.$TIMESTAMP"
    print_step "Backing up $ASOUND_CONF to $BACKUP_FILE..."
    cp "$ASOUND_CONF" "$BACKUP_FILE"
    print_success "Backup created: $BACKUP_FILE"
else
    print_warning "$ASOUND_CONF does not exist (nothing to backup)"
fi

# Step 5: Remove or replace /etc/asound.conf
if [[ "$KEEP_MINIMAL" == true ]]; then
    # Create minimal config without softvol
    print_step "Creating minimal ALSA config (no softvol)..."
    
    # First, detect ReSpeaker card number
    RESPEAKER_CARD=$(aplay -l 2>/dev/null | grep -i -E "respeaker|arrayuac10|uac1.0" | grep -oP 'card \K\d+' | head -1 || echo "2")
    
    cat > "$ASOUND_CONF" << EOF
# Minimal ALSA Configuration - Direct hardware access (no softvol)
# Card number auto-detected: $RESPEAKER_CARD

pcm.!default {
    type asym
    playback.pcm "plughw:$RESPEAKER_CARD,0"
    capture.pcm  "plughw:$RESPEAKER_CARD,0"
}

ctl.!default {
    type hw
    card $RESPEAKER_CARD
}
EOF
    print_success "Minimal config created for card $RESPEAKER_CARD (no softvol, no AEC routing)"
else
    # Completely remove config
    if [[ -f "$ASOUND_CONF" ]]; then
        print_step "Removing $ASOUND_CONF..."
        rm "$ASOUND_CONF"
        print_success "Config file removed"
    else
        print_warning "$ASOUND_CONF does not exist (already removed)"
    fi
fi

# Step 6: Backup ALSA state file
if [[ -f "$ALSA_STATE" ]]; then
    ALSA_BACKUP="$BACKUP_DIR/asound.state.backup.$TIMESTAMP"
    print_step "Backing up ALSA state file to $ALSA_BACKUP..."
    cp "$ALSA_STATE" "$ALSA_BACKUP"
    print_success "ALSA state backed up"
fi

# Step 7: Clear ALSA state to remove Softvol control
print_step "Clearing ALSA state to remove Softvol control..."
if [[ -f "$ALSA_STATE" ]]; then
    # Option 1: Remove Softvol entries from state file
    sed -i '/Softvol/,/}/d' "$ALSA_STATE" 2>/dev/null || true
    print_success "Softvol entries removed from ALSA state"
    
    # Option 2: Alternatively, delete the entire state file (ALSA will recreate it)
    # rm "$ALSA_STATE"
else
    print_warning "ALSA state file does not exist"
fi

# Step 8: Reload ALSA to apply changes
print_step "Reloading ALSA..."
# Stop ALSA and reload
alsactl kill rescan 2>/dev/null || true
sleep 1

# Restore ALSA state (or initialize if deleted)
if [[ -f "$ALSA_STATE" ]]; then
    alsactl restore 2>/dev/null || true
else
    alsactl init 2>/dev/null || true
fi

print_success "ALSA reloaded"

# =============================================================================
# Verification
# =============================================================================
print_header "Verification"

# Check if Softvol control exists
print_step "Checking for Softvol control..."
RESPEAKER_CARD=$(aplay -l 2>/dev/null | grep -i -E "respeaker|arrayuac10|uac1.0" | grep -oP 'card \K\d+' | head -1 || echo "2")

if amixer -c "$RESPEAKER_CARD" sget Softvol &>/dev/null; then
    print_warning "Softvol control still exists (may need reboot to fully clear)"
    echo "   Run: sudo reboot"
else
    print_success "Softvol control removed successfully"
fi

# Check config file
print_step "Checking $ASOUND_CONF..."
if [[ -f "$ASOUND_CONF" ]]; then
    if grep -q "softvol" "$ASOUND_CONF" 2>/dev/null; then
        print_warning "Config file still contains softvol"
    else
        if [[ "$KEEP_MINIMAL" == true ]]; then
            print_success "Minimal config in place (no softvol)"
        else
            print_warning "Config file exists but shouldn't (manual intervention needed)"
        fi
    fi
else
    print_success "Config file removed"
fi

# =============================================================================
# Summary
# =============================================================================
print_header "Reset Complete"

echo "Backups saved to: $BACKUP_DIR"
echo ""

if [[ "$KEEP_MINIMAL" == true ]]; then
    echo "Current state:"
    echo "  • /etc/asound.conf: Minimal config (no softvol)"
    echo "  • ALSA state: Softvol control removed"
    echo ""
    echo "The system now uses direct hardware access."
else
    echo "Current state:"
    echo "  • /etc/asound.conf: REMOVED"
    echo "  • ALSA state: Softvol control removed"
    echo ""
    echo "The system will use system defaults (no ALSA customization)."
fi

echo ""
echo "Next steps:"
echo "  1. Verify Softvol is gone: amixer -c $RESPEAKER_CARD sget Softvol"
echo "  2. (Optional) Reboot to fully clear ALSA state: sudo reboot"
echo "  3. Start service to test auto-config: sudo systemctl start kin-client"
echo "  4. Check logs: sudo journalctl -u kin-client -f"
echo ""

if amixer -c "$RESPEAKER_CARD" sget Softvol &>/dev/null; then
    print_warning "If Softvol still exists, reboot is recommended: sudo reboot"
fi

print_success "Audio configuration reset complete!"
