# ReSpeaker AEC Testing Guide & Checklist

## Overview

This guide provides a systematic approach to testing and verifying Acoustic Echo Cancellation (AEC) on the ReSpeaker 4-Mic Array. Use this checklist to ensure both hardware and software are properly configured for reliable barge-in functionality.

## Prerequisites

- ReSpeaker 4-Mic Array (UAC1.0) connected via USB
- Speaker connected to ReSpeaker 3.5mm jack
- ALSA-only setup (PipeWire/PulseAudio disabled)
- Python 3 with sounddevice, numpy, sox installed
- ReSpeaker tuning tools (`usb_4_mic_array` repository)

## Quick Status Check

```bash
# Verify ReSpeaker is detected
arecord -l | grep -i "ArrayUAC10\|ReSpeaker"

# Expected: card 3: ArrayUAC10 [ReSpeaker 4 Mic Array (UAC1.0)]

# Verify 6-channel firmware
arecord -D hw:3,0 --dump-hw-params 2>&1 | grep -E "CHANNELS|RATE"
# Expected: CHANNELS: 6, RATE: 16000
```

---

## Phase 1: Hardware Verification

### ✅ Test 1.1: ReSpeaker Detection

```bash
arecord -l
```

**Expected:** Card 3 shows `ArrayUAC10 [ReSpeaker 4 Mic Array (UAC1.0)]`

**If missing:**
- Check USB connection
- Verify device is powered
- Check `dmesg | grep -i audio` for errors

---

### ✅ Test 1.2: 6-Channel Firmware Verification

```bash
arecord -D hw:3,0 --dump-hw-params 2>&1 | grep CHANNELS
```

**Expected:** `CHANNELS: 6`

**Why:** ReSpeaker AEC requires 6-channel firmware:
- Channel 0: AEC-processed audio (what we want)
- Channels 1-4: Raw microphone inputs
- Channel 5: Playback reference signal

**If not 6 channels:** Firmware may need update or wrong device selected

---

### ✅ Test 1.3: Mechanical Isolation Check

**Problem:** Speaker vibration can shake the enclosure, causing mechanical noise that bypasses AEC.

**Test (Spaghetti Test):**

```bash
cd ~/usb_4_mic_array

# Record baseline (no playback) - components OUTSIDE enclosure
arecord -D hw:3,0 -c 6 -f S16_LE -r 16000 -d 4 test_baseline.wav
sox test_baseline.wav -n remix 2 stat 2>&1 | grep "RMS amplitude"
# Note: Baseline RMS (should be <0.01)

# Record during playback - components OUTSIDE enclosure
sox -n -r 16000 -c 1 /tmp/test_tone.wav synth 3 sine 440 vol 0.8
(sleep 1 && aplay -D plughw:3,0 /tmp/test_tone.wav > /dev/null 2>&1) &
arecord -D hw:3,0 -c 6 -f S16_LE -r 16000 -d 4 test_outside.wav
wait

# Record during playback - components INSIDE enclosure
(sleep 1 && aplay -D plughw:3,0 /tmp/test_tone.wav > /dev/null 2>&1) &
arecord -D hw:3,0 -c 6 -f S16_LE -r 16000 -d 4 test_inside.wav
wait

# Compare
echo "Outside enclosure (Ch1 RMS):"
sox test_outside.wav -n remix 2 stat 2>&1 | grep "RMS amplitude"

echo "Inside enclosure (Ch1 RMS):"
sox test_inside.wav -n remix 2 stat 2>&1 | grep "RMS amplitude"
```

**Success Criteria:**
- Outside enclosure: Mic RMS < 0.20
- Inside enclosure: Mic RMS should be similar (within 0.05)
- **If inside is >0.20 higher:** Mechanical vibration detected - need isolation

**Fix:** Use foam pads, rubber grommets, or decouple speaker from case

---

## Phase 2: Software Configuration

### ✅ Test 2.1: AEC Parameters Verification

```bash
cd ~/usb_4_mic_array

# Check AEC is enabled
python tuning.py AECFREEZEONOFF
# Expected: 0 (adaptation enabled)

python tuning.py ECHOONOFF
# Expected: 1 (echo suppression ON)

# If not correct:
python tuning.py AECFREEZEONOFF 0
python tuning.py ECHOONOFF 1
```

**Critical:** Both must be correct for AEC to work.

---

### ✅ Test 2.2: AGC Gain Configuration

**Problem:** Auto Gain Control can boost echo, making AEC ineffective.

```bash
# Check current AGC status
python tuning.py AGCONOFF
python tuning.py AGCGAIN

# Disable AGC and set safe gain
python tuning.py AGCONOFF 0  # Freeze AGC (stops auto-adjustment)
python tuning.py AGCGAIN 5.0  # Set to safe level (5x = ~14dB)

# Verify it stuck
python tuning.py AGCGAIN
# Expected: ~5.0 (not 10+)
```

**Why:** High AGC gain (10-20x) amplifies everything, including echo. AEC can't subtract amplified echo effectively.

---

### ✅ Test 2.3: ALSA Configuration for Channel 0

**Problem:** Must read from Channel 0 (AEC-processed), not raw channels.

```bash
# Check current ALSA config
cat /etc/asound.conf

# Verify it extracts Channel 0
# Should contain something like:
# pcm.respeaker_aec {
#     type route
#     slave { pcm "hw:3,0"; channels 6 }
#     ttable.0.0 1
# }
```

**If missing:** See Phase 5 for correct ALSA config.

---

## Phase 3: Channel Analysis (Critical Test)

### ✅ Test 3.1: Record All Channels During Playback

**Purpose:** Verify Channel 0 is AEC-processed and Channel 5 (reference) is present.

```bash
cd ~/aec_test

# Create test tone
sox -n -r 16000 -c 1 /tmp/test_tone.wav synth 3 sine 440 vol 0.8

# Record all 6 channels while playing tone
echo "Starting recording - play tone in 2 seconds..."
(sleep 2 && aplay -D plughw:3,0 /tmp/test_tone.wav > /dev/null 2>&1) &
arecord -D hw:3,0 -c 6 -f S16_LE -r 16000 -d 6 test_channels.wav
wait

# Extract and analyze each channel
echo "---------------------------------"
echo "Ch0 (AEC Processed):"
sox test_channels.wav -n remix 1 stat 2>&1 | grep "RMS amplitude"

echo "Ch1 (Raw Mic Input):"
sox test_channels.wav -n remix 2 stat 2>&1 | grep "RMS amplitude"

echo "Ch5 (Playback Reference):"
sox test_channels.wav -n remix 6 stat 2>&1 | grep "RMS amplitude"
echo "---------------------------------"
```

**Success Criteria:**

| Channel | Expected RMS | Interpretation |
|---------|-------------|----------------|
| **Ch0 (AEC)** | **<0.20** | ✅ AEC working |
| Ch0 (AEC) | >0.30 | ❌ AEC not effective |
| **Ch1 (Raw)** | 0.15-0.40 | Normal raw mic level |
| **Ch5 (Ref)** | **0.20-0.50** | ✅ Reference present |
| Ch5 (Ref) | <0.10 | ❌ Reference missing |

**Key Ratios:**
- **Ch0 < Ch1** (by at least 30%): AEC is working
- **Ch5 > Ch1**: Reference is stronger than echo (good for AEC)
- **Ch5 ≈ 0**: Reference missing - AEC can't work

---

### ✅ Test 3.2: Gain Staging Verification

**Purpose:** Ensure reference signal (Ch5) is stronger than mic input (Ch1) for effective AEC.

```bash
cd ~/aec_test

# Record during playback
sox -n -r 16000 -c 1 /tmp/test_tone.wav synth 3 sine 440 vol 0.8
(sleep 1 && aplay -D plughw:3,0 /tmp/test_tone.wav > /dev/null 2>&1) &
arecord -D hw:3,0 -c 6 -f S16_LE -r 16000 -d 4 gain_test.wav
wait

# Extract RMS values
ref=$(sox gain_test.wav -n remix 6 stat 2>&1 | grep "RMS amplitude" | awk '{print $3}')
mic=$(sox gain_test.wav -n remix 2 stat 2>&1 | grep "RMS amplitude" | awk '{print $3}')

echo "-------------------------------------"
echo "Reference (Ch5): $ref"
echo "Mic Input (Ch1): $mic"

# Compare (requires bc)
if command -v bc >/dev/null; then
    if (( $(echo "$ref > $mic" | bc -l) )); then
        echo "✅ GOOD: Reference is louder than Mic. AEC has a chance."
    else
        echo "❌ BAD: Mic is louder. Turn down the physical amp!"
        echo "   Solution: Lower PAM8403 amplifier gain"
    fi
else
    echo "Install 'bc' for automatic comparison, or compare manually"
fi
echo "-------------------------------------"
```

**Success Criteria:**
- **Reference (Ch5) > Mic (Ch1)**: ✅ Good gain staging
- **Reference (Ch5) < Mic (Ch1)**: ❌ Need to lower amplifier gain

**Fix:** Physically turn down PAM8403 potentiometer until Ch5 > Ch1

---

## Phase 4: End-to-End AEC Test

### ✅ Test 4.1: Full AEC Test Script

**Purpose:** Simulate real-world usage with Python sounddevice.

```bash
cd ~/raspberry-pi-client-wrapper/raspberry-pi-client
source venv/bin/activate
python ~/aec_test/test_aec.py
```

**Test Procedure:**
1. **Baseline:** Stay quiet for 3 seconds
2. **During Playback:** Test tone plays for 4 seconds
3. **User Speech:** Speak for 5 seconds

**Success Criteria:**

| Condition | Expected RMS | Ratio | Status |
|-----------|-------------|-------|--------|
| Baseline (quiet) | 50-200 | - | Normal |
| During playback | **<500** | **<3x** | ✅ AEC working |
| During playback | 500-2000 | 3-10x | ⚠️ AEC partial |
| During playback | >2000 | >10x | ❌ AEC failing |
| User speaking | 1000-5000 | - | Normal |

**For barge-in to work reliably:** Ratio must be **<3x**

---

## Phase 5: ALSA Configuration

### ✅ Test 5.1: Verify Channel 0 Extraction

**Purpose:** Ensure ALSA reads from Channel 0 (AEC-processed), not raw channels.

```bash
# Test default device (should use ALSA config)
(sleep 2 && aplay -D plughw:3,0 /tmp/test_tone.wav > /dev/null 2>&1) &
arecord -D default -c 1 -f S16_LE -r 16000 -d 6 /tmp/alsa_test.wav
wait

# Check RMS
sox /tmp/alsa_test.wav -n stat 2>&1 | grep "RMS amplitude"
```

**Compare to direct Channel 0:**
```bash
# Extract Ch0 from 6-channel recording
sox test_channels.wav -c 1 ch0_direct.wav remix 1
sox ch0_direct.wav -n stat 2>&1 | grep "RMS amplitude"
```

**Success:** ALSA default RMS should be **similar** to Ch0 direct RMS (within 20%)

**If different:** ALSA config not extracting Channel 0 correctly - see fix below.

---

### ✅ Test 5.2: ALSA Configuration Template

**Correct ALSA config for Channel 0 extraction:**

```bash
sudo nano /etc/asound.conf
```

```conf
# =============================================================================
# ALSA Configuration - ReSpeaker Channel 0 (AEC-processed)
# =============================================================================

# Capture ONLY channel 0 (AEC-processed audio)
pcm.respeaker_aec {
    type route
    slave {
        pcm "hw:3,0"
        channels 6
    }
    ttable.0.0 1
}

pcm.respeaker_mono {
    type plug
    slave.pcm "respeaker_aec"
}

# Playback to ReSpeaker
pcm.respeaker_out {
    type plug
    slave.pcm "hw:3,0"
}

# Default: asymmetric (different for playback/capture)
pcm.!default {
    type asym
    playback.pcm "respeaker_out"
    capture.pcm "respeaker_mono"
}

ctl.!default {
    type hw
    card 3
}
```

**After updating:**
```bash
# Reload ALSA (or reboot)
sudo alsactl kill rescan
# OR
sudo reboot
```

---

## Phase 6: Troubleshooting

### Issue: High RMS During Playback (>2000)

**Symptoms:**
- Test shows ratio >10x
- Mic picking up speaker output

**Checklist:**
1. ✅ Is Channel 0 being read? (Test 3.1)
2. ✅ Is Reference (Ch5) present? (Test 3.1)
3. ✅ Is AEC enabled? (Test 2.1)
4. ✅ Is AGC gain too high? (Test 2.2)
5. ✅ Mechanical vibration? (Test 1.3)
6. ✅ Gain staging correct? (Test 3.2)

---

### Issue: Reference Signal Missing (Ch5 ≈ 0)

**Symptoms:**
- Ch5 RMS < 0.10 during playback
- AEC can't work without reference

**Causes:**
- Audio not routing through ReSpeaker
- Wrong ALSA device selected
- Playback going to different device (HDMI, etc.)

**Fix:**
```bash
# Verify playback device
aplay -D plughw:3,0 /tmp/test_tone.wav
# Should hear tone from speaker

# Check what device is default
aplay -D default /tmp/test_tone.wav
# Should also play through ReSpeaker
```

---

### Issue: Mechanical Vibration

**Symptoms:**
- Ch1 RMS much higher inside enclosure vs outside
- AEC can't cancel mechanical noise

**Fix:**
1. Mount ReSpeaker on foam pads
2. Decouple speaker from case (foam gasket)
3. Use soft mounting (rubber grommets) instead of rigid screws
4. Reduce bass frequencies (high-pass filter on speaker output)

---

### Issue: AGC Boosting Echo

**Symptoms:**
- Ch1 RMS very high (>0.40)
- AGCGAIN > 10

**Fix:**
```bash
cd ~/usb_4_mic_array
python tuning.py AGCONOFF 0  # Freeze AGC
python tuning.py AGCGAIN 5.0  # Set safe level
```

---

## Phase 7: Final Verification

### ✅ Test 7.1: Complete System Test

Run all tests in sequence:

```bash
# 1. Hardware check
arecord -l | grep -i "ArrayUAC10"

# 2. Channel count
arecord -D hw:3,0 --dump-hw-params 2>&1 | grep CHANNELS

# 3. AEC enabled
cd ~/usb_4_mic_array
python tuning.py AECFREEZEONOFF
python tuning.py ECHOONOFF

# 4. AGC configured
python tuning.py AGCONOFF
python tuning.py AGCGAIN

# 5. Channel analysis
cd ~/aec_test
sox -n -r 16000 -c 1 /tmp/test_tone.wav synth 3 sine 440 vol 0.8
(sleep 2 && aplay -D plughw:3,0 /tmp/test_tone.wav > /dev/null 2>&1) &
arecord -D hw:3,0 -c 6 -f S16_LE -r 16000 -d 6 final_test.wav
wait

echo "Ch0 (AEC):" && sox final_test.wav -n remix 1 stat 2>&1 | grep "RMS amplitude"
echo "Ch1 (Raw):" && sox final_test.wav -n remix 2 stat 2>&1 | grep "RMS amplitude"
echo "Ch5 (Ref):" && sox final_test.wav -n remix 6 stat 2>&1 | grep "RMS amplitude"

# 6. End-to-end test
cd ~/raspberry-pi-client-wrapper/raspberry-pi-client
source venv/bin/activate
python ~/aec_test/test_aec.py
```

**All tests passing?** ✅ AEC is configured correctly!

---

## Quick Reference: Expected Values

| Test | Metric | Good | Warning | Bad |
|------|--------|------|---------|-----|
| Ch0 RMS (during playback) | RMS amplitude | <0.20 | 0.20-0.30 | >0.30 |
| Ch5 RMS (reference) | RMS amplitude | 0.20-0.50 | 0.10-0.20 | <0.10 |
| Ch5 vs Ch1 ratio | Reference/Mic | >1.0 | 0.5-1.0 | <0.5 |
| AEC test ratio | Playback/Baseline | <3x | 3-10x | >10x |
| AGC Gain | Multiplier | 1-5 | 5-10 | >10 |

---

## Test Scripts

### Channel Analysis Script

Save as `~/aec_test/analyze_channels.sh`:

```bash
#!/bin/bash
# Analyze all 6 channels from recording

if [ -z "$1" ]; then
    echo "Usage: $0 <recording.wav>"
    exit 1
fi

FILE="$1"

echo "=========================================="
echo "Channel Analysis: $FILE"
echo "=========================================="

for ch in 1 2 3 4 5 6; do
    case $ch in
        1) label="Ch0 (AEC Processed)" ;;
        2) label="Ch1 (Raw Mic 1)" ;;
        3) label="Ch2 (Raw Mic 2)" ;;
        4) label="Ch3 (Raw Mic 3)" ;;
        5) label="Ch4 (Raw Mic 4)" ;;
        6) label="Ch5 (Reference)" ;;
    esac
    
    rms=$(sox "$FILE" -n remix $ch stat 2>&1 | grep "RMS amplitude" | awk '{print $3}')
    echo "$label: $rms"
done

echo "=========================================="
```

Usage:
```bash
chmod +x ~/aec_test/analyze_channels.sh
~/aec_test/analyze_channels.sh test_channels.wav
```

---

## Summary Checklist

Before enabling barge-in, verify:

- [ ] ReSpeaker detected (card 3)
- [ ] 6-channel firmware confirmed
- [ ] AEC enabled (AECFREEZEONOFF=0, ECHOONOFF=1)
- [ ] AGC gain set to safe level (5.0 or lower)
- [ ] Channel 0 RMS < 0.20 during playback
- [ ] Channel 5 (reference) present and > Ch1
- [ ] Mechanical isolation verified (spaghetti test)
- [ ] ALSA config extracts Channel 0
- [ ] End-to-end test shows ratio <3x
- [ ] Barge-in enabled in code (`barge_in_enabled = True`)

**All checked?** ✅ Ready for production!

---

## Revision History

- **v1.0** (Nov 2025): Initial comprehensive guide based on debugging session
- Documents all tests performed, success criteria, and troubleshooting steps
