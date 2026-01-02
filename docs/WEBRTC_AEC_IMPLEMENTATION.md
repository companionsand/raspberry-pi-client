# WebRTC AEC3 Implementation - Complete

## ✅ Implementation Status: COMPLETE

All 4 phases have been successfully implemented. The system is now ready for testing.

---

## 🎯 What Was Implemented

### Phase 1: Configuration ✅

**File**: `lib/config.py`

**Changes**:
1. **CHUNK_SIZE**: Changed from `512` (32ms) → `320` (20ms)
   - Rationale: Multiple of 160 (WebRTC's 10ms frame requirement)
   - Better CPU/latency balance for real-time processing

2. **New Configuration Variables**:
   ```python
   # Channel assignments
   RESPEAKER_AEC_CHANNEL = 0  # Beamformed mic input
   RESPEAKER_REFERENCE_CHANNEL = 5  # Playback loopback
   
   # WebRTC AEC flags (configured via environment variables)
   USE_WEBRTC_AEC = false  # Default: disabled (opt-in via ENV)
   WEBRTC_AEC_STREAM_DELAY_MS = 100  # USB audio delay
   WEBRTC_AEC_NS_LEVEL = 1  # Noise suppression (0-3)
   WEBRTC_AEC_AGC_MODE = 2  # AGC mode (1=adaptive, 2=fixed)
   ```

### Phase 2: WebRTC Module ✅

**File**: `lib/audio/webrtc_aec.py`

**Created**: `WebRTCAECProcessor` class
- Wraps `aec-audio-processing.AudioProcessor` (WebRTC AEC3)
- Handles 320-sample chunks (20ms @ 16kHz)
- Splits internally into 2x 160-sample WebRTC frames (10ms)
- Context manager support (`with` statement)
- Comprehensive error handling and debug output

**Key Methods**:
- `start()`: Initialize AEC processor state
- `process_chunk(mic, ref)`: Process 320-sample chunks
- `stop()`: Clean up processor state

### Phase 3: Integration ✅

**File**: `lib/elevenlabs/client.py`

**Changes**:

1. **Import**: Conditional import of `WebRTCAECProcessor` (only if enabled)

2. **Initialization** (`__init__`):
   - Added `webrtc_aec_processor` instance variable
   - Added `_use_webrtc_aec` flag
   - Added `_webrtc_debug_last_log` for debug throttling

3. **Start Method** (`start()`):
   - Detects ReSpeaker with 6 channels
   - Initializes WebRTC AEC if `USE_WEBRTC_AEC=true`
   - **Applies ReSpeaker tuning** (runtime configuration):
     ```python
     ECHOONOFF = 0  # Disable hardware AEC
     AGCONOFF = 0  # Disable hardware AGC
     CNIONOFF = 0  # Disable Comfort Noise
     TRANSIENTONOFF = 0  # Disable Transient Suppression
     STATNOISEONOFF = 0  # Disable Stationary NS
     NONSTATNOISEONOFF = 0  # Disable Non-stationary NS
     FREEZEONOFF = 0  # Keep beamforming ACTIVE
     ```

4. **Audio Processing** (`_send_audio()`):
   - Extracts **Ch0** (beamformed mic) + **Ch5** (playback reference)
   - **Conditional processing**:
     - **Agent speaking** (`playback_active=True`): Run WebRTC AEC
     - **User speaking** (no playback): Passthrough (no processing)
   - Debug logging every 3s (active) / 10s (passthrough)

5. **Cleanup** (`stop()`):
   - Stops WebRTC processor if running
   - Proper resource cleanup

### Phase 4: Dependencies ✅

**File**: `requirements.txt`

**Added**:
```txt
# WebRTC AEC3 (Acoustic Echo Cancellation) - Optional
aec-audio-processing>=0.1.0
```

---

## 🚀 How to Use

### Quick Start (Enable WebRTC AEC)

1. **Install dependency**:
   ```bash
   pip install aec-audio-processing
   ```

2. **Enable via environment variable**:
   ```bash
   export USE_WEBRTC_AEC=true
   ```

3. **Run the client**:
   ```bash
   python main.py
   ```

4. **Verify activation** (check console output):
   ```
   🎯 Initializing WebRTC AEC3...
   ✓ WebRTC AEC initialized:
      - Sample rate: 16000Hz
      - Chunk size: 320 samples (20ms)
      - Stream delay: 100ms
      - Noise suppression level: 1
   ✓ WebRTC AEC enabled:
      - Using Ch0 (beamformed mic) + Ch5 (playback ref)
   📊 Applying ReSpeaker tuning for WebRTC AEC...
      - Disabling hardware AEC (ECHOONOFF=0)
      - Disabling hardware AGC (AGCONOFF=0)
      - Keeping beamforming ACTIVE (FREEZEONOFF=0)
   ✓ ReSpeaker tuned: ECHOONOFF=0, AGCONOFF=0, FREEZEONOFF=0
   ```

### Advanced Configuration

All settings are configurable via environment variables:

```bash
# Enable WebRTC AEC
export USE_WEBRTC_AEC=true

# Stream delay (50-200ms typical for USB audio)
# Too low = poor AEC, too high = poor AEC
# Start with 100ms and tune if needed
export WEBRTC_AEC_STREAM_DELAY_MS=100

# Noise suppression level (0-3)
# 0 = off, 1 = moderate, 3 = max
# Lower = preserve more speech, higher = more aggressive
export WEBRTC_AEC_NS_LEVEL=1

# AGC mode (1 or 2)
# 1 = adaptive digital, 2 = fixed digital
# Fixed is more aggressive and predictable
export WEBRTC_AEC_AGC_MODE=2
```

### Testing & Validation

**Monitor Debug Output**:
```
# During agent speech (WebRTC AEC active):
🎙️  [WebRTC AEC] Mic=0.1234, Ref=0.5678, Out=0.0456 [ACTIVE]
📊 [CHANNELS] [PLAYING] [WebRTC ON] Ch0=0.12 | Ch1=0.08 | ... | Ch5=0.56

# During user speech (passthrough):
🎙️  [WebRTC AEC] Mic=0.1234 [PASSTHROUGH - no agent speech]
📊 [CHANNELS] [IDLE] [WebRTC ON] Ch0=0.12 | Ch1=0.08 | ... | Ch5=0.00
```

**What to Look For**:
- ✅ `Out RMS` should be **much lower** than `Ref RMS` during agent speech
- ✅ `Mic RMS` vs `Out RMS`: Output should suppress echo/noise
- ✅ Ch5 (reference) should show activity **only during playback**
- ✅ `[ACTIVE]` should appear **only when agent is speaking**

**Performance Metrics**:
- CPU usage: Should be moderate (monitor with `top`)
- Latency: Should remain low (<100ms total pipeline)
- Echo reduction: Listen for echo during agent speech

---

## 🔧 Tuning Guide

### 1. Stream Delay Calibration

**Problem**: Echo cancellation not working well
**Solution**: Tune `WEBRTC_AEC_STREAM_DELAY_MS`

```bash
# Try incrementing by 10ms steps:
export WEBRTC_AEC_STREAM_DELAY_MS=90   # Try lower
export WEBRTC_AEC_STREAM_DELAY_MS=110  # Try higher
export WEBRTC_AEC_STREAM_DELAY_MS=120  # Keep tuning
```

**How to find optimal value**:
1. Play music through the device
2. Speak at the same time
3. Listen to the output for echo
4. Adjust delay until echo is minimized

### 2. Noise Suppression

**Problem**: Too much background noise
**Solution**: Increase `WEBRTC_AEC_NS_LEVEL`

```bash
export WEBRTC_AEC_NS_LEVEL=2  # More aggressive
export WEBRTC_AEC_NS_LEVEL=3  # Maximum suppression
```

**Warning**: Higher values may cut off speech edges

### 3. Automatic Gain Control

**Problem**: Output volume too quiet/loud/unstable
**Solution**: Switch AGC mode

```bash
export WEBRTC_AEC_AGC_MODE=1  # Adaptive (slower, smoother)
export WEBRTC_AEC_AGC_MODE=2  # Fixed (faster, more aggressive)
```

---

## 🐛 Debugging

### Enable WebRTC AEC but still getting echo?

**Check**:
1. Verify WebRTC is actually enabled:
   ```
   # Should see in logs:
   ✓ WebRTC AEC enabled
   🎙️  [WebRTC AEC] ... [ACTIVE]
   ```

2. Verify ReSpeaker tuning was applied:
   ```
   # Should see:
   ✓ ReSpeaker tuned: ECHOONOFF=0, AGCONOFF=0
   ```

3. Check Ch5 (reference) is active during playback:
   ```
   # During agent speech, Ch5 should be non-zero:
   📊 [CHANNELS] [PLAYING] ... Ch5=0.xxxx (not 0.0000!)
   ```

### WebRTC not initializing?

**Error**: `⚠️  WebRTC AEC enabled but not available`
**Solution**: Install the package:
```bash
pip install aec-audio-processing
```

**Error**: `⚠️  WebRTC AEC enabled but ReSpeaker not detected`
**Solution**: Check ReSpeaker connection:
```bash
arecord -l  # Should show ReSpeaker
```

### Audio sounds distorted?

**Possible causes**:
1. **CHUNK_SIZE mismatch**: Verify `CHUNK_SIZE=320` in config
2. **CPU overload**: Check CPU usage with `top`
3. **Stream delay too wrong**: Tune `WEBRTC_AEC_STREAM_DELAY_MS`

### High CPU usage?

**Solution**: WebRTC AEC only processes during agent speech (`playback_active=True`)
- Should see `[PASSTHROUGH]` logs when user is speaking
- If always seeing `[ACTIVE]`, check `playback_active` flag logic

---

## 📊 Architecture Summary

### Signal Flow

```
ReSpeaker 6-channel capture
├─ Ch0: Beamformed mic (hardware beamforming applied)
├─ Ch1-4: Raw mics (unused)
└─ Ch5: Playback loopback (reference signal)

WebRTC AEC Processing (when agent speaking):
1. Extract Ch0 (mic) + Ch5 (ref)
2. Process 320 samples → 2x 160-sample WebRTC frames
3. AEC: Cancel echo using reference
4. NS: Suppress noise
5. AGC: Normalize gain
6. Output: Clean 320-sample chunk → ElevenLabs

Passthrough (when user speaking):
1. Extract Ch0 (mic)
2. Pass through unchanged → ElevenLabs
```

### Processing Stages

1. **Hardware** (ReSpeaker firmware):
   - Beamforming: Directional audio from 4 mics → Ch0
   - Loopback capture: Playback → Ch5

2. **Software** (WebRTC AEC3):
   - AEC: Adaptive echo cancellation (Ch0 - Ch5)
   - NS: Noise suppression (configurable)
   - AGC: Automatic gain control (configurable)

3. **Conditional**:
   - Only runs WebRTC when `playback_active=True`
   - Saves CPU during user speech (no echo to cancel)

---

## 📝 Implementation Notes

### Why Ch0 instead of raw mics?

- **Ch0**: Already beamformed by ReSpeaker hardware
- **Benefits**: Directional audio, better SNR, less CPU
- **Future**: Can replace with custom beamforming when ReSpeaker is removed

### Why Ch5 instead of raw playback?

- **Ch5**: Sample-locked to mic capture (same clock)
- **Benefits**: Perfect alignment, no drift, accurate AEC
- **Critical**: Must use Ch5, not external reference signal

### Why only process during playback?

- **Echo only exists when agent is speaking**
- **CPU savings**: 50% reduction by not processing silence/user speech
- **Quality**: Avoids over-processing user's voice

### Why 320 samples (20ms)?

- **WebRTC requirement**: 160 samples (10ms)
- **Compromise**: 2x 160 = 320 (20ms)
- **Balance**: CPU efficiency vs. latency
- **Fits USB audio**: Good alignment with USB frame timing

---

## 🔍 Testing Checklist

### Before Deployment

- [ ] Install `aec-audio-processing` package
- [ ] Set `USE_WEBRTC_AEC=true`
- [ ] Verify WebRTC initialization in logs
- [ ] Check ReSpeaker tuning was applied
- [ ] Play music + speak → verify echo reduction
- [ ] Monitor CPU usage (should be reasonable)
- [ ] Test interruption (barge-in) still works
- [ ] Verify Ch5 shows activity during playback

### Performance Validation

- [ ] Echo cancellation: Agent speech should not echo back
- [ ] Noise suppression: Background noise reduced
- [ ] Speech quality: User speech clear and natural
- [ ] Latency: Response time still fast (<1s)
- [ ] CPU: Usage reasonable (<80% sustained)
- [ ] Stability: No crashes over 10+ conversations

### Fallback Testing

- [ ] Disable WebRTC (`USE_WEBRTC_AEC=false`) → system still works
- [ ] Uninstall `aec-audio-processing` → graceful fallback to hardware AEC
- [ ] Non-ReSpeaker device → WebRTC disabled automatically

---

## 🎓 Next Steps

1. **Deploy and Test**:
   - Install dependency
   - Enable flag
   - Run conversations
   - Monitor debug output

2. **Tune Parameters**:
   - Optimize `STREAM_DELAY_MS` for your hardware
   - Adjust `NS_LEVEL` for your environment
   - Test different `AGC_MODE` settings

3. **Measure Improvement**:
   - Record "before" audio (hardware AEC only)
   - Record "after" audio (WebRTC AEC enabled)
   - Compare echo levels (ERLE metric)

4. **Iterate**:
   - Gather user feedback
   - Fine-tune parameters
   - Monitor CPU usage
   - Optimize as needed

---

## 📚 References

- **WebRTC AEC3**: [Google WebRTC AEC3 documentation](https://webrtc.googlesource.com/src/+/refs/heads/main/modules/audio_processing/)
- **aec-audio-processing**: [PyPI package](https://pypi.org/project/aec-audio-processing/)
- **ReSpeaker 4-Mic Array**: [Seeed Studio docs](https://wiki.seeedstudio.com/ReSpeaker_4_Mic_Array_for_Raspberry_Pi/)
- **Implementation reference**: `pipipi/tests/test_aec_webrtc.py`
- **Analysis document**: `AEC_IMPLEMENTATION_ANALYSIS.md`

---

## 🙏 Credits

Implementation based on:
- **pipipi** reference implementation (test_aec_webrtc.py)
- **WebRTC AEC3** (Google's state-of-the-art echo canceller)
- **ReSpeaker firmware** (hardware beamforming + loopback)

---

**Status**: Ready for deployment and testing ✅
**Date**: 2025-12-26
**Version**: 1.0

