# Scribe v2 Wake Word Verification

## Overview

This document describes the third-layer wake word verification system using ElevenLabs Scribe v2 realtime speech-to-text. This layer sits after VAD gating and Picovoice detection to reduce false positives.

## Architecture

The wake word detection now has **three layers**:

```
Audio Input
    ↓
┌─────────────────────┐
│  Layer 1: VAD Gate  │  ← Filters out non-speech audio
└─────────────────────┘
    ↓ (speech detected)
┌─────────────────────┐
│ Layer 2: Picovoice  │  ← Phonetic wake word detection
└─────────────────────┘
    ↓ (wake word detected)
┌─────────────────────┐
│ Layer 3: Scribe v2  │  ← ASR verification (NEW)
└─────────────────────┘
    ↓ (transcript matches)
Wake Word Accepted ✓
```

## How It Works

### 1. Ring Buffer (Continuous)
- **Size**: 3 seconds of VAD-passed audio
- **Frame Rate**: ~31.25 frames/second (512 samples @ 16kHz)
- **Storage**: Only stores audio that passed VAD gate
- **Auto-cleanup**: Oldest frames automatically dropped (deque with maxlen)

### 2. Trigger Window Extraction
When Picovoice detects the wake word:
- **Pre-trigger**: Extract last 1.5 seconds from ring buffer
- **Post-trigger**: Capture additional 0.2 seconds after detection
- **Total window**: ~1.7 seconds of audio sent to Scribe

### 3. Scribe v2 Transcription
- **API**: ElevenLabs Scribe v2 Realtime WebSocket
- **Model**: `scribe_v2_realtime`
- **Audio Format**: PCM 16kHz, 16-bit, base64 encoded
- **Mode**: Synchronous (blocks until transcription received)
- **Timeout**: 2 seconds max wait time

### 4. Fuzzy Matching
The transcript is checked against `Config.WAKE_WORD` using three methods:

#### A. Exact Substring Match
```python
if "porcupine" in "hey porcupine":
    return True  # ✓ Match
```

#### B. Fuzzy Match (Full Transcript)
```python
# Uses SequenceMatcher with 0.75 threshold
"pork you pine" ~ "porcupine"  # similarity: 0.83 ✓ Match
```

#### C. Fuzzy Match (Individual Words)
```python
# Check each word in transcript
"hey porkupine there" → ["hey", "porkupine", "there"]
"porkupine" ~ "porcupine"  # similarity: 0.90 ✓ Match
```

### 5. Fail-Open Strategy
If Scribe verification fails (API error, timeout, etc.), the wake word is **accepted anyway**. This ensures reliability:
- ✓ Picovoice detected wake word
- ✗ Scribe verification failed/errored
- → **Accept wake word** (don't block user)

## Configuration

### Always Enabled
The feature is currently **always enabled** when:
- `Config.ELEVENLABS_API_KEY` is available
- No configuration toggle required (future enhancement)

### Timing Parameters
Located in `lib/wake_word/detector.py`:

```python
# Ring buffer settings
self._ring_buffer_duration_seconds = 3.0

# Verification window
self._scribe_pre_trigger_seconds = 1.5   # Audio before trigger
self._scribe_post_trigger_seconds = 0.2  # Audio after trigger

# Timeout
self._scribe_timeout_seconds = 2.0
```

### Fuzzy Match Threshold
Located in `_fuzzy_match_wake_word()`:

```python
threshold: float = 0.75  # Minimum similarity (0.0-1.0)
```

## Telemetry & Monitoring

### OpenTelemetry Metrics
New metrics added to track verification performance:

| Metric | Type | Description |
|--------|------|-------------|
| `scribe_verifications_total` | Counter | Total verification attempts |
| `scribe_verifications_passed_total` | Counter | Verifications that matched wake word |
| `scribe_verifications_failed_total` | Counter | Verifications that didn't match |
| `scribe_api_errors_total` | Counter | API errors during verification |

### Attributes
Each metric includes:
- `device_id`: Device identifier
- `wake_word`: Expected wake word
- `transcript`: (failed verifications only) The actual transcript

### Structured Logging
Events logged via OpenTelemetry logger:

```python
# Verification passed
logger.info("scribe_verification_passed", extra={
    "transcript": "hey porcupine",
    "wake_word": "porcupine",
    "device_id": "...",
    "user_id": "..."
})

# Verification failed
logger.warning("scribe_verification_failed", extra={
    "transcript": "hey there",
    "wake_word": "porcupine",
    "device_id": "...",
    "user_id": "..."
})

# Wake word rejected
logger.warning("wake_word_rejected_by_scribe", extra={
    "wake_word": "porcupine",
    "device_id": "...",
    "user_id": "..."
})
```

### Local Statistics
When detector stops, stats are printed:

```
📊 Scribe Verification Stats:
   Total: 42
   Passed: 39 (92.9%)
   Failed: 2
   API Errors: 1
```

## Latency Impact

### Expected Latency
- **Scribe transcription**: ~200-500ms
- **Network overhead**: ~50-100ms
- **Total added latency**: ~250-600ms

This is **synchronous** - the audio callback blocks until Scribe responds. This ensures we don't proceed with a false positive.

### Tradeoff
- ✓ **Better accuracy**: Reduces false positives significantly
- ✗ **Slightly slower response**: User waits ~0.5s longer after saying wake word

## Implementation Details

### Dependencies
Added to `requirements.txt`:
```
elevenlabs>=1.0.0
```

### Key Files Modified
1. **`lib/wake_word/detector.py`**
   - Added ring buffer for audio storage
   - Added `_verify_with_scribe()` async method
   - Added `_fuzzy_match_wake_word()` matching logic
   - Modified `_audio_callback()` to call verification
   - Added telemetry metrics initialization

2. **`lib/telemetry/telemetry.py`**
   - Added 4 new metrics for Scribe verification tracking

3. **`requirements.txt`**
   - Added ElevenLabs SDK dependency

### Thread Safety
The implementation handles asyncio in a synchronous audio callback:
```python
# Get or create event loop for audio callback thread
try:
    loop = asyncio.get_event_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

# Run async verification synchronously
verified = loop.run_until_complete(self._verify_with_scribe(audio_frames))
```

## Testing & Validation

### Manual Testing
1. Say the wake word correctly → Should pass all 3 layers
2. Say similar words (false trigger) → Picovoice might trigger, but Scribe should reject
3. Test with background noise → VAD should filter, Picovoice/Scribe should handle remaining
4. Test API timeout → Should fail-open and accept wake word

### Monitoring in Production
Watch for:
- **High failure rate** (>20%) → May need to tune fuzzy match threshold
- **High API error rate** → Check ElevenLabs API status/quota
- **User complaints about slowness** → May need to reduce timeout

### Tuning Parameters
If false positives persist:
- Increase `threshold` in `_fuzzy_match_wake_word()` (e.g., 0.80 or 0.85)
- Increase `_scribe_pre_trigger_seconds` for more context

If false negatives occur:
- Decrease `threshold` (e.g., 0.70)
- Check transcription logs to see what Scribe hears

## Future Enhancements

### Configuration Toggle
Add to device config:
```json
{
  "scribe_verification_enabled": true
}
```

### Adaptive Threshold
Learn optimal threshold per user based on their accent/environment.

### Caching
Cache recent transcriptions to avoid duplicate API calls if wake word triggers twice in quick succession.

### Async Non-Blocking
Move verification to background thread and accept wake word immediately, but log mismatches for monitoring. This removes latency but keeps telemetry benefits.

## References

- [ElevenLabs Scribe v2 Documentation](https://elevenlabs.io/docs/cookbooks/speech-to-text/streaming)
- Python `difflib.SequenceMatcher` for fuzzy matching
- OpenTelemetry metrics specification

