# Kin AI Raspberry Pi Client - main.py

## Overview

`main.py` is the Raspberry Pi client that combines ALSA-only audio, full telemetry, LED feedback, and modular architecture. Previous versions are available as `old_main.py` (PipeWire-based with telemetry) and `new_main.py` (ALSA-only with LEDs, no telemetry).

## Key Features

### 1. **ALSA-Only Audio Architecture**
- Prioritizes ReSpeaker 4 Mic Array with hardware echo cancellation
- Automatically falls back to best available mic/speaker if ReSpeaker not found
- Warns user when hardware AEC is unavailable
- No PipeWire/PulseAudio dependencies

### 2. **Full Telemetry (No Metrics)**
- OpenTelemetry traces and spans
- Conversation-level trace contexts
- Trace context propagation across WebSocket messages
- Structured logging with OTEL logger
- Exception recording
- Graceful degradation if telemetry unavailable

### 3. **LED Visual Feedback**
- Complete LED controller from `new_main.py`
- States: BOOT, IDLE, WAKE_WORD_DETECTED, CONVERSATION, ERROR, OFF
- Animated patterns designed for elderly users

### 4. **Signal Handling**
- `SIGUSR1`: Terminates current conversation only (graceful)
- `SIGINT/SIGTERM`: Full shutdown (graceful if not in conversation)
- Proper cleanup in all cases

### 5. **Modular Architecture**
- Clean separation of concerns
- Easy to test and maintain
- Reusable components

## Project Structure

```
raspberry-pi-client/
├── main.py               # Entry point with KinClient
└── lib/
    ├── __init__.py
    ├── config.py         # Configuration from env vars
    ├── auth.py           # Device authentication (Ed25519)
    ├── audio/
    │   ├── __init__.py
    │   ├── device_detection.py  # ALSA device detection
    │   └── led_controller.py    # LED visual feedback
    ├── wake_word/
    │   ├── __init__.py
    │   └── detector.py   # Porcupine wake word detection
    ├── elevenlabs/
    │   ├── __init__.py
    │   └── client.py     # ElevenLabs conversation client
    ├── orchestrator/
    │   ├── __init__.py
    │   └── client.py     # Orchestrator WebSocket client
    └── telemetry/
        ├── __init__.py
        └── telemetry.py  # OpenTelemetry setup (moved from root)
```

## Usage

```bash
# Run the client
python main.py

# Or if executable:
./main.py

# Send signals
kill -SIGUSR1 <pid>  # End conversation only
kill -SIGINT <pid>   # Full shutdown (or Ctrl+C)
```

## Environment Variables

Required credentials (provisioned via admin portal):

- `DEVICE_ID` - Unique device identifier (UUID)
- `DEVICE_PRIVATE_KEY` - Ed25519 private key for device authentication (base64-encoded)

Optional configuration:

- `SKIP_WIFI_SETUP` - Enable/disable WiFi setup mode (default: `true`)
- `CONVERSATION_ORCHESTRATOR_URL` - WebSocket URL for orchestrator (default: hardcoded in Config)
- `OTEL_ENABLED` - Enable OpenTelemetry (default: `true`)
- `OTEL_EXPORTER_ENDPOINT` - OTEL collector endpoint (default: `http://localhost:4318`)
- `ENV` - Deployment environment (default: `production`)

Note: Most settings (API keys, wake word, LED settings) are fetched from backend after authentication.
- `WAKE_WORD` - Wake word to detect (default: "porcupine")
- `LED_ENABLED` - Enable LED feedback (default: "true")
- `OTEL_ENABLED` - Enable OpenTelemetry (default: "true")
- `OTEL_EXPORTER_ENDPOINT` - OTEL exporter endpoint (default: "http://localhost:4318")
- `ENV` - Environment name (default: "production")

## What's Different from Previous Versions?

### ✅ Kept from old_main.py (PipeWire version):
- Full telemetry with traces, spans, and structured logging
- Log level set to INFO
- Trace context propagation
- Conversation-level traces
- Signal differentiation (SIGUSR1 vs SIGINT/SIGTERM)
- Graceful cleanup

### ✅ Kept from new_main.py:
- ALSA-only architecture
- ReSpeaker hardware AEC support
- LED visual feedback
- LED animations

### ❌ Removed:
- Metrics (as requested, will be added later)
- Heartbeat messages (as requested)
- PipeWire/PulseAudio support

### ✨ New:
- Modular architecture with clean separation
- Automatic fallback to best available audio devices
- Better error handling and logging
- Easier to test and maintain

## Telemetry Details

The telemetry system is heavily copied from `old_main.py` to ensure it works correctly:

1. **Logging setup**: Log level is explicitly set to INFO before any imports
2. **Trace context**: Properly injected into WebSocket messages for distributed tracing
3. **Conversation traces**: Each conversation gets its own root trace span
4. **Structured logging**: All log messages include relevant context (user_id, device_id, etc.)
5. **Exception recording**: Errors are properly recorded with stack traces
6. **Graceful degradation**: System works without telemetry if unavailable

## Testing

To test the refactored code:

1. **Audio detection**: Verify ReSpeaker is detected or fallback works
2. **Wake word**: Test wake word detection with detected microphone
3. **Conversation**: Verify full conversation flow works
4. **Signals**: Test SIGUSR1 (conversation only) and SIGINT (full shutdown)
5. **LEDs**: Verify LED states if ReSpeaker is available
6. **Telemetry**: Check OTEL traces in your observability backend

## Migration from Previous Versions

If you were using `old_main.py` (PipeWire-based) or `new_main.py` (ALSA without telemetry):

```bash
# Now just use:
python main.py
```

All environment variables remain the same! The new version combines the best of both.

## Future Enhancements

- Add metrics (planned for later)
- Add unit tests for each module
- Add integration tests
- Consider adding configuration file support
- Add more LED patterns

