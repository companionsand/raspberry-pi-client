# Kin AI Raspberry Pi Client - main.py

## Overview

`main.py` is the Raspberry Pi client that combines ALSA-only audio, full telemetry, LED feedback, and modular architecture.

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

- Complete LED controller for ReSpeaker
- States: BOOT, IDLE, WAKE_WORD_DETECTED, CONVERSATION, ERROR, OFF
- Animated patterns designed for elderly users

### 4. **Voice Feedback System**

- Voice guidance during device startup and setup
- Pre-recorded voice messages for key events:
  - "Starting up" when client initializes
  - "No internet detected, entering setup mode" when connectivity fails
  - "Device not paired, entering setup mode" when pairing is required
  - "Join Kin underscore Setup WiFi and enter WiFi credentials" when HTTP server is ready
- Graceful degradation if voice message files are missing
- Uses blocking playback to ensure messages are heard
- Compatible with all ALSA audio devices

### 5. **Signal Handling**

- `SIGUSR1`: Terminates current conversation only (graceful)
- `SIGINT/SIGTERM`: Full shutdown (graceful if not in conversation)
- Proper cleanup in all cases

### 6. **Modular Architecture**

- Clean separation of concerns
- Easy to test and maintain
- Reusable components

## Project Structure

```
raspberry-pi-client/
├── main.py               # Entry point with KinClient
├── scripts/
│   └── generate_voice_messages.py  # Voice message generation helper
└── lib/
    ├── __init__.py
    ├── config.py         # Configuration from env vars
    ├── auth.py           # Device authentication (Ed25519)
    ├── audio/
    │   ├── __init__.py
    │   ├── device_detection.py  # ALSA device detection
    │   └── led_controller.py    # LED visual feedback
    ├── voice_feedback/
    │   ├── __init__.py
    │   ├── voice_feedback.py    # Voice feedback system
    │   └── voice_messages/      # Pre-recorded voice message files
    │       ├── README.md         # Audio format specifications
    │       ├── startup.wav
    │       ├── no_internet.wav
    │       ├── device_not_paired.wav
    │       └── wifi_setup_ready.wav
    ├── wake_word/
    │   ├── __init__.py
    │   └── detector.py   # Porcupine wake word detection
    ├── elevenlabs/
    │   ├── __init__.py
    │   └── client.py     # ElevenLabs conversation client
    ├── orchestrator/
    │   ├── __init__.py
    │   └── client.py     # Orchestrator WebSocket client
    ├── wifi_setup/
    │   ├── __init__.py
    │   ├── manager.py    # WiFi setup orchestration
    │   ├── access_point.py
    │   ├── http_server.py
    │   ├── network_connector.py
    │   └── connectivity.py
    └── telemetry/
        ├── __init__.py
        └── telemetry.py  # OpenTelemetry setup
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
  - When enabled, device creates "Kin_Setup" WiFi network (password: `kinsetup123`)
  - Connect to it and visit http://192.168.4.1:8080 to configure
- `CONVERSATION_ORCHESTRATOR_URL` - WebSocket URL for orchestrator (default: hardcoded in Config)
- `OTEL_ENABLED` - Enable OpenTelemetry (default: `true`)
- `OTEL_EXPORTER_ENDPOINT` - OTEL collector endpoint (default: `http://localhost:4318`)
- `ENV` - Deployment environment (default: `production`)

Note: Most settings (API keys, wake word, LED settings) are fetched from backend after authentication.

## Voice Feedback Setup

The voice feedback system uses pre-recorded WAV files to provide voice guidance. To set up voice feedback:

### Option 1: Generate Voice Message Files (Recommended)

Use the provided script to generate all voice message files using ElevenLabs API:

```bash
# Set your ElevenLabs API key
export ELEVENLABS_API_KEY="your-api-key-here"

# Generate voice message files
python scripts/generate_voice_messages.py

# Or specify a custom voice:
python scripts/generate_voice_messages.py --voice your-voice-id
```

### Option 2: Manual Recording

Record your own voice messages and convert them to the required format:

```bash
# Required format: 16kHz, mono, 16-bit PCM WAV
ffmpeg -i input.mp3 -ar 16000 -ac 1 -sample_fmt s16 lib/voice_feedback/voice_messages/startup.wav
```

### Required Voice Message Files

- `startup.wav` - "Starting up"
- `no_internet.wav` - "No internet detected, entering setup mode"
- `device_not_paired.wav` - "Device not paired, entering setup mode"
- `wifi_setup_ready.wav` - "Join Kin underscore Setup WiFi and enter WiFi credentials"

### Testing Voice Message Files

```bash
# Verify audio format
file lib/voice_feedback/voice_messages/startup.wav

# Play voice message file
aplay lib/voice_feedback/voice_messages/startup.wav
```

**Note**: If voice message files are missing, the system will continue to work normally but without voice guidance. Warnings will be logged for each missing file.

## Telemetry Details

The telemetry system provides comprehensive observability:

1. **Logging setup**: Log level is explicitly set to INFO before any imports
2. **Trace context**: Properly injected into WebSocket messages for distributed tracing
3. **Conversation traces**: Each conversation gets its own root trace span
4. **Structured logging**: All log messages include relevant context (user_id, device_id, etc.)
5. **Exception recording**: Errors are properly recorded with stack traces
6. **Graceful degradation**: System works without telemetry if unavailable

## Testing

To test the client:

1. **Audio detection**: Verify ReSpeaker is detected or fallback works
2. **Voice feedback**: Test that voice messages play during startup and setup
3. **Wake word**: Test wake word detection with detected microphone
4. **Conversation**: Verify full conversation flow works
5. **Signals**: Test SIGUSR1 (conversation only) and SIGINT (full shutdown)
6. **LEDs**: Verify LED states if ReSpeaker is available
7. **Telemetry**: Check OTEL traces in your observability backend

## Future Enhancements

- Add metrics (planned for later)
- Add unit tests for each module
- Add integration tests
- Consider adding configuration file support
- Add more LED patterns
