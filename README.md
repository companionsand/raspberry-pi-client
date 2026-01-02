# Kin AI Raspberry Pi Client

## Overview

The Kin AI Raspberry Pi Client is a comprehensive voice assistant client that combines ALSA-only audio, full telemetry, LED feedback, human presence detection, and a modular architecture. It supports both reactive (wake word) and proactive (server-initiated) conversations.

## Key Features

### 1. **ALSA-Only Audio Architecture**

- Prioritizes ReSpeaker 4 Mic Array with hardware echo cancellation
- **Auto-detects ReSpeaker ALSA card number** (works with card 1, 2, 3, etc.)
- **Auto-configures `/etc/asound.conf`** with correct card number, softvol control, and AEC channel routing
- **Auto-sets volume** from backend configuration (`speaker_volume_percent`, 0-100)
- **ReSpeaker hardware tuning** - Applies optimized AEC, AGC, and noise suppression parameters from backend
- **Optional WebRTC AEC** - Software echo cancellation on top of hardware (opt-in via `USE_WEBRTC_AEC=true`)
- Automatically falls back to best available mic/speaker if ReSpeaker not found
- Warns user when hardware AEC is unavailable
- No PipeWire/Pulseaudio dependencies

- See [Audio Setup Guide](docs/RASPBERRY_PI_SETUP.md#3-audio-alsa-only-respeaker-as-default) and [AEC Testing Guide](docs/AEC_TESTING_GUIDE.md) for detailed configuration
- For WebRTC AEC configuration, see [WebRTC AEC Implementation](docs/WEBRTC_AEC_IMPLEMENTATION.md)

### 2. **Full Telemetry & Observability**

- OpenTelemetry traces and spans
- Conversation-level trace contexts (separate root traces per conversation)
- Trace context propagation across WebSocket messages (for distributed tracing)
- Structured logging with OTEL logger
- Exception recording with stack traces
- Stdout/stderr redirection to telemetry backend
- Graceful degradation if telemetry unavailable

- See [Telemetry Reference](docs/TELEMETRY_REFERENCE.md) for detailed telemetry information

### 3. **LED Visual Feedback**

- Complete LED controller for ReSpeaker
- States: BOOT, IDLE, WAKE_WORD_DETECTED, CONVERSATION, ERROR, OFF
- Animated patterns designed for elderly users
- Audio-reactive feedback during conversations
- Configurable brightness and enable/disable from backend

- See [LED States Reference](docs/LED_STATES_REFERENCE.md) for complete state documentation

### 4. **Voice Feedback System**

- Voice guidance during device startup and setup
- Pre-recorded voice messages for key events:
  - "Starting up" when client initializes
  - "No internet detected, entering setup mode" when connectivity fails
  - "Device not paired, entering setup mode" when pairing is required
  - "Join Kin underscore Setup WiFi and enter WiFi credentials" when HTTP server is ready
- **Quiet hours support** - Skips startup message during 8pm-10am to avoid noise
- Graceful degradation if voice message files are missing
- Uses blocking playback to ensure messages are heard
- Compatible with all ALSA audio devices

- See [Voice Messages Guide](docs/voice_messages.md) for voice message setup and format specifications

### 5. **Human Presence Detection**

- Background audio analysis using YAMNet ONNX model
- Weighted scoring system for 100+ human-related audio events
- Runs on 5-second duty cycle without blocking main thread
- Shares audio stream with wake word detector (no additional mic access)
- Sends presence detections to orchestrator for proactive conversations
- Configurable threshold and event weights from backend
- Detects speech, footsteps, doors, coughs, and other human activity

- See [Human Presence Detection Documentation](docs/human_presence_detection.md) for detailed configuration and troubleshooting

### 6. **Context Management**

- Location data via WiFi triangulation (Google Geolocation API)
- Dynamic variables for conversations:
  - Current date, time, day of week, timezone
  - Location: latitude, longitude, city, state, country
- Fetches location once at startup (with timeout)
- Force refresh on reconnection
- Graceful degradation if location unavailable

### 7. **Wake Word Detection**

- Porcupine wake word detection
- ASR-based similarity matching for flexible wake words
- Configurable similarity threshold from backend
- Fast response path using cached agent details (no orchestrator round-trip)
- Falls back to orchestrator request if cache unavailable

### 8. **Conversation Types**

- **Reactive conversations**: Triggered by wake word detection
  - Fast path: Uses cached default reactive agent (instant response)
  - Fallback: Requests agent details from orchestrator
- **Proactive conversations**: Server-initiated via `start_conversation` message
  - Supports trace context propagation for distributed tracing
  - Can interrupt idle state to start conversation

### 9. **WiFi Setup**

- Automatic WiFi setup mode when internet unavailable
- Creates "Kin_Setup" WiFi access point (password: `kinsetup123`)
- Web interface at `http://192.168.4.1:8080` for configuration
- Retry logic with up to 3 attempts
- Automatic WiFi connection deletion on failure (prevents deadlock)
- Configurable via `SKIP_WIFI_SETUP` env var or cached config
- Graceful fallback if WiFi setup module unavailable
- **Note**: WiFi setup requires sudo privileges for network management. If prompted for a password, enter your Raspberry Pi user password (the password for the user account running the script). For production deployments, consider configuring passwordless sudo for `nmcli` commands.

- See [Raspberry Pi Setup Guide](docs/RASPBERRY_PI_SETUP.md#6-wifi-setup-mode-optional---for-devices-without-initial-network-access) for WiFi setup instructions

### 10. **Device Pairing**

- Links device to a user account in the backend system
- Requires 4-digit pairing code from admin portal
- Pairing code is collected via WiFi setup web interface (when device has no internet) or can be provided when device has internet but is unpaired
- Device must be paired before it can:
  - Start conversations
  - Access user-specific settings
  - Receive proactive conversations
  - Function as a personal assistant
- Pairing is separate from WiFi connectivity - a device can have internet but still need pairing

**Note**: Currently, pairing code collection is combined with WiFi setup in the same web interface for convenience. However, pairing and WiFi are conceptually separate:
- **WiFi Setup**: Provides network connectivity (only needed when device has no internet)
- **Device Pairing**: Links device to user account (always needed if device is unpaired)

### 11. **Authentication & Configuration**

- Ed25519 device authentication
- JWT token management with automatic refresh
- Runtime configuration from backend (wake word, LED settings, API keys, etc.)
- Configuration caching for offline boot capability
- Token refresh monitoring in background

### 12. **Activity Tracking**

- Updates `.last_activity` file timestamp for wrapper idle monitoring
- Tracks wake word detections and conversation activity
- Enables wrapper to detect device activity for power management

### 13. **Signal Handling**

- `SIGINT/SIGTERM`: Full shutdown (graceful if not in conversation)
  - If in conversation: Ends current conversation, then shuts down
  - If idle: Shuts down immediately
- Proper cleanup in all cases (LEDs, audio streams, WebSocket connections)

### 14. **Modular Architecture**

- Clean separation of concerns
- Easy to test and maintain
- Reusable components
- Graceful degradation for optional modules

## Project Structure

```
raspberry-pi-client/
├── main.py                    # Entry point with KinClient
├── scripts/
│   └── generate_voice_messages.py  # Voice message generation helper
├── models/                    # ML models
│   ├── yamnet.onnx           # YAMNet ONNX model for presence detection
│   ├── yamnet_class_map.csv  # YAMNet class mappings
│   └── silero_vad.onnx       # Silero VAD model
└── lib/
    ├── __init__.py
    ├── config.py              # Configuration from env vars and backend
    ├── auth.py                # Device authentication (Ed25519)
    ├── device_auth.py         # Device authentication helpers
    ├── audio/
    │   ├── __init__.py
    │   ├── device_detection.py  # ALSA device detection
    │   ├── led_controller.py    # LED visual feedback
    │   ├── webrtc_aec.py        # WebRTC AEC (optional software AEC)
    │   └── respeaker/
    │       ├── __init__.py
    │       ├── respeaker.py     # ReSpeaker controller
    │       └── usb_4_mic_array/
    │           ├── __init__.py
    │           └── tuning.py     # ReSpeaker hardware tuning
    ├── audio/
    │   ├── ...
    │   └── voice_messages/      # Pre-recorded voice message files (voice feedback)
    │       ├── startup.wav
    │       ├── no_internet.wav
    │       ├── device_not_paired.wav
    │       └── wifi_setup_ready.wav
    ├── wake_word/
    │   ├── __init__.py
    │   └── detector.py         # Porcupine wake word detection
    ├── presence_detection/
    │   ├── __init__.py
    │   ├── detector.py         # Human presence detection (YAMNet)
    │   └── standalone_pi.py     # Standalone presence detection script
    ├── activity/
    │   ├── __init__.py
    │   └── monitor.py           # Activity monitoring (legacy)
    ├── elevenlabs/
    │   ├── __init__.py
    │   └── client.py           # ElevenLabs conversation client
    ├── orchestrator/
    │   ├── __init__.py
    │   └── client.py           # Orchestrator WebSocket client
    ├── agent/
    │   ├── orchestrator.py      # Orchestrator WebSocket client
    │   ├── context.py           # Location/context data manager
    │   ├── elevenlabs.py        # ElevenLabs conversation client
    │   └── tools/
    │       └── location/        # WiFi geolocation for agent context
    │           ├── __init__.py
    │           ├── fetcher.py   # Location fetching (WiFi triangulation)
    │           └── wifi_location.py  # WiFi location helpers
    ├── wifi_setup/
    │   ├── __init__.py
    │   ├── manager.py           # WiFi setup orchestration
    │   ├── access_point.py      # WiFi access point creation
    │   ├── http_server.py       # Web server for setup UI
    │   ├── network_connector.py # Network connection management
    │   └── connectivity.py      # Connectivity checking
    └── telemetry/
        ├── __init__.py
        ├── telemetry.py         # OpenTelemetry setup
        └── stdout_redirect.py   # Stdout/stderr redirection to OTEL
```

## Installation

### Prerequisites

- Python 3.11 or higher
- `uv` package manager (see installation instructions below)

### Installing uv

Install `uv` using one of the following methods:

**On Linux/macOS:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**On Windows:**
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**Or via pip:**
```bash
pip install uv
```

**Or via Homebrew (macOS):**
```bash
brew install uv
```

For more installation options, see the [official uv documentation](https://github.com/astral-sh/uv).

### Setting Up the Project

1. Clone the repository:
```bash
git clone https://github.com/companionsand/raspberry-pi-client
cd raspberry-pi-client
```

2. Install dependencies using `uv`:
```bash
uv sync
```

Or use the Makefile:
```bash
make install
```

This will create a virtual environment and install all dependencies from `pyproject.toml`.

## Usage

### Using Make Commands

The project includes a Makefile with convenient commands for common tasks:

```bash
# Show all available commands
make help

# Install dependencies
make install
# or
make setup

# Run the main client application
make run

# Generate voice message files (requires ELEVENLABS_API_KEY)
ELEVENLABS_API_KEY=your-key make generate-voice-messages

# Test voice message files (shows file information)
make test-voice-messages
```

### Direct Usage

You can also run commands directly:

```bash
# Run the client using uv
uv run main.py

# Send signals
kill -SIGINT <pid>   # Full shutdown (or Ctrl+C)
# Note: If in conversation, this will end the conversation first, then shut down
```

## Environment Variables

### Required Credentials (provisioned via admin portal)

- `DEVICE_ID` - Unique device identifier (UUID)
- `DEVICE_PRIVATE_KEY` - Ed25519 private key for device authentication (base64-encoded)

### Optional Configuration

- `SKIP_WIFI_SETUP` - Enable/disable WiFi setup mode (default: `false` = allow setup)
  - When `false`, device creates "Kin_Setup" WiFi network (password: `kinsetup123`) if no internet
  - Connect to it and visit http://192.168.4.1:8080 to configure WiFi credentials
  - **Note**: The WiFi setup interface also collects pairing code, but pairing and WiFi are separate concepts
  - Can also be set via cached config from backend
- `OTEL_ENABLED` - Enable OpenTelemetry (default: `true`)
- `OTEL_EXPORTER_ENDPOINT` - OTEL collector endpoint (default: `http://localhost:4318`)
  - Note: Always uses local collector; wrapper forwards to central endpoint
- `ENV` - Deployment environment (default: `production`)

### WebRTC AEC Configuration (Optional)

- `USE_WEBRTC_AEC` - Enable WebRTC software AEC on top of hardware (default: `false`)
  - Set to `true` to enable software echo cancellation
  - Useful for improving echo cancellation beyond hardware capabilities
- `WEBRTC_AEC_STREAM_DELAY_MS` - USB audio delay in milliseconds (default: `100`)
  - Typical range: 50-200ms depending on USB audio device
- `WEBRTC_AEC_NS_LEVEL` - Noise suppression level 0-3 (default: `1`)
  - 0 = off, 1 = moderate, 3 = maximum
- `WEBRTC_AEC_AGC_MODE` - AGC mode (default: `2`)
  - 1 = adaptive, 2 = fixed digital

### Runtime Configuration

Most settings are fetched from backend after authentication:
- API keys (ElevenLabs, Picovoice, Google)
- Wake word and ASR similarity threshold
- LED settings (enabled, brightness)
- Speaker volume (0-100%)
- Presence detection threshold and YAMNet weights
- ReSpeaker hardware tuning parameters
- Default reactive agent (for fast wake word response)
- Logging settings (debug log toggles)

Configuration is cached to `~/.kin_config.json` for offline boot capability.

## Voice Feedback Setup

The voice feedback system uses pre-recorded WAV files to provide voice guidance. To set up voice feedback:

### Option 1: Generate Voice Message Files (Recommended)

Use the provided script to generate all voice message files using ElevenLabs API:

```bash
# Set your ElevenLabs API key
export ELEVENLABS_API_KEY="your-api-key-here"

# Generate voice message files using Makefile
make generate-voice-messages

# Or run directly with uv:
uv run scripts/generate_voice_messages.py

# Or specify a custom voice:
uv run scripts/generate_voice_messages.py --voice your-voice-id
```

### Option 2: Manual Recording

Record your own voice messages and convert them to the required format:

```bash
# Required format: 16kHz, mono, 16-bit PCM WAV
ffmpeg -i input.mp3 -ar 16000 -ac 1 -sample_fmt s16 lib/audio/voice_messages/startup.wav
```

### Required Voice Message Files

- `startup.wav` - "Starting up"
- `no_internet.wav` - "No internet detected, entering setup mode"
- `device_not_paired.wav` - "Device not paired, entering setup mode"
- `wifi_setup_ready.wav` - "Join Kin underscore Setup WiFi and enter WiFi credentials"

### Testing Voice Message Files

```bash
# Test all voice message files (shows file information)
make test-voice-messages

# Or verify audio format manually
file lib/audio/voice_messages/startup.wav

# Play voice message file
aplay lib/audio/voice_messages/startup.wav
```

**Note**: If voice message files are missing, the system will continue to work normally but without voice guidance. Warnings will be logged for each missing file.

## Telemetry Details

The telemetry system provides comprehensive observability:

1. **Logging setup**: Log level is explicitly set to INFO before any imports
2. **Trace context**: Properly injected into WebSocket messages for distributed tracing
3. **Conversation traces**: Each conversation gets its own root trace span (not child of main trace)
4. **Structured logging**: All log messages include relevant context (user_id, device_id, etc.)
5. **Exception recording**: Errors are properly recorded with stack traces
6. **Stdout/stderr redirection**: All print() output is captured and sent to telemetry backend
7. **Graceful degradation**: System works without telemetry if unavailable

### Trace Context Propagation

- **Reactive conversations**: Trace context is created locally and propagated to orchestrator
- **Proactive conversations**: Trace context is extracted from `start_conversation` message and used for the conversation
- This enables end-to-end distributed tracing across services

## Testing

To test the client:

1. **Audio detection**: Verify ReSpeaker is detected or fallback works
2. **Voice feedback**: Test that voice messages play during startup and setup
   - Verify quiet hours (8pm-10am) skip startup message
3. **Wake word**: Test wake word detection with detected microphone
   - Verify fast path with cached agent details
   - Verify fallback to orchestrator request
4. **Presence detection**: Verify human presence detection runs in background
   - Check that detections are sent to orchestrator
5. **Conversation**: Verify full conversation flow works
   - Test reactive (wake word) conversations
   - Test proactive (server-initiated) conversations
6. **WiFi setup**: Test WiFi setup mode when internet unavailable
   - Verify retry logic and pairing code flow
   - Verify WiFi connection deletion on failure
7. **Signals**: Test SIGINT (full shutdown)
   - Verify graceful shutdown when idle
   - Verify conversation termination when in conversation
8. **LEDs**: Verify LED states if ReSpeaker is available
   - Test all states: BOOT, IDLE, WAKE_WORD_DETECTED, CONVERSATION, ERROR, OFF
9. **Telemetry**: Check OTEL traces in your observability backend
   - Verify conversation-level traces
   - Verify trace context propagation
10. **Location**: Verify location data is fetched and available for conversations
11. **Token refresh**: Verify JWT token refresh works in background
12. **Activity tracking**: Verify `.last_activity` file is updated

## Architecture Notes

### Fast Wake Word Response

The client implements a fast path for wake word detection:
- Backend provides `default_reactive_agent` in configuration
- Client caches agent ID and WebSocket URL
- On wake word detection, client immediately starts conversation (no orchestrator round-trip)
- Falls back to traditional flow if cache unavailable

### Audio Stream Sharing

- Wake word detector opens the microphone stream
- Presence detector receives audio via `feed_audio()` callback
- This avoids conflicts from multiple processes accessing the mic
- Both run concurrently without blocking each other

### Configuration Caching

- Full device configuration is cached to `~/.kin_config.json` after successful authentication
- Cache includes `skip_wifi_setup` and other critical settings
- Allows device to boot with cached config if internet unavailable
- Cache is updated on each successful authentication

### ReSpeaker Hardware Tuning

- Backend provides optimized ReSpeaker parameters (AEC, AGC, noise suppression)
- Parameters are applied on startup via `ReSpeakerController`
- Settings are optimized based on testing (prevents filter saturation, echo leakage)
- Critical: NLAEC must remain disabled (enabling causes device bricking)

- See [AEC Testing Guide](docs/AEC_TESTING_GUIDE.md) for detailed tuning parameters and testing procedures

## Documentation

For detailed information on specific features, see:

- [Raspberry Pi Setup Guide](docs/RASPBERRY_PI_SETUP.md) - Quick setup and installation
- [Production Reliability Guide](docs/RASPBERRY_PI_RELIABILITY.md) - Production-grade settings for 24/7 operation
- [AEC Testing Guide](docs/AEC_TESTING_GUIDE.md) - Comprehensive AEC testing and troubleshooting
- [WebRTC AEC Implementation](docs/WEBRTC_AEC_IMPLEMENTATION.md) - WebRTC software AEC configuration
- [LED States Reference](docs/LED_STATES_REFERENCE.md) - Complete LED state documentation
- [Human Presence Detection](docs/human_presence_detection.md) - Presence detection configuration
- [Voice Messages Guide](docs/voice_messages.md) - Voice message setup and format
- [Telemetry Reference](docs/TELEMETRY_REFERENCE.md) - Telemetry and observability details

## Future Enhancements

- Add metrics (planned for later)
- Add unit tests for each module
- Add integration tests
- Consider adding configuration file support
- Add more LED patterns
