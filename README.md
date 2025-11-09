# Kin AI Raspberry Pi Client

A minimalistic Raspberry Pi client for Kin AI companion devices. Features wake word detection using Porcupine and real-time conversation via ElevenLabs WebSocket API, with communication to the conversation-orchestrator backend.

## Features

- **Wake Word Detection**: Uses Porcupine with built-in "Porcupine" keyword
- **Echo Cancellation**: PipeWire-based AEC for barge-in capability
- **Real-time Conversation**: WebSocket streaming with ElevenLabs
- **Backend Communication**: WebSocket connection to conversation-orchestrator
- **Heartbeat Monitoring**: Periodic status updates to backend
- **Conversation Management**: Handles start/end notifications and user-initiated conversations

## Architecture

```
┌─────────────────────────────────────────┐
│         Raspberry Pi 5                  │
│  ┌───────────────────────────────────┐  │
│  │  main.py                          │  │
│  │  - Wake word detection (Porcupine)│  │
│  │  - Conversation client (WebSocket)│  │
│  │  - Audio routing (PipeWire AEC)   │  │
│  │  - Orchestrator communication     │  │
│  └───────────────────────────────────┘  │
└─────────────────────────────────────────┘
              ↕ WebSocket
┌─────────────────────────────────────────┐
│   Conversation Orchestrator (Backend)   │
│  - Agent management                     │
│  - Trigger monitoring                  │
│  - Conversation coordination            │
└─────────────────────────────────────────┘
```

## Prerequisites

### Hardware
- Raspberry Pi 5 (recommended) or Pi 4
- USB microphone
- USB speaker or Bluetooth audio
- MicroSD card (32GB+ recommended)
- Power supply (5V/3A)

### Software
- Raspberry Pi OS Lite (64-bit)
- Python 3.11+
- PipeWire audio system
- Network connectivity (WiFi/Ethernet)

## Quick Setup

### 1. Install System Dependencies

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install audio dependencies
sudo apt install -y python3-pip python3-venv portaudio19-dev \
                    python3-pyaudio alsa-utils pipewire wireplumber \
                    libspa-0.2-modules

# Verify PipeWire is running
systemctl --user status pipewire
```

### 2. Setup Python Environment

```bash
# Clone or copy raspberry-pi-client directory to Pi
cd ~/kin-voice-ai/raspberry-pi-client

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure Audio (PipeWire AEC)

```bash
# Restart PipeWire to ensure AEC modules are loaded
systemctl --user restart wireplumber
systemctl --user restart pipewire pipewire-pulse

# Verify echo cancellation nodes exist
pactl list short sources | grep echo_cancel
pactl list short sinks | grep echo_cancel

# Expected output:
# echo_cancel.mic
# echo_cancel.speaker
```

If AEC nodes are missing:
```bash
# Install additional PipeWire modules
sudo apt install -y libspa-0.2-modules
systemctl --user restart wireplumber
```

### 4. Environment Configuration

Create `.env` file:
```bash
# Device credentials
DEVICE_ID=your-device-id-here
USER_ID=your-user-id-here
AUTH_TOKEN=your-supabase-jwt-token-here

# Backend
CONVERSATION_ORCHESTRATOR_URL=ws://localhost:8001/ws

# ElevenLabs API
ELEVENLABS_API_KEY=your-elevenlabs-api-key-here

# Wake word detection
PICOVOICE_ACCESS_KEY=your-picovoice-access-key-here
WAKE_WORD=porcupine  # Built-in keyword

# Audio devices (optional)
MIC_DEVICE=echo_cancel.mic
SPEAKER_DEVICE=echo_cancel.speaker
```

### 5. Test Audio Setup

```bash
# Test microphone recording
arecord -D plughw:CARD=USB,DEV=0 -f cd -d 3 test.wav

# Test speaker playback
aplay test.wav

# Test AEC routing
pactl list short sources
pactl list short sinks
```

## Usage

### Run the Client

```bash
# Activate virtual environment
source venv/bin/activate

# Run the client
python main.py
```

### User-Initiated Conversations

1. Say the wake word ("Porcupine" by default)
2. Client sends `user_initiated` request to orchestrator
3. Orchestrator responds with `agent_details` message
4. Client connects to ElevenLabs WebSocket
5. Conversation begins

### Trigger-Initiated Conversations

1. Orchestrator detects a trigger is ready
2. Orchestrator sends `start_conversation` message
3. Client connects to ElevenLabs WebSocket
4. Conversation begins

### Heartbeat Messages

The client automatically sends heartbeat messages every 10 seconds to the orchestrator, including:
- `user_id`
- `device_id`
- `device_status` (online/offline/error/setup)
- `timestamp`

### Conversation End Detection

Conversations end when:
- **Normal**: WebSocket connection closes normally
- **Silence**: 30 seconds of no audio activity
- **User-initiated**: User sends termination signal (SIGUSR1)
- **Network failure**: Connection error occurs

## Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `DEVICE_ID` | Unique device identifier | Yes |
| `USER_ID` | User ID from Supabase Auth | Yes |
| `AUTH_TOKEN` | Supabase JWT token | Yes |
| `CONVERSATION_ORCHESTRATOR_URL` | WebSocket URL for orchestrator | Yes |
| `ELEVENLABS_API_KEY` | ElevenLabs API key | Yes |
| `PICOVOICE_ACCESS_KEY` | Picovoice access key | Yes |
| `WAKE_WORD` | Wake word keyword (default: "porcupine") | No |
| `MIC_DEVICE` | Microphone device name (default: "echo_cancel.mic") | No |
| `SPEAKER_DEVICE` | Speaker device name (default: "echo_cancel.speaker") | No |

### Audio Settings

- **Sample Rate**: 16000 Hz (matches ElevenLabs requirements)
- **Channels**: 1 (mono)
- **Chunk Size**: 512 frames (~32ms for low latency)

## Message Protocol

### Messages to Orchestrator

#### Authentication
```json
{
  "type": "auth",
  "token": "supabase-jwt-token",
  "device_id": "device-id",
  "user_id": "user-id"
}
```

#### User Initiated
```json
{
  "type": "user_initiated",
  "user_id": "user-id",
  "device_id": "device-id"
}
```

#### Heartbeat
```json
{
  "type": "heartbeat",
  "user_id": "user-id",
  "device_id": "device-id",
  "device_status": "online",
  "timestamp": "2024-01-01T00:00:00Z"
}
```

#### Conversation Start
```json
{
  "type": "conversation_start",
  "conversation_id": "uuid",
  "elevenlabs_conversation_id": "elevenlabs-id",
  "agent_id": "agent-id",
  "device_id": "device-id",
  "user_id": "user-id",
  "start_time": "2024-01-01T00:00:00Z"
}
```

#### Conversation End
```json
{
  "type": "conversation_end",
  "conversation_id": "uuid",
  "elevenlabs_conversation_id": "elevenlabs-id",
  "agent_id": "agent-id",
  "device_id": "device-id",
  "user_id": "user-id",
  "end_time": "2024-01-01T00:00:00Z",
  "end_reason": "normal|silence|network_failure|user_terminated"
}
```

### Messages from Orchestrator

#### Connection Confirmation
```json
{
  "type": "connected",
  "message": "Connected successfully"
}
```

#### Agent Details
```json
{
  "type": "agent_details",
  "agent_id": "agent-id",
  "device_id": "device-id",
  "user_id": "user-id",
  "web_socket_url": "wss://api.elevenlabs.io/v1/convai/conversation?agent_id=..."
}
```

#### Start Conversation (Trigger-based)
```json
{
  "type": "start_conversation",
  "agent_id": "agent-id",
  "device_id": "device-id",
  "user_id": "user-id",
  "web_socket_url": "wss://api.elevenlabs.io/v1/convai/conversation?agent_id=..."
}
```

#### Error
```json
{
  "type": "error",
  "message": "Error description"
}
```

## Troubleshooting

### Audio Issues

**Problem**: No audio input/output
```bash
# Check audio devices
pactl list short sources
pactl list short sinks

# Test microphone
arecord -D plughw:CARD=USB,DEV=0 -f cd -d 3 test.wav

# Test speaker
aplay test.wav
```

**Problem**: Echo cancellation not working
```bash
# Restart PipeWire services
systemctl --user restart wireplumber
systemctl --user restart pipewire pipewire-pulse

# Verify AEC nodes exist
pactl list short sources | grep echo_cancel
```

### Connection Issues

**Problem**: Cannot connect to orchestrator
- Check `CONVERSATION_ORCHESTRATOR_URL` is correct
- Verify network connectivity
- Check orchestrator is running
- Verify `AUTH_TOKEN` is valid

**Problem**: Authentication fails
- Verify `AUTH_TOKEN` is a valid Supabase JWT token
- Check token hasn't expired
- Verify `USER_ID` matches token user

### Wake Word Issues

**Problem**: Wake word not detected
- Check `PICOVOICE_ACCESS_KEY` is valid
- Verify microphone is working
- Adjust sensitivity in code (default: 0.7)
- Check audio device is correct

## Notes

- The client assumes the device is already set up and registered
- Device setup code will be added later
- User termination signal: `kill -USR1 <pid>`
- Heartbeat interval: 10 seconds (configurable via `Config.HEARTBEAT_INTERVAL`)
- Silence timeout: 30 seconds (configurable in `ElevenLabsConversationClient`)

## Security

- Never commit `.env` file to version control
- Keep `AUTH_TOKEN` secure and rotate regularly
- Use HTTPS/WSS in production
- Validate all environment variables before running

## Project Structure

```
raspberry-pi-client/
├── main.py                 # Main application
├── requirements.txt        # Python dependencies
├── .env.example           # Example environment variables
├── .gitignore             # Git ignore rules
└── README.md              # This file
```

## License

Copyright © 2025 Kin Voice AI. All rights reserved.
