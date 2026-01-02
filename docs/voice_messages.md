# Voice Messages Directory

This directory contains pre-recorded voice messages for voice guidance during device startup and setup.

## Required Voice Message Files

The following WAV files are needed for the voice feedback system:

- `startup.wav` - "Starting up"
- `no_internet.wav` - "No internet detected, entering setup mode"
- `device_not_paired.wav` - "Device not paired, entering setup mode"
- `wifi_setup_ready.wav` - "Join Kin underscore Setup WiFi and enter WiFi credentials"

## Audio Format Specifications

All voice message files should meet these specifications:

- **Format**: WAV (PCM)
- **Sample Rate**: 16000 Hz (16 kHz)
- **Channels**: 1 (Mono)
- **Bit Depth**: 16-bit
- **Duration**: 3-5 seconds maximum per message

## Generating Voice Message Files

### Option 1: Use the Generation Script

Run the provided script to generate all voice messages using ElevenLabs API:

```bash
cd /path/to/raspberry-pi-client
python scripts/generate_voice_messages.py
```

The script will:
1. Read your ElevenLabs API key from environment variable `ELEVENLABS_API_KEY`
2. Generate all required voice messages
3. Save them in this directory with the correct format

### Option 2: Manual Recording

You can record your own voice messages:

1. Record each message using any recording software
2. Convert to the correct format using `ffmpeg`:

```bash
ffmpeg -i input.mp3 -ar 16000 -ac 1 -sample_fmt s16 startup.wav
```

### Option 3: Use Online TTS Services

Use any text-to-speech service (Google Cloud TTS, Amazon Polly, etc.) and convert the output to the required format.

## Testing Voice Message Files

To test if a voice message file is in the correct format:

```bash
# Check file info
file startup.wav
# Should show: RIFF (little-endian) data, WAVE audio, Microsoft PCM, 16 bit, mono 16000 Hz

# Play the file
aplay startup.wav
```

## Graceful Degradation

If voice message files are missing, the system will continue to operate normally but without voice guidance. A warning will be logged for each missing file.
