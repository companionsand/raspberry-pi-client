# Interruption Handling (Barge-In)

## Overview

The Raspberry Pi client supports **user interruption (barge-in)**, allowing users to interrupt the AI agent mid-response by speaking. Interruption detection is handled **server-side by ElevenLabs**, which sends an `interruption` message when it detects user speech during agent playback.

## How It Works

### Architecture

```
ElevenLabs Server (handles interruption detection)
         │
         │  detects user speech during agent audio
         ▼
sends 'interruption' message via WebSocket
         │
         ▼
┌─────────────────────────────────────────────────────┐
│                  Raspberry Pi Client                 │
│                                                      │
│  Input stream (mic) ──► WebSocket ──► ElevenLabs    │
│                                                      │
│  Audio Queue ◄── ElevenLabs audio chunks            │
│       │                                              │
│       ▼                                              │
│  Playback Task ──► Output stream (speaker)          │
│       │                                              │
│  On 'interruption' message:                         │
│    1. Set playback_active = False                   │
│    2. Clear audio queue                             │
│    3. Reset LED to listening state                  │
└─────────────────────────────────────────────────────┘
```

### Key Points

1. **Server-side detection**: ElevenLabs receives the user's audio stream and detects when they speak during agent playback
2. **WebSocket message**: When interruption is detected, ElevenLabs sends a message with `type: 'interruption'`
3. **Client response**: The client immediately stops playback and clears queued audio

## Implementation Details

### Key Components

#### 1. Separate Input/Output Streams

- **Input stream** (`InputStream`): Microphone only - sends audio to ElevenLabs
- **Output stream** (`OutputStream`): Speaker only - plays agent audio

Separate streams allow clean interruption without affecting the microphone input.

#### 2. Audio Queue (`asyncio.Queue`)

- All incoming agent audio chunks are placed in a queue
- Decouples audio reception from playback
- Can be cleared instantly on interruption

#### 3. Playback Task (`_play_audio`)

- Runs concurrently with send/receive tasks
- Consumes audio chunks from queue and writes to stream
- Plays audio in small sub-chunks (256 samples = ~16ms) for fast interruption response
- Respects `playback_active` flag to stop mid-playback

#### 4. Interruption Handler (in `_receive_messages`)

When an `interruption` message is received:

```python
elif data.get('type') == 'interruption' or 'interruption' in data:
    # Stop playback immediately
    self.playback_active = False
    
    # Clear the audio queue
    while not self.audio_queue.empty():
        self.audio_queue.get_nowait()
    
    # Reset chunk count for next agent turn
    self._chunk_count = 0
    
    # Reset LED to conversation/listening state
    if self.led_controller:
        self.led_controller.set_state(self.led_controller.STATE_CONVERSATION)
```

## Behavior

### Normal Flow

1. Agent audio arrives from ElevenLabs
2. Audio chunk is queued
3. Playback task dequeues and plays chunk
4. Process continues until conversation ends

### Interruption Flow

1. User starts speaking during agent playback
2. ElevenLabs detects user speech (server-side VAD)
3. ElevenLabs sends `interruption` message
4. Client receives message and:
   - Stops current audio playback
   - Clears all queued audio chunks
   - Updates LED to listening state
5. ElevenLabs processes user's new input
6. Agent generates new response (or continues based on context)

### Edge Cases Handled

- **Empty queue**: Queue drain succeeds silently
- **Late-arriving chunks**: Continue to queue normally after interruption
- **Conversation end**: All tasks cancelled cleanly via `running` flag

## Logs

Interruption events are logged:

```json
{
  "event": "elevenlabs_interruption_received",
  "conversation_id": "...",
  "queue_size_before": 5,
  "user_id": "...",
  "device_id": "..."
}
```

## Performance

- **Latency**: Determined by ElevenLabs server-side detection (~100-300ms typical)
- **Playback stop**: Instant once message received (within current 16ms sub-chunk)
- **Memory**: Queue size bounded by ElevenLabs chunk rate (~10-20 chunks max)

## Historical Note

Previously, this client had **local VAD-based barge-in detection** that ran on-device. This was disabled in favor of ElevenLabs server-side detection because:

1. ElevenLabs already does VAD server-side and sends interruption messages
2. Server-side detection is more reliable (no echo cancellation issues)
3. Simpler client code with less state to manage
4. Consistent behavior across different hardware configurations

The local VAD is still used for **LED state management** (showing when user is speaking), but not for interruption detection.

---

**Last Updated**: December 2025  
**Status**: ✅ Working - Server-side interruption detection  
**Applies to**: raspberry-pi-client
