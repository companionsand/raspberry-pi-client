# Complete LED States Reference

## All LED States Overview

| State                     | Code | Color                   | Pattern           | Brightness | Duration/Timing | Purpose                         |
| ------------------------- | ---- | ----------------------- | ----------------- | ---------- | --------------- | ------------------------------- |
| **OFF**                   | 0    | None                    | Off               | 0%         | -               | Shutdown/cleanup                |
| **BOOT**                  | 1    | Soft Amber (244,162,97) | Smooth pulse      | 20-40%     | 2s cycle        | Device starting up              |
| **IDLE**                  | 2    | White (255,255,255)     | Breathing         | 20-100%    | 3s cycle        | Ready, waiting for wake word    |
| **WAKE_WORD_DETECTED**    | 3    | Amber/Orange/White      | Rapid color burst | 100%       | 1.5s burst      | Wake word detected!             |
| **CONVERSATION**          | 4    | Amber/Gold (255,180,20) | Pulsing           | 0-70%      | 1.5s cycle      | User speaking, device listening |
| **ERROR**                 | 5    | Soft Red (255,107,107)  | Slow blink        | 40%        | 2s on/off       | Error condition                 |
| **SPEAKING**              | 6    | White (255,255,255)     | Audio-reactive    | 20-100%    | Real-time       | Agent speaking                  |
| **THINKING**              | 7    | Amber/Gold (255,180,20) | Fast pulse        | 15-70%     | 0.5s cycle      | Processing response             |
| **WIFI_SETUP**            | 8    | Soft Amber (244,162,97) | Slow blink        | 60%        | 2s on/off       | WiFi setup mode active          |
| **ATTEMPTING_CONNECTION** | 9    | Soft Amber (244,162,97) | Fast blink        | 60%        | 0.5s on/off     | Connecting to WiFi/pairing      |

## State Transition Flow

### Normal Operation Flow

```
OFF → BOOT → IDLE → WAKE_WORD_DETECTED → CONVERSATION ⇄ SPEAKING/THINKING → IDLE
                ↑                                                                  ↓
                └──────────────────────────────────────────────────────────────────┘
                                    (conversation ends)
```

### WiFi Setup Flow

```
BOOT → WIFI_SETUP → ATTEMPTING_CONNECTION → {SUCCESS: IDLE | FAILURE: WIFI_SETUP}
```

### Error Handling

```
Any state → ERROR → (after timeout) → Previous state or IDLE
```

## Detailed State Descriptions

### OFF (0)

- **When**: System shutdown, cleanup
- **Visual**: All LEDs turned off
- **Notes**: Final state before power down

### BOOT (1)

- **When**: System initialization, component loading
- **Visual**: Gentle amber pulse, creates warm startup feeling
- **Pattern**: Sine wave smoothly oscillating between 20-40% brightness
- **Timing**: 2-second cycle, 50ms updates
- **Design**: Calming, not startling for elderly users

### IDLE (2)

- **When**: System ready, waiting for wake word
- **Visual**: White breathing effect, clear presence indicator
- **Pattern**: Sine wave oscillating between 20-100% brightness
- **Timing**: 3-second cycle (slower than conversation for calm feel)
- **Design**: Clear visibility that device is ready and listening

### WAKE_WORD_DETECTED (3)

- **When**: Wake word ("Hey Kin") detected
- **Visual**: Exciting color burst cycling through amber, orange, warm white
- **Pattern**: Rapid color cycling at full 100% brightness
- **Timing**: 1.5 seconds total, changes color every ~160ms
- **Design**: Attention-grabbing, confirms wake word detection
- **Colors**: Chosen to be visible through red-tinted ReSpeaker glass

### CONVERSATION (4)

- **When**: User is speaking, device is actively listening
- **Visual**: Amber/gold pulsing, faster than idle
- **Pattern**: Sine wave oscillating from 0-70% brightness
- **Timing**: 1.5-second cycle (faster than idle, shows engagement)
- **Design**: Warm, inviting color showing device is engaged

### ERROR (5)

- **When**: Connection issues, authentication failures, system errors
- **Visual**: Soft red slow blink
- **Pattern**: Simple on/off blink
- **Timing**: 2 seconds on, 2 seconds off
- **Brightness**: 40% (soft, not alarming)
- **Design**: Non-harsh error indicator, gives time to read message

### SPEAKING (6)

- **When**: Agent is speaking/responding to user
- **Visual**: White LEDs that pulse with voice energy
- **Pattern**: Audio-reactive based on RMS amplitude of audio
- **Brightness**: 20-100% (base 20%, peaks at 100% for loud speech)
- **Design**: Creates natural visual feedback synchronized with voice
- **Notes**: Uses gamma curve (^1.5) for punchier visual response

### THINKING (7)

- **When**: User paused, agent is preparing response
- **Visual**: Fast amber pulse, shows "processing"
- **Pattern**: Sine wave oscillating 15-70% brightness
- **Timing**: 0.5-second cycle (3x faster than conversation)
- **Design**: Fast pulse indicates active processing, keeps minimum brightness to avoid strobe effect

### WIFI_SETUP (8) ⭐ NEW

- **When**: WiFi setup mode active, AP and HTTP server running
- **Visual**: Soft amber slow blink
- **Pattern**: Simple on/off blink
- **Timing**: 2 seconds on, 2 seconds off
- **Brightness**: 60% (clear but not harsh)
- **Design**: Calm, patient waiting pattern for user to configure WiFi

### ATTEMPTING_CONNECTION (9) ⭐ NEW

- **When**: Attempting to connect to WiFi and authenticate with backend
- **Visual**: Soft amber fast blink
- **Pattern**: Simple on/off blink
- **Timing**: 0.5 seconds on, 0.5 seconds off
- **Brightness**: 60% (same as WIFI_SETUP for consistency)
- **Design**: Fast blink conveys active connection attempt

## Color Palette Rationale

### Warm Colors (Amber/Gold)

- Used for: BOOT, CONVERSATION, THINKING, WIFI_SETUP, ATTEMPTING_CONNECTION
- Purpose: Create welcoming, non-threatening presence
- Psychology: Warm, comforting, inviting interaction

### White

- Used for: IDLE, SPEAKING
- Purpose: Clear, neutral presence and voice visualization
- Psychology: Clean, attentive, ready

### Red (Soft)

- Used for: ERROR
- Purpose: Signal attention needed without alarm
- Psychology: Notice me, but not emergency
- Note: Intentionally kept soft (40% brightness) to avoid startling elderly users

## Design Philosophy

1. **Elderly-Friendly**: All patterns avoid harsh brightness or rapid flashing that could be startling
2. **Clear State Differentiation**: Each state has distinct pattern recognizable at a glance
3. **Smooth Animations**: Sine wave transitions for calm, professional feel
4. **Appropriate Brightness**: Most states use 20-70% brightness range
5. **Color Temperature**: Warm colors (amber) dominate for comfort
6. **Audio-Reactive Speaking**: Creates natural connection between voice and visuals
7. **Red-Tinted Glass Compatibility**: Colors chosen to remain visible through ReSpeaker's red glass cover

## Implementation Notes

- All animations run as async tasks
- State changes cleanly cancel previous animations
- Audio timeout detection for SPEAKING → CONVERSATION transition
- Update rate: 30-100ms for smooth visual experience
- Uses pixel_ring library for ReSpeaker hardware control


