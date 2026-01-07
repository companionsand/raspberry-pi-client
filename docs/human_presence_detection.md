# Human Presence Detection

This module provides human presence detection using the YAMNet audio classification model with weighted scoring.

## Overview

The `HumanPresenceDetector` runs in the background, sampling audio every 5 seconds and using a weighted classification approach to detect human presence. Unlike the existing `ActivityMonitor` which looks for specific target classes, this detector uses a comprehensive weighted scoring system across 100+ human-related audio classes.

**Note**: This detector uses ONNX Runtime exclusively for better performance and consistency.

## How It Works

1. **Audio Capture**: Every 5 seconds, captures ~1 second of audio (15,600 samples at 16kHz)
2. **Classification**: Runs YAMNet inference to get probabilities for 521 audio classes
3. **Weighted Scoring**: Calculates a weighted sum using human-presence class weights
4. **Detection**: If weighted score exceeds threshold (default 0.3), logs "human_detected"

**Note**: When using `standalone_pi.py`, the detector logs every cycle with the weighted score. For quieter operation, use the `--quiet` flag.

### Weighted Class Categories

- **Speech & Voice** (1.0): Speech, Conversation, Whispering, Shouting
- **Human Sounds** (0.9-0.95): Laughter, Crying, Coughing, Sneezing, Breathing
- **Movement** (0.85): Walking, Running, Shuffling
- **Activities** (0.6-0.7): Doors, Dishes, Keys, Typing, Writing
- **Ambient** (0.5-0.65): Rustling, Home appliances

## Setup

### 1. Convert YAMNet Model to ONNX (Required)

The detector requires the YAMNet model in ONNX format. Convert it using:

```bash
# Install dependencies (one-time)
pip install tf2onnx tensorflow

# Run conversion script
python scripts/convert_yamnet_to_onnx.py
```

This will create `models/yamnet.onnx` from the existing `models/yamnet.tflite`.

### Alternative: Manual Conversion

If the script fails, convert manually:

```bash
cd models
python -m tf2onnx.convert \
    --tflite yamnet.tflite \
    --output yamnet.onnx \
    --opset 13
```

## Usage

### Basic Usage

```python
from lib.detection import HumanPresenceDetector

# Initialize detector
detector = HumanPresenceDetector(
    mic_device_index=None,  # Use default mic
    threshold=0.3           # Detection threshold
)

# Start background detection
detector.start()

# ... detector runs in background every 30s ...

# Stop when done
detector.stop()
```

### Integration Example

```python
from lib.detection import HumanPresenceDetector

class MyApp:
    def __init__(self):
        # Initialize with specific microphone
        self.presence_detector = HumanPresenceDetector(
            mic_device_index=2,
            threshold=0.25  # Lower threshold = more sensitive
        )

    def start(self):
        # Start presence detection
        self.presence_detector.start()

        # Your app code...

    def cleanup(self):
        self.presence_detector.cleanup()
```

## Configuration

### Detection Threshold

The `threshold` parameter controls sensitivity:

- **0.2**: Very sensitive (may have false positives)
- **0.3**: Balanced (default, recommended)
- **0.4**: Conservative (fewer false positives, may miss some presence)

### Duty Cycle

The detection runs every 30 seconds by default. To change this, modify `DUTY_CYCLE_SECONDS` in `detector.py`.

### Custom Weights

To adjust class weights, modify `HUMAN_PRESENCE_WEIGHTS` in `detector.py`. Weights range from 0.0 to 1.0.

## Logging

When human presence is detected, the detector logs:

```
[HUMAN DETECTED] Weighted score: 0.456
  Top contributors: Speech (0.82), Conversation (0.45), Walk, footsteps (0.23)
```

With structured logging (via telemetry):

```python
{
    "event": "human_detected",
    "weighted_score": 0.456,
    "top_classes": ["Speech (0.82)", "Conversation (0.45)", "Walk, footsteps (0.23)"],
    "threshold": 0.3
}
```

## Performance

- **CPU Usage**: Minimal (~0.5% on Raspberry Pi 4)
- **Duty Cycle**: 5 seconds between checks
- **Inference Time**: ~100-200ms per check
- **Memory**: ~50MB (model + runtime)
- **Time Coverage**: 20% (samples 1s every 5s)

## Comparison with ActivityMonitor

| Feature          | ActivityMonitor             | HumanPresenceDetector  |
| ---------------- | --------------------------- | ---------------------- |
| Model Format     | TFLite                      | ONNX (required)        |
| Duty Cycle       | 5 seconds                   | 5 seconds              |
| Detection Method | Top class matching          | Weighted scoring       |
| Target Classes   | 10 specific classes         | 100+ weighted classes  |
| Use Case         | Specific activity detection | General human presence |

Both can run simultaneously if needed (they use different models and approaches).

## Troubleshooting

### Model Not Found Error

```
FileNotFoundError: YAMNet ONNX model not found: /path/to/models/yamnet.onnx
```

**Solution**: Run the conversion script (see Setup section above)

### Model Not Found Error

```
FileNotFoundError: YAMNet ONNX model not found
```

**Solution**: Convert the model first:

```bash
python scripts/convert_yamnet_to_onnx.py
```

### ONNX Runtime Error

```
Error: ONNX Runtime not installed
```

**Solution**: Install onnxruntime:

```bash
pip install onnxruntime
```

### High False Positive Rate

**Solution**: Increase the threshold:

```python
detector = HumanPresenceDetector(threshold=0.4)
```

### Missing Detections

**Solution**: Lower the threshold or check microphone placement:

```python
detector = HumanPresenceDetector(threshold=0.2)
```

## Technical Details

### Audio Preprocessing

- Sample Rate: 16kHz (YAMNet requirement)
- Sample Length: 15,600 samples (~0.975 seconds)
- Format: Float32, normalized to [-1.0, 1.0]
- Channels: Mono

### Model Details

- Input Shape: [15600]
- Output Shape: [521] (class probabilities)
- Framework: ONNX Runtime
- Execution Provider: CPU (optimized for Raspberry Pi)

### Weighted Scoring Formula

```
weighted_score = Σ(P(class_i) × weight_i) for all human-related classes

where:
  P(class_i) = probability of class i from YAMNet
  weight_i = predefined weight for class i (0.0-1.0)
```

## Examples

### Silent Room

```
Weighted score: 0.05 (below threshold, no detection)
```

### Conversation

```
[HUMAN DETECTED] Weighted score: 0.78
  Top contributors: Speech (0.85), Conversation (0.72), Laughter (0.34)
```

### Walking

```
[HUMAN DETECTED] Weighted score: 0.42
  Top contributors: Walk, footsteps (0.68), Door (0.32), Keys jangling (0.18)
```

### Background Music (No Humans)

```
Weighted score: 0.12 (music classes not weighted, below threshold)
```
