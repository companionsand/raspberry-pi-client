# Raspberry Pi Client Installation Instructions

## Issue: tflite-runtime Dependency Conflict

If you get this error when running `pip install -r requirements.txt`:

```
ERROR: Cannot install -r ./requirements.txt (line 2) because these package versions have conflicting dependencies.

The conflict is caused by:
    openwakeword 0.6.0 depends on tflite-runtime<3 and >=2.8.0; platform_system == "Linux"
```

This happens because pip's dependency resolver has trouble with the platform-specific markers even on Linux.

---

## Solution 1: Install Without Dependency Checks (Recommended)

```bash
cd raspberry-pi-client

# 1. Install openwakeword without checking dependencies
pip install openwakeword --no-deps

# 2. Install tflite-runtime separately (for ARM64 Raspberry Pi)
pip install tflite-runtime

# 3. Install remaining dependencies
pip install pyaudio sounddevice numpy scipy websockets certifi python-dotenv cryptography requests aiohttp pixel-ring elevenlabs onnxruntime

# 4. Verify installation
python -c "from openwakeword.model import Model; print('✅ openWakeWord installed')"
```

---

## Solution 2: Use Legacy Resolver

```bash
cd raspberry-pi-client

# Install with pip's old dependency resolver
pip install -r requirements.txt --use-deprecated=legacy-resolver
```

---

## Solution 3: Install TFLite Runtime First

Sometimes installing tflite-runtime before openwakeword helps:

```bash
cd raspberry-pi-client

# 1. Install tflite-runtime first
pip install tflite-runtime

# 2. Now install everything else
pip install -r requirements.txt
```

---

## Solution 4: Use ONNX Instead of TFLite

If TFLite continues to cause issues, you can use ONNX runtime instead:

```bash
cd raspberry-pi-client

# 1. Install openwakeword without deps
pip install openwakeword --no-deps

# 2. Install with ONNX instead of TFLite
pip install onnxruntime

# 3. Install remaining dependencies (skip tflite-runtime)
pip install pyaudio sounddevice numpy scipy websockets certifi python-dotenv cryptography requests aiohttp pixel-ring elevenlabs

# 4. Update detector.py to use ONNX
# (Model will automatically use ONNX if tflite is not available)
```

---

## Platform-Specific Notes

### Raspberry Pi OS (32-bit)

- Use `tflite-runtime` for better performance
- Pre-built wheels available from piwheels

### Raspberry Pi OS (64-bit)

- `tflite-runtime` may need manual compilation
- ONNX runtime is easier (pre-built wheels available)
- Performance is similar on 64-bit systems

### Check Your Architecture

```bash
# Check if you're running 32-bit or 64-bit
uname -m

# armv7l = 32-bit (use tflite-runtime)
# aarch64 = 64-bit (onnx or tflite both work)
```

---

## Recommended Installation by Pi Version

### Raspberry Pi 4 (64-bit OS)

```bash
cd raspberry-pi-client

# Use ONNX for easier installation
pip install openwakeword --no-deps
pip install onnxruntime numpy scipy pyaudio sounddevice websockets certifi python-dotenv cryptography requests aiohttp pixel-ring elevenlabs

# Verify
python -c "from openwakeword.model import Model; import onnxruntime; print('✅ Ready with ONNX')"
```

### Raspberry Pi 4 (32-bit OS) or Raspberry Pi 3

```bash
cd raspberry-pi-client

# Use TFLite for better performance
pip install openwakeword --no-deps
pip install tflite-runtime numpy scipy pyaudio sounddevice websockets certifi python-dotenv cryptography requests aiohttp pixel-ring elevenlabs onnxruntime

# Verify
python -c "from openwakeword.model import Model; import tflite_runtime; print('✅ Ready with TFLite')"
```

---

## Verification Steps

After installation, verify everything is working:

```bash
# 1. Check model file exists
ls -lh models/hey_mycroft.tflite
# Should show: ~840 KB file

# 2. Check VAD model exists
ls -lh models/silero_vad.onnx
# Should show: ~2 MB file

# 3. Test Python imports
python3 << EOF
from openwakeword.model import Model
import onnxruntime
import numpy
import sounddevice
print('✅ All core dependencies installed')
EOF

# 4. Test model loading
python3 << EOF
from openwakeword.model import Model
model = Model(wakeword_models=['models/hey_mycroft.tflite'])
print(f'✅ Model loaded: {list(model.models.keys())}')
EOF

# 5. Run the client
python3 main.py
```

Expected output when starting:

```
🎤 Initializing wake word detection...
   Wake word: 'hey mycroft'
   Loading model: /home/pi/raspberry-pi-client/models/hey_mycroft.tflite
   Model loaded: ['hey_mycroft']
   ✓ VAD gate initialized (reduces false wake word triggers)
   ✓ Listening for wake word...
```

---

## Troubleshooting

### Issue: Cannot install tflite-runtime

**On 64-bit Raspberry Pi OS:**

```bash
# tflite-runtime has limited 64-bit support
# Use ONNX instead (it's easier and works great)
pip install onnxruntime
```

**On 32-bit Raspberry Pi OS:**

```bash
# Use piwheels mirror (usually automatic on Raspberry Pi OS)
pip install tflite-runtime --index-url https://www.piwheels.org/simple
```

### Issue: Import error for tflite_runtime

```bash
# Check what's installed
pip list | grep -E "(tflite|onnx)"

# If tflite-runtime missing, install it
pip install tflite-runtime

# Or use ONNX as fallback
pip install onnxruntime
```

### Issue: "No module named 'openwakeword'"

```bash
# Reinstall openwakeword
pip uninstall openwakeword
pip install openwakeword --no-deps

# Verify
python3 -c "import openwakeword; print(openwakeword.__version__)"
```

### Issue: Audio device errors

```bash
# Install ALSA development libraries
sudo apt-get update
sudo apt-get install libasound2-dev portaudio19-dev

# Reinstall pyaudio
pip uninstall pyaudio
pip install pyaudio

# Test audio devices
python3 -m sounddevice
```

### Issue: Model file not found

```bash
# Re-download the model
cd raspberry-pi-client
curl -L -o models/hey_mycroft.tflite \
  https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/hey_mycroft_v0.1.tflite

# Verify size
ls -lh models/hey_mycroft.tflite  # Should be ~840 KB
```

### Issue: VAD model not found

```bash
# The VAD model should already exist
ls -lh models/silero_vad.onnx

# If missing, you need to download it separately
# (Contact dev team for VAD model source)
```

---

## Performance Optimization

### For Best Performance on Raspberry Pi 4:

1. **Use 64-bit OS** if possible
2. **Use ONNX runtime** for easier setup
3. **Enable hardware acceleration** (if available)
4. **Increase swap** if experiencing memory issues:
   ```bash
   sudo dphys-swapfile swapoff
   sudo nano /etc/dphys-swapfile
   # Set CONF_SWAPSIZE=1024
   sudo dphys-swapfile setup
   sudo dphys-swapfile swapon
   ```

### CPU Usage Expectations:

- **Idle (waiting for wake word)**: <5% CPU
- **During speech (VAD + wake word detection)**: 20-30% CPU
- **During conversation**: 15-25% CPU

If CPU usage is higher, check:

- VAD is properly initialized (reduces unnecessary processing)
- No other heavy processes running
- Proper audio device configuration

---

## Quick Start (TL;DR)

**Fastest method to get running:**

```bash
cd raspberry-pi-client

# Install everything without dependency checks
pip install openwakeword --no-deps
pip install onnxruntime tflite-runtime numpy scipy pyaudio sounddevice websockets certifi python-dotenv cryptography requests aiohttp pixel-ring elevenlabs

# Verify
python3 -c "from openwakeword.model import Model; print('✅ Ready')"

# Run
python3 main.py
# Say: "Hey Mycroft"
```

---

## Next Steps

Once installed and verified:

1. **Configure device credentials**: Edit `.env` file with your DEVICE_ID and DEVICE_PRIVATE_KEY
2. **Test wake word**: Run `python3 main.py` and say "Hey Mycroft"
3. **Follow testing guide**: See `../OPENWAKEWORD_TESTING.md` for comprehensive tests
4. **Monitor performance**: Check CPU/memory usage during operation
5. **Tune if needed**: Adjust detection thresholds in `lib/wake_word/detector.py`

---

## Additional Resources

- **Main documentation**: `../OPENWAKEWORD_MIGRATION.md`
- **Testing guide**: `../OPENWAKEWORD_TESTING.md`
- **General installation help**: `../INSTALLATION_FIX.md`
- **openWakeWord docs**: https://github.com/dscripka/openWakeWord

---

## Getting Help

If you continue to have installation issues:

1. Check your Python version: `python3 --version` (should be 3.9+)
2. Check your Pi model: `cat /proc/cpuinfo | grep Model`
3. Check your OS version: `cat /etc/os-release`
4. Try in a fresh virtual environment
5. Check openWakeWord GitHub issues: https://github.com/dscripka/openWakeWord/issues

For device-specific issues, include this info when asking for help:

```bash
# System info
uname -a
python3 --version
pip --version
cat /proc/cpuinfo | grep Model
```
