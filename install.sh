#!/bin/bash
# Installation for Raspberry Pi client
# Handles openwakeword tflite-runtime dependency conflict

set -e

echo "========================================="
echo "Raspberry Pi Client - Installation"
echo "========================================="
echo ""

# Clean install of openwakeword
echo "Cleaning previous openwakeword installation..."
pip3 uninstall -y openwakeword 2>/dev/null || true

# Install all dependencies (including openwakeword's)
echo "Installing dependencies..."
pip3 install onnxruntime numpy scipy tqdm scikit-learn
pip3 install pyaudio sounddevice websockets certifi python-dotenv 
pip3 install cryptography requests aiohttp pixel-ring elevenlabs
pip3 install opentelemetry-api==1.28.2 opentelemetry-sdk==1.28.2 opentelemetry-exporter-otlp-proto-http==1.28.2

# Install openwakeword with --no-deps from PyPI ONLY (not piwheels)
# (piwheels version is missing resource files like melspectrogram.onnx)
echo ""
echo "Installing openwakeword from PyPI (not piwheels)..."
pip3 install --no-deps --no-index --index-url https://pypi.org/simple/ openwakeword

# Verify installation
echo ""
echo "Verifying installation..."
python3 -c "
import openwakeword
import onnxruntime
print('✓ Packages installed')

# Check if melspectrogram model exists
import os
oww_path = os.path.dirname(openwakeword.__file__)
melspec_path = os.path.join(oww_path, 'resources', 'models', 'melspectrogram.onnx')
if os.path.exists(melspec_path):
    print('✓ openWakeWord resource files found')
else:
    print('✗ Missing melspectrogram.onnx - reinstalling...')
    exit(1)
" || {
    echo ""
    echo "Package files missing, trying full reinstall from PyPI (bypassing piwheels)..."
    pip3 uninstall -y openwakeword
    pip3 install --force-reinstall --no-cache-dir --no-deps --no-index --index-url https://pypi.org/simple/ openwakeword
}

echo ""
echo "✅ Installation complete!"
echo ""
echo "Next: python3 main.py"
