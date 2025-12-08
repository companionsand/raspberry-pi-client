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
# Download directly from PyPI using curl to bypass piwheels
echo ""
echo "Downloading openwakeword directly from PyPI..."
curl -L -o /tmp/openwakeword.whl https://files.pythonhosted.org/packages/py3/o/openwakeword/openwakeword-0.6.0-py3-none-any.whl
echo "Installing openwakeword from downloaded file..."
pip3 install --no-deps /tmp/openwakeword.whl
rm /tmp/openwakeword.whl

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
    curl -L -o /tmp/openwakeword.whl https://files.pythonhosted.org/packages/py3/o/openwakeword/openwakeword-0.6.0-py3-none-any.whl
    pip3 install --no-deps --force-reinstall /tmp/openwakeword.whl
    rm /tmp/openwakeword.whl
}

echo ""
echo "✅ Installation complete!"
echo ""
echo "Next: python3 main.py"
