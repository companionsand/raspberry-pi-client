#!/bin/bash
# Automated installation script for Raspberry Pi client
# This handles the dependency conflict with openwakeword gracefully

set -e  # Exit on error

echo "========================================="
echo "Raspberry Pi Client - Installation"
echo "========================================="
echo ""

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 9 ]); then
    echo "❌ Error: Python 3.9+ required (found $PYTHON_VERSION)"
    exit 1
fi

echo "✓ Python version: $PYTHON_VERSION"
echo ""

# Try standard installation first
echo "Attempting standard installation..."
if pip3 install -r requirements.txt 2>/dev/null; then
    echo ""
    echo "✅ Installation successful!"
else
    echo ""
    echo "⚠️  Standard installation failed (dependency conflict)"
    echo "   Using alternative installation method..."
    echo ""
    
    # Fallback: Use locked requirements
    if [ -f "requirements-lock.txt" ]; then
        echo "Installing from requirements-lock.txt..."
        pip3 install -r requirements-lock.txt
    else
        # Manual installation as last resort
        echo "Installing dependencies manually..."
        
        # Install core dependencies first
        pip3 install 'numpy>=1.24.0' 'scipy>=1.11.0'
        
        # Install ONNX runtime (primary runtime for all Python versions)
        pip3 install 'onnxruntime>=1.16.0'
        
        # Try to install tflite-runtime (optional, for older Python versions)
        echo "Attempting to install tflite-runtime (optional)..."
        if pip3 install 'tflite-runtime>=2.8.0,<3' 2>/dev/null; then
            echo "✓ tflite-runtime installed"
        else
            echo "⚠️  tflite-runtime not available (will use onnxruntime)"
        fi
        
        # Install openwakeword
        pip3 install 'openwakeword>=0.5.0'
        
        # Install remaining dependencies
        pip3 install pyaudio sounddevice websockets certifi python-dotenv cryptography requests aiohttp pixel-ring elevenlabs
        pip3 install opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp-proto-http
    fi
    
    echo ""
    echo "✅ Installation complete (using fallback method)"
fi

echo ""
echo "========================================="
echo "Verifying installation..."
echo "========================================="
echo ""

# Verify core imports
python3 << 'EOF'
try:
    from openwakeword.model import Model
    print("✓ openWakeWord")
except ImportError as e:
    print(f"✗ openWakeWord: {e}")
    exit(1)

try:
    import onnxruntime
    print("✓ ONNX Runtime")
except ImportError as e:
    print(f"✗ ONNX Runtime: {e}")
    exit(1)

try:
    import tflite_runtime
    print("✓ TFLite Runtime (optional)")
except ImportError:
    print("⚠ TFLite Runtime not available (using ONNX)")

try:
    import sounddevice
    print("✓ sounddevice")
except ImportError as e:
    print(f"✗ sounddevice: {e}")
    exit(1)

try:
    import numpy
    print("✓ NumPy")
except ImportError as e:
    print(f"✗ NumPy: {e}")
    exit(1)

print("\n✅ All core dependencies verified")
EOF

echo ""
echo "========================================="
echo "Installation Summary"
echo "========================================="
echo ""
echo "✅ Dependencies installed successfully"
echo ""
echo "Next steps:"
echo "  1. Configure .env with your device credentials"
echo "  2. Run: python3 main.py"
echo "  3. Say: 'Hey Mycroft'"
echo ""
echo "For troubleshooting, see INSTALL_INSTRUCTIONS.md"
echo ""
