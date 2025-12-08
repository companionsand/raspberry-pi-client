#!/bin/bash
# Simple installation for Raspberry Pi client
# Uses --no-deps for openwakeword to avoid tflite-runtime conflict

set -e

echo "Installing Raspberry Pi client dependencies..."
echo ""

# Install everything except openwakeword
pip3 install onnxruntime numpy scipy pyaudio sounddevice websockets certifi python-dotenv cryptography requests aiohttp pixel-ring elevenlabs opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp-proto-http

# Install openwakeword without its dependencies (avoids tflite-runtime)
pip3 install --no-deps openwakeword

echo ""
echo "✅ Installation complete!"
echo ""
echo "Run: python3 main.py"
