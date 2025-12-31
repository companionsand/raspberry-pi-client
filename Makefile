.PHONY: help install setup run generate-voice-messages test-voice-messages

# Default target
help:
	@echo "Kin AI Raspberry Pi Client - Available Commands:"
	@echo ""
	@echo "  make install          - Install dependencies using uv sync"
	@echo "  make setup            - Alias for install"
	@echo "  make run              - Run the main client application"
	@echo "  make generate-voice-messages - Generate voice message files (requires ELEVENLABS_API_KEY)"
	@echo "  make test-voice-messages - Test voice message files (requires aplay)"
	@echo ""
	@echo "Environment Variables:"
	@echo "  Required: DEVICE_ID, DEVICE_PRIVATE_KEY"
	@echo "  Optional: SKIP_WIFI_SETUP, OTEL_ENABLED, USE_WEBRTC_AEC, etc."
	@echo ""
	@echo "See README.md for detailed documentation."

# Install dependencies
install: check-uv
	@echo "Installing dependencies with uv..."
	uv sync

# Alias for install
setup: install

# Run the main application
run: check-uv
	@echo "Running Kin AI Raspberry Pi Client..."
	uv run main.py

# Generate voice message files
generate-voice-messages: check-uv
	@echo "Generating voice message files..."
	@if [ -z "$$ELEVENLABS_API_KEY" ]; then \
		echo "Error: ELEVENLABS_API_KEY environment variable is not set"; \
		echo "Usage: ELEVENLABS_API_KEY=your-key make generate-voice-messages"; \
		exit 1; \
	fi
	uv run scripts/generate_voice_messages.py

# Test voice message files (requires aplay on Linux/Raspberry Pi)
test-voice-messages:
	@echo "Testing voice message files..."
	@for file in lib/voice_feedback/voice_messages/*.wav; do \
		if [ -f "$$file" ]; then \
			echo "Testing: $$file"; \
			file "$$file" || true; \
		fi; \
	done
	@echo ""
	@echo "To play a voice message, run:"
	@echo "  aplay lib/voice_feedback/voice_messages/startup.wav"
