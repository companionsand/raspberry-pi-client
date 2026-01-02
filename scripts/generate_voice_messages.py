#!/usr/bin/env python3
"""
Generate Voice Messages for Raspberry Pi Client

This script generates pre-recorded voice messages using the ElevenLabs API.
The audio files are saved in the correct format (16kHz, mono, 16-bit PCM WAV)
for use with the voice feedback system.

Requirements:
- elevenlabs package (pip install elevenlabs)
- ELEVENLABS_API_KEY environment variable or passed as argument

Usage:
    python generate_voice_messages.py [--api-key YOUR_KEY] [--voice VOICE_ID]
"""

import os
import sys
import argparse
import wave
from pathlib import Path
from typing import Optional

try:
    from elevenlabs.client import ElevenLabs
    from elevenlabs import save
except ImportError:
    print("Error: elevenlabs package not found")
    print("Install it with: pip install elevenlabs")
    sys.exit(1)

# Voice messages to generate
MESSAGES = {
    "startup": "Starting up",
    "no_internet": "No internet detected, entering setup mode",
    "device_not_paired": "Device not paired, entering setup mode",
    "wifi_setup_ready": "Join Kin underscore Setup WiFi and enter WiFi credentials"
}

# Target audio format
TARGET_SAMPLE_RATE = 16000  # 16 kHz
TARGET_CHANNELS = 1  # Mono
TARGET_SAMPLE_WIDTH = 2  # 16-bit


def generate_audio_file(
    client: ElevenLabs,
    text: str,
    output_path: Path,
    voice_id: Optional[str] = None
) -> bool:
    """
    Generate a voice message file using ElevenLabs API.
    
    Args:
        client: ElevenLabs client instance
        text: Text to convert to speech
        output_path: Path where to save the audio file
        voice_id: Optional voice ID (uses default if not provided)
        
    Returns:
        True if successful, False otherwise
    """
    try:
        print(f"  Generating: {text}")
        
        # Generate audio using ElevenLabs (new API)
        if voice_id:
            audio_generator = client.text_to_speech.convert(
                voice_id=voice_id,
                text=text,
                model_id="eleven_monolingual_v1"
            )
        else:
            # Use default voice (Rachel)
            audio_generator = client.text_to_speech.convert(
                voice_id="21m00Tcm4TlvDq8ikWAM",  # Rachel - default voice
                text=text,
                model_id="eleven_monolingual_v1"
            )
        
        # Save to temporary file first (ElevenLabs returns MP3)
        temp_path = output_path.with_suffix('.mp3')
        save(audio_generator, str(temp_path))
        
        # Convert to WAV with correct format using pydub
        try:
            from pydub import AudioSegment
            
            # Load MP3
            audio = AudioSegment.from_mp3(str(temp_path))
            
            # Convert to target format (16kHz, mono, 16-bit)
            audio = audio.set_frame_rate(TARGET_SAMPLE_RATE)
            audio = audio.set_channels(TARGET_CHANNELS)
            audio = audio.set_sample_width(TARGET_SAMPLE_WIDTH)
            
            # Export as WAV
            audio.export(
                str(output_path),
                format="wav",
                parameters=["-acodec", "pcm_s16le"]
            )
            
            # Clean up temp file
            temp_path.unlink()
            
            print(f"  ✓ Saved: {output_path}")
            return True
            
        except ImportError:
            print("  Warning: pydub not installed, using ffmpeg directly")
            
            # Fallback to ffmpeg command
            import subprocess
            
            result = subprocess.run([
                'ffmpeg', '-i', str(temp_path),
                '-ar', str(TARGET_SAMPLE_RATE),
                '-ac', str(TARGET_CHANNELS),
                '-sample_fmt', 's16',
                '-y',  # Overwrite output file
                str(output_path)
            ], capture_output=True, text=True)
            
            # Clean up temp file
            temp_path.unlink()
            
            if result.returncode == 0:
                print(f"  ✓ Saved: {output_path}")
                return True
            else:
                print(f"  ✗ ffmpeg error: {result.stderr}")
                return False
        
    except Exception as e:
        print(f"  ✗ Error: {type(e).__name__}: {e}")
        return False


def verify_audio_format(file_path: Path) -> bool:
    """
    Verify that a voice message file has the correct format.
    
    Args:
        file_path: Path to audio file
        
    Returns:
        True if format is correct, False otherwise
    """
    try:
        with wave.open(str(file_path), 'rb') as wf:
            channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            sample_rate = wf.getframerate()
            
            if channels != TARGET_CHANNELS:
                print(f"  ⚠ Warning: Expected {TARGET_CHANNELS} channel(s), got {channels}")
                return False
            
            if sample_width != TARGET_SAMPLE_WIDTH:
                print(f"  ⚠ Warning: Expected {TARGET_SAMPLE_WIDTH} byte sample width, got {sample_width}")
                return False
            
            if sample_rate != TARGET_SAMPLE_RATE:
                print(f"  ⚠ Warning: Expected {TARGET_SAMPLE_RATE} Hz sample rate, got {sample_rate}")
                return False
            
            return True
            
    except Exception as e:
        print(f"  ✗ Verification failed: {type(e).__name__}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Generate voice messages for Raspberry Pi client"
    )
    parser.add_argument(
        '--api-key',
        help='ElevenLabs API key (or set ELEVENLABS_API_KEY env var)',
        default=os.getenv('ELEVENLABS_API_KEY')
    )
    parser.add_argument(
        '--voice',
        help='ElevenLabs voice ID (optional, uses default if not specified)',
        default=None
    )
    parser.add_argument(
        '--output-dir',
        help='Output directory for voice message files',
        default=None
    )
    
    args = parser.parse_args()
    
    # Validate API key
    if not args.api_key:
        print("Error: ElevenLabs API key is required")
        print("Set ELEVENLABS_API_KEY environment variable or use --api-key argument")
        sys.exit(1)
    
    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        # Default to lib/audio/voice_messages directory
        script_dir = Path(__file__).parent
        output_dir = script_dir.parent / "lib" / "audio" / "voice_messages"
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Generating Voice Messages for Raspberry Pi Client")
    print("=" * 60)
    print(f"Output directory: {output_dir}")
    print(f"Voice ID: {args.voice or 'default'}")
    print(f"Target format: {TARGET_SAMPLE_RATE} Hz, {TARGET_CHANNELS} channel(s), {TARGET_SAMPLE_WIDTH * 8}-bit")
    print()
    
    # Initialize ElevenLabs client
    try:
        client = ElevenLabs(api_key=args.api_key)
    except Exception as e:
        print(f"Error initializing ElevenLabs client: {e}")
        sys.exit(1)
    
    # Generate each voice message
    success_count = 0
    total_count = len(MESSAGES)
    
    for message_name, message_text in MESSAGES.items():
        print(f"\n[{success_count + 1}/{total_count}] {message_name}.wav")
        
        output_path = output_dir / f"{message_name}.wav"
        
        # Generate voice message file
        if generate_audio_file(client, message_text, output_path, args.voice):
            # Verify format
            if verify_audio_format(output_path):
                print(f"  ✓ Format verified")
                success_count += 1
            else:
                print(f"  ⚠ Format verification failed")
        else:
            print(f"  ✗ Generation failed")
    
    # Summary
    print()
    print("=" * 60)
    print(f"Generation complete: {success_count}/{total_count} files")
    print("=" * 60)
    
    if success_count == total_count:
        print("✓ All voice messages generated successfully!")
        print()
        print("You can now use the voice feedback system on your Raspberry Pi.")
        sys.exit(0)
    else:
        print(f"⚠ {total_count - success_count} file(s) failed to generate")
        print()
        print("The system will still work but some voice messages may be missing.")
        sys.exit(1)


if __name__ == "__main__":
    main()

