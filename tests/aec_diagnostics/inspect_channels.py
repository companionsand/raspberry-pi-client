import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
from scipy import signal
import time
import os
import argparse

# Configuration
SAMPLE_RATE = 16000
OUTPUT_DIR = "aec_recordings"
VOICE_FILE = "/usr/share/sounds/alsa/Front_Center.wav"
VOICE_REPETITIONS = 3

def load_voice_sample(voice_file=VOICE_FILE, repetitions=VOICE_REPETITIONS, volume=1.0):
    """Load voice sample and repeat it for testing"""
    if not os.path.exists(voice_file):
        print(f"⚠️  Voice file not found: {voice_file}")
        print("   Falling back to generated tone")
        # Fallback to tone generation
        duration = 5.0
        t = np.linspace(0, duration, int(duration * SAMPLE_RATE), False)
        tone = np.sin(2 * np.pi * 440.0 * t)
        envelope = 0.5 * (1 + np.sin(2 * np.pi * 2 * t))
        mask = np.ones_like(t)
        mask[:int(1.0*SAMPLE_RATE)] = 0
        mask[-int(1.0*SAMPLE_RATE):] = 0
        return (tone * envelope * mask * 0.5 * volume).astype(np.float32)
    
    # Load the voice sample
    sr, audio = wav.read(voice_file)
    
    # Convert to float32 and normalize
    if audio.dtype == np.int16:
        audio_float = audio.astype(np.float32) / 32768.0
    elif audio.dtype == np.int32:
        audio_float = audio.astype(np.float32) / 2147483648.0
    else:
        audio_float = audio.astype(np.float32)
    
    # Handle multi-channel: take first channel if stereo
    if len(audio_float.shape) > 1:
        audio_float = audio_float[:, 0]
    
    # Resample if not 16kHz
    if sr != SAMPLE_RATE:
        num_samples = int(len(audio_float) * SAMPLE_RATE / sr)
        audio_float = signal.resample(audio_float, num_samples).astype(np.float32)
    
    # Repeat the sample
    repeated = np.tile(audio_float, repetitions)
    
    # Apply volume
    repeated = repeated * volume
    
    print(f"✓ Loaded voice sample: {voice_file}")
    print(f"  Original sample rate: {sr}Hz, duration: {len(audio_float)/SAMPLE_RATE:.2f}s")
    print(f"  Repeating {repetitions} times for {len(repeated)/SAMPLE_RATE:.2f}s total")
    print(f"  Volume: {int(volume * 100)}%")
    
    return repeated

def find_respeaker():
    """Finds ReSpeaker device index"""
    devices = sd.query_devices()
    for i, dev in enumerate(devices):
        if 'respeaker' in dev['name'].lower() or 'arrayuac10' in dev['name'].lower():
            # We strictly need the device that supports 6 input channels
            if dev['max_input_channels'] >= 6:
                return i, dev['name']
    return None, None

def run_test(volume=1.0):
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    print("🔍 Searching for ReSpeaker (6-channel mode)...")
    dev_idx, dev_name = find_respeaker()
    
    if dev_idx is None:
        print("❌ ReSpeaker (6-channel) not found!")
        print("   Check 'arecord -L'. Ensure firmware is 6-channel.")
        return

    print(f"✅ Found: {dev_name} (Index {dev_idx})")

    # Prepare signal
    print("📊 Loading voice sample...")
    playback_audio = load_voice_sample(volume=volume)
    duration = len(playback_audio) / SAMPLE_RATE
    
    print(f"\n🎙  RECORDING {duration:.1f}s...")
    print("   Signal: Voice sample (repeated 3x)")
    print("   ACTION: Please remain quiet for this test.")
    
    # Play and Record simultaneously
    # We use the same device for input and output to ensure sync if possible,
    # but usually ReSpeaker is split. Let's try using the found index for both
    # if it supports output, otherwise system default for output.
    
    try:
        # Note: We record 6 channels to see:
        # Ch 0: AEC processed
        # Ch 1-4: Raw Mics
        # Ch 5: Reference (Playback Loopback)
        recording = sd.playrec(
            playback_audio, 
            samplerate=SAMPLE_RATE, 
            channels=6, 
            dtype='float32',
            device=(dev_idx, dev_idx) # Try using ReSpeaker for both
        )
        sd.wait()
    except Exception as e:
        print(f"\n⚠️  Error using ReSpeaker for playback: {e}")
        print("   Falling back to default output device (System Speaker)...")
        # If this happens, ReSpeaker might not be set as default output, 
        # which is a common cause of AEC failure (No reference signal!)
        recording = sd.playrec(
            playback_audio, 
            samplerate=SAMPLE_RATE, 
            channels=6, 
            dtype='float32',
            device=(dev_idx, None) # Input: ReSpeaker, Output: Default
        )
        sd.wait()

    print("💾 Saving channels to ./aec_recordings/ ...")
    
    # Save full mix
    wav.write(f"{OUTPUT_DIR}/full_mix.wav", SAMPLE_RATE, recording)

    # Split and save individual channels
    channel_names = [
        "ch0_aec_processed",
        "ch1_raw_mic",
        "ch2_raw_mic",
        "ch3_raw_mic",
        "ch4_raw_mic",
        "ch5_reference_signal"
    ]

    rms_values = []

    for i in range(6):
        ch_data = recording[:, i]
        filename = f"{OUTPUT_DIR}/{channel_names[i]}.wav"
        wav.write(filename, SAMPLE_RATE, ch_data)
        
        rms = np.sqrt(np.mean(ch_data**2))
        rms_values.append(rms)
        
        # Visual bar
        bar = "█" * int(rms * 100)
        print(f"   {channel_names[i]:<20} | RMS: {rms:.4f} | {bar}")

    print("\n🧐 DIAGNOSIS GUIDE:")
    print("1. LISTEN to 'ch5_reference_signal.wav'")
    print("   - If SILENT/QUIET: Hardware AEC has NO reference. Fix audio routing.")
    print("   - If LOUD: Good. The ReSpeaker 'hears' the speaker.")
    print("2. LISTEN to 'ch0_aec_processed.wav' vs 'ch1_raw_mic.wav'")
    print("   - Ch0 should have drastically less voice/echo than Ch1.")
    print("   - If Ch0 sounds exactly like Ch1: AEC is OFF or NOT WORKING.")
    print("\n💡 To play the recordings:")
    print("   aplay -D plughw:3,0 aec_recordings/ch5_reference_signal.wav")
    print("   aplay -D plughw:3,0 aec_recordings/ch1_raw_mic.wav")
    print("   aplay -D plughw:3,0 aec_recordings/ch0_aec_processed.wav")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test ReSpeaker AEC with voice sample')
    parser.add_argument('--volume', type=int, default=100, help='Volume percentage (0-100)')
    args = parser.parse_args()
    
    volume = args.volume / 100.0  # Convert percentage to 0.0-1.0
    run_test(volume=volume)
