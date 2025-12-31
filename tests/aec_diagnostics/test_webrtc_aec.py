"""
Test PulseAudio WebRTC AEC on ReSpeaker recordings

This script tests PulseAudio's echo-cancel module by:
1. Loading pre-recorded audio from inspect_channels.py
2. Playing it back through PulseAudio while recording
3. Comparing the echo cancellation effectiveness

Prerequisites:
- PulseAudio with echo-cancel module enabled
- Run 'pulseaudio --start' before running this script

Usage:
    1. First run: python tests/aec_diagnostics/inspect_channels.py
    2. Configure PulseAudio: Edit /etc/pulse/default.pa
       Add: load-module module-echo-cancel aec_method=webrtc
    3. Restart: pulseaudio -k && pulseaudio --start
    4. Then run: python tests/aec_diagnostics/test_webrtc_aec.py
"""

import numpy as np
import scipy.io.wavfile as wav
import scipy.signal as signal
import sounddevice as sd
import os
import sys
import time
import subprocess

# PulseAudio's echo-cancel sink/source typically run at 32kHz mono
TARGET_SAMPLE_RATE = 32000
SAMPLE_RATE = 16000
OUTPUT_DIR = "aec_recordings"


def check_pulseaudio():
    """Check if PulseAudio is running and has echo-cancel module"""
    try:
        result = subprocess.run(['pactl', 'list', 'modules'], 
                              capture_output=True, text=True, timeout=5)
        
        if 'module-echo-cancel' in result.stdout:
            print("✓ PulseAudio echo-cancel module is loaded")
            
            # Extract echo-cancel sink name
            for line in result.stdout.split('\n'):
                if 'echo-cancel' in line.lower() and 'sink' in line.lower():
                    print(f"  {line.strip()}")
            
            return True
        else:
            print("❌ PulseAudio echo-cancel module NOT loaded!")
            print("\nTo enable echo cancellation:")
            print("1. Edit /etc/pulse/default.pa")
            print("2. Add this line:")
            print("   load-module module-echo-cancel aec_method=webrtc")
            print("3. Restart: pulseaudio -k && pulseaudio --start")
            return False
            
    except FileNotFoundError:
        print("❌ PulseAudio not found! Install it:")
        print("   sudo apt-get install pulseaudio")
        return False
    except Exception as e:
        print(f"⚠ Error checking PulseAudio: {e}")
        return False


def find_pulse_devices():
    """
    Find PulseAudio echo-cancel input and ReSpeaker (or echo-cancel) output.
    Returns (input_dev, output_dev) where each can be a device name (preferred) or index.
    """
    devices = sd.query_devices()
    input_dev = None
    output_dev = None
    
    # Prefer names; sounddevice supports PulseAudio device names directly.
    for idx, dev in enumerate(devices):
        name = dev['name'].lower()
        if input_dev is None and ('aec' in name and 'mic' in name):
            input_dev = dev['name']  # use name
            print(f"✓ Found echo-cancel mic: {dev['name']} (index {idx})")
        if output_dev is None and ('echo-cancel' in name and dev['max_output_channels'] > 0):
            output_dev = dev['name']  # echo-cancel sink
            print(f"✓ Found echo-cancel sink: {dev['name']} (index {idx})")
    
    # Fallbacks
    if input_dev is None:
        for idx, dev in enumerate(devices):
            if dev['max_input_channels'] > 0 and 'pulse' in dev['name'].lower():
                input_dev = dev['name']
                print(f"⚠ Using PulseAudio default input: {dev['name']} (index {idx})")
                break
    
    if output_dev is None:
        for idx, dev in enumerate(devices):
            if dev['max_output_channels'] > 0 and 'respeaker' in dev['name'].lower():
                output_dev = dev['name']
                print(f"⚠ Using ReSpeaker output: {dev['name']} (index {idx})")
                break
    
    # Final fallback: None (system default)
    if input_dev is None:
        print("⚠ aec_mic not found; will use system default input (may lack AEC)")
    if output_dev is None:
        print("⚠ echo-cancel/ReSpeaker output not found; will use system default output")
    
    return input_dev, output_dev


def test_pulseaudio_realtime():
    """
    Real-time test: Play reference audio while recording through PulseAudio AEC
    """
    print("\n" + "="*60)
    print("🎙️  REAL-TIME PULSEAUDIO AEC TEST")
    print("="*60)
    
    if not check_pulseaudio():
        return
    
    # Find echo-cancel input and output devices (PulseAudio)
    input_dev, output_dev = find_pulse_devices()
    
    # Load the test tone from previous recording
    ch5_file = os.path.join(OUTPUT_DIR, "ch5_reference_signal.wav")
    if not os.path.exists(ch5_file):
        print(f"❌ {ch5_file} not found!")
        print("   Run 'python tests/aec_diagnostics/inspect_channels.py' first.")
        return
    
    sr, playback_audio = wav.read(ch5_file)
    print(f"✓ Loaded playback audio: {len(playback_audio)} samples @ {sr}Hz")
    
    # Resample to PulseAudio echo-cancel rate (typically 32kHz mono)
    target_sr = TARGET_SAMPLE_RATE
    if sr != target_sr:
        num_samples = int(len(playback_audio) * target_sr / sr)
        playback_audio = signal.resample(playback_audio, num_samples).astype(np.float32)
        sr = target_sr
        print(f"↪ Resampled playback to {sr}Hz for PulseAudio AEC")
    
    # Normalize to float32
    if playback_audio.dtype == np.int16:
        playback_audio = playback_audio.astype(np.float32) / 32768.0
    
    # Record using PulseAudio while playing
    print("\n🎙️  Recording with PulseAudio echo-cancel...")
    print("   (Playing test tone through ReSpeaker speaker)")
    
    # Try to use PulseAudio device name directly (more reliable)
    try:
        # Prefer device names (PulseAudio): aec_mic for input, echo-cancel/ReSpeaker sink for output
        recording = sd.playrec(
            playback_audio,
            samplerate=sr,
            channels=1,
            dtype='float32',
            device=(input_dev, output_dev)
        )
        sd.wait()
        print(f"✓ Recorded using devices: input={input_dev}, output={output_dev}")
    except Exception as e:
        print(f"⚠ Recording with named devices failed: {e}")
        print("   Falling back to system defaults (may lack AEC)...")
        recording = sd.playrec(
            playback_audio,
            samplerate=sr,
            channels=1,
            dtype='float32'
        )
        sd.wait()
        print("⚠ Warning: Recording may not have echo cancellation!")
    
    # Save the PulseAudio AEC output
    output_file = os.path.join(OUTPUT_DIR, "pulseaudio_aec_output.wav")
    recording_int16 = (recording * 32767.0).astype(np.int16)
    wav.write(output_file, sr, recording_int16)
    print(f"✓ Saved PulseAudio AEC output: {output_file}")
    
    # Compare with hardware AEC
    ch0_file = os.path.join(OUTPUT_DIR, "ch0_aec_processed.wav")
    ch1_file = os.path.join(OUTPUT_DIR, "ch1_raw_mic.wav")
    
    print("\n📊 RMS Comparison:")
    
    rms_pulse = np.sqrt(np.mean(recording.flatten() ** 2))
    print(f"   PulseAudio AEC:          {rms_pulse:.4f}")
    
    if os.path.exists(ch1_file):
        _, ch1_raw = wav.read(ch1_file)
        ch1_float = ch1_raw.astype(np.float32) / 32768.0
        rms_ch1 = np.sqrt(np.mean(ch1_float ** 2))
        print(f"   Ch1 (Raw Mic):           {rms_ch1:.4f}")
        improvement_pulse = (1 - rms_pulse / rms_ch1) * 100
        print(f"   PulseAudio AEC reduction: {improvement_pulse:.1f}%")
    
    if os.path.exists(ch0_file):
        _, ch0_raw = wav.read(ch0_file)
        ch0_float = ch0_raw.astype(np.float32) / 32768.0
        rms_ch0 = np.sqrt(np.mean(ch0_float ** 2))
        print(f"   Ch0 (Hardware AEC):      {rms_ch0:.4f}")
        if os.path.exists(ch1_file):
            improvement_hw = (1 - rms_ch0 / rms_ch1) * 100
            print(f"   Hardware AEC reduction:   {improvement_hw:.1f}%")
    
    print("\n🧐 LISTEN TEST:")
    print("1. Listen to 'ch1_raw_mic.wav' - raw microphone (no AEC)")
    print("2. Listen to 'ch0_aec_processed.wav' - ReSpeaker hardware AEC")
    print("3. Listen to 'pulseaudio_aec_output.wav' - PulseAudio software AEC")
    print("4. Compare: Which has better echo cancellation?")
    print("\n💡 TIP: If PulseAudio AEC isn't working well:")
    print("   - Check: pactl list modules | grep echo-cancel")
    print("   - Try: pactl unload-module module-echo-cancel")
    print("   - Then: pactl load-module module-echo-cancel aec_method=webrtc")


def show_pulseaudio_info():
    """Show PulseAudio configuration info"""
    print("\n" + "="*60)
    print("📋 PULSEAUDIO CONFIGURATION INFO")
    print("="*60)
    
    try:
        # Show loaded modules
        result = subprocess.run(['pactl', 'list', 'modules', 'short'],
                              capture_output=True, text=True, timeout=5)
        print("\n🔌 Loaded Modules (filtered):")
        for line in result.stdout.split('\n'):
            if 'echo' in line.lower() or 'cancel' in line.lower() or 'null' in line.lower():
                print(f"   {line}")
        
        # Show sinks
        result = subprocess.run(['pactl', 'list', 'sinks', 'short'],
                              capture_output=True, text=True, timeout=5)
        print("\n🔊 Audio Sinks (outputs):")
        for line in result.stdout.split('\n'):
            if line.strip():
                print(f"   {line}")
        
        # Show sources
        result = subprocess.run(['pactl', 'list', 'sources', 'short'],
                              capture_output=True, text=True, timeout=5)
        print("\n🎤 Audio Sources (inputs):")
        for line in result.stdout.split('\n'):
            if line.strip():
                print(f"   {line}")
                
    except Exception as e:
        print(f"⚠ Could not get PulseAudio info: {e}")


def run_test():
    """Main test function"""
    
    if not os.path.exists(OUTPUT_DIR):
        print(f"❌ Directory {OUTPUT_DIR} not found!")
        print("   Run 'python tests/aec_diagnostics/inspect_channels.py' first to create recordings.")
        return
    
    # Show PulseAudio configuration
    show_pulseaudio_info()
    
    # Run real-time test
    test_pulseaudio_realtime()


if __name__ == "__main__":
    run_test()
