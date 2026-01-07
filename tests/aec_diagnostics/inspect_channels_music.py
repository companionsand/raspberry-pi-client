import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
from scipy import signal
import time
import os
import subprocess
import argparse

# Configuration
SAMPLE_RATE = 16000
OUTPUT_DIR = "aec_recordings_music"
DURATION = 5.0  # seconds
MUSIC_CACHE_FILE = os.path.join(OUTPUT_DIR, "test_music.wav")

# SoundCloud URL
SOUNDCLOUD_URL = "https://soundcloud.com/soundhelix/the-stationary-ark"

def download_music_sample():
    """Download music sample from SoundCloud using yt-dlp"""
    if os.path.exists(MUSIC_CACHE_FILE):
        print(f"✓ Using cached music file: {MUSIC_CACHE_FILE}")
        return MUSIC_CACHE_FILE
    
    # Check if yt-dlp is available
    if subprocess.run(['which', 'yt-dlp'], capture_output=True).returncode != 0:
        print("⚠️  yt-dlp not found. Install with: pip install yt-dlp")
        print("   Or: sudo apt-get install yt-dlp")
        return None
    
    print("📥 Downloading music from SoundCloud...")
    print(f"   URL: {SOUNDCLOUD_URL}")
    
    # Ensure output directory exists
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    try:
        # Download and convert to WAV, limit to 5 seconds
        # yt-dlp can extract audio from SoundCloud
        subprocess.run([
            'yt-dlp',
            '-x',  # Extract audio only
            '--audio-format', 'wav',
            '--audio-quality', '0',  # Best quality
            '-o', MUSIC_CACHE_FILE.replace('.wav', '.%(ext)s'),  # Output template
            '--postprocessor-args', f'ffmpeg:-ar {SAMPLE_RATE} -ac 1 -t {DURATION}',  # Convert to mono, 16kHz, 5s
            SOUNDCLOUD_URL
        ], check=True, capture_output=True)
        
        # yt-dlp might save with different extension, find the file
        for ext in ['wav', 'm4a', 'mp3', 'ogg']:
            temp_file = MUSIC_CACHE_FILE.replace('.wav', f'.{ext}')
            if os.path.exists(temp_file):
                if ext != 'wav':
                    # Convert to WAV if needed
                    subprocess.run([
                        'ffmpeg', '-i', temp_file,
                        '-ar', str(SAMPLE_RATE),
                        '-ac', '1',  # Mono
                        '-t', str(DURATION),
                        '-y',
                        MUSIC_CACHE_FILE
                    ], check=True, capture_output=True)
                    os.remove(temp_file)
                else:
                    os.rename(temp_file, MUSIC_CACHE_FILE)
                break
        
        if os.path.exists(MUSIC_CACHE_FILE):
            print(f"✓ Downloaded: {MUSIC_CACHE_FILE}")
            return MUSIC_CACHE_FILE
        else:
            print("⚠️  Download completed but file not found")
            return None
            
    except subprocess.CalledProcessError as e:
        print(f"⚠️  Download failed: {e}")
        if e.stderr:
            print(f"   Error: {e.stderr.decode()}")
        return None
    except Exception as e:
        print(f"⚠️  Error: {e}")
        return None

def load_music_sample(volume=1.0):
    """Load music sample from file or generate fallback"""
    music_file = download_music_sample()
    
    if music_file and os.path.exists(music_file):
        try:
            sr, audio = wav.read(music_file)
            
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
            
            # Limit to DURATION seconds
            max_samples = int(DURATION * SAMPLE_RATE)
            if len(audio_float) > max_samples:
                audio_float = audio_float[:max_samples]
            
            # Resample if not 16kHz
            if sr != SAMPLE_RATE:
                num_samples = int(len(audio_float) * SAMPLE_RATE / sr)
                audio_float = signal.resample(audio_float, num_samples).astype(np.float32)
            
            # Apply volume
            audio_float = audio_float * volume
            
            print(f"✓ Loaded music sample: {len(audio_float)/SAMPLE_RATE:.2f}s @ {SAMPLE_RATE}Hz")
            print(f"  Volume: {int(volume * 100)}%")
            return audio_float
            
        except Exception as e:
            print(f"⚠️  Error loading music file: {e}")
            print("   Falling back to generated music")
    else:
        print("⚠️  No music file available")
        print("   Falling back to generated music")
    
    # Fallback to generated music
    return generate_music_sample(DURATION, SAMPLE_RATE, volume=volume)

def generate_music_sample(duration, sr, volume=1.0):
    """Generates a 5-second musical melody (C major scale with chords)"""
    t = np.linspace(0, duration, int(duration * sr), False)
    
    # Musical notes (C major scale)
    notes = [
        261.63,  # C4
        293.66,  # D4
        329.63,  # E4
        349.23,  # F4
        392.00,  # G4
        440.00,  # A4
        493.88,  # B4
        523.25,  # C5
    ]
    
    # Create a melody by playing notes sequentially
    melody = np.zeros_like(t)
    note_duration = duration / len(notes)
    
    for i, freq in enumerate(notes):
        start_idx = int(i * note_duration * sr)
        end_idx = int((i + 1) * note_duration * sr)
        note_t = t[start_idx:end_idx]
        
        # Add the fundamental frequency
        note = np.sin(2 * np.pi * freq * note_t)
        
        # Add a harmonic for richness (like a piano)
        note += 0.3 * np.sin(2 * np.pi * freq * 2 * note_t)  # Octave
        note += 0.2 * np.sin(2 * np.pi * freq * 3 * note_t)  # Fifth
        
        # Simple envelope for each note
        envelope = np.exp(-3 * note_t / note_duration)  # Decay
        melody[start_idx:end_idx] = note * envelope
    
    # Overall amplitude modulation (like music dynamics)
    overall_envelope = 0.5 * (1 + np.sin(2 * np.pi * 0.5 * t))
    
    return (melody * overall_envelope * 0.3 * volume).astype(np.float32)


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
    print("📊 Loading music sample...")
    playback_audio = load_music_sample(volume=volume)  # Changed from generate_music_sample
    duration = len(playback_audio) / SAMPLE_RATE
    
    print(f"\n🎙  RECORDING {duration:.1f}s...")
    print("   Signal: Real music sample from SoundCloud")
    print("   ACTION: Please remain quiet for this test.")
    
    # Play and Record simultaneously
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
        recording = sd.playrec(
            playback_audio, 
            samplerate=SAMPLE_RATE, 
            channels=6, 
            dtype='float32',
            device=(dev_idx, None) # Input: ReSpeaker, Output: Default
        )
        sd.wait()

    print("💾 Saving channels to ./aec_recordings_music/ ...")
    
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
    print("   - Ch0 should have drastically less music/echo than Ch1.")
    print("   - If Ch0 sounds exactly like Ch1: AEC is OFF or NOT WORKING.")
    print("\n💡 To play the recordings:")
    print("   aplay -D plughw:3,0 aec_recordings_music/ch5_reference_signal.wav")
    print("   aplay -D plughw:3,0 aec_recordings_music/ch1_raw_mic.wav")
    print("   aplay -D plughw:3,0 aec_recordings_music/ch0_aec_processed.wav")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test ReSpeaker AEC with music sample')
    parser.add_argument('--volume', type=int, default=100, help='Volume percentage (0-100)')
    args = parser.parse_args()
    
    volume = args.volume / 100.0  # Convert percentage to 0.0-1.0
    run_test(volume=volume)
