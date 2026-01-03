import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
from scipy import signal
import time
import os
import argparse
import subprocess
import tempfile

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


def _check_gstreamer_available():
    """Check if GStreamer with webrtcdsp plugin is available."""
    import subprocess
    try:
        # Check if gst-inspect-1.0 exists and webrtcdsp plugin is available
        result = subprocess.run(
            ['gst-inspect-1.0', 'webrtcdsp'],
            capture_output=True,
            text=True,
            timeout=5
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False

def _import_aec_library():
    """Try to import an AEC (Acoustic Echo Cancellation) library.

    Tries in order:
      1. GStreamer with webrtcdsp (ARM-compatible, best option for Raspberry Pi)
      2. noisereduce (ARM-compatible, simple fallback)
      3. webrtc-audio-processing (x86 only, not ARM-compatible)

    Returns a tuple: (library_type, factory_callable)
    """
    # Try GStreamer first (ARM-compatible, full WebRTC AEC)
    if _check_gstreamer_available():
        print("ℹ️  Using GStreamer webrtcdsp for AEC (ARM-compatible, full WebRTC AEC)")
        return ('gstreamer', lambda sample_rate, frame_size: None)
    else:
        print("⚠️  GStreamer webrtcdsp not found. Install with:")
        print("    sudo apt-get install gstreamer1.0-plugins-bad")
    
    # Try noisereduce as fallback (ARM-compatible, available on PyPI)
    try:
        import noisereduce as nr  # type: ignore
        print("ℹ️  Using noisereduce for noise reduction (ARM-compatible)")
        print("⚠️  Note: This is noise reduction, not full AEC. Install GStreamer for better results.")
        return ('noisereduce', lambda sample_rate, frame_size: nr)
    except ImportError:
        pass

    # Fallback to webrtc-audio-processing (x86 only)
    try:
        import webrtc_audio_processing as wap  # type: ignore
        print("ℹ️  Using webrtc-audio-processing for AEC")
        
        # Common patterns across bindings
        if hasattr(wap, "AudioProcessing"):
            return ('webrtc', lambda sample_rate, frame_size: wap.AudioProcessing(sample_rate=sample_rate, num_channels=1))
        if hasattr(wap, "AudioProcessor"):
            return ('webrtc', lambda sample_rate, frame_size: wap.AudioProcessor(sample_rate=sample_rate, num_channels=1))
        if hasattr(wap, "create_apm"):
            return ('webrtc', lambda sample_rate, frame_size: wap.create_apm(sample_rate=sample_rate, num_channels=1))
        
        # As a last resort, expose module itself
        return ('webrtc', lambda sample_rate, frame_size: wap)
    except ImportError:
        pass

    raise ImportError("No AEC library found. Install GStreamer: sudo apt-get install gstreamer1.0-plugins-bad")

def _run_gstreamer_aec(near: np.ndarray, ref: np.ndarray, sample_rate: int) -> np.ndarray:
    """Run GStreamer WebRTC DSP for echo cancellation.
    
    Args:
        near: int16 array of near-end (mic) signal
        ref: int16 array of far-end (reference/echo) signal
        sample_rate: sample rate in Hz
    
    Returns:
        float32 array of AEC-processed output
    """
    
    # Convert int16 to float32 for return
    def i16_to_f32(x):
        return (x.astype(np.float32) / 32767.0)
    
    # Create temporary files
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as near_file, \
         tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as ref_file, \
         tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as out_file:
        
        near_path = near_file.name
        ref_path = ref_file.name
        out_path = out_file.name
    
    try:
        # Save near and reference to WAV files
        wav.write(near_path, sample_rate, near)
        wav.write(ref_path, sample_rate, ref)
        
        # GStreamer pipeline for WebRTC AEC
        # The webrtcechoprobe captures the reference (far-end) signal
        # The webrtcdsp performs AEC, noise suppression, and AGC
        pipeline = [
            'gst-launch-1.0', '-q',
            'webrtcechoprobe', 'name=probe', '!', 'queue', '!', 'fakesink',
            'filesrc', f'location={ref_path}', '!', 'wavparse', '!', 'audioconvert', '!', 'probe.sink',
            'filesrc', f'location={near_path}', '!', 'wavparse', '!', 'audioconvert', '!',
            'webrtcdsp',
            'target-level-dbfs=0',
            'compression-gain-db=9',
            'startup-min-volume=255',
            'echo-cancel=true',
            'noise-suppression=true',
            'gain-control=true',
            'extended-filter=true',
            'delay-agnostic=true',
            'high-pass-filter=true',
            'limiter=true',
            'probe=probe',
            '!', 'audioconvert', '!', 'wavenc', '!',
            'filesink', f'location={out_path}'
        ]
        
        print(f"   Running GStreamer AEC pipeline...")
        result = subprocess.run(
            pipeline,
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode != 0:
            raise RuntimeError(f"GStreamer failed: {result.stderr}")
        
        # Read processed output
        sr_out, audio_out = wav.read(out_path)
        if sr_out != sample_rate:
            raise RuntimeError(f"Sample rate mismatch: expected {sample_rate}, got {sr_out}")
        
        # Convert to float32
        if audio_out.dtype == np.int16:
            return i16_to_f32(audio_out)
        elif audio_out.dtype == np.float32:
            return audio_out
        else:
            return audio_out.astype(np.float32)
    
    finally:
        # Clean up temp files
        for path in [near_path, ref_path, out_path]:
            try:
                os.unlink(path)
            except:
                pass

def run_webrtc_aec(recording: np.ndarray, sample_rate: int, near_ch: int = 1, ref_ch: int = 5,
                  frame_ms: int = 10) -> np.ndarray:
    """Run software audio processing (AEC or noise reduction) offline on a recorded buffer.

    recording: float32 ndarray [N, 6] in [-1, 1]
    near_ch: raw mic channel index (1-4)
    ref_ch: reference channel index (typically 5)

    Returns float32 mono ndarray [N] in [-1, 1] (processed output)
    
    Processing options (in priority order):
    1. GStreamer webrtcdsp: Full WebRTC AEC (echo cancel, noise suppression, AGC)
    2. noisereduce: Noise reduction only (not full AEC)
    
    For production use, hardware AEC on channel 0 is still recommended.
    """
    if recording.ndim != 2:
        raise ValueError("recording must be 2D (samples, channels)")
    if sample_rate not in (8000, 16000, 32000, 48000):
        raise ValueError("AEC typically supports 8/16/32/48 kHz sample rates")

    # Prepare frames
    frame_len = int(sample_rate * frame_ms / 1000)
    if frame_len <= 0:
        raise ValueError("invalid frame length")
    n = recording.shape[0]
    n_frames = n // frame_len
    if n_frames == 0:
        raise ValueError("recording too short for frame processing")

    # Convert float [-1,1] to int16
    def f32_to_i16(x):
        x = np.clip(x, -1.0, 1.0)
        return (x * 32767.0).astype(np.int16)

    def i16_to_f32(x):
        return (x.astype(np.float32) / 32767.0)

    near = f32_to_i16(recording[:n_frames * frame_len, near_ch])
    ref = f32_to_i16(recording[:n_frames * frame_len, ref_ch])

    # Get AEC library and create processor
    lib_type, factory = _import_aec_library()
    aec = factory(sample_rate, frame_len)

    out_frames = []

    if lib_type == 'gstreamer':
        # GStreamer WebRTC DSP processing
        # This provides true WebRTC AEC using the webrtcdsp plugin
        return _run_gstreamer_aec(near, ref, sample_rate)

    elif lib_type == 'noisereduce':
        # NoiseReduce API: reduce_noise(audio_clip, noise_clip, sr=sample_rate)
        # We use the reference signal as the "noise profile"
        # This is not true AEC but can help reduce echo-like artifacts
        near_f32 = i16_to_f32(near)
        ref_f32 = i16_to_f32(ref)
        
        # Use first 0.5 seconds of reference as noise profile
        noise_profile_len = min(int(sample_rate * 0.5), len(ref_f32))
        
        try:
            # Newer noisereduce API
            reduced = aec.reduce_noise(
                y=near_f32,
                sr=sample_rate,
                y_noise=ref_f32[:noise_profile_len],
                stationary=False,
                prop_decrease=0.8
            )
        except TypeError:
            # Older noisereduce API
            reduced = aec.reduce_noise(
                audio_clip=near_f32,
                noise_clip=ref_f32[:noise_profile_len],
                verbose=False
            )
        
        return reduced

    elif lib_type == 'webrtc':
        # WebRTC API: process_reverse_stream(far_end), process_stream(near_end)
        stream_delay_ms = 0
        
        for i in range(n_frames):
            a = slice(i * frame_len, (i + 1) * frame_len)
            ref_frame = ref[a]
            near_frame = near[a]

            # Far-end first
            if hasattr(aec, "process_reverse_stream"):
                aec.process_reverse_stream(ref_frame)
            elif hasattr(aec, "ProcessReverseStream"):
                aec.ProcessReverseStream(ref_frame)
            else:
                raise RuntimeError("webrtc binding has no process_reverse_stream")

            # Then near-end
            if hasattr(aec, "process_stream"):
                try:
                    out_i16 = aec.process_stream(near_frame, stream_delay_ms=stream_delay_ms)
                except TypeError:
                    out_i16 = aec.process_stream(near_frame)
            elif hasattr(aec, "ProcessStream"):
                try:
                    out_i16 = aec.ProcessStream(near_frame, stream_delay_ms=stream_delay_ms)
                except TypeError:
                    out_i16 = aec.ProcessStream(near_frame)
            else:
                raise RuntimeError("webrtc binding has no process_stream")

            # Handle various return types
            if out_i16 is None:
                out_i16 = near_frame
            if isinstance(out_i16, (bytes, bytearray)):
                out_i16 = np.frombuffer(out_i16, dtype=np.int16)
            out_i16 = np.asarray(out_i16, dtype=np.int16)
            out_frames.append(i16_to_f32(out_i16))

    else:
        raise RuntimeError(f"Unknown library type: {lib_type}")

    out = np.concatenate(out_frames, axis=0)
    # Pad to original length
    if out.shape[0] < recording.shape[0]:
        out = np.pad(out, (0, recording.shape[0] - out.shape[0]))
    return out

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


    # -----------------------------
    # Optional: Software AEC (using GStreamer WebRTC or noisereduce) for comparison
    # Uses: near-end mic (default ch1) + far-end reference (ch5)
    # Produces: ch_software_processed.wav
    # GStreamer provides full WebRTC AEC (echo cancel + noise suppression + AGC)
    # Hardware AEC (ch0) is still preferred for production use.
    # -----------------------------
    try:
        software_processed_out = run_webrtc_aec(recording, SAMPLE_RATE, near_ch=1, ref_ch=5)
        software_processed_file = f"{OUTPUT_DIR}/ch_software_processed.wav"
        wav.write(software_processed_file, SAMPLE_RATE, software_processed_out.astype(np.float32))
        rms = np.sqrt(np.mean(software_processed_out**2))
        print(f"   {'ch_software_processed':<20} | RMS: {rms:.4f}")
        print(f"✅ Wrote software-processed output: {software_processed_file}")
    except Exception as e:
        print(f"\n⚠️  Software AEC skipped: {e}")
        print("   To enable WebRTC AEC, install GStreamer:")
        print("     - sudo apt-get install gstreamer1.0-plugins-bad")
        print("   Or for basic noise reduction:")
        print("     - pip install noisereduce")
        print("   For best echo cancellation, use the hardware AEC on ch0.")
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
