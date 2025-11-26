"""Wake word detection using Porcupine with Silero VAD gating"""

import os
import pvporcupine
import sounddevice as sd
import numpy as np
import onnxruntime as ort
from lib.config import Config


class WakeWordDetector:
    """Porcupine-based wake word detection with telemetry"""
    
    def __init__(self, mic_device_index=None):
        """
        Initialize wake word detector.
        
        Args:
            mic_device_index: Audio device index for microphone (None for default)
        """
        self.porcupine = None
        self.audio_stream = None
        self.detected = False
        self.running = False
        self.mic_device_index = mic_device_index
        self.logger = Config.LOGGER
        
        # -------------------------------------------------------------------------
        # Silero VAD Gate - Reduces false positives by only processing speech
        # -------------------------------------------------------------------------
        self._vad_session = None
        self._vad_enabled = False
        
        # Locate model: project_root/models/silero_vad.onnx
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        model_path = os.path.join(project_root, 'models', 'silero_vad.onnx')
        
        if os.path.exists(model_path):
            try:
                self._vad_session = ort.InferenceSession(model_path)
                self._vad_enabled = True
                print("✓ VAD gate initialized (reduces false wake word triggers)")
            except Exception as e:
                print(f"⚠ VAD gate init failed: {e} - falling back to ungated detection")
        else:
            print(f"⚠ VAD model not found at {model_path} - falling back to ungated detection")
        
        # VAD state: LSTM hidden states preserved across frames for context
        self._vad_h = np.zeros((2, 1, 64), dtype=np.float32)
        self._vad_c = np.zeros((2, 1, 64), dtype=np.float32)
        self._vad_threshold = 0.5  # Speech probability threshold
        
    def start(self):
        """Initialize Porcupine and start listening"""
        if self.running:
            return

        # Reset detection flag each time we enter listening mode
        self.detected = False

        print(f"\n🎤 Initializing wake word detection...")
        print(f"   Wake word: '{Config.WAKE_WORD}'")
        
        try:
            # Initialize Porcupine with built-in keyword
            self.porcupine = pvporcupine.create(
                access_key=Config.PICOVOICE_ACCESS_KEY,
                keywords=[Config.WAKE_WORD],
                sensitivities=[0.7]  # 0.0 (least sensitive) to 1.0 (most sensitive)
            )
            
            # Query devices for logging and error messages
            devices = sd.query_devices()
            
            # Log device being used
            if self.mic_device_index is not None:
                dev = devices[self.mic_device_index]
                print(f"   Using microphone: {dev['name']} (index {self.mic_device_index})")
            else:
                print(f"   Using default microphone")
            
            print(f"   Required sample rate: {self.porcupine.sample_rate} Hz")
            
            # Start audio stream for wake word detection
            self.audio_stream = sd.InputStream(
                device=self.mic_device_index,
                channels=Config.CHANNELS,
                samplerate=self.porcupine.sample_rate,
                blocksize=self.porcupine.frame_length,
                dtype='int16',
                callback=self._audio_callback
            )
            
            self.audio_stream.start()
            self.running = True
            print(f"✓ Listening for wake word...")
            
            if self.logger:
                self.logger.info(
                    "listening_for_wake_word",
                    extra={
                        "wake_word": Config.WAKE_WORD,
                        "user_id": Config.USER_ID,
                        "device_id": Config.DEVICE_ID
                    }
                )
        except sd.PortAudioError as e:
            # Handle audio device errors with helpful message
            error_msg = str(e)
            if "Invalid sample rate" in error_msg or "paInvalidSampleRate" in error_msg:
                # Get device name for error message
                try:
                    device_name = devices[self.mic_device_index]['name'] if self.mic_device_index is not None else 'default'
                except:
                    device_name = f"index {self.mic_device_index}" if self.mic_device_index is not None else "default"
                
                print(f"\n✗ ERROR: Microphone doesn't support required sample rate")
                print(f"   Required: {self.porcupine.sample_rate} Hz (Porcupine requirement)")
                print(f"   Device: {device_name}")
                print(f"\n   Solutions:")
                print(f"   1. Use a different USB microphone that supports 16kHz")
                print(f"   2. Use ReSpeaker 4 Mic Array (recommended - has hardware AEC)")
                print(f"   3. Check device supported rates with: python -m sounddevice")
                
                if self.logger:
                    self.logger.error(
                        "wake_word_unsupported_sample_rate",
                        extra={
                            "error": error_msg,
                            "required_rate": self.porcupine.sample_rate,
                            "device_index": self.mic_device_index,
                            "user_id": Config.USER_ID
                        }
                    )
            else:
                print(f"\n✗ ERROR: Audio device error: {error_msg}")
                if self.logger:
                    self.logger.error(
                        "wake_word_audio_device_error",
                        extra={
                            "error": error_msg,
                            "user_id": Config.USER_ID
                        },
                        exc_info=True
                    )
            
            self.stop()
            raise
        except Exception as e:
            # Ensure partially-initialized resources are cleaned up
            if self.logger:
                self.logger.error(
                    "wake_word_detection_start_failed",
                    extra={
                        "error": str(e),
                        "user_id": Config.USER_ID
                    },
                    exc_info=True
                )
            self.stop()
            raise
        
    def _audio_callback(self, indata, frames, time_info, status):
        """Process audio frames for wake word detection with VAD gating"""
        if status:
            print(f"⚠ Audio status: {status}")
        
        # Convert to the format Porcupine expects (int16)
        audio_frame = indata[:, 0].astype(np.int16)
        
        # -------------------------------------------------------------------------
        # VAD Gate: Only process with Porcupine if speech is detected
        # -------------------------------------------------------------------------
        if self._vad_enabled and self._vad_session:
            # Convert to float32 normalized [-1, 1] for Silero VAD
            audio_float = audio_frame.astype(np.float32) / 32768.0
            
            # Run VAD inference
            try:
                ort_inputs = {
                    'input': audio_float.reshape(1, -1),
                    'sr': np.array([16000], dtype=np.int64),
                    'h': self._vad_h,
                    'c': self._vad_c
                }
                output, self._vad_h, self._vad_c = self._vad_session.run(None, ort_inputs)
                speech_prob = output[0][0]
                
                # Gate: Skip Porcupine if no speech detected
                if speech_prob < self._vad_threshold:
                    return  # Noise/silence - don't waste CPU on Porcupine
            except Exception:
                pass  # VAD failed - fall through to Porcupine anyway
        
        # -------------------------------------------------------------------------
        # Porcupine: Check for wake word (only if VAD passed or VAD disabled)
        # -------------------------------------------------------------------------
        keyword_index = self.porcupine.process(audio_frame)
        
        if keyword_index >= 0:
            print(f"\n🎯 Wake word '{Config.WAKE_WORD}' detected!")
            self.detected = True
            
            if self.logger:
                self.logger.info(
                    "wake_word_detected",
                    extra={
                        "wake_word": Config.WAKE_WORD,
                        "user_id": Config.USER_ID,
                        "device_id": Config.DEVICE_ID
                    }
                )
    
    def stop(self):
        """Stop wake word detection"""
        if self.audio_stream:
            self.audio_stream.stop()
            self.audio_stream.close()
            self.audio_stream = None
        
        if self.porcupine:
            self.porcupine.delete()
            self.porcupine = None
        
        self.running = False
    
    def cleanup(self):
        """Clean up resources"""
        self.stop()

