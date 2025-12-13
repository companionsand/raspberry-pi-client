"""
Voice Feedback Module

Provides voice guidance during device startup, connectivity checks, and WiFi setup.
Uses pre-recorded audio files for consistent, reliable feedback.
"""

import os
import logging
import wave
from typing import Optional
import numpy as np
import sounddevice as sd
from pathlib import Path

logger = logging.getLogger(__name__)


class VoiceFeedback:
    """Plays pre-recorded voice messages through the speaker device"""
    
    def __init__(self, speaker_device_index: Optional[int] = None, voice_messages_dir: Optional[str] = None):
        """
        Initialize voice feedback system.
        
        Args:
            speaker_device_index: Speaker device index (None for system default)
            voice_messages_dir: Directory containing voice message files (defaults to ./voice_messages)
        """
        self.speaker_device_index = speaker_device_index
        
        # Determine voice messages directory
        if voice_messages_dir is None:
            # Default to voice_messages directory in the same folder as this module
            module_dir = Path(__file__).parent
            self.voice_messages_dir = module_dir / "voice_messages"
        else:
            self.voice_messages_dir = Path(voice_messages_dir)
        
        # Cache for loaded audio files (message_name -> (audio_data, sample_rate))
        self._audio_cache = {}
        
        logger.info(f"[VoiceFeedback] Initialized with voice messages directory: {self.voice_messages_dir}")
    
    def play(self, message_name: str) -> bool:
        """
        Play a pre-recorded voice message.
        
        Args:
            message_name: Name of the message (without .wav extension)
            
        Returns:
            True if playback succeeded, False otherwise
        """
        try:
            # Load audio file (from cache or disk)
            audio_data, sample_rate = self._get_audio(message_name)
            
            if audio_data is None:
                logger.warning(f"[VoiceFeedback] No audio data available for '{message_name}'")
                return False
            
            # Play audio (blocking)
            logger.info(f"[VoiceFeedback] Playing '{message_name}'")
            sd.play(audio_data, samplerate=sample_rate, device=self.speaker_device_index, blocking=True)
            logger.info(f"[VoiceFeedback] Finished playing '{message_name}'")
            
            return True
            
        except Exception as e:
            logger.error(f"[VoiceFeedback] Error playing '{message_name}': {type(e).__name__}: {e}")
            return False
    
    def _get_audio(self, message_name: str) -> tuple[Optional[np.ndarray], int]:
        """
        Get audio data for a message (from cache or load from disk).
        
        Args:
            message_name: Name of the message (without .wav extension)
            
        Returns:
            Tuple of (audio_data, sample_rate) or (None, 0) if not available
        """
        # Check cache first
        if message_name in self._audio_cache:
            return self._audio_cache[message_name]
        
        # Load from disk
        audio_file = self.voice_messages_dir / f"{message_name}.wav"
        
        if not audio_file.exists():
            logger.warning(f"[VoiceFeedback] Voice message file not found: {audio_file}")
            # Cache the failure to avoid repeated warnings
            self._audio_cache[message_name] = (None, 0)
            return (None, 0)
        
        try:
            audio_data, sample_rate = self._load_audio_file(audio_file)
            
            # Cache the loaded audio
            self._audio_cache[message_name] = (audio_data, sample_rate)
            
            logger.info(f"[VoiceFeedback] Loaded voice message: {audio_file} ({sample_rate} Hz, {len(audio_data)} samples)")
            
            return (audio_data, sample_rate)
            
        except Exception as e:
            logger.error(f"[VoiceFeedback] Failed to load {audio_file}: {type(e).__name__}: {e}")
            # Cache the failure
            self._audio_cache[message_name] = (None, 0)
            return (None, 0)
    
    def _load_audio_file(self, filepath: Path) -> tuple[np.ndarray, int]:
        """
        Load a WAV file and return audio data as numpy array.
        
        Args:
            filepath: Path to WAV file
            
        Returns:
            Tuple of (audio_data, sample_rate)
        """
        with wave.open(str(filepath), 'rb') as wf:
            # Get audio parameters
            n_channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            sample_rate = wf.getframerate()
            n_frames = wf.getnframes()
            
            # Read audio data
            audio_bytes = wf.readframes(n_frames)
            
            # Convert to numpy array
            if sample_width == 1:
                # 8-bit unsigned
                dtype = np.uint8
                audio_data = np.frombuffer(audio_bytes, dtype=dtype)
                # Convert to float32 in range [-1, 1]
                audio_data = (audio_data.astype(np.float32) - 128) / 128.0
            elif sample_width == 2:
                # 16-bit signed
                dtype = np.int16
                audio_data = np.frombuffer(audio_bytes, dtype=dtype)
                # Convert to float32 in range [-1, 1]
                audio_data = audio_data.astype(np.float32) / 32768.0
            elif sample_width == 4:
                # 32-bit signed
                dtype = np.int32
                audio_data = np.frombuffer(audio_bytes, dtype=dtype)
                # Convert to float32 in range [-1, 1]
                audio_data = audio_data.astype(np.float32) / 2147483648.0
            else:
                raise ValueError(f"Unsupported sample width: {sample_width}")
            
            # If stereo, convert to mono by averaging channels
            if n_channels == 2:
                audio_data = audio_data.reshape(-1, 2).mean(axis=1)
            elif n_channels > 2:
                # Multi-channel: take first channel
                audio_data = audio_data.reshape(-1, n_channels)[:, 0]
            
            return (audio_data, sample_rate)
    
    def preload_all(self):
        """
        Preload all voice message files in the voice messages directory.
        
        This can be called during initialization to front-load any I/O errors.
        """
        if not self.voice_messages_dir.exists():
            logger.warning(f"[VoiceFeedback] Voice messages directory not found: {self.voice_messages_dir}")
            return
        
        logger.info(f"[VoiceFeedback] Preloading voice messages from {self.voice_messages_dir}")
        
        for audio_file in self.voice_messages_dir.glob("*.wav"):
            message_name = audio_file.stem
            self._get_audio(message_name)

