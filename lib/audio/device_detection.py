"""
Audio device detection for ALSA-only architecture.

Prioritizes ReSpeaker 4 Mic Array with hardware AEC.
Falls back to best available mic/speaker if ReSpeaker not found.
"""

import sounddevice as sd
from lib.config import Config


def get_audio_devices():
    """
    Detect and return audio device indices for input and output.
    
    Priority:
    1. ReSpeaker 4 Mic Array (hardware AEC available)
    2. Best available mic/speaker (no AEC - warning issued)
    
    Returns:
        tuple: (mic_device_index, speaker_device_index, has_hardware_aec)
               Returns (None, None, False) if using ALSA defaults
    """
    logger = Config.LOGGER
    
    devices = sd.query_devices()
    respeaker_idx = None
    
    # Search for ReSpeaker device
    for idx, dev in enumerate(devices):
        # Check if this is the ReSpeaker device
        if any(keyword in dev['name'].lower() 
               for keyword in ['respeaker', 'arrayuac10', 'uac1.0']):
            # Verify it has both input and output channels
            if dev['max_input_channels'] > 0 and dev['max_output_channels'] > 0:
                respeaker_idx = idx
                print(f"✓ ReSpeaker detected: {dev['name']} (index {idx})")
                print(f"  Hardware echo cancellation: Available")
                print(f"  Input channels: {dev['max_input_channels']}")
                print(f"  Output channels: {dev['max_output_channels']}")
                
                if logger:
                    logger.info(
                        "respeaker_detected",
                        extra={
                            "device_name": dev['name'],
                            "device_index": idx,
                            "input_channels": dev['max_input_channels'],
                            "output_channels": dev['max_output_channels'],
                            "hardware_aec": True,
                            "user_id": Config.USER_ID
                        }
                    )
                
                return respeaker_idx, respeaker_idx, True
    
    # ReSpeaker not found - fall back to system default
    print("⚠ ReSpeaker not detected - using system default audio")
    print("  Hardware echo cancellation: NOT available")
    print("  Note: Barge-in may not work properly without AEC")
    print("  System will auto-select audio devices via ALSA/PulseAudio")
    
    if logger:
        logger.warning(
            "fallback_audio_devices",
            extra={
                "respeaker_found": False,
                "hardware_aec": False,
                "using_system_default": True,
                "user_id": Config.USER_ID
            }
        )
    
    # Return None for both - let sounddevice/ALSA/PulseAudio choose the best device
    # This allows the audio system to route through the appropriate interface (e.g., IEC958 vs USB Audio)
    # that supports the required sample rate
    return None, None, False

