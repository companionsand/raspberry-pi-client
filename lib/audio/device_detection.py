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
    
    # ReSpeaker not found - fall back to default devices
    print("⚠ ReSpeaker not detected - using default audio devices")
    print("  Hardware echo cancellation: NOT available")
    print("  Note: Barge-in may not work properly without AEC")
    
    # Get default devices
    default_input = sd.default.device[0]
    default_output = sd.default.device[1]
    
    if default_input is not None:
        input_dev = devices[default_input]
        print(f"  Default microphone: {input_dev['name']} (index {default_input})")
    else:
        print(f"  Default microphone: system default")
    
    if default_output is not None:
        output_dev = devices[default_output]
        print(f"  Default speaker: {output_dev['name']} (index {default_output})")
    else:
        print(f"  Default speaker: system default")
    
    if logger:
        logger.warning(
            "fallback_audio_devices",
            extra={
                "respeaker_found": False,
                "hardware_aec": False,
                "default_input": default_input,
                "default_output": default_output,
                "user_id": Config.USER_ID
            }
        )
    
    return default_input, default_output, False


def verify_audio_setup():
    """
    Verify ALSA audio setup and log available devices.
    
    This function verifies device detection and logs information for debugging.
    Actual device selection is handled by get_audio_devices().
    """
    print("\n🔊 Verifying ALSA audio setup...")
    print("  Audio Architecture:")
    print("    • ALSA-only (PipeWire/PulseAudio disabled)")
    print("    • Priority: ReSpeaker 4 Mic Array v2.0")
    print("    • Fallback: Best available mic/speaker")
    
    try:
        # Query available audio devices via sounddevice
        devices = sd.query_devices()
        print("\n  Available ALSA devices:")
        
        for idx, dev in enumerate(devices):
            # Check if this is the ReSpeaker
            is_respeaker = any(keyword in dev['name'].lower() 
                             for keyword in ['respeaker', 'arrayuac10', 'uac1.0'])
            
            if is_respeaker:
                print(f"    ✓ [{idx}] {dev['name']} (ReSpeaker - hardware AEC)")
                print(f"        Input channels: {dev['max_input_channels']}")
                print(f"        Output channels: {dev['max_output_channels']}")
            else:
                # Log other devices for debugging
                in_ch = dev['max_input_channels']
                out_ch = dev['max_output_channels']
                caps = []
                if in_ch > 0:
                    caps.append(f"in:{in_ch}")
                if out_ch > 0:
                    caps.append(f"out:{out_ch}")
                cap_str = f" ({', '.join(caps)})" if caps else ""
                print(f"      [{idx}] {dev['name']}{cap_str}")
        
        print()  # Blank line for readability
            
    except Exception as e:
        print(f"  ⚠ Could not query audio devices: {e}")
        print("    This may indicate ALSA configuration issues")
        
        logger = Config.LOGGER
        if logger:
            logger.error(
                "audio_device_query_failed",
                extra={
                    "error": str(e),
                    "user_id": Config.USER_ID
                },
                exc_info=True
            )

