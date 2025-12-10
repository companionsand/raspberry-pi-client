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
    
    Strategy:
    1. Detect ReSpeaker 4 Mic Array
    2. Use ALSA default device (None, None) to route through /etc/asound.conf
    3. ALSA config extracts Channel 0 (AEC-processed audio) from ReSpeaker
    
    Returns:
        tuple: (mic_device_index, speaker_device_index, has_hardware_aec)
               Returns (None, None, True) when ReSpeaker found (uses ALSA Ch0 routing)
               Returns (None, None, False) if ReSpeaker not found (fallback)
    """
    import os
    
    logger = Config.LOGGER
    
    devices = sd.query_devices()
    respeaker_found = False
    respeaker_info = None
    
    # Search for ReSpeaker device
    for idx, dev in enumerate(devices):
        # Check if this is the ReSpeaker device
        if any(keyword in dev['name'].lower() 
               for keyword in ['respeaker', 'arrayuac10', 'uac1.0']):
            # Note: We check for output channels (speaker) - input channels may show 0
            # if device is busy or sounddevice can't query it properly
            if dev['max_output_channels'] > 0:
                respeaker_found = True
                respeaker_info = {
                    'index': idx,
                    'name': dev['name'],
                    'input_channels': dev['max_input_channels'],
                    'output_channels': dev['max_output_channels']
                }
                break
    
    if respeaker_found:
        # Check if ALSA config exists and mentions Ch0/Channel 0
        alsa_config_path = '/etc/asound.conf'
        using_ch0 = False
        ch0_info = ""
        
        if os.path.exists(alsa_config_path):
            try:
                with open(alsa_config_path, 'r') as f:
                    alsa_config = f.read()
                    # Check if config mentions Ch0 or Channel 0 extraction
                    if 'ch0' in alsa_config.lower() or 'channel 0' in alsa_config.lower() or 'channel.0' in alsa_config.lower():
                        using_ch0 = True
                        ch0_info = " (Ch0 extraction configured)"
            except Exception:
                pass
        
        print(f"✓ ReSpeaker detected: {respeaker_info['name']} (hardware index {respeaker_info['index']})")
        print(f"  Hardware echo cancellation: Available")
        print(f"  ReSpeaker channels: {respeaker_info['input_channels']} in, {respeaker_info['output_channels']} out")
        print(f"  Audio routing: ALSA default device → /etc/asound.conf")
        if using_ch0:
            print(f"  ✓ Using Channel 0 (AEC-processed audio) from ReSpeaker{ch0_info}")
        else:
            print(f"  ⚠ ALSA config may not be extracting Ch0 - check /etc/asound.conf")
        print(f"  Playback: ReSpeaker 3.5mm jack (for AEC reference signal)")
        
        if logger:
            logger.info(
                "respeaker_detected_alsa_routing",
                extra={
                    "device_name": respeaker_info['name'],
                    "hardware_index": respeaker_info['index'],
                    "input_channels": respeaker_info['input_channels'],
                    "output_channels": respeaker_info['output_channels'],
                    "alsa_config_exists": os.path.exists(alsa_config_path),
                    "using_ch0": using_ch0,
                    "hardware_aec": True,
                    "routing": "alsa_default",
                    "user_id": Config.USER_ID
                }
            )
        
        # Return None for both - use ALSA default which routes through /etc/asound.conf
        # This ensures we get Ch0 (AEC-processed) instead of raw hardware channels
        return None, None, True
    
    # ReSpeaker not found - fall back to system default
    print("⚠ ReSpeaker not detected - using system default audio")
    print("  Hardware echo cancellation: NOT available")
    print("  Note: Barge-in may not work properly without AEC")
    print("  System will auto-select audio devices via ALSA/PulseAudio")
    print("  Channel routing: Not using ReSpeaker Ch0 (device not found)")
    
    if logger:
        logger.warning(
            "fallback_audio_devices",
            extra={
                "respeaker_found": False,
                "hardware_aec": False,
                "using_system_default": True,
                "using_ch0": False,
                "user_id": Config.USER_ID
            }
        )
    
    # Return None for both - let sounddevice/ALSA/PulseAudio choose the best device
    return None, None, False


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

