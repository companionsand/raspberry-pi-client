#!/usr/bin/env python3
"""
Kin AI Raspberry Pi Client
===========================
Minimalistic wake word detection + conversation client for Raspberry Pi.

Features:
- Wake word detection using Porcupine ("Porcupine" keyword)
- Real-time conversation via ElevenLabs WebSocket API
- ReSpeaker hardware echo cancellation (ALSA-only)
- Communication with conversation-orchestrator via WebSocket
- Supabase authentication on startup

Audio Architecture:
- ALSA-only (PipeWire/PulseAudio disabled)
- Single device: ReSpeaker 4 Mic Array v2.0 (USB)
- Capture: ReSpeaker microphones
- Playback: ReSpeaker 3.5mm jack → powered speaker
- Echo cancellation: ReSpeaker built-in hardware AEC

Usage:
    python main.py

Requirements:
    - Raspberry Pi OS with ALSA (PipeWire/PulseAudio disabled)
    - ReSpeaker 4 Mic Array v2.0 with powered speaker on 3.5mm jack
    - /etc/asound.conf configured to use ReSpeaker as default
    - Environment variables: DEVICE_ID, SUPABASE_URL, SUPABASE_ANON_KEY, EMAIL, PASSWORD, 
                            CONVERSATION_ORCHESTRATOR_URL, ELEVENLABS_API_KEY, PICOVOICE_ACCESS_KEY
"""

import os
import sys
import signal
import time
import json
import base64
import asyncio
import subprocess
import uuid
import math
from datetime import datetime, timezone
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import required packages
try:
    import pvporcupine
    import sounddevice as sd
    import numpy as np
    import websockets
    import certifi
    import ssl
    from supabase import create_client, Client
except ImportError as e:
    print(f"✗ Missing required package: {e}")
    print("Install dependencies: pip install -r requirements.txt")
    sys.exit(1)


# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    """Application configuration from environment variables"""
    
    # Device credentials
    DEVICE_ID = os.getenv("DEVICE_ID")
    
    # Supabase authentication
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")
    EMAIL = os.getenv("EMAIL")
    PASSWORD = os.getenv("PASSWORD")
    
    # These will be set after authentication
    USER_ID = None
    AUTH_TOKEN = None
    
    # Backend
    CONVERSATION_ORCHESTRATOR_URL = os.getenv("CONVERSATION_ORCHESTRATOR_URL", "ws://localhost:8001/ws")
    
    # ElevenLabs API
    ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
    
    # Wake word detection
    PICOVOICE_ACCESS_KEY = os.getenv("PICOVOICE_ACCESS_KEY")
    WAKE_WORD = os.getenv("WAKE_WORD", "porcupine")
    
    # Audio settings (ALSA-only, single ReSpeaker device for both capture and playback)
    # No device selection needed - ALSA default (/etc/asound.conf) routes to ReSpeaker
    # ReSpeaker handles both microphone input and 3.5mm jack output
    SAMPLE_RATE = 16000  # 16kHz for both capture and playback
    CHANNELS = 1  # Mono (ReSpeaker AEC expects mono reference)
    CHUNK_SIZE = 512  # ~32ms frames for low latency
    
    # Heartbeat interval
    HEARTBEAT_INTERVAL = 10  # seconds
    
    # LED settings (optional - for ReSpeaker visual feedback)
    LED_ENABLED = os.getenv("LED_ENABLED", "true").lower() == "true"
    LED_BRIGHTNESS = int(os.getenv("LED_BRIGHTNESS", "60"))  # 0-100, default 60% for subtlety
    
    @classmethod
    def validate(cls):
        """Validate required configuration"""
        missing = []
        
        if not cls.DEVICE_ID:
            missing.append("DEVICE_ID")
        if not cls.SUPABASE_URL:
            missing.append("SUPABASE_URL")
        if not cls.SUPABASE_ANON_KEY:
            missing.append("SUPABASE_ANON_KEY")
        if not cls.EMAIL:
            missing.append("EMAIL")
        if not cls.PASSWORD:
            missing.append("PASSWORD")
        if not cls.ELEVENLABS_API_KEY:
            missing.append("ELEVENLABS_API_KEY")
        if not cls.PICOVOICE_ACCESS_KEY:
            missing.append("PICOVOICE_ACCESS_KEY")
            
        if missing:
            print(f"✗ Missing required environment variables: {', '.join(missing)}")
            print("Create a .env file with required credentials")
            sys.exit(1)


# =============================================================================
# AUTHENTICATION
# =============================================================================

def authenticate_with_supabase():
    """
    Authenticate with Supabase and fetch auth token and user ID.
    Sets Config.AUTH_TOKEN and Config.USER_ID on success.
    """
    print("\n🔐 Authenticating with Supabase...")
    
    try:
        # Create Supabase client
        supabase: Client = create_client(Config.SUPABASE_URL, Config.SUPABASE_ANON_KEY)
        
        # Sign in with email and password
        response = supabase.auth.sign_in_with_password({
            "email": Config.EMAIL,
            "password": Config.PASSWORD
        })
        
        # Extract auth token and user ID
        if response.user and response.session:
            Config.AUTH_TOKEN = response.session.access_token
            Config.USER_ID = response.user.id
            print(f"✓ Successfully authenticated")
            print(f"   User ID: {Config.USER_ID}")
            return True
        else:
            print("✗ Authentication failed: No user or session returned")
            return False
            
    except Exception as e:
        print(f"✗ Authentication error: {e}")
        return False


# =============================================================================
# AUDIO SETUP (ALSA-only, ReSpeaker hardware AEC)
# =============================================================================
#
# AUDIO ARCHITECTURE OVERVIEW:
# ============================
#
# Hardware:
# ---------
# • ReSpeaker 4 Mic Array v2.0 (USB) - Single device for all audio I/O
# • 4 microphones on ReSpeaker → capture audio
# • 3.5mm jack on ReSpeaker → powered speaker (playback)
# • Built-in hardware AEC (Acoustic Echo Cancellation) on ReSpeaker DSP
#
# Software Stack:
# ---------------
# • ALSA ONLY - PipeWire and PulseAudio are disabled at system level
# • No software AEC (no WebRTC echo cancellation between devices)
# • No device selection in application code (device=None everywhere)
# • /etc/asound.conf configures ALSA default → ReSpeaker for both capture/playback
#
# Audio Flow:
# -----------
# 1. User speaks → ReSpeaker microphones
# 2. Captured audio (16kHz mono) → Python app via ALSA
# 3. Wake word detection (Porcupine) or conversation (ElevenLabs)
# 4. TTS audio from ElevenLabs → Python app
# 5. Playback (16kHz mono) → ReSpeaker 3.5mm jack → powered speaker
# 6. ReSpeaker hardware AEC cancels speaker audio from microphone input
#
# Echo Cancellation:
# ------------------
# • Hardware-based: ReSpeaker DSP performs AEC internally
# • Mono playback: AEC expects mono reference signal (not stereo)
# • No multi-device software AEC (no PipeWire module-echo-cancel)
#
# Configuration Requirements:
# ---------------------------
# 1. /etc/asound.conf must set ReSpeaker as ALSA default
# 2. PipeWire/PulseAudio services must be stopped and disabled
# 3. ReSpeaker must be only audio device (or explicitly default)
#
# =============================================================================

def verify_audio_setup():
    """
    Verify ALSA audio setup and log available devices.
    
    This function does NOT configure audio routing - that's handled by:
    1. /etc/asound.conf (sets ALSA default to ReSpeaker)
    2. ReSpeaker hardware AEC (built into the device)
    
    We simply verify the ReSpeaker is detected and log device info for debugging.
    """
    print("\n🔊 Verifying ALSA audio setup...")
    print("  Audio Architecture:")
    print("    • ALSA-only (PipeWire/PulseAudio disabled)")
    print("    • Single device: ReSpeaker 4 Mic Array v2.0")
    print("    • Capture: ReSpeaker microphones")
    print("    • Playback: ReSpeaker 3.5mm jack → powered speaker")
    print("    • Echo cancellation: ReSpeaker hardware AEC")
    print("    • Routing: /etc/asound.conf → ReSpeaker default")
    
    try:
        # Query available audio devices via sounddevice
        devices = sd.query_devices()
        print("\n  Available ALSA devices:")
        
        respeaker_found = False
        for idx, dev in enumerate(devices):
            # Check if this is the ReSpeaker
            is_respeaker = any(keyword in dev['name'].lower() 
                             for keyword in ['respeaker', 'arrayuac10', 'uac1.0'])
            
            if is_respeaker:
                respeaker_found = True
                print(f"    ✓ [{idx}] {dev['name']} (ReSpeaker)")
                print(f"        Input channels: {dev['max_input_channels']}")
                print(f"        Output channels: {dev['max_output_channels']}")
            else:
                # Log other devices for debugging
                print(f"      [{idx}] {dev['name']}")
        
        if respeaker_found:
            print("\n  ✓ ReSpeaker detected")
            print("  ✓ Using ALSA default device (via /etc/asound.conf)")
        else:
            print("\n  ⚠ Warning: ReSpeaker not detected")
            print("    Check USB connection and run: arecord -l")
            
    except Exception as e:
        print(f"  ⚠ Could not query audio devices: {e}")
        print("    This may indicate ALSA configuration issues")


def get_audio_device_index(device_name: str = None, kind: str = "input") -> None:
    """
    Returns None to use ALSA default device (ReSpeaker).
    
    In ALSA-only mode with /etc/asound.conf configured, we don't select devices by index.
    Instead, we use device=None in sounddevice, which lets ALSA route to the default
    device (ReSpeaker for both capture and playback).
    
    Args:
        device_name: Ignored (kept for backward compatibility)
        kind: Ignored (kept for backward compatibility)
    
    Returns:
        None (instructs sounddevice to use ALSA default)
    """
    # Always return None - ALSA handles routing via /etc/asound.conf
    return None


# =============================================================================
# LED CONTROL (ReSpeaker visual feedback)
# =============================================================================

class LEDController:
    """
    Manages ReSpeaker LED ring for visual feedback.
    
    Design Philosophy:
    - Subtle, calming patterns for elderly users
    - Warm colors to create a reassuring presence
    - Low brightness (20-40%) to avoid being startling
    - Clear state differentiation with smooth animations
    
    LED States:
    - BOOT: Soft amber pulse (20-40% brightness) during startup
    - IDLE: Soft breathing (20-40% brightness) when ready
    - WAKE_WORD_DETECTED: Blue ring with high brightness (awakening signal)
    - CONVERSATION: Pulsating white during active conversation
    - ERROR: Soft red slow blink (2s on/off) for issues
    - OFF: No lights during shutdown
    """
    
    # State constants
    STATE_OFF = 0
    STATE_BOOT = 1
    STATE_IDLE = 2
    STATE_WAKE_WORD_DETECTED = 3
    STATE_CONVERSATION = 4
    STATE_ERROR = 5
    
    # Color palette (RGB, 0-255)
    # Colors are applied with brightness multipliers for visibility
    COLORS = {
        'boot': (244, 162, 97),         # Soft amber - warm startup
        'idle': (255, 255, 255),        # White - clear presence
        'wake_word_1': (100, 149, 237), # Cornflower blue
        'wake_word_2': (255, 228, 181), # Warm white
        'wake_word_3': (244, 162, 97),  # Amber
        'conversation': (0, 255, 0), # Blue - active engagement
        'error': (255, 107, 107),       # Soft red - non-alarming indicator
        'off': (0, 0, 0)                # No light
    }
    
    def __init__(self, enabled: bool = True):
        """
        Initialize LED controller.
        
        Args:
            enabled: Enable LED control (set False to disable all LED feedback)
        """
        self.enabled = enabled
        self.current_state = self.STATE_OFF
        self.pixel_ring = None
        self.animation_task = None  # Async task for animations (breathing, pulsing, blinking)
        
        if not self.enabled:
            return
        
        # Try to import pixel_ring library for ReSpeaker
        try:
            from pixel_ring import pixel_ring
            self.pixel_ring = pixel_ring
            self.pixel_ring.off()
            print("✓ LED controller initialized")
        except ImportError:
            print("⚠ pixel_ring not available - LED control disabled")
            print("  Install with: pip install pixel-ring")
            self.enabled = False
    
    def set_state(self, state: int):
        """
        Set LED visual state with appropriate pattern.
        
        Args:
            state: One of STATE_* constants (OFF, BOOT, IDLE, WAKE_WORD_DETECTED, CONVERSATION, ERROR)
        """
        # Skip if LEDs disabled or unavailable
        if not self.enabled or not self.pixel_ring:
            return
        
        # Skip if already in this state (avoid unnecessary updates)
        if state == self.current_state:
            return
        
        self.current_state = state
        
        # Log state change with expected pattern
        state_names = {
            self.STATE_OFF: "OFF (no lights)",
            self.STATE_BOOT: "BOOT (soft amber pulse)",
            self.STATE_IDLE: "IDLE (white breathing 5-40%)",
            self.STATE_WAKE_WORD_DETECTED: "WAKE_WORD_DETECTED (crazy color burst!)",
            self.STATE_CONVERSATION: "CONVERSATION (pulsating blue 20-50%)",
            self.STATE_ERROR: "ERROR (soft red blink)"
        }
        print(f"💡 LED: {state_names.get(state, f'UNKNOWN({state})')}")
        
        # Stop any running animation when changing states
        self._stop_animation()
        
        # Apply visual pattern based on state
        if state == self.STATE_OFF:
            # Turn off all LEDs
            self.pixel_ring.off()
        
        elif state == self.STATE_BOOT:
            # Soft amber pulse during startup (20-40% brightness)
            self._start_animation(self._boot_pulse_loop)
        
        elif state == self.STATE_IDLE:
            # Soft breathing animation (20-40% brightness)
            self._start_animation(self._idle_breathing_loop)
        
        elif state == self.STATE_WAKE_WORD_DETECTED:
            # Crazy color burst for 1-2 seconds (awakening signal)
            self._start_animation(self._wake_word_burst_loop)
        
        elif state == self.STATE_CONVERSATION:
            # Pulsating white during conversation
            self._start_animation(self._conversation_pulse_loop)
        
        elif state == self.STATE_ERROR:
            # Soft red slow blink (2s on/off)
            self._start_animation(self._error_blink_loop)
    
    # -------------------------------------------------------------------------
    # Animation Control Methods
    # -------------------------------------------------------------------------
    
    def _stop_animation(self):
        """Stop any running animation task"""
        if self.animation_task and not self.animation_task.done():
            self.animation_task.cancel()
            self.animation_task = None
    
    def _start_animation(self, animation_func):
        """Start an animation as an async task"""
        # Stop any existing animation first
        self._stop_animation()
        
        # Create and start the animation task
        try:
            loop = asyncio.get_event_loop()
            self.animation_task = loop.create_task(animation_func())
        except RuntimeError:
            # If no event loop exists, create one (shouldn't happen in normal flow)
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self.animation_task = loop.create_task(animation_func())
    
    def _rgb_to_int_with_brightness(self, rgb_tuple, brightness):
        """
        Convert RGB tuple to integer color with brightness multiplier.
        
        Args:
            rgb_tuple: (R, G, B) tuple with values 0-255
            brightness: Multiplier 0.0-1.0
        
        Returns:
            Integer color value for pixel_ring.mono()
        """
        r, g, b = rgb_tuple
        r = int(r * brightness)
        g = int(g * brightness)
        b = int(b * brightness)
        return (r << 16) | (g << 8) | b
    
    # -------------------------------------------------------------------------
    # Animation Loop Methods
    # -------------------------------------------------------------------------
    
    async def _boot_pulse_loop(self):
        """
        Soft amber pulse during boot (20-40% brightness).
        Slow, calming pulse to indicate system is starting up.
        """
        if not self.enabled or not self.pixel_ring:
            return
        
        CYCLE_SECONDS = 2.0  # 2 second pulse cycle
        UPDATE_INTERVAL = 0.05  # 50ms updates for smooth animation
        MIN_BRIGHTNESS = 0.2  # 20% minimum
        MAX_BRIGHTNESS = 0.4  # 40% maximum
        
        base_color = self.COLORS['boot']
        start_time = time.time()
        
        try:
            while self.current_state == self.STATE_BOOT:
                elapsed = time.time() - start_time
                cycle_position = (elapsed % CYCLE_SECONDS) / CYCLE_SECONDS
                
                # Sine wave for smooth pulsing
                sine_value = (math.sin(cycle_position * 2 * math.pi) + 1) / 2
                brightness = MIN_BRIGHTNESS + (sine_value * (MAX_BRIGHTNESS - MIN_BRIGHTNESS))
                
                color = self._rgb_to_int_with_brightness(base_color, brightness)
                self.pixel_ring.mono(color)
                
                await asyncio.sleep(UPDATE_INTERVAL)
                
        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"  ⚠ LED boot animation error: {e}")
    
    async def _idle_breathing_loop(self):
        """
        White breathing effect for IDLE state (5-40% brightness).
        Clear, visible presence indicating device is ready.
        """
        if not self.enabled or not self.pixel_ring:
            return
        
        CYCLE_SECONDS = 4.0  # 4 second breathing cycle (calm, steady)
        UPDATE_INTERVAL = 0.05  # 50ms updates
        MIN_BRIGHTNESS = 0.05  # 5% minimum (very subtle low point)
        MAX_BRIGHTNESS = 0.4  # 40% maximum (clearly visible)
        
        base_color = self.COLORS['idle']  # White
        start_time = time.time()
        
        try:
            while self.current_state == self.STATE_IDLE:
                elapsed = time.time() - start_time
                cycle_position = (elapsed % CYCLE_SECONDS) / CYCLE_SECONDS
                
                # Sine wave for smooth breathing
                sine_value = (math.sin(cycle_position * 2 * math.pi) + 1) / 2
                brightness = MIN_BRIGHTNESS + (sine_value * (MAX_BRIGHTNESS - MIN_BRIGHTNESS))
                
                color = self._rgb_to_int_with_brightness(base_color, brightness)
                self.pixel_ring.mono(color)
                
                await asyncio.sleep(UPDATE_INTERVAL)
                
        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"  ⚠ LED idle animation error: {e}")
    
    async def _wake_word_burst_loop(self):
        """
        Crazy color burst for wake word detection (1-2 seconds).
        Rapid alternating colors and brightness to create excitement.
        """
        if not self.enabled or not self.pixel_ring:
            return
        
        BURST_DURATION = 1.5  # 1.5 seconds of crazy animation
        UPDATE_INTERVAL = 0.08  # 80ms updates (fast, energetic)
        
        # Color sequence for burst
        colors = [
            self.COLORS['wake_word_1'],  # Blue
            self.COLORS['wake_word_2'],  # Warm white
            self.COLORS['wake_word_3'],  # Amber
        ]
        
        start_time = time.time()
        color_index = 0
        
        try:
            while (time.time() - start_time) < BURST_DURATION:
                # Alternate between 0% and 80% brightness rapidly
                elapsed = time.time() - start_time
                
                # Fast toggle between bright and off
                if int(elapsed / UPDATE_INTERVAL) % 2 == 0:
                    brightness = 0.8  # 80% - bright
                else:
                    brightness = 0.0  # 0% - off
                
                # Cycle through colors
                current_color = colors[color_index % len(colors)]
                color = self._rgb_to_int_with_brightness(current_color, brightness)
                self.pixel_ring.mono(color)
                
                # Change color every 3 updates
                if int(elapsed / UPDATE_INTERVAL) % 3 == 0:
                    color_index += 1
                
                await asyncio.sleep(UPDATE_INTERVAL)
            
            # After burst, turn off briefly before next state
            self.pixel_ring.off()
                
        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"  ⚠ LED wake word animation error: {e}")
    
    async def _conversation_pulse_loop(self):
        """
        Pulsating blue during active conversation (20-50% brightness).
        Indicates device is engaged in conversation with user.
        """
        if not self.enabled or not self.pixel_ring:
            return
        
        CYCLE_SECONDS = 2.0  # 2 second pulse (faster than idle, shows activity)
        UPDATE_INTERVAL = 0.05  # 50ms updates
        MIN_BRIGHTNESS = 0.2  # 20% minimum
        MAX_BRIGHTNESS = 0.5  # 50% maximum
        
        base_color = self.COLORS['conversation']  # Blue
        start_time = time.time()
        
        try:
            while self.current_state == self.STATE_CONVERSATION:
                elapsed = time.time() - start_time
                cycle_position = (elapsed % CYCLE_SECONDS) / CYCLE_SECONDS
                
                # Sine wave for smooth pulsing
                sine_value = (math.sin(cycle_position * 2 * math.pi) + 1) / 2
                brightness = MIN_BRIGHTNESS + (sine_value * (MAX_BRIGHTNESS - MIN_BRIGHTNESS))
                
                color = self._rgb_to_int_with_brightness(base_color, brightness)
                self.pixel_ring.mono(color)
                
                await asyncio.sleep(UPDATE_INTERVAL)
                
        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"  ⚠ LED conversation animation error: {e}")
    
    async def _error_blink_loop(self):
        """
        Soft red slow blink for errors (2s on/off).
        Non-alarming indicator that something needs attention.
        """
        if not self.enabled or not self.pixel_ring:
            return
        
        BLINK_INTERVAL = 2.0  # 2 seconds on, 2 seconds off
        BRIGHTNESS = 0.4  # 40% brightness (soft, not harsh)
        
        base_color = self.COLORS['error']
        color = self._rgb_to_int_with_brightness(base_color, BRIGHTNESS)
        
        try:
            while self.current_state == self.STATE_ERROR:
                # Turn on
                self.pixel_ring.mono(color)
                await asyncio.sleep(BLINK_INTERVAL)
                
                # Turn off
                if self.current_state == self.STATE_ERROR:  # Check state hasn't changed
                    self.pixel_ring.off()
                    await asyncio.sleep(BLINK_INTERVAL)
                
        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"  ⚠ LED error animation error: {e}")
    
    def cleanup(self):
        """Turn off LEDs during shutdown"""
        # Stop any running animation
        self._stop_animation()
        
        # Turn off LEDs
        if self.enabled and self.pixel_ring:
            self.pixel_ring.off()


# =============================================================================
# WAKE WORD DETECTION (from old-raspberry-pi-client)
# =============================================================================

class WakeWordDetector:
    """Porcupine-based wake word detection"""
    
    def __init__(self):
        self.porcupine = None
        self.audio_stream = None
        self.detected = False
        self.running = False
        
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
            
            # Find ReSpeaker device index
            # Critical: device=None doesn't always use ALSA default correctly
            devices = sd.query_devices()
            respeaker_idx = None
            
            for idx, dev in enumerate(devices):
                # Check if this is the ReSpeaker device
                if any(keyword in dev['name'].lower() 
                       for keyword in ['respeaker', 'arrayuac10', 'uac1.0']):
                    # Verify it has input channels
                    if dev['max_input_channels'] > 0:
                        respeaker_idx = idx
                        print(f"   Using ReSpeaker microphone: {dev['name']} (index {idx})")
                        break
            
            if respeaker_idx is None:
                print("   ⚠ Warning: ReSpeaker not found, using default input device")
            
            # Start audio stream for wake word detection
            # Use explicit ReSpeaker device index for reliable capture
            self.audio_stream = sd.InputStream(
                device=respeaker_idx,  # Explicit ReSpeaker device
                channels=Config.CHANNELS,
                samplerate=self.porcupine.sample_rate,
                blocksize=self.porcupine.frame_length,
                dtype='int16',
                callback=self._audio_callback
            )
            
            self.audio_stream.start()
            self.running = True
            print(f"✓ Listening for wake word...")
        except Exception:
            # Ensure partially-initialized resources are cleaned up
            self.stop()
            raise
        
    def _audio_callback(self, indata, frames, time_info, status):
        """Process audio frames for wake word detection"""
        if status:
            print(f"⚠ Audio status: {status}")
        
        # Convert to the format Porcupine expects
        audio_frame = indata[:, 0].astype(np.int16)
        
        # Process with Porcupine
        keyword_index = self.porcupine.process(audio_frame)
        
        if keyword_index >= 0:
            print(f"\n🎯 Wake word '{Config.WAKE_WORD}' detected!")
            self.detected = True
    
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


# =============================================================================
# CONVERSATION ORCHESTRATOR CLIENT
# =============================================================================

class OrchestratorClient:
    """WebSocket client for conversation-orchestrator"""
    
    def __init__(self):
        self.websocket = None
        self.connected = False
        self.running = False
        
    async def connect(self):
        """Connect to conversation-orchestrator"""
        print(f"\n🔌 Connecting to conversation-orchestrator...")
        print(f"   URL: {Config.CONVERSATION_ORCHESTRATOR_URL}")
        
        try:
            # Create SSL context if using wss://
            ssl_context = None
            if Config.CONVERSATION_ORCHESTRATOR_URL.startswith("wss://"):
                ssl_context = ssl.create_default_context(cafile=certifi.where())
            
            # Connect to WebSocket
            self.websocket = await websockets.connect(
                Config.CONVERSATION_ORCHESTRATOR_URL,
                ssl=ssl_context
            )
            
            # Wait for connection acceptance (FastAPI accepts first)
            # Then send authentication
            await self.websocket.send(json.dumps({
                "type": "auth",
                "token": Config.AUTH_TOKEN,
                "device_id": Config.DEVICE_ID,
                "user_id": Config.USER_ID,
            }))
            
            # Wait for connection confirmation
            response = await self.websocket.recv()
            data = json.loads(response)
            
            if data.get("type") == "connected":
                self.connected = True
                self.running = True
                print("✓ Connected to conversation-orchestrator")
                return True
            else:
                print(f"✗ Connection failed: {data}")
                return False
                
        except Exception as e:
            print(f"✗ Connection error: {e}")
            return False
    
    async def send_reactive(self):
        """Send reactive conversation request"""
        if not self.connected:
            return
        
        message = {
            "type": "reactive",
            "user_id": Config.USER_ID,
            "device_id": Config.DEVICE_ID,
        }
        
        await self.websocket.send(json.dumps(message))
        print("✓ Sent reactive request")
    
    async def send_heartbeat(self, device_status: str = "online"):
        """Send heartbeat message"""
        if not self.connected:
            return
        
        message = {
            "type": "heartbeat",
            "user_id": Config.USER_ID,
            "device_id": Config.DEVICE_ID,
            "device_status": device_status,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        
        await self.websocket.send(json.dumps(message))
    
    async def send_conversation_start(
        self, conversation_id: str, elevenlabs_conversation_id: str, agent_id: str
    ):
        """Send conversation start notification"""
        if not self.connected:
            return
        
        message = {
            "type": "conversation_start",
            "conversation_id": conversation_id,
            "elevenlabs_conversation_id": elevenlabs_conversation_id,
            "agent_id": agent_id,
            "device_id": Config.DEVICE_ID,
            "user_id": Config.USER_ID,
            "start_time": datetime.now(timezone.utc).isoformat(),
        }
        
        await self.websocket.send(json.dumps(message))
        print("✓ Sent conversation_start notification")
    
    async def send_conversation_end(
        self, conversation_id: str, elevenlabs_conversation_id: str, 
        agent_id: str, end_reason: str
    ):
        """Send conversation end notification"""
        if not self.connected:
            return
        
        message = {
            "type": "conversation_end",
            "conversation_id": conversation_id,
            "elevenlabs_conversation_id": elevenlabs_conversation_id,
            "agent_id": agent_id,
            "device_id": Config.DEVICE_ID,
            "user_id": Config.USER_ID,
            "end_time": datetime.now(timezone.utc).isoformat(),
            "end_reason": end_reason,
        }
        
        await self.websocket.send(json.dumps(message))
        print("✓ Sent conversation_end notification")
    
    async def receive_message(self):
        """Receive and return a message from orchestrator"""
        if not self.connected:
            return None
        
        try:
            message = await asyncio.wait_for(self.websocket.recv(), timeout=0.1)
            return json.loads(message)
        except asyncio.TimeoutError:
            return None
        except websockets.exceptions.ConnectionClosed as e:
            # Normal closure (1000/1001) is expected, don't spam logs
            if e.code in (1000, 1001):
                # Clean disconnect, mark as disconnected
                self.connected = False
                return None
            else:
                # Abnormal closure, log it
                print(f"✗ Connection closed unexpectedly: {e.code} - {e.reason}")
                self.connected = False
                return None
        except Exception as e:
            # Only log unexpected errors
            print(f"✗ Receive error: {e}")
            return None
    
    async def disconnect(self):
        """Disconnect from orchestrator"""
        self.connected = False
        self.running = False
        if self.websocket:
            await self.websocket.close()


# =============================================================================
# ELEVENLABS CONVERSATION CLIENT
# =============================================================================

class ElevenLabsConversationClient:
    """WebSocket-based conversation client for ElevenLabs"""
    
    def __init__(self, web_socket_url: str, agent_id: str, user_terminate_flag=None):
        self.web_socket_url = web_socket_url
        self.agent_id = agent_id
        self.websocket = None
        self.audio_stream = None
        self.running = False
        self.conversation_id = str(uuid.uuid4())
        self.elevenlabs_conversation_id = None
        self.end_reason = "normal"
        self.silence_timeout = 30.0  # seconds
        self.last_audio_time = None
        self.user_terminate_flag = user_terminate_flag  # Reference to shared flag
        
    async def start(self, orchestrator_client: OrchestratorClient):
        """Start a conversation session"""
        print(f"\n💬 Starting conversation...")
        print(f"   Agent ID: {self.agent_id}")
        print(f"   Conversation ID: {self.conversation_id}")
        
        # Add API key to WebSocket URL
        ws_url = f"{self.web_socket_url}&api_key={Config.ELEVENLABS_API_KEY}"
        
        # Find ReSpeaker device index
        # Critical: device=None doesn't always use ALSA default correctly
        # We must explicitly find and use the ReSpeaker device
        devices = sd.query_devices()
        respeaker_idx = None
        
        for idx, dev in enumerate(devices):
            # Check if this is the ReSpeaker device
            if any(keyword in dev['name'].lower() 
                   for keyword in ['respeaker', 'arrayuac10', 'uac1.0']):
                # Verify it has both input and output channels
                if dev['max_input_channels'] > 0 and dev['max_output_channels'] > 0:
                    respeaker_idx = idx
                    print(f"   Audio: Using ReSpeaker device index {idx}")
                    print(f"          {dev['name']}")
                    print(f"          Input channels: {dev['max_input_channels']}, Output: {dev['max_output_channels']}")
                    break
        
        if respeaker_idx is None:
            print("   ⚠ Warning: ReSpeaker not found, using default device")
            # This will likely fail but let it try
        
        # Create SSL context
        ssl_context = ssl.create_default_context(cafile=certifi.where())
        
        try:
            # Open audio stream for conversation
            # Use same ReSpeaker device for both input (mic) and output (3.5mm jack)
            self.audio_stream = sd.Stream(
                device=(respeaker_idx, respeaker_idx) if respeaker_idx is not None else None,
                samplerate=Config.SAMPLE_RATE,
                channels=Config.CHANNELS,
                dtype='int16',
                blocksize=Config.CHUNK_SIZE
            )
            self.audio_stream.start()
            
            # Connect to WebSocket
            async with websockets.connect(ws_url, ssl=ssl_context) as websocket:
                self.websocket = websocket
                self.running = True
                self.last_audio_time = time.time()
                
                print("✓ Connected to ElevenLabs")
                
                # Send conversation initiation
                await websocket.send(json.dumps({
                    "type": "conversation_initiation_client_data"
                }))
                
                print("✓ Conversation started - speak now!")
                
                # Run send and receive tasks concurrently
                send_task = asyncio.create_task(self._send_audio())
                receive_task = asyncio.create_task(self._receive_messages(orchestrator_client))
                
                # Wait for either task to complete
                done, pending = await asyncio.wait(
                    [send_task, receive_task],
                    return_when=asyncio.FIRST_COMPLETED
                )
                
                # Cancel pending tasks
                for task in pending:
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
                
        except Exception as e:
            print(f"✗ Conversation error: {e}")
            self.end_reason = "network_failure"
        finally:
            await self.stop(orchestrator_client)
    
    async def _send_audio(self):
        """Send microphone audio to WebSocket"""
        try:
            while self.running:
                # Read audio from microphone
                audio_data, _ = self.audio_stream.read(Config.CHUNK_SIZE)
                
                # Check for audio activity (simple energy detection)
                audio_energy = np.abs(audio_data).mean()
                if audio_energy > 100:  # Threshold for detecting speech
                    self.last_audio_time = time.time()
                
                # Encode as base64 and send
                audio_b64 = base64.b64encode(audio_data.tobytes()).decode('utf-8')
                message = {"user_audio_chunk": audio_b64}
                
                await self.websocket.send(json.dumps(message))
                
                # Check for silence timeout
                if self.last_audio_time and (time.time() - self.last_audio_time) > self.silence_timeout:
                    print("\n⏱️  Silence timeout - ending conversation")
                    self.end_reason = "silence"
                    self.running = False
                    break
                
                # Check for user termination
                if self.user_terminate_flag and self.user_terminate_flag[0]:
                    print("\n🛑 User termination - ending conversation")
                    self.end_reason = "user_terminated"
                    self.running = False
                    break
                
                # Small delay to prevent overwhelming the connection
                await asyncio.sleep(0.01)
                
        except asyncio.CancelledError:
            self.end_reason = "user_terminated"
            self.running = False
                
        except Exception as e:
            print(f"✗ Send error: {e}")
            self.end_reason = "network_failure"
            self.running = False
    
    async def _receive_messages(self, orchestrator_client: OrchestratorClient):
        """Receive and process messages from WebSocket"""
        try:
            while self.running:
                message = await self.websocket.recv()
                
                # Parse JSON message
                data = json.loads(message)
                
                # Handle different message types
                if 'conversation_initiation_metadata_event' in data:
                    metadata = data['conversation_initiation_metadata_event']
                    self.elevenlabs_conversation_id = metadata.get('conversation_id', None)
                    print(f"   ElevenLabs Conversation ID: {self.elevenlabs_conversation_id}")
                    
                    # Send conversation start notification
                    await orchestrator_client.send_conversation_start(
                        conversation_id=self.conversation_id,
                        elevenlabs_conversation_id=self.elevenlabs_conversation_id or "",
                        agent_id=self.agent_id,
                    )
                
                elif 'user_transcription_event' in data:
                    transcript = data['user_transcription_event'].get('user_transcript', '')
                    if transcript:
                        print(f"👤 You: {transcript}")
                        self.last_audio_time = time.time()  # Reset silence timer
                
                elif 'agent_response_event' in data:
                    response = data['agent_response_event'].get('agent_response', '')
                    if response:
                        print(f"🤖 Agent: {response}")
                
                elif 'audio_event' in data:
                    # Decode and play agent audio
                    audio_b64 = data['audio_event'].get('audio_base_64', '')
                    if audio_b64:
                        audio_bytes = base64.b64decode(audio_b64)
                        audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
                        self.audio_stream.write(audio_array)
                        self.last_audio_time = time.time()  # Reset silence timer
                
                elif data.get('type') == 'ping':
                    # Respond to ping to keep connection alive
                    ping_event = data.get('ping_event', {})
                    event_id = ping_event.get('event_id')
                    if event_id is not None:
                        await self.websocket.send(json.dumps({
                            'type': 'pong',
                            'event_id': event_id
                        }))
                
                elif data.get('type') == 'error':
                    error_msg = data.get('message', 'Unknown error')
                    print(f"✗ Server error: {error_msg}")
                    self.end_reason = "error"
                    self.running = False
                
                # Check for user termination
                if self.user_terminate_flag and self.user_terminate_flag[0]:
                    print("\n🛑 User termination - ending conversation")
                    self.end_reason = "user_terminated"
                    self.running = False
                    break
                    
        except websockets.exceptions.ConnectionClosed:
            print("\n✓ Conversation ended (connection closed)")
            if self.user_terminate_flag and self.user_terminate_flag[0]:
                self.end_reason = "user_terminated"
            else:
                self.end_reason = "normal"
            self.running = False
        except asyncio.CancelledError:
            self.end_reason = "user_terminated"
            self.running = False
        except Exception as e:
            print(f"✗ Receive error: {e}")
            self.end_reason = "network_failure"
            self.running = False
    
    async def stop(self, orchestrator_client: OrchestratorClient):
        """Stop the conversation"""
        self.running = False
        
        if self.audio_stream:
            self.audio_stream.stop()
            self.audio_stream.close()
        
        # Send conversation end notification
        await orchestrator_client.send_conversation_end(
            conversation_id=self.conversation_id,
            elevenlabs_conversation_id=self.elevenlabs_conversation_id or "",
            agent_id=self.agent_id,
            end_reason=self.end_reason,
        )
        
        if self.websocket:
            await self.websocket.close()


# =============================================================================
# MAIN APPLICATION
# =============================================================================

class KinClient:
    """Main application controller"""
    
    def __init__(self):
        self.wake_detector = WakeWordDetector()
        self.orchestrator_client = OrchestratorClient()
        self.led_controller = LEDController(enabled=Config.LED_ENABLED)  # Visual feedback via ReSpeaker LEDs
        self.running = True
        self.conversation_active = False
        self.awaiting_agent_details = False
        self.user_terminate = [False]  # Use list for mutable reference
        self.shutdown_requested = False
        
        # Setup signal handlers
        signal.signal(signal.SIGUSR1, self._handle_terminate_signal)
        signal.signal(signal.SIGINT, self._handle_interrupt_signal)
        signal.signal(signal.SIGTERM, self._handle_interrupt_signal)
    
    def _handle_terminate_signal(self, sig, frame):
        """Handle user-initiated termination signal"""
        print("\n🛑 User termination signal received")
        self.user_terminate[0] = True
    
    def _handle_interrupt_signal(self, sig, frame):
        """Handle interrupt/termination signals (Ctrl+C, SIGTERM)."""
        signal_name = getattr(signal, "Signals", lambda s: s)(sig)
        print(f"\n🛑 Received {signal_name} - shutting down...")
        
        # Force immediate shutdown
        self.running = False
        self.shutdown_requested = True
        self.user_terminate[0] = True
        
        # Stop wake word detector immediately
        if self.wake_detector:
            self.wake_detector.stop()
        
        # Raise KeyboardInterrupt to break out of async loops
        raise KeyboardInterrupt()
    
    def _resume_wake_word_detection(self):
        """Start wake word detector again if it is not already running."""
        if self.wake_detector.running:
            return
        
        self.wake_detector.start()
        print(f"\n✓ Listening for '{Config.WAKE_WORD}' again...")
    
    async def run(self):
        """Main application loop"""
        print("\n" + "="*60)
        print("🎙️  Kin AI Raspberry Pi Client")
        print("="*60)
        
        # Show boot state during startup
        self.led_controller.set_state(LEDController.STATE_BOOT)
        
        # Validate configuration
        Config.validate()
        
        # Authenticate with Supabase
        if not authenticate_with_supabase():
            self.led_controller.set_state(LEDController.STATE_ERROR)  # Show auth error
            print("✗ Failed to authenticate with Supabase")
            return
        
        # Verify audio setup (ALSA-only, no configuration needed)
        verify_audio_setup()
        
        # Connect to conversation-orchestrator
        connected = await self.orchestrator_client.connect()
        if not connected:
            self.led_controller.set_state(LEDController.STATE_ERROR)  # Show connection error
            print("✗ Failed to connect to conversation-orchestrator")
            return
        
        # Start wake word detection
        self.wake_detector.start()
        
        # System ready - show idle state (soft breathing, ready for wake word)
        self.led_controller.set_state(LEDController.STATE_IDLE)
        
        print("\n" + "="*60)
        print(f"✓ Ready! Say '{Config.WAKE_WORD}' to start a conversation")
        print("  Press Ctrl+C to exit")
        print("="*60 + "\n")
        
        # Start heartbeat task
        heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        
        # Main loop
        try:
            while self.running:
                # Check for shutdown request (from Ctrl+C or other signals)
                if self.shutdown_requested:
                    print("🛑 Shutdown requested, exiting main loop...")
                    break
                
                # Check if wake word was detected
                if self.wake_detector.detected and not self.conversation_active:
                    self.wake_detector.detected = False
                    
                    # Show wake word detected state (blue ring)
                    self.led_controller.set_state(LEDController.STATE_WAKE_WORD_DETECTED)
                    
                    # Stop wake word detection during conversation
                    self.wake_detector.stop()
                    
                    # Handle conversation
                    await self._handle_conversation()
                
                # Check for messages from orchestrator
                message = await self.orchestrator_client.receive_message()
                if message:
                    await self._handle_orchestrator_message(message)
                
                # Handle trigger-initiated conversations (start_conversation message)
                # This is handled in _handle_orchestrator_message
                
                # Small sleep to prevent busy loop
                await asyncio.sleep(0.1)
                
        except KeyboardInterrupt:
            print("\n\n🛑 Keyboard interrupt - shutting down...")
        except Exception as e:
            print(f"\n\n✗ Unexpected error: {e}")
        finally:
            print("🧹 Cleaning up...")
            await self.cleanup()
            heartbeat_task.cancel()
            try:
                await heartbeat_task
            except asyncio.CancelledError:
                pass
    
    async def _heartbeat_loop(self):
        """Send heartbeat messages periodically"""
        while self.running:
            try:
                await asyncio.sleep(Config.HEARTBEAT_INTERVAL)
                if self.running:
                    await self.orchestrator_client.send_heartbeat("online")
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"✗ Heartbeat error: {e}")
    
    async def _handle_conversation(self):
        """Handle a single conversation session"""
        self.conversation_active = True
        self.awaiting_agent_details = True
        self.user_terminate[0] = False
        
        # Send reactive request (wake word already detected, blue ring showing)
        await self.orchestrator_client.send_reactive()
        
        # Wait for agent_details message (with timeout)
        timeout = 10.0  # seconds
        start_time = time.time()
        
        while self.conversation_active and (time.time() - start_time) < timeout:
            # Check for messages from orchestrator
            message = await self.orchestrator_client.receive_message()
            if message:
                await self._handle_orchestrator_message(message)
                if not self.conversation_active:
                    break
            
            # Small sleep to prevent busy loop
            await asyncio.sleep(0.1)
        
        if self.awaiting_agent_details:
            print("✗ Timeout waiting for agent details")
            self.led_controller.set_state(LEDController.STATE_ERROR)  # Show timeout error
            time.sleep(2)  # Brief pause to show error state
            self.conversation_active = False
            self.awaiting_agent_details = False
            self._resume_wake_word_detection()
            self.led_controller.set_state(LEDController.STATE_IDLE)  # Back to idle
        
    async def _handle_orchestrator_message(self, message: dict):
        """Handle messages from conversation-orchestrator"""
        message_type = message.get("type")
        
        if message_type == "agent_details" or message_type == "start_conversation":
            # Check if conversation is already active
            if self.conversation_active and not self.awaiting_agent_details:
                print("⚠ Conversation already active, ignoring new request")
                return
            
            # Clear pending flag once we have agent details
            self.awaiting_agent_details = False
            
            # Extract agent details
            agent_id = message.get("agent_id")
            web_socket_url = message.get("web_socket_url")
            
            if not agent_id or not web_socket_url:
                print("✗ Invalid agent details received")
                return
            
            print(f"✓ Received agent details: {agent_id}")
            
            # Mark conversation as active
            self.conversation_active = True
            self.user_terminate[0] = False
            
            # Stop wake word detection during conversation
            self.wake_detector.stop()
            
            # Show conversation state (pulsating white - active conversation)
            self.led_controller.set_state(LEDController.STATE_CONVERSATION)
            
            # Start ElevenLabs conversation
            client = ElevenLabsConversationClient(
                web_socket_url, 
                agent_id,
                user_terminate_flag=self.user_terminate
            )
            await client.start(self.orchestrator_client)
            
            # Check if user terminated
            if self.user_terminate[0]:
                print("✓ User terminated conversation")
            
            # Resume wake word detection
            self._resume_wake_word_detection()
            
            # Back to idle state (soft breathing, ready for wake word)
            self.led_controller.set_state(LEDController.STATE_IDLE)
            
            self.conversation_active = False
            self.user_terminate[0] = False
            
        elif message_type == "error":
            error_msg = message.get("message", "Unknown error")
            print(f"✗ Orchestrator error: {error_msg}")
            self.led_controller.set_state(LEDController.STATE_ERROR)  # Show error state
            time.sleep(2)  # Brief pause to show error
            self.conversation_active = False
            self.awaiting_agent_details = False
            
            # If we were waiting on a conversation that failed, resume wake word detection
            self._resume_wake_word_detection()
            self.led_controller.set_state(LEDController.STATE_IDLE)  # Back to idle
    
    async def cleanup(self):
        """Clean up resources - force immediate shutdown"""
        print("  Stopping all services...")
        self.running = False
        
        # Stop LED animations and turn off
        try:
            self.led_controller.set_state(LEDController.STATE_OFF)
            self.led_controller.cleanup()
        except Exception as e:
            print(f"  ⚠ LED cleanup error: {e}")
        
        # Stop wake word detector
        try:
            self.wake_detector.cleanup()
        except Exception as e:
            print(f"  ⚠ Wake detector cleanup error: {e}")
        
        # Disconnect from orchestrator
        try:
            await self.orchestrator_client.disconnect()
        except Exception as e:
            print(f"  ⚠ Orchestrator disconnect error: {e}")
        
        print("✓ Cleanup complete")
        print("👋 Goodbye!\n")


# =============================================================================
# ENTRY POINT
# =============================================================================

def main():
    """Application entry point"""
    # Run the client
    client = KinClient()
    asyncio.run(client.run())


if __name__ == "__main__":
    main()