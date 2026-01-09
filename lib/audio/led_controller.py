"""
LED Controller for ReSpeaker visual feedback.

Manages ReSpeaker LED ring with various states and animations.
"""

import asyncio
import time
import math
import numpy as np
from lib.config import Config


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
    - IDLE: Soft breathing (20-100% brightness) in white when ready
    - WAKE_WORD_DETECTED: Very visible burst at 100% brightness with amber/gold/orange/white
    - CONVERSATION: Pulsating amber/gold (0-70% brightness) when listening, fast pulse (1.5s)
    - SPEAKING: White at 20-100% brightness, beating with voice energy
    - THINKING: Fast amber pulse (15-70% brightness, 0.5s cycle) when preparing response
    - ERROR: Soft red slow blink (2s on/off) for issues
    - WIFI_SETUP: Soft amber slow blink (60% brightness, 2s on/off) during WiFi setup mode
    - ATTEMPTING_CONNECTION: Soft amber fast blink (60% brightness, 0.5s on/off) when connecting
    - OFF: No lights during shutdown
    """
    
    # State constants
    STATE_OFF = 0
    STATE_BOOT = 1
    STATE_IDLE = 2
    STATE_WAKE_WORD_DETECTED = 3
    STATE_CONVERSATION = 4  # LISTENING: User is speaking, slow amber breathing
    STATE_ERROR = 5
    STATE_SPEAKING = 6      # SPEAKING: Agent speaking, white audio-reactive
    STATE_THINKING = 7      # THINKING: User paused, agent preparing response, fast amber pulse
    STATE_WIFI_SETUP = 8    # WIFI_SETUP: Device in WiFi setup mode, ready for configuration
    STATE_ATTEMPTING_CONNECTION = 9  # ATTEMPTING_CONNECTION: Attempting WiFi/pairing connection
    STATE_MUSIC = 10  # MUSIC: Music visualization mode - LEDs react to music playback
    
    # Color palette (RGB, 0-255)
    # Colors are applied with brightness multipliers for visibility
    COLORS = {
        'boot': (244, 162, 97),         # Soft amber - warm startup
        'idle': (255, 255, 255),        # White - clear presence
        'wake_word_amber': (255, 180, 20),  # Amber/Gold - visible through red tinted glass
        'wake_word_orange': (255, 120, 0),  # Orange - visible through red tinted glass
        'wake_word_white': (255, 228, 181),  # Warm white - visible through red tinted glass
        'conversation': (255, 180, 20),  # Amber/Gold - warm listening state
        'speaking': (255, 255, 255),     # White - clear speaking state
        'error': (255, 107, 107),       # Soft red - non-alarming indicator
        'off': (0, 0, 0),               # No light
        # Music visualization colors (visible through red tinted glass)
        'music_1': (255, 100, 0),       # Orange
        'music_2': (255, 50, 150),      # Pink/Magenta
        'music_3': (255, 200, 50),      # Gold
    }
    
    # -------------------------------------------------------------------------
    # Configuration Constants
    # -------------------------------------------------------------------------
    
    # Audio timeout: how long to wait after last audio chunk before returning to CONVERSATION
    # This handles the case where agent stops speaking (no more audio chunks arrive)
    AUDIO_TIMEOUT_MS = 400  # 400ms - quick but not too twitchy
    
    # Silence threshold: RMS value below which we consider audio "silent"
    # ElevenLabs TTS audio has baseline RMS ~3000-5000, so this catches true pauses
    SILENCE_THRESHOLD = 2500
    
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
        
        # Audio timeout tracking for SPEAKING -> CONVERSATION transition
        # When audio stops arriving, we need to detect that and return to CONVERSATION
        self._last_audio_time = None  # Timestamp of last audio chunk received
        self._audio_timeout_task = None  # Task that monitors for audio timeout
        
        if not self.enabled:
            return
        
        # Try to import pixel_ring library for ReSpeaker
        try:
            from pixel_ring import pixel_ring
            self.pixel_ring = pixel_ring
            self.pixel_ring.off()
            print("✓ LED controller initialized")
        except ImportError:
            print("⚠ pixel_ring library not available - LED control disabled")
            print("  Install with: pip install pixel-ring")
            self.enabled = False
        except (FileNotFoundError, PermissionError, OSError) as e:
            print("⚠ LED hardware not detected - LED control disabled")
            print(f"  Reason: {e}")
            print("  Note: LED control requires ReSpeaker with LED ring hardware")
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
        
        # Track previous state for cleanup
        previous_state = self.current_state
        self.current_state = state
        
        # Log state change with expected pattern
        state_names = {
            self.STATE_OFF: "OFF (no lights)",
            self.STATE_BOOT: "BOOT (soft amber pulse)",
            self.STATE_IDLE: "IDLE (white breathing 20-100%)",
            self.STATE_WAKE_WORD_DETECTED: "WAKE_WORD_DETECTED (100% brightness burst)",
            self.STATE_CONVERSATION: "LISTENING (slow amber breathing, user speaking)",
            self.STATE_THINKING: "THINKING (fast amber pulse, preparing response)",
            self.STATE_SPEAKING: "SPEAKING (white audio-reactive, agent talking)",
            self.STATE_ERROR: "ERROR (soft red blink)",
            self.STATE_WIFI_SETUP: "WIFI_SETUP (soft amber slow blink 60%, ready for configuration)",
            self.STATE_ATTEMPTING_CONNECTION: "ATTEMPTING_CONNECTION (soft amber fast blink 60%, connecting)",
            self.STATE_MUSIC: "MUSIC (audio-reactive music visualization)"
        }
        if Config.SHOW_LED_STATE_LOGS:
            print(f"💡 LED: {state_names.get(state, f'UNKNOWN({state})')}")
        
        # Stop any running animation when changing states
        self._stop_animation()
        
        # Clean up audio timeout tracking when leaving SPEAKING state
        if previous_state == self.STATE_SPEAKING:
            self._stop_audio_timeout()
            self._last_audio_time = None
        
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
            # LISTENING: Slow amber breathing while user speaks
            self._start_animation(self._conversation_pulse_loop)
        
        elif state == self.STATE_THINKING:
            # THINKING: Fast amber pulse while agent prepares response
            self._start_animation(self._thinking_pulse_loop)
        
        elif state == self.STATE_SPEAKING:
            # SPEAKING: Audio-reactive white - no animation loop needed
            # LEDs will be updated directly by update_speaking_leds()
            pass
        
        elif state == self.STATE_ERROR:
            # Soft red slow blink (2s on/off)
            self._start_animation(self._error_blink_loop)
        
        elif state == self.STATE_WIFI_SETUP:
            # WIFI_SETUP: Soft amber slow blink (2s on/off, 60% brightness)
            self._start_animation(self._wifi_setup_blink_loop)
        
        elif state == self.STATE_ATTEMPTING_CONNECTION:
            # ATTEMPTING_CONNECTION: Soft amber fast blink (0.5s on/off, 60% brightness)
            self._start_animation(self._attempting_connection_blink_loop)
        
        elif state == self.STATE_MUSIC:
            # MUSIC: Audio-reactive visualization - no animation loop needed
            # LEDs will be updated directly by update_music_leds()
            # Start with a base color to show music mode is active
            self._start_animation(self._music_idle_loop)
    
    # -------------------------------------------------------------------------
    # Animation Control Methods
    # -------------------------------------------------------------------------
    
    def _stop_animation(self):
        """Stop any running animation task"""
        if self.animation_task and not self.animation_task.done():
            self.animation_task.cancel()
            self.animation_task = None
    
    def _stop_audio_timeout(self):
        """Stop the audio timeout monitoring task"""
        if self._audio_timeout_task and not self._audio_timeout_task.done():
            self._audio_timeout_task.cancel()
            self._audio_timeout_task = None
    
    def _start_audio_timeout_monitor(self):
        """
        Start monitoring for audio timeout.
        Called when entering SPEAKING state to detect when audio stops arriving.
        """
        # Don't start if already running
        if self._audio_timeout_task and not self._audio_timeout_task.done():
            return
        
        try:
            loop = asyncio.get_event_loop()
            self._audio_timeout_task = loop.create_task(self._audio_timeout_loop())
        except RuntimeError:
            # No event loop - shouldn't happen in normal operation
            pass
    
    async def _audio_timeout_loop(self):
        """
        Monitor for audio timeout while in SPEAKING state.
        
        When agent stops speaking, audio chunks stop arriving. This loop detects
        that gap and transitions back to CONVERSATION (listening) state.
        
        This is the PRIMARY mechanism for SPEAKING -> CONVERSATION transition.
        """
        CHECK_INTERVAL_MS = 100  # Check every 100ms
        
        try:
            while self.current_state == self.STATE_SPEAKING:
                await asyncio.sleep(CHECK_INTERVAL_MS / 1000.0)
                
                # Check if audio has timed out
                if self._last_audio_time is not None:
                    elapsed_ms = (time.time() - self._last_audio_time) * 1000
                    
                    if elapsed_ms > self.AUDIO_TIMEOUT_MS:
                        # No audio for AUDIO_TIMEOUT_MS -> agent stopped speaking
                        if Config.SHOW_LED_STATE_LOGS:
                            print(f"💡 LED: audio timeout ({elapsed_ms:.0f}ms) -> returning to CONVERSATION")
                        self.set_state(self.STATE_CONVERSATION)
                        break
                        
        except asyncio.CancelledError:
            pass  # Normal cancellation when state changes
    
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
        White breathing effect for IDLE state (20-100% brightness).
        Clear, visible presence indicating device is ready.
        """
        if not self.enabled or not self.pixel_ring:
            return
        
        CYCLE_SECONDS = 3.0  # 4 second breathing cycle (calm, steady)
        UPDATE_INTERVAL = 0.1  # 50ms updates
        MIN_BRIGHTNESS = 0.2  # 20% minimum
        MAX_BRIGHTNESS = 1  # 100% maximum
        
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
        Very visible color burst for wake word detection at 100% brightness.
        Uses amber/gold, orange, and white colors visible through red tinted glass.
        Creates an exciting, attention-grabbing visual signal.
        """
        if not self.enabled or not self.pixel_ring:
            return
        
        BURST_DURATION = 1.5  # 1.5 seconds of burst animation
        UPDATE_INTERVAL = 0.08  # 80ms updates (fast, energetic)
        BRIGHTNESS = 1.0  # 100% brightness for maximum visibility
        
        # Color sequence for burst - visible through red tinted glass
        colors = [
            self.COLORS['wake_word_amber'],   # Amber/Gold (255, 180, 20)
            self.COLORS['wake_word_orange'],   # Orange (255, 120, 0)
            self.COLORS['wake_word_white'],   # Warm white (255, 228, 181)
        ]
        
        start_time = time.time()
        color_index = 0
        
        try:
            while (time.time() - start_time) < BURST_DURATION:
                elapsed = time.time() - start_time
                
                # Cycle through colors rapidly at full brightness
                current_color = colors[color_index % len(colors)]
                color = self._rgb_to_int_with_brightness(current_color, BRIGHTNESS)
                self.pixel_ring.mono(color)
                
                # Change color every 2 updates for rapid color cycling
                if int(elapsed / UPDATE_INTERVAL) % 2 == 0:
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
        Pulsating amber/gold during conversation when listening (0-70% brightness).
        Faster pulse rate (1.5s cycle) compared to idle for more active feel.
        Indicates device is engaged in conversation but not currently speaking.
        Warm, inviting color that shows the device is ready to listen.
        """
        if not self.enabled or not self.pixel_ring:
            return
        
        CYCLE_SECONDS = 1.5  # 1.5 second pulse (much faster than idle, shows activity)
        UPDATE_INTERVAL = 0.05  # 50ms updates
        MIN_BRIGHTNESS = 0.0  # 0% minimum (dims to black)
        MAX_BRIGHTNESS = 0.7  # 70% maximum
        
        base_color = self.COLORS['conversation']  # Amber/Gold (255, 180, 20)
        start_time = time.time()
        
        # Diagnostic: Log when conversation pulse loop starts
        if Config.SHOW_LED_STATE_LOGS:
            print("💡 LED diag: starting CONVERSATION pulse loop (0-70%, 1.5s cycle)")
        
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
    
    async def _thinking_pulse_loop(self):
        """
        Fast amber pulse during THINKING state (0-70% brightness, 0.5s cycle).
        
        Indicates the agent is preparing a response after user stopped speaking.
        Faster than CONVERSATION pulse to show "processing" activity.
        """
        if not self.enabled or not self.pixel_ring:
            return
        
        CYCLE_SECONDS = 0.5   # Fast 0.5s cycle (3x faster than conversation)
        UPDATE_INTERVAL = 0.03  # 30ms updates for smooth fast animation
        MIN_BRIGHTNESS = 0.15  # 15% minimum - keeps light "alive", avoids strobe effect
        MAX_BRIGHTNESS = 0.7   # 70% maximum
        
        base_color = self.COLORS['conversation']  # Same amber/gold color
        start_time = time.time()
        
        try:
            while self.current_state == self.STATE_THINKING:
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
            print(f"  ⚠ LED thinking animation error: {e}")
    
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
    
    async def _wifi_setup_blink_loop(self):
        """
        Soft amber slow blink for WiFi setup mode (2s on/off, 60% brightness).
        Indicates device is in WiFi setup mode with hotspot active and ready for configuration.
        """
        if not self.enabled or not self.pixel_ring:
            return
        
        BLINK_INTERVAL = 2.0  # 2 seconds on, 2 seconds off
        BRIGHTNESS = 0.6  # 60% brightness
        
        base_color = self.COLORS['boot']  # Soft amber (244, 162, 97)
        color = self._rgb_to_int_with_brightness(base_color, BRIGHTNESS)
        
        try:
            while self.current_state == self.STATE_WIFI_SETUP:
                # Turn on
                self.pixel_ring.mono(color)
                await asyncio.sleep(BLINK_INTERVAL)
                
                # Turn off
                if self.current_state == self.STATE_WIFI_SETUP:  # Check state hasn't changed
                    self.pixel_ring.off()
                    await asyncio.sleep(BLINK_INTERVAL)
                
        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"  ⚠ LED WiFi setup animation error: {e}")
    
    async def _attempting_connection_blink_loop(self):
        """
        Soft amber fast blink for connection attempts (0.5s on/off, 60% brightness).
        Indicates device is attempting to connect to WiFi and pair with backend.
        """
        if not self.enabled or not self.pixel_ring:
            return
        
        BLINK_INTERVAL = 0.5  # 0.5 seconds on, 0.5 seconds off (fast blink)
        BRIGHTNESS = 0.6  # 60% brightness
        
        base_color = self.COLORS['boot']  # Soft amber (244, 162, 97)
        color = self._rgb_to_int_with_brightness(base_color, BRIGHTNESS)
        
        try:
            while self.current_state == self.STATE_ATTEMPTING_CONNECTION:
                # Turn on
                self.pixel_ring.mono(color)
                await asyncio.sleep(BLINK_INTERVAL)
                
                # Turn off
                if self.current_state == self.STATE_ATTEMPTING_CONNECTION:  # Check state hasn't changed
                    self.pixel_ring.off()
                    await asyncio.sleep(BLINK_INTERVAL)
                
        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"  ⚠ LED attempting connection animation error: {e}")
    
    async def _music_idle_loop(self):
        """
        Gentle breathing animation when in music mode but no audio detected.
        Uses warm colors to indicate music mode is active.
        """
        if not self.enabled or not self.pixel_ring:
            return
        
        CYCLE_SECONDS = 2.0  # 2 second breathing cycle
        UPDATE_INTERVAL = 0.05  # 50ms updates
        MIN_BRIGHTNESS = 0.1  # 10% minimum
        MAX_BRIGHTNESS = 0.4  # 40% maximum
        
        base_color = self.COLORS['music_1']  # Orange
        start_time = time.time()
        
        try:
            while self.current_state == self.STATE_MUSIC:
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
            print(f"  ⚠ LED music idle animation error: {e}")
    
    def update_music_leds(self, ref_audio):
        """
        Update LEDs based on music energy from Ch5 (playback reference channel).
        
        Creates colorful, energetic visualization that reacts to music playback.
        Color cycles through warm tones visible through the red-tinted ReSpeaker glass.
        
        Args:
            ref_audio: numpy array of int16 audio samples from Ch5 (playback loopback)
        """
        # Skip if LEDs are disabled or unavailable
        if not self.enabled or not self.pixel_ring:
            return
        
        # Only process during music state
        if self.current_state != self.STATE_MUSIC:
            return
        
        # Calculate RMS energy from reference audio
        rms = np.sqrt(np.mean(ref_audio.astype(float) ** 2))
        
        # Skip if silence/very quiet (no music playing or between tracks)
        # Music typically has higher baseline than speech
        if rms < 800:
            # Let the idle animation handle quiet periods
            return
        
        # Stop idle animation when music is playing (we're handling LEDs directly)
        if self.animation_task and not self.animation_task.done():
            self.animation_task.cancel()
            self.animation_task = None
        
        # -------------------------------------------------------------------------
        # Calculate brightness from music energy
        # -------------------------------------------------------------------------
        
        # Normalize: music streams typically peak around 20000-30000 RMS
        normalized_energy = min(rms / 25000.0, 1.0)
        
        # Apply gamma curve for punchier visual response to beats
        # Lower gamma = more responsive to quiet parts, higher = only react to loud
        normalized_energy = normalized_energy ** 1.2
        
        # Map to brightness range: 15% (base) to 100% (peak)
        brightness = 0.15 + (0.85 * normalized_energy)
        brightness = max(0.15, min(1.0, brightness))
        
        # -------------------------------------------------------------------------
        # Color cycling for fun effect
        # -------------------------------------------------------------------------
        
        # Cycle through colors based on time (creates flowing effect)
        t = time.time()
        color_phase = (t * 1.5) % 3.0  # Complete cycle every 2 seconds
        
        # Blend between colors for smooth transitions
        if color_phase < 1.0:
            # Blend music_1 (orange) -> music_2 (pink)
            blend = color_phase
            c1 = self.COLORS['music_1']
            c2 = self.COLORS['music_2']
        elif color_phase < 2.0:
            # Blend music_2 (pink) -> music_3 (gold)
            blend = color_phase - 1.0
            c1 = self.COLORS['music_2']
            c2 = self.COLORS['music_3']
        else:
            # Blend music_3 (gold) -> music_1 (orange)
            blend = color_phase - 2.0
            c1 = self.COLORS['music_3']
            c2 = self.COLORS['music_1']
        
        # Linear interpolation between colors
        r = int(c1[0] * (1 - blend) + c2[0] * blend)
        g = int(c1[1] * (1 - blend) + c2[1] * blend)
        b = int(c1[2] * (1 - blend) + c2[2] * blend)
        
        # Apply brightness
        r = int(r * brightness)
        g = int(g * brightness)
        b = int(b * brightness)
        
        # Set all LEDs to this color
        color_int = (r << 16) | (g << 8) | b
        self.pixel_ring.mono(color_int)
    
    def update_speaking_leds(self, audio_chunk):
        """
        Update LEDs based on audio amplitude (voice energy).
        
        Called for every audio chunk from the agent. Creates real-time
        audio-reactive LED visualization (white beating with voice).
        
        State transitions:
        - CONVERSATION -> SPEAKING: When audio with sufficient energy arrives
        - SPEAKING -> CONVERSATION: Via audio timeout (see _audio_timeout_loop)
        
        Args:
            audio_chunk: numpy array of int16 audio samples from the agent
        """
        # Skip if LEDs are disabled or unavailable
        if not self.enabled or not self.pixel_ring:
            return
        
        # Only process during conversation, thinking, or speaking states
        # Prevents LED flashes from buffered audio during other states
        if self.current_state not in [self.STATE_CONVERSATION, self.STATE_THINKING, self.STATE_SPEAKING]:
            return
        
        # Always update timestamp when we receive audio (for timeout detection)
        self._last_audio_time = time.time()
        
        # Calculate RMS (Root Mean Square) amplitude for voice energy
        rms = np.sqrt(np.mean(audio_chunk.astype(float) ** 2))
        
        # Check if audio is below silence threshold
        # This is a SECONDARY check - primary transition is via audio timeout
        if rms < self.SILENCE_THRESHOLD:
            # Low energy audio chunk - don't update LEDs, let timeout handle transition
            return
        
        # Audio has sufficient energy - transition to SPEAKING if needed
        if self.current_state != self.STATE_SPEAKING:
            self.set_state(self.STATE_SPEAKING)
            # Start timeout monitor to detect when agent stops speaking
            self._start_audio_timeout_monitor()
        
        # -------------------------------------------------------------------------
        # Calculate brightness from audio energy
        # -------------------------------------------------------------------------
        
        # Normalize: typical TTS speech peaks around 15000-20000 RMS
        normalized_energy = min(rms / 20000.0, 1.0)
        
        # Apply gamma curve for punchier visual response
        # Makes quiet sounds dimmer and loud sounds more dramatic
        normalized_energy = normalized_energy ** 1.5
        
        # Map to brightness range: 20% (base) to 100% (peak)
        brightness = 0.2 + (0.8 * normalized_energy)
        brightness = max(0.2, min(1.0, brightness))
        
        # Apply white color at calculated brightness
        color = self._rgb_to_int_with_brightness(self.COLORS['speaking'], brightness)
        self.pixel_ring.mono(color)
    
    def cleanup(self):
        """Turn off LEDs during shutdown"""
        # Stop any running tasks
        self._stop_animation()
        self._stop_audio_timeout()
        
        # Reset tracking state
        self._last_audio_time = None
        
        # Turn off LEDs
        if self.enabled and self.pixel_ring:
            self.pixel_ring.off()

