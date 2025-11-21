"""
LED Controller for ReSpeaker visual feedback.

Manages ReSpeaker LED ring with various states and animations.
"""

import asyncio
import time
import math


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
        'conversation': (0, 255, 0),    # Green - active engagement
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
            self.STATE_CONVERSATION: "CONVERSATION (pulsating green 20-50%)",
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
        Pulsating green during active conversation (20-50% brightness).
        Indicates device is engaged in conversation with user.
        """
        if not self.enabled or not self.pixel_ring:
            return
        
        CYCLE_SECONDS = 2.0  # 2 second pulse (faster than idle, shows activity)
        UPDATE_INTERVAL = 0.05  # 50ms updates
        MIN_BRIGHTNESS = 0.2  # 20% minimum
        MAX_BRIGHTNESS = 0.5  # 50% maximum
        
        base_color = self.COLORS['conversation']  # Green
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

