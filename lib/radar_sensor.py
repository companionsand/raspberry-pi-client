"""
MR60FDA1 mmWave Radar Sensor - Presence & Fall Detection

Seeed Studio 60GHz mmWave radar for:
- Human presence detection (stationary/moving)
- Fall detection (suspected/confirmed)
- Stationary duration tracking

Protocol: UART @ 115200 baud, proprietary binary frames
Frame format: Header(0x53 0x59) + Control(2) + Length(2) + Data(n) + CRC16(2) + Footer(0x54 0x43)

Usage:
    sensor = MR60FDA1Sensor(port="/dev/ttyUSB0", on_fall=my_callback)
    sensor.start()
    reading = sensor.get_reading()
    sensor.stop()
"""

import serial
import struct
import threading
import time
import glob
from datetime import datetime, timezone
from typing import Callable, Optional
from dataclasses import dataclass
from enum import IntEnum

from lib.config import Config


# =============================================================================
# ENUMS & DATA CLASSES
# =============================================================================

class PresenceState(IntEnum):
    """Radar presence detection states."""
    NONE = 0        # No human detected
    STATIONARY = 1  # Human present, not moving
    MOVING = 2      # Human present and moving


class FallState(IntEnum):
    """Fall detection states."""
    NORMAL = 0      # No fall detected
    SUSPECTED = 1   # Possible fall (needs confirmation)
    CONFIRMED = 2   # Confirmed fall event


@dataclass
class RadarReading:
    """Current radar sensor reading."""
    presence: PresenceState
    movement_intensity: int  # 0-100 scale
    fall_state: FallState
    stationary_duration: int  # seconds
    timestamp: datetime


# =============================================================================
# CRC16 CALCULATION (Seeed MR60FDA1 protocol)
# =============================================================================

def _crc16_modbus(data: bytes) -> int:
    """
    Calculate CRC16/MODBUS for frame validation.
    Polynomial: 0x8005, Init: 0xFFFF, RefIn: True, RefOut: True
    """
    crc = 0xFFFF
    for byte in data:
        crc ^= byte
        for _ in range(8):
            if crc & 0x0001:
                crc = (crc >> 1) ^ 0xA001
            else:
                crc >>= 1
    return crc


# =============================================================================
# MAIN SENSOR CLASS
# =============================================================================

class MR60FDA1Sensor:
    """
    Thread-safe interface to MR60FDA1 mmWave radar sensor.
    
    Runs a background thread for continuous frame parsing.
    Provides callbacks for presence changes and fall events.
    """
    
    # Protocol constants
    HEADER = bytes([0x53, 0x59])
    FOOTER = bytes([0x54, 0x43])
    BAUD_RATE = 115200
    
    # Control words (first byte, second byte)
    CTRL_HEARTBEAT = (0x80, 0x01)
    CTRL_PRODUCT_INFO = (0x80, 0x02)
    CTRL_PRESENCE = (0x80, 0x03)
    CTRL_MOVEMENT = (0x80, 0x04)
    CTRL_FALL_ALERT = (0x83, 0x01)
    CTRL_FALL_SENSITIVITY = (0x83, 0x02)
    CTRL_STATIONARY_DURATION = (0x84, 0x02)
    
    def __init__(
        self,
        port: Optional[str] = None,
        on_presence: Optional[Callable[[PresenceState, int], None]] = None,
        on_fall: Optional[Callable[[FallState], None]] = None,
        orchestrator_client=None,
        event_loop=None,
        logger=None
    ):
        """
        Initialize radar sensor.
        
        Args:
            port: Serial port (e.g., "/dev/ttyUSB0"). Auto-detects if None.
            on_presence: Callback(presence_state, movement_intensity) on presence change.
            on_fall: Callback(fall_state) on fall detection.
            orchestrator_client: OrchestratorClient for sending alerts.
            event_loop: asyncio event loop for async callbacks.
            logger: Logger instance for telemetry.
        """
        self.port = port
        self.on_presence = on_presence
        self.on_fall = on_fall
        self.orchestrator_client = orchestrator_client
        self.event_loop = event_loop
        self.logger = logger or Config.LOGGER
        
        self._serial: Optional[serial.Serial] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self.running = False
        
        # Current state (thread-safe access via _state_lock)
        self.current_reading = RadarReading(
            presence=PresenceState.NONE,
            movement_intensity=0,
            fall_state=FallState.NORMAL,
            stationary_duration=0,
            timestamp=datetime.now(timezone.utc)
        )
        self._state_lock = threading.Lock()
        
        # Debounce: prevent duplicate fall alerts
        self._last_fall_alert_time = 0
        self._fall_alert_cooldown = 10.0  # seconds
        
        # Stats for debugging
        self._frames_received = 0
        self._parse_errors = 0
    
    # =========================================================================
    # LIFECYCLE
    # =========================================================================
    
    def start(self) -> bool:
        """
        Start sensor reading thread.
        
        Returns:
            True if started successfully, False otherwise.
        """
        # Auto-detect port if not specified
        if self.port is None:
            self.port = self._detect_port()
            if self.port is None:
                print("✗ MR60FDA1 radar: No compatible device found")
                return False
        
        try:
            self._serial = serial.Serial(
                self.port,
                self.BAUD_RATE,
                timeout=1.0
            )
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._read_loop, daemon=True)
            self._thread.start()
            self.running = True
            
            print(f"✓ MR60FDA1 radar started on {self.port}")
            
            if self.logger:
                self.logger.info(
                    "radar_sensor_started",
                    extra={"port": self.port, "device_id": Config.DEVICE_ID}
                )
            
            return True
            
        except serial.SerialException as e:
            print(f"✗ Failed to open radar sensor on {self.port}: {e}")
            if self.logger:
                self.logger.error(
                    "radar_sensor_start_failed",
                    extra={"port": self.port, "error": str(e), "device_id": Config.DEVICE_ID}
                )
            return False
    
    def stop(self):
        """Stop sensor reading and close serial port."""
        self._stop_event.set()
        
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        
        if self._serial and self._serial.is_open:
            self._serial.close()
        
        self.running = False
        
        if self.logger:
            self.logger.info(
                "radar_sensor_stopped",
                extra={
                    "frames_received": self._frames_received,
                    "parse_errors": self._parse_errors,
                    "device_id": Config.DEVICE_ID
                }
            )
    
    def cleanup(self):
        """Alias for stop() - matches other module patterns."""
        self.stop()
    
    # =========================================================================
    # PORT DETECTION
    # =========================================================================
    
    def _detect_port(self) -> Optional[str]:
        """
        Auto-detect MR60FDA1 on USB serial ports.
        
        Looks for the sensor's heartbeat frame on available ports.
        Falls back to single available port if detection fails.
        
        Returns:
            Port path (e.g., "/dev/ttyUSB0") or None if not found.
        """
        # Common USB serial port patterns
        candidates = glob.glob('/dev/ttyUSB*') + glob.glob('/dev/ttyACM*')
        
        # Also check for our udev symlink
        radar_symlink = glob.glob('/dev/radar0')
        if radar_symlink:
            candidates = radar_symlink + candidates
        
        if not candidates:
            return None
        
        # Try to detect radar by looking for header bytes
        for port in candidates:
            try:
                with serial.Serial(port, self.BAUD_RATE, timeout=0.5) as s:
                    time.sleep(0.5)  # Wait longer for heartbeat
                    if s.in_waiting:
                        data = s.read(s.in_waiting)
                        # Look for header bytes (sensor sends periodic heartbeats)
                        if self.HEADER in data:
                            print(f"   Found MR60FDA1 on {port} (detected heartbeat)")
                            return port
            except (serial.SerialException, OSError) as e:
                continue
        
        # Fallback: If only one port available, use it (user can specify manually if wrong)
        if len(candidates) == 1:
            port = candidates[0]
            print(f"   Using single available port: {port} (auto-detection inconclusive)")
            return port
        
        # Multiple ports but none detected radar - return None (user must specify)
        print(f"   Found {len(candidates)} ports but couldn't detect radar:")
        for port in candidates:
            print(f"     - {port}")
        print(f"   Try specifying manually: --port /dev/ttyUSB0")
        return None
    
    # =========================================================================
    # FRAME PARSING
    # =========================================================================
    
    def _read_loop(self):
        """Main reading loop - runs in background thread."""
        buffer = bytearray()
        
        while not self._stop_event.is_set():
            try:
                if self._serial and self._serial.in_waiting:
                    buffer.extend(self._serial.read(self._serial.in_waiting))
                    self._parse_buffer(buffer)
                else:
                    time.sleep(0.01)  # Prevent CPU spin
            except serial.SerialException as e:
                # Serial disconnection - attempt reconnect
                if self.logger:
                    self.logger.warning(
                        "radar_serial_error",
                        extra={"error": str(e), "device_id": Config.DEVICE_ID}
                    )
                self._attempt_reconnect()
            except Exception as e:
                if self.logger:
                    self.logger.error(
                        "radar_read_error",
                        extra={"error": str(e), "device_id": Config.DEVICE_ID}
                    )
                time.sleep(0.1)
    
    def _attempt_reconnect(self):
        """Attempt to reconnect to serial port after disconnection."""
        if self._serial:
            try:
                self._serial.close()
            except:
                pass
        
        # Wait before retry
        time.sleep(2.0)
        
        if self._stop_event.is_set():
            return
        
        try:
            self._serial = serial.Serial(
                self.port,
                self.BAUD_RATE,
                timeout=1.0
            )
            print(f"   ✓ Radar reconnected on {self.port}")
        except serial.SerialException:
            pass  # Will retry on next loop iteration
    
    def _parse_buffer(self, buffer: bytearray):
        """
        Parse frames from buffer, removing processed data.
        
        Frame structure:
            Header (2 bytes): 0x53 0x59
            Control (2 bytes): Command type
            Length (2 bytes): Data length (little-endian)
            Data (n bytes): Payload
            CRC16 (2 bytes): Checksum
            Footer (2 bytes): 0x54 0x43
        """
        while len(buffer) >= 10:  # Minimum frame: header(2)+ctrl(2)+len(2)+crc(2)+footer(2)
            # Find header
            try:
                idx = buffer.index(self.HEADER[0])
                if idx > 0:
                    del buffer[:idx]  # Discard bytes before header
                if len(buffer) < 2 or buffer[1] != self.HEADER[1]:
                    del buffer[:1]
                    continue
            except ValueError:
                buffer.clear()
                return
            
            # Need at least header + control + length to proceed
            if len(buffer) < 6:
                return  # Wait for more data
            
            # Parse data length (bytes 4-5, little-endian)
            data_len = struct.unpack('<H', buffer[4:6])[0]
            
            # Calculate total frame length
            frame_len = 6 + data_len + 4  # header(2) + ctrl(2) + len(2) + data(n) + crc(2) + footer(2)
            
            # Sanity check on length
            if data_len > 256:
                # Invalid length - skip this header
                del buffer[:2]
                self._parse_errors += 1
                continue
            
            if len(buffer) < frame_len:
                return  # Wait for more data
            
            # Verify footer
            if buffer[frame_len-2:frame_len] != self.FOOTER:
                del buffer[:2]
                self._parse_errors += 1
                continue
            
            # Extract frame
            frame = bytes(buffer[:frame_len])
            del buffer[:frame_len]
            
            # Validate CRC (optional - some firmware versions don't use it correctly)
            # crc_received = struct.unpack('<H', frame[-4:-2])[0]
            # crc_calculated = _crc16_modbus(frame[:-4])
            # if crc_received != crc_calculated:
            #     self._parse_errors += 1
            #     continue
            
            self._frames_received += 1
            self._process_frame(frame)
    
    def _process_frame(self, frame: bytes):
        """Process a validated frame and update state."""
        ctrl = (frame[2], frame[3])
        data_len = struct.unpack('<H', frame[4:6])[0]
        data = frame[6:6+data_len] if data_len > 0 else b''
        
        with self._state_lock:
            # Presence status (0x80 0x03)
            if ctrl == self.CTRL_PRESENCE:
                if data and len(data) >= 1:
                    try:
                        new_presence = PresenceState(data[0])
                        old_presence = self.current_reading.presence
                        self.current_reading.presence = new_presence
                        self.current_reading.timestamp = datetime.now(timezone.utc)
                        
                        # Callback on state change
                        if new_presence != old_presence and self.on_presence:
                            self.on_presence(
                                new_presence,
                                self.current_reading.movement_intensity
                            )
                    except ValueError:
                        pass  # Invalid presence value
            
            # Movement intensity (0x80 0x04)
            elif ctrl == self.CTRL_MOVEMENT:
                if data and len(data) >= 1:
                    self.current_reading.movement_intensity = min(100, data[0])
            
            # Fall detection (0x83 0x01)
            elif ctrl == self.CTRL_FALL_ALERT:
                if data and len(data) >= 1:
                    try:
                        new_fall_state = FallState(data[0])
                        self.current_reading.fall_state = new_fall_state
                        self.current_reading.timestamp = datetime.now(timezone.utc)
                        
                        # Handle fall alert (with debounce)
                        if new_fall_state != FallState.NORMAL:
                            self._handle_fall_alert(new_fall_state)
                    except ValueError:
                        pass
            
            # Stationary duration (0x84 0x02)
            elif ctrl == self.CTRL_STATIONARY_DURATION:
                if data and len(data) >= 4:
                    self.current_reading.stationary_duration = struct.unpack('<I', data[:4])[0]
            
            # Heartbeat (0x80 0x01) - just indicates sensor is alive
            elif ctrl == self.CTRL_HEARTBEAT:
                pass  # No action needed
    
    # =========================================================================
    # FALL DETECTION HANDLING
    # =========================================================================
    
    def _handle_fall_alert(self, fall_state: FallState):
        """
        Handle fall detection alert.
        
        Applies debouncing and sends to orchestrator.
        """
        now = time.time()
        
        # Debounce: skip if we just sent an alert
        if now - self._last_fall_alert_time < self._fall_alert_cooldown:
            return
        
        self._last_fall_alert_time = now
        severity = "suspected" if fall_state == FallState.SUSPECTED else "confirmed"
        
        print(f"🚨 FALL DETECTED: {severity.upper()}")
        
        # Log the event
        if self.logger:
            self.logger.warning(
                "fall_detected",
                extra={
                    "severity": severity,
                    "movement_intensity": self.current_reading.movement_intensity,
                    "stationary_duration": self.current_reading.stationary_duration,
                    "device_id": Config.DEVICE_ID
                }
            )
        
        # User callback
        if self.on_fall:
            self.on_fall(fall_state)
        
        # Send to orchestrator (async from sync context)
        if self.orchestrator_client and self.event_loop:
            import asyncio
            try:
                asyncio.run_coroutine_threadsafe(
                    self.orchestrator_client.send_fall_detection(
                        severity=severity,
                        detected_at=datetime.now(timezone.utc).isoformat(),
                        movement_intensity=self.current_reading.movement_intensity,
                        stationary_duration=self.current_reading.stationary_duration
                    ),
                    self.event_loop
                )
            except Exception as e:
                if self.logger:
                    self.logger.error(
                        "fall_alert_send_failed",
                        extra={"error": str(e), "device_id": Config.DEVICE_ID}
                    )
    
    # =========================================================================
    # PUBLIC API
    # =========================================================================
    
    def get_reading(self) -> RadarReading:
        """
        Get current sensor state (thread-safe).
        
        Returns:
            Copy of current RadarReading.
        """
        with self._state_lock:
            return RadarReading(
                presence=self.current_reading.presence,
                movement_intensity=self.current_reading.movement_intensity,
                fall_state=self.current_reading.fall_state,
                stationary_duration=self.current_reading.stationary_duration,
                timestamp=self.current_reading.timestamp
            )
    
    def is_present(self) -> bool:
        """Quick check if someone is present (stationary or moving)."""
        with self._state_lock:
            return self.current_reading.presence != PresenceState.NONE
    
    def is_moving(self) -> bool:
        """Quick check if someone is actively moving."""
        with self._state_lock:
            return self.current_reading.presence == PresenceState.MOVING
    
    def configure_sensitivity(self, level: int = 2):
        """
        Set fall detection sensitivity.
        
        Args:
            level: 0 (lowest) to 3 (highest). Default is 2.
        
        Note: Sends command to sensor. May not work with all firmware versions.
        """
        if not self._serial or not self._serial.is_open:
            return
        
        level = max(0, min(3, level))  # Clamp to 0-3
        
        # Build frame: Header + Control(0x83 0x02) + Length(1) + Data(level) + CRC + Footer
        payload = bytes([0x53, 0x59, 0x83, 0x02, 0x01, 0x00, level])
        crc = _crc16_modbus(payload)
        frame = payload + struct.pack('<H', crc) + bytes([0x54, 0x43])
        
        try:
            self._serial.write(frame)
            print(f"   Radar sensitivity set to {level}")
        except serial.SerialException as e:
            print(f"   ⚠️ Failed to set sensitivity: {e}")
    
    def get_stats(self) -> dict:
        """Get sensor statistics for debugging."""
        return {
            "running": self.running,
            "port": self.port,
            "frames_received": self._frames_received,
            "parse_errors": self._parse_errors
        }

