#!/usr/bin/env python3
"""
MR60FDA1 Radar Sensor Test Script

Run this to test the radar sensor independently from main.py.
Shows real-time presence, movement, and fall detection.

Usage:
    python scripts/test_radar.py
    python scripts/test_radar.py --port /dev/ttyUSB0
"""

import sys
import os
import time
import argparse

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.radar_sensor import MR60FDA1Sensor, PresenceState, FallState


# ANSI colors for terminal output
class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"
    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_YELLOW = "\033[43m"


def clear_screen():
    """Clear terminal screen."""
    os.system('clear' if os.name != 'nt' else 'cls')


def presence_to_color(presence: PresenceState) -> str:
    """Get color for presence state."""
    if presence == PresenceState.NONE:
        return Colors.WHITE
    elif presence == PresenceState.STATIONARY:
        return Colors.YELLOW
    else:  # MOVING
        return Colors.GREEN


def presence_to_bar(presence: PresenceState, intensity: int) -> str:
    """Generate ASCII bar for presence/movement."""
    color = presence_to_color(presence)
    bar_len = intensity // 5  # 0-100 -> 0-20 chars
    bar = "█" * bar_len + "░" * (20 - bar_len)
    return f"{color}{bar}{Colors.RESET}"


def fall_state_display(fall_state: FallState) -> str:
    """Get display string for fall state."""
    if fall_state == FallState.NORMAL:
        return f"{Colors.GREEN}✓ Normal{Colors.RESET}"
    elif fall_state == FallState.SUSPECTED:
        return f"{Colors.YELLOW}{Colors.BOLD}⚠ SUSPECTED FALL{Colors.RESET}"
    else:  # CONFIRMED
        return f"{Colors.BG_RED}{Colors.WHITE}{Colors.BOLD}🚨 CONFIRMED FALL 🚨{Colors.RESET}"


def on_presence_change(presence: PresenceState, intensity: int):
    """Callback for presence changes."""
    state_name = presence.name
    print(f"\n{Colors.CYAN}[EVENT]{Colors.RESET} Presence changed: {state_name} (intensity: {intensity})")


def on_fall_detected(fall_state: FallState):
    """Callback for fall detection."""
    if fall_state == FallState.SUSPECTED:
        print(f"\n{Colors.YELLOW}{Colors.BOLD}[ALERT] ⚠️  Suspected fall detected!{Colors.RESET}")
    else:
        print(f"\n{Colors.BG_RED}{Colors.WHITE}{Colors.BOLD}[ALERT] 🚨 CONFIRMED FALL DETECTED! 🚨{Colors.RESET}")


def run_test(port: str = None, refresh_rate: float = 0.5):
    """Run the radar test with live display."""
    print(f"\n{Colors.CYAN}{'='*60}{Colors.RESET}")
    print(f"{Colors.BOLD}  MR60FDA1 mmWave Radar Sensor Test{Colors.RESET}")
    print(f"{Colors.CYAN}{'='*60}{Colors.RESET}\n")
    
    # Initialize sensor
    sensor = MR60FDA1Sensor(
        port=port,
        on_presence=on_presence_change,
        on_fall=on_fall_detected
    )
    
    if not sensor.start():
        print(f"\n{Colors.RED}✗ Failed to start radar sensor{Colors.RESET}")
        print(f"  Check that the sensor is connected and the port is correct.")
        print(f"  Try: ls /dev/ttyUSB* or ls /dev/ttyACM*")
        return
    
    print(f"{Colors.GREEN}✓ Radar sensor started on {sensor.port}{Colors.RESET}")
    print(f"\nPress Ctrl+C to exit\n")
    print("-" * 60)
    
    try:
        last_display_time = 0
        while True:
            time.sleep(0.1)
            
            # Update display periodically
            now = time.time()
            if now - last_display_time >= refresh_rate:
                last_display_time = now
                reading = sensor.get_reading()
                
                # Build display
                presence_name = reading.presence.name.ljust(10)
                presence_color = presence_to_color(reading.presence)
                movement_bar = presence_to_bar(reading.presence, reading.movement_intensity)
                fall_display = fall_state_display(reading.fall_state)
                
                # Format stationary duration
                stat_mins = reading.stationary_duration // 60
                stat_secs = reading.stationary_duration % 60
                stat_time = f"{stat_mins}m {stat_secs}s" if stat_mins > 0 else f"{stat_secs}s"
                
                # Stats
                stats = sensor.get_stats()
                
                # Print status line (overwrites previous)
                print(f"\r{Colors.BOLD}Presence:{Colors.RESET} {presence_color}{presence_name}{Colors.RESET} "
                      f"| {Colors.BOLD}Movement:{Colors.RESET} {movement_bar} {reading.movement_intensity:3d}% "
                      f"| {Colors.BOLD}Stationary:{Colors.RESET} {stat_time:>8s} "
                      f"| {fall_display} "
                      f"| Frames: {stats['frames_received']}", end="", flush=True)
                
    except KeyboardInterrupt:
        print(f"\n\n{Colors.YELLOW}Stopping...{Colors.RESET}")
    finally:
        sensor.stop()
        stats = sensor.get_stats()
        print(f"\n{Colors.CYAN}{'='*60}{Colors.RESET}")
        print(f"Session Stats:")
        print(f"  Frames received: {stats['frames_received']}")
        print(f"  Parse errors: {stats['parse_errors']}")
        print(f"{Colors.CYAN}{'='*60}{Colors.RESET}\n")


def main():
    parser = argparse.ArgumentParser(description='Test MR60FDA1 Radar Sensor')
    parser.add_argument('--port', '-p', type=str, default=None,
                        help='Serial port (e.g., /dev/ttyUSB0). Auto-detects if not specified.')
    parser.add_argument('--refresh', '-r', type=float, default=0.5,
                        help='Display refresh rate in seconds (default: 0.5)')
    args = parser.parse_args()
    
    run_test(port=args.port, refresh_rate=args.refresh)


if __name__ == "__main__":
    main()

