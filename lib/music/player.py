"""Music player using mpv for streaming audio"""

import subprocess
import os
import signal
from typing import Optional
from lib.config import Config


# Default radio stream URLs
# Note: BBC streams are geo-restricted to UK. Use a VPN or alternative stream outside UK.
# SomaFM Groove Salad is a reliable fallback (ambient/chill music)
DEFAULT_STREAM_URL = "https://ice2.somafm.com/groovesalad-128-mp3"

# BBC Radio 6 Music - works in UK only (may require VPN outside UK)
BBC_RADIO_6_MUSIC_URL = "https://stream.live.vc.bbcmedia.co.uk/bbc_6music"


class MusicPlayer:
    """
    Music player that streams audio using mpv.
    
    Uses mpv subprocess for reliable streaming of internet radio.
    Designed for non-blocking operation with explicit start/stop control.
    """
    
    def __init__(self, speaker_device_index: Optional[int] = None):
        """
        Initialize music player.
        
        Args:
            speaker_device_index: ALSA device index for audio output (None for default)
        """
        self.speaker_device_index = speaker_device_index
        self.process: Optional[subprocess.Popen] = None
        self.is_playing = False
        self.logger = Config.LOGGER
        
    def start(self, stream_url: str = DEFAULT_STREAM_URL) -> bool:
        """
        Start playing the audio stream.
        
        Args:
            stream_url: URL of the audio stream to play
            
        Returns:
            True if playback started successfully, False otherwise
        """
        if self.is_playing:
            print("⚠️  Music is already playing")
            return True
        
        try:
            # Build mpv command
            # --no-video: Audio only (no video window)
            # --really-quiet: Suppress all output
            # --no-terminal: Don't use terminal for input
            # --volume=100: Full volume (system volume controls actual level)
            cmd = [
                "mpv",
                "--no-video",
                "--really-quiet",
                "--no-terminal",
                "--volume=100",
                stream_url
            ]
            
            # Audio device selection
            # On Mac (CoreAudio): Use default output, no special config needed
            # On Linux (ALSA): Optionally specify device
            import sys
            if sys.platform == "linux" and self.speaker_device_index is not None:
                # Use ALSA device on Linux
                cmd.insert(-1, "--audio-device=alsa")
            
            print(f"🎵 Starting music stream: {stream_url}")
            print(f"   Command: {' '.join(cmd)}")
            
            # Start mpv process
            # Capture stderr to detect startup errors
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                stdin=subprocess.DEVNULL,
                preexec_fn=os.setsid if os.name != 'nt' else None  # Create new process group for clean termination (Unix only)
            )
            
            # Wait briefly and check if process is still running
            import time
            time.sleep(0.5)
            
            if self.process.poll() is not None:
                # Process already exited - something went wrong
                stderr_output = self.process.stderr.read().decode('utf-8', errors='ignore') if self.process.stderr else ""
                print(f"✗ mpv exited immediately with code {self.process.returncode}")
                if stderr_output:
                    print(f"   Error: {stderr_output[:200]}")
                self.process = None
                return False
            
            self.is_playing = True
            
            if self.logger:
                self.logger.info(
                    "music_playback_started",
                    extra={
                        "stream_url": stream_url,
                        "pid": self.process.pid,
                        "device_id": Config.DEVICE_ID
                    }
                )
            
            print(f"✓ Music playing (PID: {self.process.pid})")
            print(f"   Stream URL: {stream_url}")
            return True
            
        except FileNotFoundError:
            print("✗ mpv not found. Install with: sudo apt install mpv")
            if self.logger:
                self.logger.error(
                    "music_player_mpv_not_found",
                    extra={"device_id": Config.DEVICE_ID}
                )
            return False
            
        except Exception as e:
            print(f"✗ Failed to start music: {e}")
            if self.logger:
                self.logger.error(
                    "music_playback_start_failed",
                    extra={
                        "error": str(e),
                        "device_id": Config.DEVICE_ID
                    }
                )
            return False
    
    def stop(self) -> bool:
        """
        Stop the music playback.
        
        Returns:
            True if stopped successfully, False if nothing was playing
        """
        if not self.is_playing or self.process is None:
            print("ℹ️  No music is currently playing")
            return False
        
        try:
            print("🛑 Stopping music...")
            
            # Send SIGTERM to the process group (kills mpv and any child processes)
            os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
            
            # Wait for process to terminate (with timeout)
            try:
                self.process.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                # Force kill if it doesn't terminate gracefully
                print("⚠️  Force killing mpv process")
                os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
                self.process.wait(timeout=1.0)
            
            if self.logger:
                self.logger.info(
                    "music_playback_stopped",
                    extra={"device_id": Config.DEVICE_ID}
                )
            
            print("✓ Music stopped")
            
        except ProcessLookupError:
            # Process already terminated
            print("ℹ️  Music process already terminated")
            
        except Exception as e:
            print(f"⚠️  Error stopping music: {e}")
            if self.logger:
                self.logger.error(
                    "music_playback_stop_error",
                    extra={
                        "error": str(e),
                        "device_id": Config.DEVICE_ID
                    }
                )
        
        finally:
            self.process = None
            self.is_playing = False
        
        return True
    
    def is_active(self) -> bool:
        """Check if music is currently playing."""
        if not self.is_playing or self.process is None:
            return False
        
        # Check if process is still running
        poll_result = self.process.poll()
        if poll_result is not None:
            # Process has terminated
            self.is_playing = False
            self.process = None
            return False
        
        return True
    
    def cleanup(self):
        """Clean up resources (call on shutdown)."""
        self.stop()

