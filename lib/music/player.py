"""
Music Player

Streams internet radio using mpv subprocess.
Supports genre-based and query-based station selection via StationRegistry.

Usage:
    player = MusicPlayer()
    player.play_genre("jazz")      # Play jazz station
    player.play_query("BBC Radio") # Search for station
    player.play_default()          # Play any station
    player.stop()
    player.pause()
    player.resume()
    player.set_volume(50)
"""

import subprocess
import os
import signal
import sys
import time
from typing import Optional

from lib.config import Config
from lib.music.stations import StationRegistry, Station


# Default fallback URL (used only if registry fails completely)
DEFAULT_STREAM_URL = "https://ice2.somafm.com/groovesalad-128-mp3"


class MusicPlayer:
    """
    Music player that streams internet radio using mpv.
    
    Uses StationRegistry to find stations by genre or search query.
    Plays audio via mpv subprocess for reliable streaming.
    """
    
    def __init__(self, speaker_device_index: Optional[int] = None):
        """
        Initialize music player.
        
        Args:
            speaker_device_index: ALSA device index for audio output (Linux only)
        """
        self.speaker_device_index = speaker_device_index
        self.process: Optional[subprocess.Popen] = None
        self.is_playing = False
        self.is_paused = False
        self.current_station: Optional[Station] = None
        self.logger = Config.LOGGER
        self._registry = StationRegistry()
        self._volume = 100  # Current volume level (0-100)
    
    # =========================================================================
    # HIGH-LEVEL PLAYBACK METHODS (use these)
    # =========================================================================
    
    def play_genre(self, genre: str) -> bool:
        """
        Play a station matching the genre (e.g., 'jazz', 'rock', 'classical').
        
        Returns:
            True if playback started, False otherwise
        """
        station = self._registry.find_by_genre(genre)
        if station:
            print(f"🎵 Genre '{genre}' → {station.name} ({station.source})")
            self.current_station = station
            return self.start(station.url)
        return self._play_fallback()
    
    def play_query(self, query: str) -> bool:
        """
        Play a station matching the search query (e.g., 'Frank Sinatra', 'BBC Radio').
        
        Returns:
            True if playback started, False otherwise
        """
        station = self._registry.find_by_query(query)
        if station:
            print(f"🎵 Query '{query}' → {station.name} ({station.source})")
            self.current_station = station
            return self.start(station.url)
        return self._play_fallback()
    
    def play_default(self) -> bool:
        """
        Play a default station (for generic 'Play music' requests).
        
        Returns:
            True if playback started, False otherwise
        """
        station = self._registry.get_default()
        print(f"🎵 Default → {station.name} ({station.source})")
        self.current_station = station
        return self.start(station.url)
    
    def _play_fallback(self) -> bool:
        """Play hardcoded fallback if registry fails"""
        print("⚠️  No station found, using fallback")
        self.current_station = None
        return self.start(DEFAULT_STREAM_URL)
    
    # =========================================================================
    # LOW-LEVEL PLAYBACK (internal)
    # =========================================================================
    
    def start(self, stream_url: str) -> bool:
        """
        Start playing the audio stream (low-level).
        
        Use play_genre(), play_query(), or play_default() instead.
        
        Args:
            stream_url: URL of the audio stream to play
            
        Returns:
            True if playback started successfully
        """
        if self.is_playing:
            print("⚠️  Music is already playing")
            return True
        
        try:
            # Build mpv command with IPC socket for control
            cmd = [
                "mpv",
                "--no-video",       # Audio only
                "--really-quiet",   # Suppress output
                "--no-terminal",    # No terminal input
                f"--volume={self._volume}",
                "--input-ipc-server=/tmp/mpv-music-socket",  # IPC for pause/resume/volume
                stream_url
            ]
            
            # ALSA device selection (Linux only)
            # Use 'alsa/default' which routes through /etc/asound.conf
            # The ALSA config now uses dmix for software mixing, allowing multiple clients
            # This allows both AudioManager and mpv to play audio simultaneously
            if sys.platform == "linux":
                cmd.insert(-1, "--audio-device=alsa/default")
            
            print(f"   Stream: {stream_url}")
            
            # Start mpv process in new process group for clean termination
            # Don't suppress stderr initially - we need to see errors
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                stdin=subprocess.DEVNULL,
                preexec_fn=os.setsid if os.name != 'nt' else None
            )
            
            # Brief wait to check for immediate failure
            time.sleep(0.5)
            
            # Check if process is still running
            if self.process.poll() is not None:
                # Process exited - read stderr to see what went wrong
                stderr = ""
                try:
                    if self.process.stderr:
                        stderr = self.process.stderr.read().decode('utf-8', errors='ignore')
                except Exception as e:
                    stderr = f"Could not read stderr: {e}"
                
                print(f"✗ mpv exited with code {self.process.returncode}")
                if stderr:
                    print(f"   Error: {stderr[:500]}")  # Show more of the error
                else:
                    print(f"   No error output captured")
                self.process = None
                return False
            
            # Process is still running - check again after a bit more time
            # Sometimes mpv takes a moment to initialize
            time.sleep(0.5)
            if self.process.poll() is not None:
                stderr = ""
                try:
                    if self.process.stderr:
                        stderr = self.process.stderr.read().decode('utf-8', errors='ignore')
                except Exception as e:
                    stderr = f"Could not read stderr: {e}"
                
                print(f"✗ mpv exited after initialization (code {self.process.returncode})")
                if stderr:
                    print(f"   Error: {stderr[:500]}")
                self.process = None
                return False
            
            self.is_playing = True
            self.is_paused = False
            
            if self.logger:
                self.logger.info(
                    "music_playback_started",
                    extra={
                        "stream_url": stream_url,
                        "station_name": self.current_station.name if self.current_station else None,
                        "station_source": self.current_station.source if self.current_station else None,
                        "pid": self.process.pid,
                        "device_id": Config.DEVICE_ID
                    }
                )
            
            print(f"✓ Playing (PID: {self.process.pid})")
            return True
            
        except FileNotFoundError:
            print("✗ mpv not found. Install: brew install mpv (Mac) or sudo apt install mpv (Linux)")
            if self.logger:
                self.logger.error("music_player_mpv_not_found", extra={"device_id": Config.DEVICE_ID})
            return False
            
        except Exception as e:
            print(f"✗ Failed to start: {e}")
            if self.logger:
                self.logger.error("music_playback_start_failed", extra={"error": str(e), "device_id": Config.DEVICE_ID})
            return False
    
    def stop(self) -> bool:
        """
        Stop music playback.
        
        Returns:
            True if stopped, False if nothing was playing
        """
        if not self.is_playing or self.process is None:
            print("ℹ️  No music playing")
            return False
        
        try:
            print("🛑 Stopping music...")
            
            # Kill process group (mpv and children)
            os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
            
            try:
                self.process.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                print("⚠️  Force killing")
                os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
                self.process.wait(timeout=1.0)
            
            if self.logger:
                self.logger.info("music_playback_stopped", extra={"device_id": Config.DEVICE_ID})
            
            print("✓ Stopped")
            
        except ProcessLookupError:
            print("ℹ️  Process already terminated")
            
        except Exception as e:
            print(f"⚠️  Error stopping: {e}")
            if self.logger:
                self.logger.error("music_playback_stop_error", extra={"error": str(e), "device_id": Config.DEVICE_ID})
        
        finally:
            self.process = None
            self.is_playing = False
            self.is_paused = False
            self.current_station = None
            # Clean up IPC socket
            try:
                os.remove("/tmp/mpv-music-socket")
            except:
                pass
        
        return True
    
    def pause(self) -> bool:
        """
        Pause music playback.
        
        Returns:
            True if paused, False if not playing or already paused
        """
        if not self.is_playing or self.is_paused:
            return False
        
        try:
            self._send_mpv_command('{"command": ["set_property", "pause", true]}')
            self.is_paused = True
            print("⏸️  Paused")
            if self.logger:
                self.logger.info("music_playback_paused", extra={"device_id": Config.DEVICE_ID})
            return True
        except Exception as e:
            print(f"⚠️  Error pausing: {e}")
            return False
    
    def resume(self) -> bool:
        """
        Resume paused music playback.
        
        Returns:
            True if resumed, False if not paused
        """
        if not self.is_playing or not self.is_paused:
            return False
        
        try:
            self._send_mpv_command('{"command": ["set_property", "pause", false]}')
            self.is_paused = False
            print("▶️  Resumed")
            if self.logger:
                self.logger.info("music_playback_resumed", extra={"device_id": Config.DEVICE_ID})
            return True
        except Exception as e:
            print(f"⚠️  Error resuming: {e}")
            return False
    
    def volume_up(self, step: int = 10) -> bool:
        """
        Increase volume by step amount.
        
        Args:
            step: Volume increase amount (default 10)
            
        Returns:
            True if volume changed, False otherwise
        """
        new_volume = min(100, self._volume + step)
        return self.set_volume(new_volume)
    
    def volume_down(self, step: int = 10) -> bool:
        """
        Decrease volume by step amount.
        
        Args:
            step: Volume decrease amount (default 10)
            
        Returns:
            True if volume changed, False otherwise
        """
        new_volume = max(0, self._volume - step)
        return self.set_volume(new_volume)
    
    def set_volume(self, volume: int) -> bool:
        """
        Set volume level.
        
        Args:
            volume: Volume level (0-100)
            
        Returns:
            True if volume set, False otherwise
        """
        if not self.is_playing:
            self._volume = max(0, min(100, volume))
            return True
        
        try:
            volume = max(0, min(100, volume))
            self._send_mpv_command(f'{{"command": ["set_property", "volume", {volume}]}}')
            self._volume = volume
            print(f"🔊 Volume: {volume}%")
            if self.logger:
                self.logger.info("music_volume_changed", extra={"volume": volume, "device_id": Config.DEVICE_ID})
            return True
        except Exception as e:
            print(f"⚠️  Error setting volume: {e}")
            return False
    
    def _send_mpv_command(self, command: str):
        """Send command to mpv via IPC socket."""
        import socket
        
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            sock.connect("/tmp/mpv-music-socket")
            sock.sendall((command + "\n").encode())
        finally:
            sock.close()
    
    def is_active(self) -> bool:
        """Check if music is currently playing (not stopped)"""
        if not self.is_playing or self.process is None:
            return False
        
        if self.process.poll() is not None:
            self.is_playing = False
            self.is_paused = False
            self.process = None
            self.current_station = None
            return False
        
        return True
    
    def get_current_station(self) -> Optional[Station]:
        """Get currently playing station info"""
        return self.current_station if self.is_active() else None
    
    def cleanup(self):
        """Clean up resources (call on shutdown)"""
        self.stop()
