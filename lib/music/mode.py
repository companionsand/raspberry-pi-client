"""
Music Mode Controller

Handles music playback mode with voice command control:
- Starts the music player (mpv)
- Listens for voice commands: stop, pause, resume, volume up, volume down
- Returns to wake word mode when "stop" is detected

Also provides radio cache management for faster station lookups.
"""

import asyncio
import json
import os
from typing import TYPE_CHECKING, Optional

from lib.config import Config

from .command_detector import MusicCommand, VoiceCommandDetector
from .player import MusicPlayer

if TYPE_CHECKING:
    from lib.audio.manager import AudioManager


# Cache configuration
CACHE_FILE = os.path.expanduser("~/.kin_radio_cache.json")
DEFAULT_GENRES = [
    "jazz", "classical", "rock", "pop", "country", 
    "blues", "electronic", "ambient", "folk", "soul"
]
STATIONS_PER_GENRE = 10
VERIFY_TIMEOUT = 2.0


class MusicModeController:
    """
    Controller for music playback mode.
    
    Manages the music player and voice command detection,
    handling the full music mode lifecycle.
    """
    
    def __init__(
        self,
        audio_manager: "AudioManager",
        speaker_device_index: Optional[int] = None,
        led_controller=None,
        logger=None
    ):
        """
        Initialize the music mode controller.
        
        Args:
            audio_manager: AudioManager for voice command detection (with AEC)
            speaker_device_index: ALSA device index for speaker (for mpv)
            led_controller: Optional LED controller for visual feedback
            logger: Optional logger for structured logging
        """
        self._audio_manager = audio_manager
        self.speaker_device_index = speaker_device_index
        self.led_controller = led_controller
        self.logger = logger
        
        self._music_player: Optional[MusicPlayer] = None
        self._voice_detector: Optional[VoiceCommandDetector] = None
    
    async def run(self, genre: Optional[str] = None) -> None:
        """
        Run music playback mode until stop command is received.
        
        This method:
        1. Sets LED to music visualization state
        2. Registers callback to sync LEDs with music via Ch5 (reference channel)
        3. Starts the music player with requested genre
        4. Starts voice command detector
        5. Handles commands until "stop" is detected
        6. Cleans up resources
        
        Args:
            genre: Optional genre to play (e.g., "jazz", "rock")
        """
        # Music LED callback for reference channel audio
        self._music_led_callback = None
        
        try:
            # Set LED to music mode and register callback for music visualization
            if self.led_controller:
                # Import here to avoid circular dependency
                from lib.audio import LEDController
                self.led_controller.set_state(LEDController.STATE_MUSIC)
                
                # Register callback to receive Ch5 (playback reference) for LED sync
                def music_led_callback(ref_audio):
                    self.led_controller.update_music_leds(ref_audio)
                
                self._music_led_callback = music_led_callback
                self._audio_manager.register_ref_channel_consumer(music_led_callback)
                print("✓ Music LED visualization enabled (synced with Ch5)")
            
            # Start music player
            self._music_player = MusicPlayer(speaker_device_index=self.speaker_device_index)
            
            # Play by genre or default
            if genre:
                success = self._music_player.play_genre(genre)
                if not success:
                    print(f"⚠️ Genre '{genre}' not found, falling back to default")
                    success = self._music_player.play_default()
            else:
                success = self._music_player.play_default()
            
            if not success:
                print("✗ Failed to start music player")
                if self.logger:
                    self.logger.error(
                        "music_player_start_failed",
                        extra={
                            "genre": genre,
                            "user_id": Config.USER_ID,
                            "device_id": Config.DEVICE_ID
                        }
                    )
                return
            
            if self.logger:
                self.logger.info(
                    "music_mode_started",
                    extra={
                        "genre": genre,
                        "station": self._music_player.current_station.name if self._music_player.current_station else None,
                        "user_id": Config.USER_ID,
                        "device_id": Config.DEVICE_ID
                    }
                )
            
            # Define command handler
            def handle_command(command: MusicCommand) -> None:
                """Handle voice commands for music control."""
                if command == MusicCommand.PAUSE:
                    self._music_player.pause()
                elif command == MusicCommand.RESUME:
                    self._music_player.resume()
                elif command == MusicCommand.VOLUME_UP:
                    self._music_player.volume_up()
                elif command == MusicCommand.VOLUME_DOWN:
                    self._music_player.volume_down()
                # STOP is handled by detector returning
            
            # Start voice command detector (uses AudioManager for AEC-processed audio)
            self._voice_detector = VoiceCommandDetector(
                audio_manager=self._audio_manager,
                on_command=handle_command
            )
            
            # Run detector until STOP command
            final_command = await self._voice_detector.start()
            
            if final_command == MusicCommand.STOP:
                print("\n🛑 Stop command detected - exiting music mode")
            else:
                print("\n📻 Music mode ended")
                
        except Exception as e:
            print(f"⚠️  Music mode error: {e}")
            if self.logger:
                self.logger.error(
                    "music_mode_error",
                    extra={
                        "error": str(e),
                        "user_id": Config.USER_ID,
                        "device_id": Config.DEVICE_ID
                    }
                )
        
        finally:
            self._cleanup()
            
            if self.logger:
                self.logger.info(
                    "music_mode_ended",
                    extra={
                        "user_id": Config.USER_ID,
                        "device_id": Config.DEVICE_ID
                    }
                )
            
            print("✓ Returning to wake word mode")
    
    def _cleanup(self) -> None:
        """Clean up music player, voice detector, and LED resources."""
        # Unregister music LED callback
        if self._music_led_callback:
            self._audio_manager.unregister_ref_channel_consumer(self._music_led_callback)
            self._music_led_callback = None
        
        if self._music_player:
            self._music_player.stop()
            self._music_player = None
        
        if self._voice_detector:
            self._voice_detector.stop()
            self._voice_detector = None


async def update_radio_cache_background(logger=None) -> int:
    """
    Update radio station cache in the background.
    
    Fetches stations from Radio Browser API, verifies they're working,
    and saves to ~/.kin_radio_cache.json for faster music playback.
    
    Args:
        logger: Optional logger for structured logging
        
    Returns:
        Total number of verified stations cached
    """
    try:
        from .radio_browser import RadioBrowserClient, verify_stream
        
        print("📻 Updating radio station cache (background)...")
        
        def fetch_cache() -> int:
            client = RadioBrowserClient()
            cache = {}
            
            # Fetch popular stations (skip verification - high-vote = likely working)
            try:
                popular = client.get_top_stations(limit=30)
                cache["popular"] = [s.to_dict() for s in popular[:20]]
            except Exception as e:
                print(f"   ⚠️ Failed to fetch popular stations: {e}")
                cache["popular"] = []
            
            # Fetch by genre with verification
            for genre in DEFAULT_GENRES:
                try:
                    stations = client.search_by_tag(genre, limit=50)
                    verified = []
                    for station in stations:
                        if verify_stream(station.url, timeout=VERIFY_TIMEOUT):
                            verified.append(station.to_dict())
                            if len(verified) >= STATIONS_PER_GENRE:
                                break
                    cache[genre] = verified
                except Exception:
                    cache[genre] = []
            
            # Save cache
            with open(CACHE_FILE, 'w') as f:
                json.dump(cache, f, indent=2)
            
            total = sum(len(v) for v in cache.values())
            return total
        
        loop = asyncio.get_event_loop()
        total = await loop.run_in_executor(None, fetch_cache)
        print(f"   ✓ Radio cache updated: {total} verified stations")
        
        if logger:
            logger.info(
                "radio_cache_updated",
                extra={
                    "total_stations": total,
                    "device_id": Config.DEVICE_ID
                }
            )
        
        return total
        
    except Exception as e:
        print(f"   ⚠️ Radio cache update failed (non-fatal): {e}")
        if logger:
            logger.warning(
                "radio_cache_update_failed",
                extra={
                    "error": str(e),
                    "device_id": Config.DEVICE_ID
                }
            )
        return 0

