"""
Turn Tracker using Silero VAD

Tracks user and agent speech turns independently using Voice Activity Detection.
Maintains separate turn lists that are merged at conversation end.
"""

import os
import time
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Optional
from datetime import datetime

import numpy as np
import onnxruntime as ort


class Speaker(Enum):
    """Speaker type for turns"""
    USER = "user"
    AGENT = "agent"


@dataclass
class Transcript:
    """Represents a transcript event (user speech or agent response text)"""
    speaker: Speaker
    text: str
    timestamp: float  # Unix timestamp when received


@dataclass
class Turn:
    """Represents a single speech turn"""
    speaker: Speaker
    start_time: float  # Unix timestamp
    end_time: Optional[float] = None  # Unix timestamp, None if still speaking
    
    @property
    def duration(self) -> Optional[float]:
        """Duration in seconds, None if turn hasn't ended"""
        if self.end_time is None:
            return None
        return self.end_time - self.start_time
    
    def __repr__(self):
        duration_str = f"{self.duration:.2f}s" if self.duration else "ongoing"
        start_str = datetime.fromtimestamp(self.start_time).strftime("%H:%M:%S.%f")[:-3]
        end_str = datetime.fromtimestamp(self.end_time).strftime("%H:%M:%S.%f")[:-3] if self.end_time else "..."
        return f"Turn({self.speaker.value}, {start_str} → {end_str}, {duration_str})"


class VADState:
    """
    VAD state machine for a single audio source (user or agent).
    
    State machine:
        IDLE → (speech detected) → SPEAKING → (silence > timeout) → IDLE
                                      ↑                   |
                                      |___(speech resumes within debounce)
    """
    
    # Silero VAD requires fixed chunk size of 512 samples at 16kHz
    VAD_CHUNK_SIZE = 512
    
    def __init__(
        self,
        speaker: Speaker,
        vad_session: ort.InferenceSession,
        sample_rate: int = 16000,
        vad_threshold: float = 0.5,
        silence_timeout: float = 2.5,
        min_turn_duration: float = 0.2,
        min_speech_onset: float = 0.15,
        debounce_window: float = 0.5,
    ):
        """
        Initialize VAD state for a speaker.
        
        Args:
            speaker: USER or AGENT
            vad_session: ONNX InferenceSession for Silero VAD
            sample_rate: Audio sample rate (default 16000)
            vad_threshold: Probability threshold for speech detection (0.0-1.0)
            silence_timeout: Seconds of silence before ending a turn
            min_turn_duration: Minimum turn duration in seconds (filter noise)
            min_speech_onset: Minimum continuous speech before starting turn (filter micro-triggers)
            debounce_window: Seconds to wait before confirming turn end
        """
        self.speaker = speaker
        self.vad_session = vad_session
        self.sample_rate = sample_rate
        self.vad_threshold = vad_threshold
        self.silence_timeout = silence_timeout
        self.min_turn_duration = min_turn_duration
        self.min_speech_onset = min_speech_onset
        self.debounce_window = debounce_window
        
        # VAD state tensor for Silero VAD v5: shape (2, 1, 128)
        self._vad_state = np.zeros((2, 1, 128), dtype=np.float32)
        
        # Audio buffer for accumulating variable-length chunks
        # Stores (receive_timestamp, audio_samples) to track actual timing
        self._audio_buffer = np.array([], dtype=np.int16)
        self._audio_buffer_timestamps: List[tuple] = []  # [(timestamp, sample_count), ...]
        self._samples_processed = 0  # Track how many samples we've processed
        
        # Turn tracking state
        self.is_speaking = False
        self.current_turn_start: Optional[float] = None
        self.last_speech_time: Optional[float] = None
        self._speech_onset_time: Optional[float] = None  # When speech first detected (before turn confirmed)
        self._onset_grace_period = 0.2  # Don't reset onset for 200ms of silence (handles network gaps)
        self.turns: List[Turn] = []
        
        # Frame-based onset tracking (more robust than time-based for buffered audio)
        self._consecutive_speech_frames = 0
        self._consecutive_silence_frames = 0
        # Number of consecutive speech frames needed to start a turn (~80ms at 512 samples/chunk @ 16kHz)
        self._min_onset_frames = max(1, int(self.min_speech_onset * self.sample_rate / self.VAD_CHUNK_SIZE))
        # Number of consecutive silence frames before resetting onset (tolerance for blips)
        self._silence_reset_frames = 3  # ~96ms of silence needed to reset onset
        
    def reset_vad_state(self):
        """Reset VAD internal state (call between conversations)"""
        self._vad_state = np.zeros((2, 1, 128), dtype=np.float32)
        self._audio_buffer = np.array([], dtype=np.int16)
        self._audio_buffer_timestamps = []
        self._samples_processed = 0
        self._speech_onset_time = None
        self._consecutive_speech_frames = 0
        self._consecutive_silence_frames = 0
        
    def process_audio(self, audio_data: np.ndarray, receive_time: Optional[float] = None) -> Optional[Turn]:
        """
        Process an audio chunk and update turn state.
        
        Buffers incoming audio and processes in fixed-size chunks (512 samples)
        that Silero VAD requires.
        
        Args:
            audio_data: Audio samples as int16 numpy array (variable length OK)
            receive_time: Optional timestamp when audio was received (default: now)
            
        Returns:
            A completed Turn if one just ended, None otherwise
        """
        if receive_time is None:
            receive_time = time.time()
        
        # Track when this audio chunk was received
        incoming_samples = len(audio_data.flatten())
        buffer_start = len(self._audio_buffer)
        self._audio_buffer_timestamps.append((receive_time, buffer_start, incoming_samples))
        
        # Add incoming audio to buffer
        self._audio_buffer = np.concatenate([self._audio_buffer, audio_data.flatten()])
        
        completed_turn = None
        
        # Process all complete chunks in buffer
        while len(self._audio_buffer) >= self.VAD_CHUNK_SIZE:
            # Calculate timestamp for this chunk based on when its samples were received
            chunk_time = self._get_timestamp_for_sample(self._samples_processed)
            
            # Extract one chunk
            chunk = self._audio_buffer[:self.VAD_CHUNK_SIZE]
            self._audio_buffer = self._audio_buffer[self.VAD_CHUNK_SIZE:]
            self._samples_processed += self.VAD_CHUNK_SIZE
            
            # Process this chunk with correct timestamp
            result = self._process_chunk(chunk, chunk_time)
            if result is not None:
                completed_turn = result
        
        return completed_turn
    
    def _get_timestamp_for_sample(self, sample_index: int) -> float:
        """Get the receive timestamp for a given sample index in the buffer."""
        # Find which timestamp entry this sample belongs to
        for receive_time, start_idx, num_samples in self._audio_buffer_timestamps:
            if start_idx <= sample_index < start_idx + num_samples:
                # Interpolate within the chunk based on sample position
                offset_samples = sample_index - start_idx
                offset_seconds = offset_samples / self.sample_rate
                return receive_time + offset_seconds
        
        # Fallback to current time if not found (shouldn't happen)
        return time.time()
    
    def _process_chunk(self, audio_chunk: np.ndarray, chunk_time: Optional[float] = None) -> Optional[Turn]:
        """
        Process a single fixed-size audio chunk (512 samples).
        
        Args:
            audio_chunk: Exactly VAD_CHUNK_SIZE samples as int16
            chunk_time: Timestamp for this chunk (default: now)
            
        Returns:
            A completed Turn if one just ended, None otherwise
        """
        current_time = chunk_time if chunk_time is not None else time.time()
        
        # Convert int16 to float32 normalized [-1, 1]
        audio_float = audio_chunk.astype(np.float32) / 32768.0
        
        # Run Silero VAD inference
        try:
            ort_inputs = {
                'input': audio_float.reshape(1, -1),
                'sr': np.array([self.sample_rate], dtype=np.int64),
                'state': self._vad_state
            }
            outs = self.vad_session.run(None, ort_inputs)
            speech_prob = outs[0][0][0]
            self._vad_state = outs[1]  # Update state for next inference
            
        except Exception as e:
            # Fallback to energy-based detection
            speech_prob = 1.0 if np.abs(audio_float).mean() > 0.01 else 0.0
        
        is_speech = speech_prob > self.vad_threshold
        completed_turn = None
        
        if is_speech:
            self.last_speech_time = current_time
            self._consecutive_speech_frames += 1
            self._consecutive_silence_frames = 0  # Reset silence counter
            
            if not self.is_speaking:
                # Track speech onset (first detection before confirming turn)
                if self._speech_onset_time is None:
                    self._speech_onset_time = current_time
                
                # Use frame counting for onset detection (more robust for buffered audio)
                if self._consecutive_speech_frames >= self._min_onset_frames:
                    # Confirmed turn start
                    self.is_speaking = True
                    self.current_turn_start = self._speech_onset_time  # Use onset time, not current
                    self._log_turn_event("START", actual_time=self.current_turn_start)
                    self._speech_onset_time = None
                    self._consecutive_speech_frames = 0
                
        else:
            # No speech detected
            self._consecutive_silence_frames += 1
            
            # Only reset onset after sustained silence (tolerates single low-prob blips)
            if self._speech_onset_time is not None and not self.is_speaking:
                if self._consecutive_silence_frames >= self._silence_reset_frames:
                    self._speech_onset_time = None
                    self._consecutive_speech_frames = 0
            
            # Check for turn end if we're in a turn
            if self.is_speaking and self.last_speech_time is not None:
                silence_duration = current_time - self.last_speech_time
                
                # Check if silence exceeds timeout
                if silence_duration > self.silence_timeout:
                    # End the turn
                    turn_duration = self.last_speech_time - self.current_turn_start
                    
                    # Only record if turn meets minimum duration
                    if turn_duration >= self.min_turn_duration:
                        completed_turn = Turn(
                            speaker=self.speaker,
                            start_time=self.current_turn_start,
                            end_time=self.last_speech_time
                        )
                        self.turns.append(completed_turn)
                        self._log_turn_event("END", completed_turn.duration, actual_time=self.last_speech_time)
                    # Silently discard turns that are too short (no log spam)
                    
                    # Reset state
                    self.is_speaking = False
                    self.current_turn_start = None
        
        return completed_turn
    
    def finalize(self) -> Optional[Turn]:
        """
        Finalize any ongoing turn at conversation end.
        
        Processes any remaining buffered audio, then closes any open turn.
        Also captures turns that were in the "onset phase" (speech detected
        but not yet confirmed as a turn).
        
        Returns:
            The final turn if one was in progress, None otherwise
        """
        # Process any remaining audio in buffer (pad with zeros if needed)
        if len(self._audio_buffer) > 0:
            # Get timestamp for remaining samples
            chunk_time = self._get_timestamp_for_sample(self._samples_processed) if self._audio_buffer_timestamps else time.time()
            
            # Pad to full chunk size
            padding = self.VAD_CHUNK_SIZE - len(self._audio_buffer)
            if padding > 0:
                self._audio_buffer = np.concatenate([
                    self._audio_buffer,
                    np.zeros(padding, dtype=np.int16)
                ])
            self._process_chunk(self._audio_buffer, chunk_time)
            self._audio_buffer = np.array([], dtype=np.int16)
            self._audio_buffer_timestamps = []
        
        final_turn = None
        
        # Case 1: Turn already started (is_speaking=True)
        if self.is_speaking and self.current_turn_start is not None:
            end_time = self.last_speech_time or time.time()
            turn_duration = end_time - self.current_turn_start
            
            if turn_duration >= self.min_turn_duration:
                final_turn = Turn(
                    speaker=self.speaker,
                    start_time=self.current_turn_start,
                    end_time=end_time
                )
                self.turns.append(final_turn)
                self._log_turn_event("END (finalized)", final_turn.duration, actual_time=end_time)
        
        # Case 2: Speech detected but turn not yet confirmed (onset phase)
        # This catches short final utterances that didn't meet onset threshold
        elif self._speech_onset_time is not None and self.last_speech_time is not None:
            end_time = self.last_speech_time
            turn_duration = end_time - self._speech_onset_time
            
            # Use a lower threshold for finalization (half of normal min_turn_duration)
            if turn_duration >= self.min_turn_duration * 0.5:
                final_turn = Turn(
                    speaker=self.speaker,
                    start_time=self._speech_onset_time,
                    end_time=end_time
                )
                self.turns.append(final_turn)
                self._log_turn_event("END (onset finalized)", final_turn.duration, actual_time=end_time)
            
        self.is_speaking = False
        self.current_turn_start = None
        self._speech_onset_time = None
        return final_turn
    
    def check_silence_timeout(self) -> Optional[Turn]:
        """
        Check if silence timeout has been exceeded without new audio.
        
        Call this periodically (e.g., every 100ms) for sources that don't
        continuously provide audio (like agent audio which only arrives
        when the agent is speaking).
        
        Returns:
            A completed Turn if silence timeout was exceeded, None otherwise
        """
        if not self.is_speaking or self.last_speech_time is None:
            return None
        
        current_time = time.time()
        silence_duration = current_time - self.last_speech_time
        
        if silence_duration > self.silence_timeout:
            # End the turn
            turn_duration = self.last_speech_time - self.current_turn_start
            
            # Only record if turn meets minimum duration
            if turn_duration >= self.min_turn_duration:
                completed_turn = Turn(
                    speaker=self.speaker,
                    start_time=self.current_turn_start,
                    end_time=self.last_speech_time
                )
                self.turns.append(completed_turn)
                self._log_turn_event("END", completed_turn.duration)
                
                # Reset state
                self.is_speaking = False
                self.current_turn_start = None
                return completed_turn
            else:
                # Turn too short, silently discard
                self.is_speaking = False
                self.current_turn_start = None
        
        return None
    
    def _log_turn_event(self, event: str, duration: Optional[float] = None, actual_time: Optional[float] = None):
        """Log a turn event to console with actual speech timestamp"""
        # Use actual speech time if provided, otherwise current time
        if actual_time is not None:
            timestamp = datetime.fromtimestamp(actual_time).strftime("%H:%M:%S.%f")[:-3]
        else:
            timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        
        emoji = "🎤" if self.speaker == Speaker.USER else "🔊"
        speaker_name = self.speaker.value.upper()
        
        if duration is not None:
            print(f"[{timestamp}] {emoji} {speaker_name}_{event} ({duration:.2f}s)")
        else:
            print(f"[{timestamp}] {emoji} {speaker_name}_{event}")


class TurnTracker:
    """
    Main turn tracker that manages VAD for both user and agent.
    
    Maintains separate turn lists for user (mic) and agent (speaker output),
    then merges them at conversation end.
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        vad_threshold: float = 0.5,
        # User settings (mic input) - allow short utterances
        user_silence_timeout: float = 2.5,
        user_min_turn_duration: float = 0.15,
        user_min_speech_onset: float = 0.08,
        # Agent settings (speaker output) - needs longer for streaming
        agent_silence_timeout: float = 2.5,
        agent_min_turn_duration: float = 0.2,
        agent_min_speech_onset: float = 0.08,
        debounce_window: float = 0.5,
    ):
        """
        Initialize turn tracker.
        
        Args:
            sample_rate: Audio sample rate (default 16000)
            vad_threshold: Probability threshold for speech detection (0.0-1.0)
            user_silence_timeout: Seconds of silence before ending user turn
            user_min_turn_duration: Minimum user turn duration (allow short "Yep")
            user_min_speech_onset: Minimum continuous speech to start user turn
            agent_silence_timeout: Seconds of silence before ending agent turn
            agent_min_turn_duration: Minimum agent turn duration
            agent_min_speech_onset: Minimum continuous speech to start agent turn
            debounce_window: Seconds to wait before confirming turn end
        """
        self.sample_rate = sample_rate
        self.vad_threshold = vad_threshold
        self.debounce_window = debounce_window
        
        # Load Silero VAD model
        self.vad_session = None
        self._load_vad_model()
        
        # Initialize VAD states for user and agent with different settings
        self.user_vad: Optional[VADState] = None
        self.agent_vad: Optional[VADState] = None
        
        if self.vad_session:
            self.user_vad = VADState(
                speaker=Speaker.USER,
                vad_session=self.vad_session,
                sample_rate=sample_rate,
                vad_threshold=vad_threshold,
                silence_timeout=user_silence_timeout,
                min_turn_duration=user_min_turn_duration,
                min_speech_onset=user_min_speech_onset,
                debounce_window=debounce_window,
            )
            self.agent_vad = VADState(
                speaker=Speaker.AGENT,
                vad_session=self.vad_session,
                sample_rate=sample_rate,
                vad_threshold=vad_threshold,
                silence_timeout=agent_silence_timeout,
                min_turn_duration=agent_min_turn_duration,
                min_speech_onset=agent_min_speech_onset,
                debounce_window=debounce_window,
            )
            print("✓ TurnTracker initialized with Silero VAD")
        else:
            print("⚠ TurnTracker: VAD model not loaded, turn tracking disabled")
        
        # Conversation tracking
        self.conversation_start_time: Optional[float] = None
        self.conversation_end_time: Optional[float] = None
        self.merged_turns: List[Turn] = []
        
        # Transcript tracking (separate from VAD-detected turns)
        self.user_transcripts: List[Transcript] = []
        self.agent_responses: List[Transcript] = []
        
        # Latency tracking (agent response time after user finishes speaking)
        self.latencies: List[float] = []  # List of latency values in seconds
        
    def _load_vad_model(self):
        """Load Silero VAD ONNX model"""
        # Locate model: project_root/models/silero_vad.onnx
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        model_path = os.path.join(project_root, 'models', 'silero_vad.onnx')
        
        if os.path.exists(model_path):
            try:
                self.vad_session = ort.InferenceSession(model_path)
            except Exception as e:
                print(f"⚠ TurnTracker: Failed to load VAD model: {e}")
        else:
            print(f"⚠ TurnTracker: VAD model not found at {model_path}")
    
    def start(self):
        """Start tracking a new conversation"""
        self.conversation_start_time = time.time()
        self.conversation_end_time = None
        self.merged_turns = []
        self.user_transcripts = []
        self.agent_responses = []
        self.latencies = []
        
        # Reset VAD states
        if self.user_vad:
            self.user_vad.reset_vad_state()
            self.user_vad.turns = []
            self.user_vad.is_speaking = False
            self.user_vad.current_turn_start = None
            self.user_vad.last_speech_time = None
            
        if self.agent_vad:
            self.agent_vad.reset_vad_state()
            self.agent_vad.turns = []
            self.agent_vad.is_speaking = False
            self.agent_vad.current_turn_start = None
            self.agent_vad.last_speech_time = None
        
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        print(f"\n[{timestamp}] 📊 TurnTracker started")
    
    def process_user_audio(self, audio_data: np.ndarray, receive_time: Optional[float] = None) -> Optional[Turn]:
        """
        Process user (microphone) audio chunk.
        
        Args:
            audio_data: Audio samples as int16 numpy array
            receive_time: Optional timestamp when audio was received
            
        Returns:
            Completed Turn if one just ended, None otherwise
        """
        if self.user_vad is None:
            return None
        return self.user_vad.process_audio(audio_data, receive_time)
    
    def process_agent_audio(self, audio_data: np.ndarray, receive_time: Optional[float] = None) -> Optional[Turn]:
        """
        Process agent (speaker) audio chunk.
        
        Args:
            audio_data: Audio samples as int16 numpy array
            receive_time: Optional timestamp when audio was received
            
        Returns:
            Completed Turn if one just ended, None otherwise
        """
        if self.agent_vad is None:
            return None
        return self.agent_vad.process_audio(audio_data, receive_time)
    
    def check_agent_silence(self) -> Optional[Turn]:
        """
        Check if agent has been silent long enough to end a turn.
        
        Call this periodically (e.g., every 100ms) since agent audio
        only arrives when the agent is speaking.
        
        Returns:
            Completed Turn if silence timeout was exceeded, None otherwise
        """
        if self.agent_vad is None:
            return None
        return self.agent_vad.check_silence_timeout()
    
    def record_user_transcript(self, text: str):
        """
        Record a user transcript event.
        
        Args:
            text: The transcribed user speech
        """
        if text and text.strip():
            transcript = Transcript(
                speaker=Speaker.USER,
                text=text.strip(),
                timestamp=time.time()
            )
            self.user_transcripts.append(transcript)
    
    def record_agent_response(self, text: str):
        """
        Record an agent response event.
        
        Args:
            text: The agent's response text
        """
        if text and text.strip():
            transcript = Transcript(
                speaker=Speaker.AGENT,
                text=text.strip(),
                timestamp=time.time()
            )
            self.agent_responses.append(transcript)
    
    def finalize(self) -> List[Turn]:
        """
        Finalize turn tracking and merge all turns.
        
        Call this at conversation end to:
        1. Close any ongoing turns
        2. Merge user and agent turns sorted by start time
        3. Print conversation summary
        
        Returns:
            List of all turns sorted by start time
        """
        self.conversation_end_time = time.time()
        
        # Finalize any ongoing turns
        if self.user_vad:
            self.user_vad.finalize()
        if self.agent_vad:
            self.agent_vad.finalize()
        
        # Merge turns from both sources
        all_turns = []
        if self.user_vad:
            all_turns.extend(self.user_vad.turns)
        if self.agent_vad:
            all_turns.extend(self.agent_vad.turns)
        
        # Sort by start time
        self.merged_turns = sorted(all_turns, key=lambda t: t.start_time)
        
        # Calculate latencies (time from user turn end to next agent turn start)
        self._calculate_latencies()
        
        # Print summary
        self._print_summary()
        
        return self.merged_turns
    
    def _calculate_latencies(self):
        """
        Calculate response latencies for each user→agent turn transition.
        
        Latency = agent_turn.start_time - previous_user_turn.end_time
        Only counts when an agent turn immediately follows a user turn.
        """
        self.latencies = []
        
        for i in range(1, len(self.merged_turns)):
            prev_turn = self.merged_turns[i - 1]
            curr_turn = self.merged_turns[i]
            
            # Check for user→agent transition
            if prev_turn.speaker == Speaker.USER and curr_turn.speaker == Speaker.AGENT:
                if prev_turn.end_time is not None:
                    latency = curr_turn.start_time - prev_turn.end_time
                    # Only count positive latencies (agent responded after user finished)
                    if latency > 0:
                        self.latencies.append(latency)
    
    def _print_summary(self):
        """Print conversation turn summary with reconciled transcripts"""
        print("\n" + "=" * 60)
        print("📊 TURN TRACKER SUMMARY")
        print("=" * 60)
        
        # Reconcile transcripts with turns
        reconciled = self._reconcile_transcripts()
        
        if not reconciled:
            print("No turns recorded.")
            print("=" * 60 + "\n")
            return
        
        # Stats
        user_turns = [r for r in reconciled if r['speaker'] == Speaker.USER and r.get('turns')]
        agent_turns = [r for r in reconciled if r['speaker'] == Speaker.AGENT and r.get('turns')]
        
        user_total_time = sum(
            sum(t.duration or 0 for t in r['turns']) 
            for r in user_turns
        )
        agent_total_time = sum(
            sum(t.duration or 0 for t in r['turns']) 
            for r in agent_turns
        )
        
        conversation_duration = (self.conversation_end_time - self.conversation_start_time) if self.conversation_start_time and self.conversation_end_time else 0
        
        print(f"\nConversation duration: {conversation_duration:.1f}s")
        print(f"Total entries: {len(reconciled)}")
        print(f"  User:  {len(user_turns)} ({user_total_time:.1f}s speech)")
        print(f"  Agent: {len(agent_turns)} ({agent_total_time:.1f}s speech)")
        
        # Latency stats
        if self.latencies:
            import statistics
            min_lat = min(self.latencies)
            max_lat = max(self.latencies)
            mean_lat = statistics.mean(self.latencies)
            median_lat = statistics.median(self.latencies)
            print(f"\nResponse Latency (user_end → agent_start):")
            print(f"  Samples: {len(self.latencies)}")
            print(f"  Min:    {min_lat*1000:.0f}ms")
            print(f"  Max:    {max_lat*1000:.0f}ms")
            print(f"  Mean:   {mean_lat*1000:.0f}ms")
            print(f"  Median: {median_lat*1000:.0f}ms")
        
        print(f"\nReconciled Timeline:")
        print("-" * 60)
        
        for i, entry in enumerate(reconciled, 1):
            emoji = "🎤" if entry['speaker'] == Speaker.USER else "🔊"
            speaker = entry['speaker'].value.upper()
            
            if entry.get('turns'):
                # Has VAD-detected turn(s)
                first_turn = entry['turns'][0]
                last_turn = entry['turns'][-1]
                start_str = datetime.fromtimestamp(first_turn.start_time).strftime("%H:%M:%S.%f")[:-3]
                end_str = datetime.fromtimestamp(last_turn.end_time).strftime("%H:%M:%S.%f")[:-3] if last_turn.end_time else "..."
                total_duration = sum(t.duration or 0 for t in entry['turns'])
                
                # Show if multiple turns were merged
                merge_note = f" [{len(entry['turns'])} segments]" if len(entry['turns']) > 1 else ""
                
                print(f"  {i:2}. {emoji} {speaker:5} | {start_str} → {end_str} | {total_duration:.2f}s{merge_note}")
            else:
                # Transcript only (no matching turn)
                ts = datetime.fromtimestamp(entry['transcript'].timestamp).strftime("%H:%M:%S.%f")[:-3]
                print(f"  {i:2}. {emoji} {speaker:5} | {ts} (transcript only, no VAD turn)")
            
            # Show transcript text if available
            if entry.get('transcript'):
                text = entry['transcript'].text
                # Truncate long text
                if len(text) > 70:
                    text = text[:67] + "..."
                print(f"      \"{text}\"")
            else:
                print(f"      (no transcript)")
        
        print("=" * 60 + "\n")
    
    def _reconcile_transcripts(self) -> List[dict]:
        """
        Reconcile VAD-detected turns with transcript events.
        
        Matches transcripts to turns by sequence (first transcript → first turn(s)).
        Adjacent turns may be merged if they correspond to a single transcript.
        
        Returns:
            List of reconciled entries, each with:
            - speaker: Speaker.USER or Speaker.AGENT
            - turns: List[Turn] (VAD-detected turns, possibly multiple merged)
            - transcript: Optional[Transcript]
        """
        # Separate turns by speaker
        user_turns = [t for t in self.merged_turns if t.speaker == Speaker.USER]
        agent_turns = [t for t in self.merged_turns if t.speaker == Speaker.AGENT]
        
        # Match user transcripts to user turns
        user_entries = self._match_transcripts_to_turns(
            self.user_transcripts, user_turns, Speaker.USER
        )
        
        # Match agent responses to agent turns
        agent_entries = self._match_transcripts_to_turns(
            self.agent_responses, agent_turns, Speaker.AGENT
        )
        
        # Merge all entries and sort by timestamp
        all_entries = user_entries + agent_entries
        
        def get_entry_time(entry):
            if entry.get('turns'):
                return entry['turns'][0].start_time
            elif entry.get('transcript'):
                return entry['transcript'].timestamp
            return 0
        
        return sorted(all_entries, key=get_entry_time)
    
    def _match_transcripts_to_turns(
        self, 
        transcripts: List[Transcript], 
        turns: List[Turn],
        speaker: Speaker
    ) -> List[dict]:
        """
        Match transcripts to turns for a single speaker using timestamp proximity.
        """
        entries = []
        
        if not transcripts and not turns:
            return entries
        
        # If no transcripts, just return turns without text
        if not transcripts:
            for turn in turns:
                entries.append({
                    'speaker': speaker,
                    'turns': [turn],
                    'transcript': None
                })
            return entries
        
        # If no turns, just return transcripts
        if not turns:
            for transcript in transcripts:
                entries.append({
                    'speaker': speaker,
                    'turns': [],
                    'transcript': transcript
                })
            return entries
        
        # Match transcripts to turns by timestamp proximity
        MAX_TRANSCRIPT_DELAY = 15.0  # seconds - transcripts can be delayed
        
        used_turns = set()  # Track which turns have been matched
        transcript_to_turns = {}  # transcript index -> list of turn indices
        
        for t_idx, transcript in enumerate(transcripts):
            best_turn_idx = None
            best_score = float('inf')
            
            for turn_idx, turn in enumerate(turns):
                if turn_idx in used_turns:
                    continue
                
                turn_end = turn.end_time or turn.start_time
                
                # Score: how close is transcript timestamp to turn end?
                time_diff = transcript.timestamp - turn_end
                
                if time_diff >= 0 and time_diff < MAX_TRANSCRIPT_DELAY:
                    score = time_diff
                elif time_diff < 0 and time_diff > -2.0:
                    score = abs(time_diff) + 5.0
                else:
                    continue
                
                if score < best_score:
                    best_score = score
                    best_turn_idx = turn_idx
            
            if best_turn_idx is not None:
                used_turns.add(best_turn_idx)
                transcript_to_turns[t_idx] = [best_turn_idx]
            else:
                transcript_to_turns[t_idx] = []
        
        # Build entries for matched transcripts
        for t_idx, transcript in enumerate(transcripts):
            turn_indices = transcript_to_turns.get(t_idx, [])
            matched_turns = [turns[i] for i in turn_indices]
            
            entries.append({
                'speaker': speaker,
                'turns': matched_turns,
                'transcript': transcript
            })
        
        # Add unmatched turns (no transcript)
        for turn_idx, turn in enumerate(turns):
            if turn_idx not in used_turns:
                entries.append({
                    'speaker': speaker,
                    'turns': [turn],
                    'transcript': None
                })
        
        return entries
    
    def get_turns(self) -> List[Turn]:
        """Get the merged turn list (call after finalize)"""
        return self.merged_turns

