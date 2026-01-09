"""
Presence Fusion - Combines Audio (YAMNet) + Radar signals

Fuses multiple presence detection sources for robust detection:
- Audio: YAMNet-based sound classification (speech, footsteps, etc.)
- Radar: mmWave presence sensing (stationary, moving)

Advantages:
- Radar catches silent presence (reading, sleeping, watching TV)
- Audio catches activity outside radar range (another room)
- Combined = near-zero false negatives

Usage:
    fused = fuse_presence(
        audio_score=0.7,
        audio_threshold=0.5,
        radar_present=True,
        radar_moving=False
    )
    print(fused.state, fused.confidence)
"""

from dataclasses import dataclass
from enum import IntEnum
from typing import Optional
from datetime import datetime, timezone


# =============================================================================
# ENUMS & DATA CLASSES
# =============================================================================

class FusedPresenceState(IntEnum):
    """Combined presence detection state."""
    ABSENT = 0           # No presence detected by any source
    LIKELY_PRESENT = 1   # Single source detected presence
    CONFIRMED_PRESENT = 2  # Multiple sources agree on presence


@dataclass
class FusedPresence:
    """Result of presence fusion."""
    state: FusedPresenceState
    audio_score: float       # 0-1, from YAMNet
    radar_state: int         # 0=none, 1=stationary, 2=moving
    confidence: float        # 0-1, combined confidence
    primary_source: str      # "audio", "radar", or "both"
    timestamp: datetime


# =============================================================================
# FUSION LOGIC
# =============================================================================

def fuse_presence(
    audio_score: float,
    audio_threshold: float,
    radar_present: bool,
    radar_moving: bool,
    audio_weight: float = 0.4,
    radar_weight: float = 0.6
) -> FusedPresence:
    """
    Combine audio and radar presence signals.
    
    Args:
        audio_score: YAMNet presence probability (0-1).
        audio_threshold: Threshold for audio detection.
        radar_present: Whether radar detects presence.
        radar_moving: Whether radar detects movement.
        audio_weight: Weight for audio in confidence calculation.
        radar_weight: Weight for radar in confidence calculation.
    
    Returns:
        FusedPresence with combined state and confidence.
    
    Detection Logic:
        - Both sources agree → CONFIRMED_PRESENT (highest confidence)
        - Only radar → LIKELY_PRESENT (good for silent presence)
        - Only audio → LIKELY_PRESENT (person may be out of radar range)
        - Neither → ABSENT
    """
    audio_detected = audio_score >= audio_threshold
    
    # Normalize weights
    total_weight = audio_weight + radar_weight
    audio_w = audio_weight / total_weight
    radar_w = radar_weight / total_weight
    
    # Radar state value
    radar_state = 0
    if radar_present:
        radar_state = 2 if radar_moving else 1
    
    # Calculate radar confidence contribution
    radar_confidence = 0.0
    if radar_moving:
        radar_confidence = 0.95  # High confidence for movement
    elif radar_present:
        radar_confidence = 0.7   # Good confidence for stationary
    
    # Fusion logic
    if audio_detected and radar_present:
        # Both sources agree - highest confidence
        # Boost confidence if radar shows movement
        base_confidence = (audio_score * audio_w) + (radar_confidence * radar_w)
        confidence = min(1.0, base_confidence * 1.1)  # Small boost for agreement
        
        return FusedPresence(
            state=FusedPresenceState.CONFIRMED_PRESENT,
            audio_score=audio_score,
            radar_state=radar_state,
            confidence=confidence,
            primary_source="both",
            timestamp=datetime.now(timezone.utc)
        )
    
    elif radar_present:
        # Radar only - good for silent presence (sleeping, reading)
        # This is a key strength: detects presence when audio is silent
        confidence = radar_confidence * radar_w * 1.3  # Boost since radar is reliable
        confidence = min(0.85, confidence)  # Cap at 85% for single source
        
        return FusedPresence(
            state=FusedPresenceState.LIKELY_PRESENT,
            audio_score=audio_score,
            radar_state=radar_state,
            confidence=confidence,
            primary_source="radar",
            timestamp=datetime.now(timezone.utc)
        )
    
    elif audio_detected:
        # Audio only - person may be out of radar range
        # Or radar has blind spots (furniture blocking)
        confidence = audio_score * audio_w * 1.2  # Slight boost
        confidence = min(0.75, confidence)  # Cap lower since audio alone is less reliable
        
        return FusedPresence(
            state=FusedPresenceState.LIKELY_PRESENT,
            audio_score=audio_score,
            radar_state=0,
            confidence=confidence,
            primary_source="audio",
            timestamp=datetime.now(timezone.utc)
        )
    
    else:
        # Neither source detects presence
        return FusedPresence(
            state=FusedPresenceState.ABSENT,
            audio_score=audio_score,
            radar_state=0,
            confidence=0.0,
            primary_source="none",
            timestamp=datetime.now(timezone.utc)
        )


# =============================================================================
# PRESENCE TRACKER (Stateful)
# =============================================================================

class PresenceFusionTracker:
    """
    Stateful presence tracker with hysteresis.
    
    Prevents rapid state flickering by requiring consistent
    readings before changing state.
    """
    
    def __init__(
        self,
        audio_threshold: float = 0.5,
        enter_confirmations: int = 2,
        exit_confirmations: int = 3,
        audio_weight: float = 0.4,
        radar_weight: float = 0.6
    ):
        """
        Initialize tracker.
        
        Args:
            audio_threshold: Threshold for audio presence detection.
            enter_confirmations: Consecutive readings to confirm presence.
            exit_confirmations: Consecutive readings to confirm absence.
            audio_weight: Weight for audio in fusion.
            radar_weight: Weight for radar in fusion.
        """
        self.audio_threshold = audio_threshold
        self.enter_confirmations = enter_confirmations
        self.exit_confirmations = exit_confirmations
        self.audio_weight = audio_weight
        self.radar_weight = radar_weight
        
        # State tracking
        self._current_state = FusedPresenceState.ABSENT
        self._confirmation_count = 0
        self._pending_state: Optional[FusedPresenceState] = None
        self._last_fused: Optional[FusedPresence] = None
        
        # Statistics
        self.state_changes = 0
        self.readings_processed = 0
    
    def update(
        self,
        audio_score: float,
        radar_present: bool,
        radar_moving: bool
    ) -> FusedPresence:
        """
        Process new sensor readings and update state.
        
        Args:
            audio_score: Current audio presence score (0-1).
            radar_present: Whether radar detects presence.
            radar_moving: Whether radar detects movement.
        
        Returns:
            Current fused presence (may not change immediately due to hysteresis).
        """
        self.readings_processed += 1
        
        # Get raw fusion result
        fused = fuse_presence(
            audio_score=audio_score,
            audio_threshold=self.audio_threshold,
            radar_present=radar_present,
            radar_moving=radar_moving,
            audio_weight=self.audio_weight,
            radar_weight=self.radar_weight
        )
        
        # Determine target state
        target_state = fused.state
        
        # Hysteresis logic
        if target_state != self._current_state:
            # State change requested
            if target_state == self._pending_state:
                # Same pending state - increment counter
                self._confirmation_count += 1
                
                # Check if enough confirmations
                required = (
                    self.enter_confirmations 
                    if target_state != FusedPresenceState.ABSENT 
                    else self.exit_confirmations
                )
                
                if self._confirmation_count >= required:
                    # Confirm state change
                    self._current_state = target_state
                    self._pending_state = None
                    self._confirmation_count = 0
                    self.state_changes += 1
            else:
                # New pending state
                self._pending_state = target_state
                self._confirmation_count = 1
        else:
            # Matches current state - reset pending
            self._pending_state = None
            self._confirmation_count = 0
        
        # Build result with current (possibly unchanged) state
        self._last_fused = FusedPresence(
            state=self._current_state,
            audio_score=fused.audio_score,
            radar_state=fused.radar_state,
            confidence=fused.confidence if self._current_state == fused.state else fused.confidence * 0.8,
            primary_source=fused.primary_source,
            timestamp=fused.timestamp
        )
        
        return self._last_fused
    
    def get_state(self) -> FusedPresenceState:
        """Get current fused presence state."""
        return self._current_state
    
    def is_present(self) -> bool:
        """Quick check if presence is detected."""
        return self._current_state != FusedPresenceState.ABSENT
    
    def get_last_fused(self) -> Optional[FusedPresence]:
        """Get last fusion result."""
        return self._last_fused
    
    def reset(self):
        """Reset tracker to initial state."""
        self._current_state = FusedPresenceState.ABSENT
        self._confirmation_count = 0
        self._pending_state = None
        self._last_fused = None
    
    def get_stats(self) -> dict:
        """Get tracker statistics."""
        return {
            "current_state": self._current_state.name,
            "state_changes": self.state_changes,
            "readings_processed": self.readings_processed,
            "pending_state": self._pending_state.name if self._pending_state else None,
            "confirmation_count": self._confirmation_count
        }

