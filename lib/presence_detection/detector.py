"""
Human presence detection using YAMNet ONNX model with weighted scoring.

Runs on a background thread with a 5-second duty cycle to detect
human presence through audio classification with weighted scoring.
"""

import os
import csv
import time
import asyncio
import threading
from datetime import datetime, timezone
import numpy as np
import sounddevice as sd
import onnxruntime as ort
from lib.config import Config


# -------------------------------------------------------------------------
# NOTE: Weights are now loaded dynamically from runtime config
# The hardcoded weights below are kept as fallback only
# -------------------------------------------------------------------------
FALLBACK_HUMAN_PRESENCE_WEIGHTS = {
    # Speech & Voice (1.0 - Strongest indicators)
    'Speech': 1.0,
    'Child speech, kid speaking': 1.0,
    'Conversation': 1.0,
    'Narration, monologue': 1.0,
    'Babbling': 1.0,
    'Whispering': 1.0,
    'Shout': 1.0,
    'Bellow': 1.0,
    'Whoop': 1.0,
    'Yell': 1.0,
    'Children shouting': 1.0,
    'Screaming': 1.0,
    
    # Human Sounds - Vocal (0.95)
    'Laughter': 0.95,
    'Baby laughter': 0.95,
    'Giggle': 0.95,
    'Snicker': 0.95,
    'Belly laugh': 0.95,
    'Chuckle, chortle': 0.95,
    'Crying, sobbing': 0.95,
    'Baby cry, infant cry': 0.95,
    'Whimper': 0.95,
    'Wail, moan': 0.95,
    'Sigh': 0.95,
    
    # Singing & Musical Voice (0.9)
    'Singing': 0.9,
    'Choir': 0.9,
    'Yodeling': 0.9,
    'Chant': 0.9,
    'Mantra': 0.9,
    'Child singing': 0.9,
    'Rapping': 0.9,
    'Humming': 0.9,
    
    # Human Sounds - Physical (0.9)
    'Cough': 0.9,
    'Throat clearing': 0.9,
    'Sneeze': 0.9,
    'Sniff': 0.9,
    'Breathing': 0.9,
    'Wheeze': 0.9,
    'Snoring': 0.9,
    'Gasp': 0.9,
    'Pant': 0.9,
    'Snort': 0.9,
    'Groan': 0.9,
    'Grunt': 0.9,
    'Whistling': 0.9,
    
    # Movement - Direct (0.85)
    'Walk, footsteps': 0.85,
    'Run': 0.85,
    'Shuffle': 0.85,
    
    # Eating & Body Sounds (0.8)
    'Chewing, mastication': 0.8,
    'Biting': 0.8,
    'Gargling': 0.8,
    'Stomach rumble': 0.8,
    'Burping, eructation': 0.8,
    'Hiccup': 0.8,
    'Fart': 0.8,
    'Heart sounds, heartbeat': 0.8,
    'Heart murmur': 0.8,
    
    # Hand Gestures & Actions (0.75)
    'Hands': 0.75,
    'Finger snapping': 0.75,
    'Clapping': 0.75,
    
    # Social & Group Sounds (0.8)
    'Cheering': 0.8,
    'Applause': 0.8,
    'Chatter': 0.8,
    'Crowd': 0.8,
    'Hubbub, speech noise, speech babble': 0.8,
    'Children playing': 0.8,
    
    # Door & Entry Activity (0.7)
    'Door': 0.7,
    'Doorbell': 0.7,
    'Ding-dong': 0.7,
    'Sliding door': 0.7,
    'Slam': 0.7,
    'Knock': 0.7,
    'Tap': 0.7,
    'Squeak': 0.7,
    
    # Home Activities (0.65)
    'Cupboard open or close': 0.65,
    'Drawer open or close': 0.65,
    'Dishes, pots, and pans': 0.65,
    'Cutlery, silverware': 0.65,
    'Chopping (food)': 0.65,
    'Frying (food)': 0.65,
    'Microwave oven': 0.65,
    'Blender': 0.65,
    'Water tap, faucet': 0.65,
    'Sink (filling or washing)': 0.65,
    'Bathtub (filling or washing)': 0.65,
    'Hair dryer': 0.65,
    'Toilet flush': 0.65,
    'Toothbrush': 0.65,
    'Electric toothbrush': 0.65,
    'Vacuum cleaner': 0.65,
    'Zipper (clothing)': 0.65,
    'Keys jangling': 0.65,
    'Coin (dropping)': 0.65,
    'Scissors': 0.65,
    'Electric shaver, electric razor': 0.65,
    
    # Office & Writing (0.6)
    'Shuffling cards': 0.6,
    'Typing': 0.6,
    'Typewriter': 0.6,
    'Computer keyboard': 0.6,
    'Writing': 0.6,
    
    # Communication Devices (0.55)
    'Telephone': 0.55,
    'Telephone bell ringing': 0.55,
    'Ringtone': 0.55,
    'Telephone dialing, DTMF': 0.55,
    'Alarm clock': 0.55,
    
    # Ambient Presence Indicators (0.5)
    'Rustle': 0.5,
    'Camera': 0.5,
    'Single-lens reflex camera': 0.5,
}


# -------------------------------------------------------------------------
# Audio capture settings for YAMNet
# -------------------------------------------------------------------------
SAMPLE_RATE = 16000          # YAMNet expects 16kHz
NUM_SAMPLES = 15600          # ~0.975 seconds of audio
DUTY_CYCLE_SECONDS = 5       # Run detection every 5 seconds
DETECTION_THRESHOLD = 0.3    # Weighted mean threshold for human presence (0.0-1.0)
                             # Score = Σ(prob × weight) / Σ(weight) for prob > 0.1


class HumanPresenceDetector:
    """
    Background human presence detection using YAMNet ONNX model.
    
    Samples audio every 5 seconds and uses weighted classification to detect
    human presence without blocking the main thread.
    """
    
    def __init__(self, mic_device_index=None, threshold=DETECTION_THRESHOLD, weights=None, orchestrator_client=None, on_detection=None, on_cycle=None, event_loop=None):
        """
        Initialize the human presence detector.
        
        Args:
            mic_device_index: Audio device index (None for system default).
            threshold: Detection threshold for weighted score (0.0-1.0).
            weights: Dict mapping event names to weights (0.0-1.0). If None, uses fallback weights.
            orchestrator_client: OrchestratorClient instance for sending detections.
            on_detection: Optional callback(weighted_score, top_classes) when humans detected.
            on_cycle: Optional callback(weighted_score) called every detection cycle.
            event_loop: Main asyncio event loop for sending messages (required if orchestrator_client provided).
        """
        self.mic_device_index = mic_device_index
        self.threshold = threshold
        self.logger = Config.LOGGER
        self.orchestrator_client = orchestrator_client
        self.event_loop = event_loop
        self.on_detection = on_detection
        self.on_cycle = on_cycle
        
        # Use provided weights or fallback to hardcoded weights
        self.event_weights = weights if weights is not None else FALLBACK_HUMAN_PRESENCE_WEIGHTS
        
        # Thread control
        self._thread = None
        self._stop_event = threading.Event()
        
        # Presence tracking
        self.last_detection_time = None
        self.running = False
        
        # Audio buffering (receives audio from wake word detector's stream)
        # We don't open our own stream - that would conflict with wake word detector
        self._audio_buffer = []
        self._buffer_lock = threading.Lock()
        
        # -------------------------------------------------------------------------
        # Load YAMNet ONNX model
        # -------------------------------------------------------------------------
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        model_path = os.path.join(project_root, 'models', 'yamnet.onnx')
        class_map_path = os.path.join(project_root, 'models', 'yamnet_class_map.csv')
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"YAMNet ONNX model not found: {model_path}\n"
                f"Please convert the model: python scripts/convert_yamnet_to_onnx.py"
            )
        
        if not os.path.exists(class_map_path):
            raise FileNotFoundError(f"YAMNet class map not found: {class_map_path}")
        
        # Initialize ONNX Runtime session
        self._session = ort.InferenceSession(
            model_path,
            providers=['CPUExecutionProvider']
        )
        
        # Get input/output names
        self._input_name = self._session.get_inputs()[0].name
        self._output_name = self._session.get_outputs()[0].name
        
        print(f"  Using ONNX model: {model_path}")
        
        # -------------------------------------------------------------------------
        # Load class map and create weighted class indices
        # -------------------------------------------------------------------------
        self._class_names = {}
        with open(class_map_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                self._class_names[int(row['index'])] = row['display_name']
        
        # Create mapping of class index -> weight for fast lookup
        self._class_weights = {}
        for class_idx, class_name in self._class_names.items():
            if class_name in self.event_weights:
                self._class_weights[class_idx] = self.event_weights[class_name]
        
        # Calculate sum of weights (used for weighted mean calculation)
        self._sum_of_weights = sum(self._class_weights.values())
        
        # Find weight statistics
        weights_list = list(self._class_weights.values())
        min_weight = min(weights_list) if weights_list else 0
        max_weight = max(weights_list) if weights_list else 0
        avg_weight = sum(weights_list) / len(weights_list) if weights_list else 0
        
        weights_source = "runtime config" if weights is not None else "fallback"
        print(f"✓ HumanPresenceDetector initialized")
        print(f"  - {len(self._class_names)} total YAMNet classes")
        print(f"  - {len(self._class_weights)} weighted classes for human presence")
        print(f"  - Weights source: {weights_source}")
        print(f"  - Weight range: [{min_weight:.2f}, {max_weight:.2f}], avg: {avg_weight:.2f}")
        print(f"  - Sum of weights: {self._sum_of_weights:.2f} (max possible denominator)")
        print(f"  - Detection threshold: {self.threshold} (score will be 0.0-1.0)")
        print(f"  - Confidence threshold: 0.1 (only events with >10% probability counted)")
        print(f"  - Scoring: weighted_mean = Σ(prob × weight) / Σ(weight) for prob > 0.1")
    
    def start(self):
        """Start the background presence detection thread (audio comes from wake word detector)."""
        if self.running:
            return
        
        # NOTE: We don't open our own audio stream - that would conflict with wake word detector
        # Instead, we receive audio via feed_audio() called from wake word detector's callback
        
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._duty_cycle_loop, daemon=True)
        self._thread.start()
        self.running = True
        
        print(f"✓ HumanPresenceDetector started (checking every {DUTY_CYCLE_SECONDS}s)")
        print(f"   Receiving audio from wake word detector's stream")
        
        if self.logger:
            self.logger.info(
                "human_presence_detector_started",
                extra={
                    "duty_cycle_seconds": DUTY_CYCLE_SECONDS,
                    "threshold": self.threshold,
                    "num_weighted_classes": len(self._class_weights),
                }
            )
    
    def stop(self):
        """Stop the background presence detection thread."""
        if not self.running:
            return
        
        self._stop_event.set()
        
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        
        self._thread = None
        self.running = False
        
        # Clear buffer
        with self._buffer_lock:
            self._audio_buffer.clear()
        
        print("✓ HumanPresenceDetector stopped")
    
    def feed_audio(self, audio_data: np.ndarray):
        """
        Feed audio data to the presence detector (called from wake word detector).
        
        Args:
            audio_data: Audio samples as float32 array (16kHz mono, values in [-1, 1])
        """
        if not self.running:
            return
        
        # Buffer audio data (copy to avoid corruption)
        with self._buffer_lock:
            self._audio_buffer.append(audio_data.copy())
    
    def _duty_cycle_loop(self):
        """
        Main duty cycle loop running on background thread.
        
        Sleeps for DUTY_CYCLE_SECONDS, then captures audio and runs inference.
        Uses stop_event.wait() for interruptible sleep.
        """
        while not self._stop_event.is_set():
            # Interruptible sleep - wakes immediately if stop() is called
            if self._stop_event.wait(timeout=DUTY_CYCLE_SECONDS):
                break  # Stop event was set
            
            try:
                self._detect_human_presence()
            except Exception as e:
                # Log error and continue - resilient to transient failures
                print(f"⚠ HumanPresenceDetector error: {e}")
                if self.logger:
                    self.logger.warning(
                        "human_presence_detector_error",
                        extra={"error": str(e)},
                        exc_info=True
                    )
    
    def _detect_human_presence(self):
        """
        Capture audio sample from buffer and run YAMNet inference with weighted scoring.
        
        Reads buffered audio from the persistent audio stream (similar to wake word detector).
        """
        # -------------------------------------------------------------------------
        # Get recent audio from buffer
        # -------------------------------------------------------------------------
        with self._buffer_lock:
            if not self._audio_buffer:
                # No audio data yet (stream just started)
                return
            
            # Concatenate all buffered audio chunks
            audio_data = np.concatenate(self._audio_buffer, axis=0)
            
            # Clear buffer for next cycle
            self._audio_buffer.clear()
        
        # -------------------------------------------------------------------------
        # Extract most recent NUM_SAMPLES for YAMNet
        # YAMNet expects shape [15600] with float32 values in [-1.0, 1.0]
        # -------------------------------------------------------------------------
        audio_flat = audio_data.flatten()
        
        # Take the most recent NUM_SAMPLES (or pad if not enough)
        if len(audio_flat) >= NUM_SAMPLES:
            # Use the most recent audio
            audio_flat = audio_flat[-NUM_SAMPLES:]
        else:
            # Pad with zeros if we don't have enough samples yet
            audio_flat = np.pad(audio_flat, (NUM_SAMPLES - len(audio_flat), 0))
        
        # Already float32, just ensure it's clipped to [-1.0, 1.0]
        audio_normalized = np.clip(audio_flat, -1.0, 1.0).astype(np.float32)
        
        # -------------------------------------------------------------------------
        # Run ONNX inference
        # -------------------------------------------------------------------------
        try:
            outputs = self._session.run(
                [self._output_name],
                {self._input_name: audio_normalized}
            )
            scores = outputs[0]
        except Exception as e:
            print(f"⚠ ONNX inference failed: {e}")
            return
        
        # Handle multi-frame output: YAMNet may return multiple frames, average them
        if scores.ndim > 1:
            scores = np.mean(scores, axis=0)
        
        # -------------------------------------------------------------------------
        # Calculate weighted mean for human presence
        # Only consider events with probability > 0.1 (10% confidence threshold)
        # weighted_mean = Σ(prob_i × weight_i) / Σ(weight_i) for prob_i > 0.1
        # This gives a true weighted average probability (always 0-1)
        # -------------------------------------------------------------------------
        CONFIDENCE_THRESHOLD = 0.1  # Only consider events with >10% probability
        
        weighted_sum = 0.0
        total_weights = 0.0
        contributing_classes = []
        
        for class_idx, weight in self._class_weights.items():
            class_prob = float(scores[class_idx])
            
            # Only include events with probability > threshold
            if class_prob > CONFIDENCE_THRESHOLD:
                contribution = class_prob * weight
                weighted_sum += contribution
                total_weights += weight
                
                # Track for logging
                contributing_classes.append((
                    self._class_names[class_idx],
                    class_prob,
                    contribution
                ))
        
        # Calculate weighted mean (normalized by sum of weights of contributing events)
        # If no events above threshold, score is 0
        weighted_score = weighted_sum / total_weights if total_weights > 0 else 0.0
        
        # Sort by contribution
        contributing_classes.sort(key=lambda x: x[2], reverse=True)
        
        # -------------------------------------------------------------------------
        # Call cycle callback (if provided) with weighted score
        # -------------------------------------------------------------------------
        if self.on_cycle:
            try:
                self.on_cycle(weighted_score)
            except Exception as e:
                print(f"⚠ Cycle callback error: {e}")
        
        # -------------------------------------------------------------------------
        # Diagnostic logging: Show score every cycle for threshold tuning
        # -------------------------------------------------------------------------
        timestamp = datetime.now().strftime('%H:%M:%S')
        num_contributing = len(contributing_classes)
        
        if weighted_score >= self.threshold:
            # Human detected - show green indicator
            top_3 = [f"{name} ({prob:.2f})" for name, prob, _ in contributing_classes[:3]]
            print(f"[{timestamp}] 🟢 Presence: {weighted_score:.3f} (DETECTED) | {num_contributing} events | Top: {', '.join(top_3) if top_3 else 'none'}")
        else:
            # Below threshold - show grey indicator with top classes for tuning (if debug logs enabled)
            if Config.SHOW_PRESENCE_DETECTION_DEBUG_LOGS:
                top_3 = [f"{name} ({prob:.2f})" for name, prob, _ in contributing_classes[:3]]
                status = f"{num_contributing} events" if num_contributing > 0 else "silence"
                print(f"[PRESENCE_DEBUG] ⚪ Presence: {weighted_score:.3f} | {status} | Top: {', '.join(top_3) if top_3 else 'none'}")
        
        # -------------------------------------------------------------------------
        # Check if weighted score exceeds threshold
        # -------------------------------------------------------------------------
        if weighted_score >= self.threshold:
            self.last_detection_time = time.time()
            
            # Get top 5 contributing classes for sending to orchestrator
            # Calculate percent_contribution: (contribution / weighted_sum) × 100
            # This shows each event's percentage contribution to the final weighted score
            top_events = [
                {
                    "event": name,
                    "percent_contribution": float((contribution / weighted_sum) * 100) if weighted_sum > 0 else 0.0
                }
                for name, prob, contribution in contributing_classes[:5]
            ]
            
            # Get top 3 for logging
            top_classes = [
                f"{name} ({prob:.2f})"
                for name, prob, _ in contributing_classes[:3]
            ]
            
            # Additional detail logging (main cycle log is printed above)
            print(f"  ↳ Sending to orchestrator: probability={weighted_score*100:.1f}%, top_events={len(top_events)}")
            
            if self.logger:
                self.logger.info(
                    "human_detected",
                    extra={
                        "weighted_score": weighted_score,
                        "top_classes": top_classes,
                        "threshold": self.threshold,
                    }
                )
            
            # Send detection to orchestrator via WebSocket
            if self.orchestrator_client and self.event_loop:
                try:
                    # Convert weighted mean (0-1) to percentage (0-100)
                    # Since weighted_score is now a weighted mean, it's always 0-1
                    normalized_percent = float(weighted_score * 100)
                    
                    # Send asynchronously from background thread using stored event loop
                    asyncio.run_coroutine_threadsafe(
                        self.orchestrator_client.send_presence_detection(
                            probability=normalized_percent,
                            detected_at=datetime.now(timezone.utc).isoformat(),
                            top_events=top_events
                        ),
                        self.event_loop
                    )
                except Exception as e:
                    print(f"⚠ Failed to send detection to orchestrator: {e}")
                    if self.logger:
                        self.logger.warning(
                            "presence_detection_send_failed",
                            extra={"error": str(e)},
                            exc_info=True
                        )
            
            # Call detection callback (if provided)
            if self.on_detection:
                try:
                    self.on_detection(weighted_score, top_classes)
                except Exception as e:
                    print(f"⚠ Detection callback error: {e}")
    
    def cleanup(self):
        """Clean up resources."""
        self.stop()
