"""
Human presence detection using YAMNet ONNX model with weighted scoring.

Runs on a background thread with a 30-second duty cycle to detect
human presence through audio classification with weighted scoring.
"""

import os
import csv
import time
import threading
import numpy as np
import sounddevice as sd
import onnxruntime as ort
from lib.config import Config


# -------------------------------------------------------------------------
# Weighted class mapping for human presence detection
# Higher weights = stronger indicator of human presence
# -------------------------------------------------------------------------
HUMAN_PRESENCE_WEIGHTS = {
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
DETECTION_THRESHOLD = 0.3    # Weighted score threshold for human presence


class HumanPresenceDetector:
    """
    Background human presence detection using YAMNet ONNX model.
    
    Samples audio every 5 seconds and uses weighted classification to detect
    human presence without blocking the main thread.
    """
    
    def __init__(self, mic_device_index=None, threshold=DETECTION_THRESHOLD, on_detection=None, on_cycle=None):
        """
        Initialize the human presence detector.
        
        Args:
            mic_device_index: Audio device index (None for system default).
            threshold: Detection threshold for weighted score (0.0-1.0).
            on_detection: Optional callback(weighted_score, top_classes) when humans detected.
            on_cycle: Optional callback(weighted_score) called every detection cycle.
        """
        self.mic_device_index = mic_device_index
        self.threshold = threshold
        self.logger = Config.LOGGER
        self.on_detection = on_detection
        self.on_cycle = on_cycle
        
        # Thread control
        self._thread = None
        self._stop_event = threading.Event()
        
        # Presence tracking
        self.last_detection_time = None
        self.running = False
        
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
            if class_name in HUMAN_PRESENCE_WEIGHTS:
                self._class_weights[class_idx] = HUMAN_PRESENCE_WEIGHTS[class_name]
        
        print(f"✓ HumanPresenceDetector initialized")
        print(f"  - {len(self._class_names)} total classes")
        print(f"  - {len(self._class_weights)} weighted classes for human presence")
        print(f"  - Detection threshold: {self.threshold}")
    
    def start(self):
        """Start the background presence detection thread."""
        if self.running:
            return
        
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._duty_cycle_loop, daemon=True)
        self._thread.start()
        self.running = True
        
        print(f"✓ HumanPresenceDetector started (checking every {DUTY_CYCLE_SECONDS}s)")
        
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
        
        print("✓ HumanPresenceDetector stopped")
    
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
        Capture audio sample and run YAMNet inference with weighted scoring.
        
        Opens mic, captures NUM_SAMPLES, closes mic, then processes.
        This open/close pattern avoids hogging the mic driver.
        """
        # -------------------------------------------------------------------------
        # Capture audio (blocking read, then release mic)
        # -------------------------------------------------------------------------
        try:
            audio_data = sd.rec(
                frames=NUM_SAMPLES,
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype='float32',
                device=self.mic_device_index,
                blocking=True
            )
        except sd.PortAudioError as e:
            print(f"⚠ Mic capture failed: {e}")
            return
        
        # -------------------------------------------------------------------------
        # Normalize and reshape for YAMNet input
        # YAMNet expects shape [15600] with float32 values in [-1.0, 1.0]
        # -------------------------------------------------------------------------
        audio_flat = audio_data.flatten()
        
        # Ensure correct length (pad or trim if necessary)
        if len(audio_flat) < NUM_SAMPLES:
            audio_flat = np.pad(audio_flat, (0, NUM_SAMPLES - len(audio_flat)))
        elif len(audio_flat) > NUM_SAMPLES:
            audio_flat = audio_flat[:NUM_SAMPLES]
        
        # Normalize to [-1.0, 1.0]
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
        # Calculate weighted score for human presence
        # -------------------------------------------------------------------------
        weighted_score = 0.0
        contributing_classes = []
        
        for class_idx, weight in self._class_weights.items():
            class_prob = float(scores[class_idx])
            contribution = class_prob * weight
            weighted_score += contribution
            
            # Track top contributing classes for logging
            if class_prob > 0.1:  # Only track significant contributions
                contributing_classes.append((
                    self._class_names[class_idx],
                    class_prob,
                    contribution
                ))
        
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
        # Check if weighted score exceeds threshold
        # -------------------------------------------------------------------------
        if weighted_score >= self.threshold:
            self.last_detection_time = time.time()
            
            # Get top 3 contributing classes for logging
            top_classes = [
                f"{name} ({prob:.2f})"
                for name, prob, _ in contributing_classes[:3]
            ]
            
            print(f"[HUMAN DETECTED] Weighted score: {weighted_score:.3f}")
            print(f"  Top contributors: {', '.join(top_classes)}")
            
            if self.logger:
                self.logger.info(
                    "human_detected",
                    extra={
                        "weighted_score": weighted_score,
                        "top_classes": top_classes,
                        "threshold": self.threshold,
                    }
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
