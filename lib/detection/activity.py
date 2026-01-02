"""
Activity detection using YAMNet for human presence awareness.

⚠️  LEGACY CODE - NOT CURRENTLY USED ⚠️
This module is kept for reference only. It has been replaced by HumanPresenceDetector
which uses a more sophisticated weighted scoring approach.

Runs on a background thread with a 5-second duty cycle to detect
sounds like footsteps, doors, coughs, etc.
"""

import os
import csv
import time
import threading
import numpy as np
import sounddevice as sd
from tflite_runtime.interpreter import Interpreter
from lib.config import Config


# -------------------------------------------------------------------------
# Target classes indicating human presence (subset of YAMNet's 521 classes)
# -------------------------------------------------------------------------
TARGET_CLASSES = {
    'Walk, footsteps',
    'Door',
    'Knock',
    'Breathing',
    'Cough',
    'Sneeze',
    'Rustle',
    'Keys jangling',
    'Dishes, pots, and pans',
}

# -------------------------------------------------------------------------
# Audio capture settings for YAMNet
# -------------------------------------------------------------------------
SAMPLE_RATE = 16000          # YAMNet expects 16kHz
NUM_SAMPLES = 15600          # ~0.975 seconds of audio
DUTY_CYCLE_SECONDS = 5       # Sleep between detection cycles
CONFIDENCE_THRESHOLD = 0.5   # Minimum score to report detection


class ActivityMonitor:
    """
    Background activity detection using YAMNet TFLite model.
    
    Periodically samples audio to detect human presence indicators
    (footsteps, doors, coughs, etc.) without blocking the main thread.
    """
    
    def __init__(self, mic_device_index=None, on_activity_detected=None):
        """
        Initialize the activity monitor.
        
        Args:
            mic_device_index: Audio device index (None for system default).
                              Using default allows PulseAudio mixing with
                              other audio consumers (e.g., WakeWordDetector).
            on_activity_detected: Optional callback(class_name, score) called
                                  when activity is detected.
        """
        self.mic_device_index = mic_device_index
        self.on_activity_detected = on_activity_detected
        self.logger = Config.LOGGER
        
        # Thread control
        self._thread = None
        self._stop_event = threading.Event()
        
        # Activity tracking
        self.last_activity_time = None
        self.running = False
        
        # -------------------------------------------------------------------------
        # Load YAMNet TFLite model
        # -------------------------------------------------------------------------
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        model_path = os.path.join(project_root, 'models', 'yamnet.tflite')
        class_map_path = os.path.join(project_root, 'models', 'yamnet_class_map.csv')
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"YAMNet model not found: {model_path}")
        if not os.path.exists(class_map_path):
            raise FileNotFoundError(f"YAMNet class map not found: {class_map_path}")
        
        # Initialize TFLite interpreter
        self._interpreter = Interpreter(model_path=model_path)
        self._interpreter.allocate_tensors()
        
        # Get input/output tensor details
        self._input_details = self._interpreter.get_input_details()
        self._output_details = self._interpreter.get_output_details()
        
        # -------------------------------------------------------------------------
        # Load class map (index -> display_name)
        # -------------------------------------------------------------------------
        self._class_names = {}
        with open(class_map_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                self._class_names[int(row['index'])] = row['display_name']
        
        print(f"✓ ActivityMonitor initialized ({len(self._class_names)} classes)")
    
    def start(self):
        """Start the background activity detection thread."""
        if self.running:
            return
        
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._duty_cycle_loop, daemon=True)
        self._thread.start()
        self.running = True
        
        print(f"✓ ActivityMonitor started (checking every {DUTY_CYCLE_SECONDS}s)")
        
        if self.logger:
            self.logger.info(
                "activity_monitor_started",
                extra={
                    "duty_cycle_seconds": DUTY_CYCLE_SECONDS,
                    "target_classes": list(TARGET_CLASSES),
                }
            )
    
    def stop(self):
        """Stop the background activity detection thread."""
        if not self.running:
            return
        
        self._stop_event.set()
        
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        
        self._thread = None
        self.running = False
        
        print("✓ ActivityMonitor stopped")
    
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
                self._detect_activity()
            except Exception as e:
                # Log error and continue - resilient to transient failures
                print(f"⚠ ActivityMonitor error: {e}")
                if self.logger:
                    self.logger.warning(
                        "activity_monitor_error",
                        extra={"error": str(e)},
                        exc_info=True
                    )
    
    def _detect_activity(self):
        """
        Capture audio sample and run YAMNet inference.
        
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
        
        # Normalize to [-1.0, 1.0] (sd.rec with float32 should already be normalized,
        # but ensure it's in the correct range)
        audio_normalized = np.clip(audio_flat, -1.0, 1.0).astype(np.float32)
        
        # -------------------------------------------------------------------------
        # Run TFLite inference
        # -------------------------------------------------------------------------
        input_index = self._input_details[0]['index']
        self._interpreter.set_tensor(input_index, audio_normalized)
        self._interpreter.invoke()
        
        # Get output scores (shape varies by model, typically [1, 521] or [N, 521])
        output_index = self._output_details[0]['index']
        scores = self._interpreter.get_tensor(output_index)
        
        # Handle multi-frame output: YAMNet may return multiple frames, average them
        if scores.ndim > 1:
            scores = np.mean(scores, axis=0)
        
        # -------------------------------------------------------------------------
        # Find top prediction
        # -------------------------------------------------------------------------
        top_index = np.argmax(scores)
        top_score = float(scores[top_index])
        top_class = self._class_names.get(top_index, f"Unknown({top_index})")
        
        # -------------------------------------------------------------------------
        # Check if it's a target class with sufficient confidence
        # -------------------------------------------------------------------------
        if top_class in TARGET_CLASSES and top_score > CONFIDENCE_THRESHOLD:
            self.last_activity_time = time.time()
            
            print(f"[ACTIVITY] Detected {top_class} ({top_score:.2f})")
            
            if self.logger:
                self.logger.info(
                    "activity_detected",
                    extra={
                        "class_name": top_class,
                        "score": top_score,
                    }
                )
            
            # Invoke callback if provided
            if self.on_activity_detected:
                try:
                    self.on_activity_detected(top_class, top_score)
                except Exception as e:
                    print(f"⚠ Activity callback error: {e}")
    
    def cleanup(self):
        """Clean up resources."""
        self.stop()

