"""ReSpeaker 4-Mic Array hardware initialization and tuning"""

import os
import subprocess
import logging
from typing import Dict, Optional


class ReSpeakerController:
    """
    Controls ReSpeaker 4-Mic Array tuning parameters via USB.
    
    Uses the vendored tuning.py script (Python 3.13 compatible) from the 
    ReSpeaker usb_4_mic_array repository to apply hardware DSP parameters 
    for acoustic echo cancellation and gain control.
    """
    
    # Use vendored tuning script (located in same package)
    TUNING_SCRIPT = os.path.join(
        os.path.dirname(__file__), 
        "usb_4_mic_array", 
        "tuning.py"
    )
    
    # Default configuration (matches wrapper defaults)
    DEFAULT_CONFIG = {
        "agc_gain": 3.0,
        "agc_on_off": 0,
        "aec_freeze_on_off": 0,
        "echo_on_off": 1,
        "hpf_on_off": 1,
        "stat_noise_on_off": 1,
        "gamma_e": 2.0,
        "gamma_enl": 3.0,
        "gamma_etail": 2.0
    }
    
    def __init__(self, config: Optional[Dict] = None, logger: Optional[logging.Logger] = None):
        """
        Initialize ReSpeaker controller.
        
        Args:
            config: Dict with tuning parameters (uses defaults if None)
            logger: Logger instance (creates one if None)
        """
        self.config = config or self.DEFAULT_CONFIG
        self.logger = logger or logging.getLogger(__name__)
        
    def is_available(self) -> bool:
        """Check if ReSpeaker tools are available"""
        return os.path.exists(self.TUNING_SCRIPT)
    
    def initialize(self) -> bool:
        """
        Apply all ReSpeaker tuning parameters.
        
        Returns:
            True if all parameters applied successfully, False otherwise
        """
        if not self.is_available():
            self.logger.warning("ReSpeaker tuning script not found at %s", self.TUNING_SCRIPT)
            return False
        
        self.logger.info("Applying ReSpeaker configuration:")
        for param, value in self.config.items():
            self.logger.info("  %s: %s", param, value)
        
        # Apply parameters in specific order for best results
        success = True
        
        # 1. Freeze AGC first (prevent auto-adjustment)
        success &= self._apply_parameter("AGCONOFF", self.config.get("agc_on_off", 0))
        
        # 2. Set AGC gain
        success &= self._apply_parameter("AGCGAIN", self.config.get("agc_gain", 3.0))
        
        # 3. Enable AEC adaptation
        success &= self._apply_parameter("AECFREEZEONOFF", self.config.get("aec_freeze_on_off", 0))
        
        # 4. Enable echo suppression
        success &= self._apply_parameter("ECHOONOFF", self.config.get("echo_on_off", 1))
        
        # 5. Enable high-pass filter
        success &= self._apply_parameter("HPFONOFF", self.config.get("hpf_on_off", 1))
        
        # 6. Enable stationary noise suppression
        success &= self._apply_parameter("STATNOISEONOFF", self.config.get("stat_noise_on_off", 1))
        
        # 7. Set echo suppression gamma parameters
        success &= self._apply_parameter("GAMMA_E", self.config.get("gamma_e", 2.0))
        success &= self._apply_parameter("GAMMA_ENL", self.config.get("gamma_enl", 3.0))
        success &= self._apply_parameter("GAMMA_ETAIL", self.config.get("gamma_etail", 2.0))
        
        if success:
            self.logger.info("ReSpeaker initialization complete")
        else:
            self.logger.warning("Some ReSpeaker parameters failed to apply")
        
        return success
    
    def _apply_parameter(self, param_name: str, param_value) -> bool:
        """
        Apply a single tuning parameter and verify it.
        
        Args:
            param_name: Parameter name (e.g., "AGCGAIN")
            param_value: Value to set
            
        Returns:
            True if successfully applied and verified, False otherwise
        """
        try:
            # Apply the parameter (use python3 explicitly - tuning.py needs Python 3 + pyusb)
            result = subprocess.run(
                ["sudo", "python3", self.TUNING_SCRIPT, param_name, str(param_value)],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode != 0:
                self.logger.error("Failed to set %s: %s", param_name, result.stderr)
                return False
            
            # Verify it was set correctly
            verify_result = subprocess.run(
                ["sudo", "python3", self.TUNING_SCRIPT, param_name],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if verify_result.returncode == 0:
                output = verify_result.stdout.strip().split('\n')[-1].strip()
                # Parse output format: "PARAMNAME: value" -> extract just "value"
                if ': ' in output:
                    current_value = output.split(': ', 1)[1].strip()
                else:
                    current_value = output
                
                # For numeric values, compare with tolerance
                if self._values_match(current_value, str(param_value)):
                    self.logger.info("✓ %s set to %s (verified: %s)", param_name, param_value, current_value)
                    return True
                else:
                    self.logger.warning("✗ %s verification failed (expected: %s, got: %s)", 
                                      param_name, param_value, current_value)
                    return False
            else:
                self.logger.warning("Could not verify %s", param_name)
                return True  # Assume success if we can't verify
                
        except subprocess.TimeoutExpired:
            self.logger.error("Timeout setting %s", param_name)
            return False
        except Exception as e:
            self.logger.error("Error setting %s: %s", param_name, e)
            return False
    
    def _values_match(self, actual: str, expected: str, tolerance: float = 0.1) -> bool:
        """Check if two values match (with tolerance for floats)"""
        try:
            actual_float = float(actual)
            expected_float = float(expected)
            
            # For integer values (1 vs 1.0), use exact comparison
            if actual_float == expected_float:
                return True
            
            # For floating point values, use tolerance-based comparison
            # Avoid division by zero
            if expected_float == 0:
                return abs(actual_float) <= tolerance
            
            return abs(actual_float - expected_float) <= abs(expected_float * tolerance)
        except ValueError:
            # Not numeric, do string comparison
            return actual == expected
