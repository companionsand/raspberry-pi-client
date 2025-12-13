#!/usr/bin/env python3
"""
Standalone human presence detector for Raspberry Pi.

This script runs the HumanPresenceDetector as a standalone service,
logging human presence events to console and optionally to a file.

Usage:
    # Run indefinitely
    python3 lib/presence_detection/standalone_pi.py
    
    # Run for specific duration
    python3 lib/presence_detection/standalone_pi.py --duration 3600  # 1 hour
    
    # Run with custom threshold
    python3 lib/presence_detection/standalone_pi.py --threshold 0.25
    
    # Run with log file
    python3 lib/presence_detection/standalone_pi.py --log-file /var/log/presence.log
    
    # Run as systemd service (see instructions below)
    sudo systemctl start presence-detector
"""

import sys
import os
import time
import signal
import argparse
import logging
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from lib.presence_detection import HumanPresenceDetector


def setup_logging(log_file=None, verbose=False):
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=handlers
    )
    
    return logging.getLogger(__name__)


def main():
    """Run standalone presence detector."""
    parser = argparse.ArgumentParser(
        description='Standalone Human Presence Detector',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run indefinitely
  python3 standalone_pi.py
  
  # Run for 1 hour
  python3 standalone_pi.py --duration 3600
  
  # Run with custom threshold
  python3 standalone_pi.py --threshold 0.25
  
  # Run with logging to file
  python3 standalone_pi.py --log-file /tmp/presence.log
  
  # Verbose output
  python3 standalone_pi.py --verbose

The detector runs every 5 seconds and only logs when humans are detected.
Press Ctrl+C to stop gracefully.
        """
    )
    
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.3,
        help='Detection threshold (0.0-1.0, default: 0.3)'
    )
    parser.add_argument(
        '--duration',
        type=int,
        default=None,
        help='Run for specified duration in seconds (default: indefinite)'
    )
    parser.add_argument(
        '--log-file',
        type=str,
        default=None,
        help='Log file path (default: console only)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    parser.add_argument(
        '--mic-device',
        type=int,
        default=None,
        help='Microphone device index (default: system default)'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Only log detections, not every cycle'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_file, args.verbose)
    
    # Print banner
    logger.info("="*60)
    logger.info("Standalone Human Presence Detector")
    logger.info("="*60)
    logger.info(f"Threshold: {args.threshold}")
    logger.info(f"Duration: {'indefinite' if args.duration is None else f'{args.duration}s'}")
    logger.info(f"Log file: {args.log_file or 'console only'}")
    logger.info(f"Mic device: {args.mic_device or 'default'}")
    logger.info("="*60)
    logger.info("")
    
    # Track statistics
    stats = {
        'start_time': time.time(),
        'detections': 0,
        'last_detection': None,
        'cycles': 0,
        'max_score': 0.0,
        'min_score': 1.0
    }
    
    # Custom callbacks
    def signal_handler(sig, frame):
        logger.info("")
        logger.info("="*60)
        logger.info("Shutting down...")
        logger.info("="*60)
        
        # Print statistics
        elapsed = time.time() - stats['start_time']
        logger.info(f"Runtime: {elapsed:.0f}s ({elapsed/60:.1f} minutes)")
        logger.info(f"Cycles: {stats['cycles']}")
        logger.info(f"Detections: {stats['detections']}")
        if stats['detections'] > 0:
            logger.info(f"Detection rate: {stats['detections']/stats['cycles']*100:.1f}%")
        if stats['last_detection']:
            logger.info(f"Last detection: {stats['last_detection']}")
        logger.info(f"Score range: [{stats['min_score']:.3f}, {stats['max_score']:.3f}]")
        logger.info("="*60)
        
        detector.cleanup()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Start detector
    def on_cycle_callback(weighted_score):
        """Called every detection cycle with the weighted score."""
        stats['cycles'] += 1
        stats['max_score'] = max(stats['max_score'], weighted_score)
        stats['min_score'] = min(stats['min_score'], weighted_score)
        
        if not args.quiet:
            # Log every cycle with weighted score
            timestamp = datetime.now().strftime('%H:%M:%S')
            if weighted_score >= args.threshold:
                logger.info(f"[{timestamp}] ✓ Score: {weighted_score:.3f} (HUMAN DETECTED)")
            else:
                logger.info(f"[{timestamp}] • Score: {weighted_score:.3f}")
    
    def on_detection_callback(weighted_score, top_classes):
        """Called when human presence detected."""
        stats['detections'] += 1
        stats['last_detection'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Initialize detector with callbacks
    logger.info("Initializing detector...")
    try:
        detector = HumanPresenceDetector(
            mic_device_index=args.mic_device,
            threshold=args.threshold,
            on_detection=on_detection_callback,
            on_cycle=on_cycle_callback
        )
    except Exception as e:
        logger.error(f"Failed to initialize detector: {e}")
        logger.error("Make sure:")
        logger.error("  1. yamnet.onnx exists in models/")
        logger.error("  2. onnxruntime is installed: pip install onnxruntime")
        logger.error("  3. Audio device is available")
        return 1
    
    logger.info("")
    
    # Setup signal handlers
    logger.info("Starting detection...")
    logger.info("The detector runs every 5 seconds")
    if args.quiet:
        logger.info("Quiet mode: Logs appear ONLY when humans are detected")
    else:
        logger.info("Logging every cycle with weighted probability")
    logger.info("Press Ctrl+C to stop")
    logger.info("")
    
    detector.start()
    
    # Run for specified duration or indefinitely
    try:
        start_time = time.time()
        
        if args.duration:
            # Run for specific duration
            logger.info(f"Running for {args.duration} seconds...")
            
            while time.time() - start_time < args.duration:
                time.sleep(1)
                
                # Show progress every 60 seconds
                elapsed = int(time.time() - start_time)
                if elapsed % 60 == 0 and elapsed > 0:
                    remaining = args.duration - elapsed
                    rate = stats['detections']/stats['cycles']*100 if stats['cycles'] > 0 else 0
                    logger.info(f"[{elapsed}s elapsed, {remaining}s remaining, {stats['detections']} detections ({rate:.1f}%)]")
        else:
            # Run indefinitely
            logger.info("Running indefinitely (Ctrl+C to stop)...")
            
            while True:
                time.sleep(60)
                
                # Show status every minute
                elapsed = int(time.time() - start_time)
                rate = stats['detections']/stats['cycles']*100 if stats['cycles'] > 0 else 0
                logger.info(f"[{elapsed/60:.0f} minutes elapsed, {stats['cycles']} cycles, {stats['detections']} detections ({rate:.1f}%)]")
    
    except KeyboardInterrupt:
        pass
    
    # Cleanup
    signal_handler(None, None)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
