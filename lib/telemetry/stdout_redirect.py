"""
Stdout/stderr redirection to Python logging for OTEL capture.

This module redirects stdout and stderr to Python's logging module so that
all print() statements and library output gets captured by OpenTelemetry's
LoggingHandler and sent to BetterStack.

The redirection parses emoji markers and text patterns to determine appropriate
log levels:
- ✓/✅ → INFO
- ⚠️ → WARNING  
- ✗/❌ → ERROR
- Default → INFO
"""

import sys
import logging
import re
from typing import Optional


class LoggingStreamHandler:
    """
    Stream handler that redirects writes to Python's logging module.
    
    This allows all print() statements to be captured by OTEL's LoggingHandler
    and sent to BetterStack, making logs in BetterStack look similar to journalctl.
    """
    
    def __init__(self, logger: logging.Logger, level: int = logging.INFO, original_stream=None):
        """
        Initialize the logging stream handler.
        
        Args:
            logger: Logger instance to send output to
            level: Default log level for messages
            original_stream: Original stream to preserve (for fallback)
        """
        self.logger = logger
        self.default_level = level
        self.original_stream = original_stream
        self.buffer = ""
        
        # Compile regex patterns for efficiency
        # Note: Patterns check for emoji at start OR keywords at word boundaries
        self.success_pattern = re.compile(r'^[✓✅]')
        self.warning_pattern = re.compile(r'^[⚠️⚠]|\bwarning\b', re.IGNORECASE)
        self.error_pattern = re.compile(r'^[✗❌]|\berror\b|\bfailed\b', re.IGNORECASE)
    
    def write(self, message: str):
        """
        Write a message to the logger.
        
        Args:
            message: Message to log
        """
        if not message:
            return
        
        # Handle standalone newlines (don't log them)
        if message == '\n':
            # If we have buffered content, flush it now
            if self.buffer:
                self._log_line(self.buffer)
                self.buffer = ""
            return
        
        # Buffer incomplete lines
        self.buffer += message
        
        # Process complete lines (split on ALL newlines)
        lines = self.buffer.split('\n')
        
        # Keep the last incomplete line in buffer
        self.buffer = lines[-1]
        
        # Log all complete lines
        for line in lines[:-1]:
            if line:  # Skip empty lines
                self._log_line(line)
    
    def _log_line(self, line: str):
        """
        Log a single line with appropriate level detection.
        
        Args:
            line: Line to log
        """
        # Strip ANSI color codes and extra whitespace
        line = self._strip_ansi(line).strip()
        
        if not line:
            return
        
        # Determine log level based on content
        level = self._detect_log_level(line)
        
        # Log with appropriate level
        self.logger.log(level, line)
    
    def _detect_log_level(self, line: str) -> int:
        """
        Detect log level from line content.
        
        Args:
            line: Line to analyze
            
        Returns:
            Log level (logging.INFO, WARNING, or ERROR)
        """
        # Check for error indicators
        if self.error_pattern.search(line):
            return logging.ERROR
        
        # Check for warning indicators
        if self.warning_pattern.search(line):
            return logging.WARNING
        
        # Check for success indicators (also INFO)
        if self.success_pattern.search(line):
            return logging.INFO
        
        # Default to INFO
        return self.default_level
    
    def _strip_ansi(self, text: str) -> str:
        """
        Strip ANSI escape codes from text.
        
        Args:
            text: Text potentially containing ANSI codes
            
        Returns:
            Text without ANSI codes
        """
        ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        return ansi_escape.sub('', text)
    
    def flush(self):
        """Flush any buffered content."""
        if self.buffer:
            self._log_line(self.buffer)
            self.buffer = ""
        
        # Also flush original stream if available
        if self.original_stream and hasattr(self.original_stream, 'flush'):
            try:
                self.original_stream.flush()
            except Exception:
                pass
    
    def isatty(self):
        """Check if this is a TTY (always False for logging stream)."""
        return False
    
    def fileno(self):
        """Return file descriptor (use original stream if available)."""
        if self.original_stream and hasattr(self.original_stream, 'fileno'):
            return self.original_stream.fileno()
        raise OSError("LoggingStreamHandler has no file descriptor")


def redirect_stdout_stderr(logger_name: str = "raspberry-pi-client"):
    """
    Redirect stdout and stderr to Python's logging module.
    
    This function should be called EARLY in the application lifecycle,
    ideally right after logging.basicConfig() and before any print() statements.
    
    Args:
        logger_name: Name of the logger to use for redirected output
        
    Returns:
        Tuple of (original_stdout, original_stderr) for restoration if needed
    """
    # Get logger for redirected output
    logger = logging.getLogger(logger_name)
    
    # Preserve original streams
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    
    # Create logging stream handlers
    stdout_handler = LoggingStreamHandler(logger, level=logging.INFO, original_stream=original_stdout)
    stderr_handler = LoggingStreamHandler(logger, level=logging.ERROR, original_stream=original_stderr)
    
    # Redirect streams
    sys.stdout = stdout_handler
    sys.stderr = stderr_handler
    
    # Log that redirection is active
    logger.info("stdout/stderr redirection activated for OTEL capture")
    
    return original_stdout, original_stderr


def restore_stdout_stderr(original_stdout, original_stderr):
    """
    Restore original stdout and stderr.
    
    Args:
        original_stdout: Original stdout stream
        original_stderr: Original stderr stream
    """
    # Flush any buffered content before restoring
    if hasattr(sys.stdout, 'flush'):
        sys.stdout.flush()
    if hasattr(sys.stderr, 'flush'):
        sys.stderr.flush()
    
    # Restore original streams
    sys.stdout = original_stdout
    sys.stderr = original_stderr


# Module-level storage for original streams (for cleanup)
_original_streams: Optional[tuple] = None


def setup_stdout_redirect(logger_name: str = "raspberry-pi-client") -> bool:
    """
    Setup stdout/stderr redirection with automatic cleanup.
    
    This is the main entry point for enabling redirection.
    
    Args:
        logger_name: Name of the logger to use
        
    Returns:
        True if redirection was setup successfully, False otherwise
    """
    global _original_streams
    
    try:
        _original_streams = redirect_stdout_stderr(logger_name)
        return True
    except Exception as e:
        # If redirection fails, log error and continue without it
        logger = logging.getLogger(logger_name)
        logger.error(f"Failed to setup stdout/stderr redirection: {e}")
        return False


def cleanup_stdout_redirect():
    """Cleanup and restore original streams."""
    global _original_streams
    
    if _original_streams:
        restore_stdout_stderr(*_original_streams)
        _original_streams = None
