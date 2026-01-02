"""
Setup Constants

Shared constants for the setup module.
"""

# Timeout values (in seconds)
CONFIG_TIMEOUT = 300  # 5 minutes
USER_READ_DELAY = 5  # Time for user to read messages
VOICE_FEEDBACK_DELAY = 3  # Time for voice feedback to complete
ERROR_DISPLAY_DELAY = 8  # Time to display error messages
RETRY_DELAY = 2  # Delay between retries
LOG_INTERVAL = 30  # Log progress every 30 seconds

# Status strings
STATUS_WAITING = "waiting"
STATUS_CONNECTING = "connecting"
STATUS_AUTHENTICATING = "authenticating"
STATUS_SUCCESS = "success"
STATUS_ERROR = "error"

