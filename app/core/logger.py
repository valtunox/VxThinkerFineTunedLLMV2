import logging
import json
import os

# Configure the logger to write to a file and console
log_directory = "app/logs"
log_file = os.path.join(log_directory, "logs.txt")

# Create logs directory if it doesn't exist
os.makedirs(log_directory, exist_ok=True)

logger = logging.getLogger()
logger.setLevel(logging.INFO) # Set a default level, can be overridden by event

# Add StreamHandler if not already present
if not any(isinstance(handler, logging.StreamHandler) for handler in logger.handlers):
    stream_handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

# Add FileHandler if not already present
if not any(isinstance(handler, logging.FileHandler) and handler.baseFilename.endswith(log_file) for handler in logger.handlers):
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

def lambda_handler(event, context=None, log_level=None):
    """
    Lambda-style logger function. Accepts an event dict with 'message' and optional 'level'.
    Can be called from anywhere in the codebase.
    """
    level_str = event.get("level", "INFO").upper()
    message = event.get("message", "")
    extra = event.get("extra", {})

    # Get the logging level constant
    level = getattr(logging, level_str, logging.INFO)

    # Set level for this specific log event if provided, otherwise use logger's default
    # Note: setting level on handler is more common, but this allows per-event level if needed
    # For simplicity and to avoid handler level conflicts, we'll just log at the specified level.

    if extra:
        logger.log(level, f"{message} | Extra: {json.dumps(extra)}")
    else:
        logger.log(level, message)

    return {
        "statusCode": 200,
        "body": json.dumps({"logged": True, "level": level_str, "message": message, "extra": extra})
    } 