"""
Structured JSON Logging Configuration (Optional Enhancement)
Adds JSON logging alongside existing logging - doesn't replace it
Enable with: VALLM_JSON_LOGGING=true
"""
import os
import json
import logging
from pathlib import Path
from datetime import datetime
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from typing import Any, Dict

# Check if JSON logging is enabled
JSON_LOGGING_ENABLED = os.getenv("VALLM_JSON_LOGGING", "false").lower() == "true"


class JSONFormatter(logging.Formatter):
    """Structured JSON formatter for production logging"""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON"""
        log_data: Dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add request ID if available
        if hasattr(record, 'request_id'):
            log_data["request_id"] = record.request_id
        
        # Add trace context if available
        if hasattr(record, 'trace_id'):
            log_data["trace_id"] = record.trace_id
        if hasattr(record, 'span_id'):
            log_data["span_id"] = record.span_id
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        # Add extra fields
        if hasattr(record, 'extra_fields'):
            log_data.update(record.extra_fields)
        
        return json.dumps(log_data)


def setup_json_logging(log_dir: Path = None, log_level: str = "INFO"):
    """
    Setup JSON logging handler (optional, alongside existing logging)
    
    Args:
        log_dir: Directory for log files
        log_level: Logging level
    """
    if not JSON_LOGGING_ENABLED:
        return None
    
    if log_dir is None:
        log_dir = Path(__file__).parent / "logs"
    log_dir.mkdir(exist_ok=True)
    
    # Get root logger
    root_logger = logging.getLogger()
    
    # JSON file handler with rotation
    json_log_file = log_dir / "app.json.log"
    json_handler = TimedRotatingFileHandler(
        filename=str(json_log_file),
        when='midnight',
        interval=1,
        backupCount=30,  # Keep 30 days
        encoding='utf-8'
    )
    json_handler.setLevel(getattr(logging, log_level.upper()))
    json_handler.setFormatter(JSONFormatter())
    
    # Add handler
    root_logger.addHandler(json_handler)
    
    logging.getLogger("vallm").info("✅ JSON logging enabled")
    
    return json_handler
