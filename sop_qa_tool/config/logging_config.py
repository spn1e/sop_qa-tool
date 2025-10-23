"""
Logging configuration with structured output and rotation.
Supports both development and production logging patterns.
"""

import logging
import logging.handlers
import sys
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
from pythonjsonlogger import jsonlogger

from .settings import get_settings


class StructuredFormatter(jsonlogger.JsonFormatter):
    """Custom JSON formatter with additional context"""
    
    def add_fields(self, log_record: Dict[str, Any], record: logging.LogRecord, message_dict: Dict[str, Any]) -> None:
        super().add_fields(log_record, record, message_dict)
        
        # Add timestamp
        log_record['timestamp'] = datetime.utcnow().isoformat() + 'Z'
        
        # Add service context
        log_record['service'] = 'sop-qa-tool'
        log_record['version'] = '1.0.0'
        
        # Add request context if available
        if hasattr(record, 'request_id'):
            log_record['request_id'] = record.request_id
        
        if hasattr(record, 'user_id'):
            log_record['user_id'] = record.user_id
        
        # Add component context
        if hasattr(record, 'component'):
            log_record['component'] = record.component
        
        # Add operation context
        if hasattr(record, 'operation'):
            log_record['operation'] = record.operation
        
        # Add performance metrics
        if hasattr(record, 'duration_ms'):
            log_record['duration_ms'] = record.duration_ms
        
        if hasattr(record, 'memory_mb'):
            log_record['memory_mb'] = record.memory_mb


class ColoredConsoleFormatter(logging.Formatter):
    """Console formatter with color coding for different log levels"""
    
    # Color codes
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
        'RESET': '\033[0m'      # Reset
    }
    
    def format(self, record: logging.LogRecord) -> str:
        # Add color to level name
        if record.levelname in self.COLORS:
            record.levelname = f"{self.COLORS[record.levelname]}{record.levelname}{self.COLORS['RESET']}"
        
        return super().format(record)


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[Path] = None,
    enable_structured: bool = True,
    enable_console: bool = True,
    max_file_size_mb: int = 10,
    backup_count: int = 5
) -> logging.Logger:
    """
    Set up logging configuration with structured output and rotation.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (optional)
        enable_structured: Enable structured JSON logging for file output
        enable_console: Enable console logging
        max_file_size_mb: Maximum log file size in MB before rotation
        backup_count: Number of backup log files to keep
    
    Returns:
        Configured logger instance
    """
    # Get root logger
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, log_level.upper()))
        
        if enable_structured:
            console_formatter = StructuredFormatter(
                fmt='%(timestamp)s %(levelname)s %(name)s %(message)s'
            )
        else:
            console_formatter = ColoredConsoleFormatter(
                fmt='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
        
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    # File handler with rotation
    if log_file:
        # Ensure log directory exists
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.handlers.RotatingFileHandler(
            filename=log_file,
            maxBytes=max_file_size_mb * 1024 * 1024,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(getattr(logging, log_level.upper()))
        
        if enable_structured:
            file_formatter = StructuredFormatter(
                fmt='%(timestamp)s %(levelname)s %(name)s %(message)s'
            )
        else:
            file_formatter = logging.Formatter(
                fmt='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
        
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the specified name"""
    return logging.getLogger(name)


def log_performance(logger: logging.Logger, operation: str, duration_ms: float, **kwargs) -> None:
    """Log performance metrics with structured data"""
    extra = {
        'operation': operation,
        'duration_ms': duration_ms,
        **kwargs
    }
    logger.info(f"Operation completed: {operation}", extra=extra)


def log_error_with_context(logger: logging.Logger, error: Exception, context: Dict[str, Any]) -> None:
    """Log error with additional context information"""
    extra = {
        'error_type': type(error).__name__,
        'error_message': str(error),
        **context
    }
    logger.error(f"Error occurred: {error}", extra=extra, exc_info=True)


def configure_uvicorn_logging(log_level: str = "INFO") -> Dict[str, Any]:
    """
    Configure Uvicorn logging to integrate with our logging setup.
    
    Args:
        log_level: Logging level for Uvicorn
    
    Returns:
        Uvicorn logging configuration dictionary
    """
    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "()": "uvicorn.logging.DefaultFormatter",
                "fmt": "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
            "structured": {
                "()": "sop_qa_tool.config.logging_config.StructuredFormatter",
                "fmt": "%(timestamp)s %(levelname)s %(name)s %(message)s"
            }
        },
        "handlers": {
            "default": {
                "formatter": "default",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
            },
            "structured": {
                "formatter": "structured",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
            }
        },
        "loggers": {
            "uvicorn": {"handlers": ["default"], "level": log_level, "propagate": False},
            "uvicorn.error": {"handlers": ["default"], "level": log_level, "propagate": False},
            "uvicorn.access": {"handlers": ["default"], "level": log_level, "propagate": False},
        },
    }


def setup_application_logging() -> logging.Logger:
    """
    Set up application logging based on settings configuration.
    
    Returns:
        Configured application logger
    """
    settings = get_settings()
    
    # Determine log file path
    log_file = None
    if settings.log_file_path:
        log_file = settings.log_file_path
    elif settings.is_local_mode():
        # Default to logs directory in local data path
        log_file = settings.local_data_path / "logs" / "sop-qa-tool.log"
    
    # Set up logging
    logger = setup_logging(
        log_level=settings.log_level,
        log_file=log_file,
        enable_structured=settings.enable_structured_logging,
        enable_console=True
    )
    
    # Log startup information
    app_logger = get_logger("sop_qa_tool")
    app_logger.info("Application logging configured", extra={
        'component': 'logging',
        'mode': settings.mode.value,
        'log_level': settings.log_level,
        'structured_logging': settings.enable_structured_logging,
        'log_file': str(log_file) if log_file else None
    })
    
    return app_logger


# Context managers for structured logging
class LogContext:
    """Context manager for adding structured logging context"""
    
    def __init__(self, logger: logging.Logger, **context):
        self.logger = logger
        self.context = context
        self.old_factory = None
    
    def __enter__(self):
        self.old_factory = logging.getLogRecordFactory()
        
        def record_factory(*args, **kwargs):
            record = self.old_factory(*args, **kwargs)
            for key, value in self.context.items():
                setattr(record, key, value)
            return record
        
        logging.setLogRecordFactory(record_factory)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        logging.setLogRecordFactory(self.old_factory)


class PerformanceLogger:
    """Context manager for performance logging"""
    
    def __init__(self, logger: logging.Logger, operation: str, **context):
        self.logger = logger
        self.operation = operation
        self.context = context
        self.start_time = None
    
    def __enter__(self):
        import time
        self.start_time = time.time()
        self.logger.debug(f"Starting operation: {self.operation}", extra=self.context)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        import time
        duration_ms = (time.time() - self.start_time) * 1000
        
        if exc_type is None:
            log_performance(self.logger, self.operation, duration_ms, **self.context)
        else:
            self.context['duration_ms'] = duration_ms
            log_error_with_context(self.logger, exc_val, self.context)