"""
Common logging configuration for AI Security Labs.

This module provides centralized logging configuration with proper formatting,
file rotation, and different log levels for development and production.
"""

import logging
import logging.config
import os
import sys
from pathlib import Path
from typing import Optional


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for console output."""
    
    # Color codes
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'        # Reset
    }
    
    def format(self, record):
        """Format log record with colors."""
        if record.levelname in self.COLORS:
            record.levelname = (
                f"{self.COLORS[record.levelname]}"
                f"{record.levelname}"
                f"{self.COLORS['RESET']}"
            )
        return super().format(record)


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    log_dir: str = "logs",
    enable_console: bool = True,
    enable_file: bool = True,
    module_name: str = "ai_security_labs"
) -> logging.Logger:
    """
    Setup centralized logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Custom log file name (optional)
        log_dir: Directory for log files
        enable_console: Enable console logging
        enable_file: Enable file logging
        module_name: Module name for logger
        
    Returns:
        Configured logger instance
    """
    # Create logs directory
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)
    
    # Default log file
    if log_file is None:
        log_file = f"{module_name}.log"
    
    log_file_path = log_path / log_file
    
    # Logging configuration
    config = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'detailed': {
                'format': (
                    '%(asctime)s | %(name)s | %(levelname)s | '
                    '%(filename)s:%(lineno)d | %(funcName)s() | %(message)s'
                ),
                'datefmt': '%Y-%m-%d %H:%M:%S'
            },
            'simple': {
                'format': '%(asctime)s | %(levelname)s | %(message)s',
                'datefmt': '%H:%M:%S'
            },
            'colored': {
                '()': ColoredFormatter,
                'format': (
                    '%(asctime)s | %(name)s | %(levelname)s | %(message)s'
                ),
                'datefmt': '%H:%M:%S'
            }
        },
        'handlers': {},
        'loggers': {
            module_name: {
                'level': log_level,
                'handlers': [],
                'propagate': False
            }
        },
        'root': {
            'level': log_level,
            'handlers': []
        }
    }
    
    # Console handler
    if enable_console:
        config['handlers']['console'] = {
            'class': 'logging.StreamHandler',
            'level': log_level,
            'formatter': 'colored',
            'stream': sys.stdout
        }
        config['loggers'][module_name]['handlers'].append('console')
        config['root']['handlers'].append('console')
    
    # File handler
    if enable_file:
        config['handlers']['file'] = {
            'class': 'logging.handlers.RotatingFileHandler',
            'level': log_level,
            'formatter': 'detailed',
            'filename': str(log_file_path),
            'maxBytes': 10 * 1024 * 1024,  # 10MB
            'backupCount': 5,
            'encoding': 'utf-8'
        }
        config['loggers'][module_name]['handlers'].append('file')
        config['root']['handlers'].append('file')
    
    # Apply configuration
    logging.config.dictConfig(config)
    
    # Get logger
    logger = logging.getLogger(module_name)
    
    # Log initial message
    logger.info(f"Logging initialized - Level: {log_level}, File: {log_file_path}")
    
    return logger


def get_logger(name: str, log_level: Optional[str] = None) -> logging.Logger:
    """
    Get a logger instance with proper configuration.
    
    Args:
        name: Logger name (usually __name__)
        log_level: Optional log level override
        
    Returns:
        Configured logger instance
    """
    # Get environment log level
    env_log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
    final_log_level = log_level or env_log_level
    
    # Setup logging if not already configured
    if not logging.getLogger().handlers:
        setup_logging(log_level=final_log_level)
    
    logger = logging.getLogger(name)
    
    if log_level:
        logger.setLevel(getattr(logging, log_level.upper()))
    
    return logger


class LoggerMixin:
    """Mixin class to add logging capabilities to any class."""
    
    @property
    def logger(self) -> logging.Logger:
        """Get logger instance for the class."""
        if not hasattr(self, '_logger'):
            self._logger = get_logger(self.__class__.__module__)
        return self._logger


def log_execution_time(func):
    """Decorator to log function execution time."""
    import time
    from functools import wraps
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        start_time = time.time()
        
        logger.debug(f"Starting {func.__name__}")
        
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.info(f"{func.__name__} completed in {execution_time:.3f}s")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"{func.__name__} failed after {execution_time:.3f}s: {e}")
            raise
    
    return wrapper


def log_method_calls(cls):
    """Class decorator to log all method calls."""
    for attr_name in dir(cls):
        attr = getattr(cls, attr_name)
        if callable(attr) and not attr_name.startswith('_'):
            setattr(cls, attr_name, log_execution_time(attr))
    return cls


# Export commonly used functions
__all__ = [
    'setup_logging',
    'get_logger', 
    'LoggerMixin',
    'log_execution_time',
    'log_method_calls'
]
