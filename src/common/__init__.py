"""
Common package for AI Security Labs.

This package contains shared utilities, logging configuration,
and common functionality used across all lab modules.
"""

from .logging_config import setup_logging, get_logger, LoggerMixin
from .utils import (
    ProjectPaths, set_reproducible_seed, get_device,
    format_bytes, format_duration, safe_create_dir, save_json, load_json,
    get_timestamp, validate_model_file, calculate_model_size, tensor_info,
    ProgressTracker
)

__version__ = "1.0.0"
__author__ = "AI Security Labs Team"

__all__ = [
    # Logging
    'setup_logging',
    'get_logger', 
    'LoggerMixin',
    
    # Utilities
    'ProjectPaths',
    'set_reproducible_seed',
    'get_device',
    'format_bytes',
    'format_duration',
    'safe_create_dir',
    'save_json',
    'load_json',
    'get_timestamp',
    'validate_model_file',
    'calculate_model_size',
    'tensor_info',
    'ProgressTracker'
]
