"""
Common utilities for AI Security Labs.

This module contains shared utilities, constants, and helper functions
used across different lab modules.
"""

import os
import json
import torch
import random
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
from datetime import datetime

from .logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class ProjectPaths:
    """Centralized project path management."""
    
    root: Path
    src: Path
    data: Path
    models: Path
    results: Path
    logs: Path
    tests: Path
    docs: Path
    
    @classmethod
    def from_root(cls, root_path: Union[str, Path]) -> 'ProjectPaths':
        """Create ProjectPaths from root directory."""
        root = Path(root_path).resolve()
        
        return cls(
            root=root,
            src=root / 'src',
            data=root / 'data',
            models=root / 'models',
            results=root / 'results',
            logs=root / 'logs',
            tests=root / 'tests',
            docs=root / 'docs'
        )
    
    def ensure_dirs(self) -> None:
        """Create all directories if they don't exist."""
        for path in [self.data, self.models, self.results, self.logs]:
            path.mkdir(exist_ok=True, parents=True)
        
        # Create subdirectories
        (self.results / 'owasp').mkdir(exist_ok=True)
        (self.results / 'owasp' / 'ml01').mkdir(exist_ok=True)
        
        logger.info("Project directories ensured")


def set_reproducible_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducible results.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Make CuDNN deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    logger.info(f"Reproducible seed set to {seed}")


def get_device(prefer_cuda: bool = True) -> torch.device:
    """
    Get the best available device for computation.
    
    Args:
        prefer_cuda: Whether to prefer CUDA if available
        
    Returns:
        PyTorch device
    """
    if prefer_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
    else:
        device = torch.device('cpu')
        logger.info("Using CPU device")
    
    return device


def format_bytes(bytes_value: int) -> str:
    """Format bytes to human readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.1f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.1f} TB"


def format_duration(seconds: float) -> str:
    """Format duration in seconds to human readable format."""
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def safe_create_dir(path: Union[str, Path]) -> Path:
    """Safely create directory with error handling."""
    path = Path(path)
    try:
        path.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Directory created: {path}")
        return path
    except Exception as e:
        logger.error(f"Failed to create directory {path}: {e}")
        raise


def save_json(data: Any, filepath: Union[str, Path], indent: int = 2) -> None:
    """Save data to JSON file with error handling."""
    filepath = Path(filepath)
    safe_create_dir(filepath.parent)
    
    try:
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=indent, default=str)
        logger.debug(f"JSON saved to {filepath}")
    except Exception as e:
        logger.error(f"Failed to save JSON to {filepath}: {e}")
        raise


def load_json(filepath: Union[str, Path]) -> Any:
    """Load data from JSON file with error handling."""
    filepath = Path(filepath)
    
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        logger.debug(f"JSON loaded from {filepath}")
        return data
    except Exception as e:
        logger.error(f"Failed to load JSON from {filepath}: {e}")
        raise


def get_timestamp(format_str: str = "%Y%m%d_%H%M%S") -> str:
    """Get current timestamp as formatted string."""
    return datetime.now().strftime(format_str)


def validate_model_file(model_path: Union[str, Path]) -> bool:
    """Validate if model file exists and is loadable."""
    model_path = Path(model_path)
    
    if not model_path.exists():
        logger.warning(f"Model file not found: {model_path}")
        return False
    
    try:
        # Try to load the model
        torch.load(model_path, map_location='cpu')
        logger.debug(f"Model file validated: {model_path}")
        return True
    except Exception as e:
        logger.error(f"Invalid model file {model_path}: {e}")
        return False


def calculate_model_size(model: torch.nn.Module) -> Tuple[int, int]:
    """
    Calculate model size information.
    
    Args:
        model: PyTorch model
        
    Returns:
        Tuple of (total_params, trainable_params)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.debug(f"Model size - Total: {total_params:,}, Trainable: {trainable_params:,}")
    
    return total_params, trainable_params


def tensor_info(tensor: torch.Tensor) -> Dict[str, Any]:
    """Get comprehensive tensor information."""
    return {
        'shape': list(tensor.shape),
        'dtype': str(tensor.dtype),
        'device': str(tensor.device),
        'requires_grad': tensor.requires_grad,
        'min': tensor.min().item() if tensor.numel() > 0 else None,
        'max': tensor.max().item() if tensor.numel() > 0 else None,
        'mean': tensor.mean().item() if tensor.numel() > 0 else None,
        'std': tensor.std().item() if tensor.numel() > 0 else None,
        'memory_mb': tensor.element_size() * tensor.numel() / (1024 * 1024)
    }


class ProgressTracker:
    """Simple progress tracking utility."""
    
    def __init__(self, total: int, description: str = "Progress"):
        """Initialize progress tracker."""
        self.total = total
        self.current = 0
        self.description = description
        self.start_time = datetime.now()
        
        logger.info(f"{description} started (total: {total})")
    
    def update(self, increment: int = 1) -> None:
        """Update progress."""
        self.current += increment
        progress = (self.current / self.total) * 100
        
        if self.current % max(1, self.total // 20) == 0 or self.current == self.total:
            elapsed = (datetime.now() - self.start_time).total_seconds()
            rate = self.current / elapsed if elapsed > 0 else 0
            
            logger.info(f"{self.description}: {self.current}/{self.total} ({progress:.1f}%) - {rate:.1f} items/s")
    
    def finish(self) -> None:
        """Mark progress as finished."""
        elapsed = (datetime.now() - self.start_time).total_seconds()
        logger.info(f"{self.description} completed in {format_duration(elapsed)}")


# Export commonly used utilities
__all__ = [
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
