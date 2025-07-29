"""
AI Security Labs - Test Package

This package contains comprehensive test suites for the AI Security Labs project,
including unit tests, integration tests, and performance benchmarks.
"""

__version__ = "1.0.0"
__author__ = "AI Security Labs Team"

# Test configuration
TEST_CONFIG = {
    'batch_size': 4,
    'num_classes': 10,
    'input_shape': (1, 28, 28),
    'epsilon_values': [0.1, 0.25, 0.3],
    'test_timeout': 300,  # seconds
    'reproducible_seed': 42
}

# Make test config available
__all__ = ['TEST_CONFIG']
