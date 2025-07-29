"""
Test configuration and fixtures for ML01 tests.
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import tempfile
import shutil

# Test configuration
BATCH_SIZE = 4
NUM_CLASSES = 10
INPUT_SHAPE = (1, 28, 28)
EPSILON_VALUES = [0.1, 0.25, 0.3]


@pytest.fixture
def device():
    """Get computation device for testing."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def sample_data(device):
    """Generate sample MNIST-like data for testing."""
    batch_size = BATCH_SIZE
    data = torch.randn(batch_size, *INPUT_SHAPE).to(device)
    labels = torch.randint(0, NUM_CLASSES, (batch_size,)).to(device)
    return data, labels


@pytest.fixture
def temp_dir():
    """Create temporary directory for test outputs."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    if temp_path.exists():
        shutil.rmtree(temp_path)


@pytest.fixture
def reproducible_seed():
    """Set reproducible random seed for testing."""
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
