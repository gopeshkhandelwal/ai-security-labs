# AI Security Labs - Complete Documentation

## Project Overview

AI Security Labs is a comprehensive educational and research platform for exploring adversarial machine learning security. This production-ready implementation provides hands-on experience with adversarial attacks, defense mechanisms, and security evaluation techniques.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Project Structure](#project-structure)
3. [ML01: Input Manipulation Lab](#ml01-input-manipulation-lab)
4. [Development Workflow](#development-workflow)
5. [API Reference](#api-reference)
6. [Security Considerations](#security-considerations)
7. [Contributing](#contributing)
8. [Troubleshooting](#troubleshooting)

## Quick Start

### Prerequisites

- Python 3.8+
- PyTorch 1.9+
- CUDA (optional, for GPU acceleration)

### Installation

```bash
# Clone repository
git clone <repository-url>
cd ai-security-labs

# Setup development environment
make setup

# Activate virtual environment
source venv/bin/activate

# Download MNIST dataset
make data-download

# Run complete ML01 lab
make ml01-full
```

## Project Structure

```
ai-security-labs/
├── src/                     # Source code
│   ├── common/             # Universal utilities
│   │   ├── __init__.py     # Common imports and utilities
│   │   ├── logging_config.py  # Logging configuration
│   │   └── utils.py        # Universal utility functions
│   └── owasp/              # OWASP ML Security Labs
│       ├── __init__.py     # OWASP package initialization
│       ├── common/         # OWASP-specific utilities
│       │   └── __init__.py # OWASP configuration management
│       └── ml01_input_manipulation/  # OWASP ML01 Lab
│           ├── __init__.py     # Lab package initialization
│           ├── model.py        # SimpleCNN model
│           ├── train_model.py  # Model training pipeline
│           ├── attack_fgsm.py  # FGSM attack implementation
│           └── defense_fgsm.py # Adversarial defense system
├── tests/                  # Test suites
├── data/                   # Datasets
├── models/                 # Trained models
├── logs/                   # Application logs
├── results/                # Experiment results
├── docs/                   # Documentation
├── requirements.txt        # Production dependencies
├── requirements-dev.txt    # Development dependencies
└── Makefile               # Development automation
```

## ML01: Input Manipulation Lab

### Overview

The ML01 lab demonstrates adversarial input manipulation attacks and defenses using the FGSM (Fast Gradient Sign Method) algorithm on MNIST digit classification.

### Components

#### 1. SimpleCNN Model (`src/ml01/model.py`)

A convolutional neural network optimized for MNIST classification with built-in security analysis capabilities.

**Key Features:**
- Lightweight CNN architecture
- Gradient computation for adversarial analysis
- Prediction confidence scoring
- Feature map visualization
- Model validation and debugging

**Usage:**
```python
from src.ml01.model import SimpleCNN

model = SimpleCNN()
predictions, confidence = model.predict(data)
gradients = model.get_gradients(data, labels)
```

#### 2. Model Training (`src/ml01/train_model.py`)

Professional training pipeline with progress tracking and model validation.

**Features:**
- Data loading and preprocessing
- Training progress visualization
- Model checkpointing
- Performance metrics tracking
- Automated model saving

**Usage:**
```python
from src.ml01.train_model import ModelTrainer

trainer = ModelTrainer()
trainer.train(epochs=10)
model = trainer.get_model()
```

#### 3. FGSM Attack (`src/ml01/attack_fgsm.py`)

Enterprise-grade implementation of Fast Gradient Sign Method attacks.

**Capabilities:**
- Single and batch adversarial example generation
- Attack success rate analysis
- Perturbation visualization
- Statistical attack analysis
- Comprehensive result reporting

**Usage:**
```python
from src.ml01.attack_fgsm import FGSMAttacker

attacker = FGSMAttacker(model, device)
adversarial_data, attack_info = attacker.fgsm_attack(data, labels, epsilon=0.25)
```

#### 4. Adversarial Defense (`src/ml01/defense_fgsm.py`)

Multi-method defense system for adversarial example detection.

**Detection Methods:**
- Gradient norm analysis
- Prediction confidence analysis
- Combined detection strategies
- ROC curve analysis
- Threshold optimization

**Usage:**
```python
from src.ml01.defense_fgsm import AdversarialDefense

defense = AdversarialDefense(model, device)
is_adversarial, detection_info = defense.detect_adversarial_combined(data, labels)
```

### Lab Workflow

1. **Model Training**
   ```bash
   make ml01-train
   ```

2. **Attack Generation**
   ```bash
   make ml01-attack
   ```

3. **Defense Evaluation**
   ```bash
   make ml01-defense
   ```

4. **Complete Pipeline**
   ```bash
   make ml01-full
   ```

## Development Workflow

### Environment Setup

```bash
# Create virtual environment
make venv

# Install dependencies
make install

# Install development tools
make install-dev

# Setup pre-commit hooks
make setup-hooks
```

### Code Quality

```bash
# Run linting
make lint

# Auto-format code
make format

# Type checking
make typecheck

# Security scan
make security-scan
```

### Testing

```bash
# Run all tests
make test

# Unit tests only
make test-unit

# Integration tests
make test-integration

# Coverage report
make test-coverage
```

### Monitoring

```bash
# View logs
make logs

# Monitor performance
make monitor

# Generate reports
make reports
```

## API Reference

### Common Utilities

#### Logging System
```python
from src.common import get_logger, log_execution_time

logger = get_logger(__name__)

@log_execution_time
def my_function():
    logger.info("Function executed")
```

#### Device Management
```python
from src.common import get_device, get_device_info

device = get_device()
info = get_device_info()
```

#### Path Management
```python
from src.common import ProjectPaths

paths = ProjectPaths()
model_path = paths.models / "simple_cnn.pth"
```

### Model API

#### SimpleCNN
```python
# Model creation
model = SimpleCNN(input_channels=1, num_classes=10)

# Prediction with confidence
predictions, confidence = model.predict(data)

# Probability distribution
probabilities = model.predict_proba(data)

# Gradient computation
gradients = model.get_gradients(data, labels)

# Feature extraction
features = model.get_feature_maps(data)

# Model summary
summary = model.summary()
```

### Attack API

#### FGSMAttacker
```python
# Attacker initialization
attacker = FGSMAttacker(model, device)

# Single attack
adversarial_data, attack_info = attacker.fgsm_attack(data, labels, epsilon=0.25)

# Batch processing
results = attacker.attack_batch(dataloader, epsilon=0.25)

# Visualization
attacker.visualize_attack(original, adversarial, target_pred, adv_pred, epsilon)

# Statistics
stats = attacker.get_attack_statistics()
```

### Defense API

#### AdversarialDefense
```python
# Defense initialization
defense = AdversarialDefense(model, device)

# Gradient-based detection
is_adv, grad_norms = defense.detect_adversarial_gradient(data, labels)

# Confidence-based detection
is_adv, confidence = defense.detect_adversarial_confidence(data)

# Combined detection
is_adv, info = defense.detect_adversarial_combined(data, labels)

# Performance evaluation
results = defense.evaluate_defense_performance(clean_data, adv_data, labels)

# ROC visualization
defense.visualize_roc_curve(results)
```

## Security Considerations

### Model Security

1. **Adversarial Robustness**
   - Regular evaluation against known attacks
   - Defense mechanism validation
   - Robustness metrics monitoring

2. **Model Integrity**
   - Checksum verification for saved models
   - Version control for model artifacts
   - Secure model deployment practices

3. **Data Security**
   - Input validation and sanitization
   - Secure data storage and transmission
   - Privacy-preserving techniques

### Code Security

1. **Dependency Management**
   - Regular security updates
   - Vulnerability scanning
   - Trusted package sources

2. **Access Control**
   - Proper file permissions
   - Secure configuration management
   - Environment isolation

3. **Logging Security**
   - No sensitive data in logs
   - Secure log storage
   - Access auditing

## Contributing

### Development Process

1. **Fork and Clone**
   ```bash
   git clone <your-fork>
   cd ai-security-labs
   ```

2. **Setup Development Environment**
   ```bash
   make setup
   make install-dev
   ```

3. **Create Feature Branch**
   ```bash
   git checkout -b feature/your-feature
   ```

4. **Development Workflow**
   ```bash
   # Make changes
   make lint          # Check code style
   make test          # Run tests
   make security-scan # Security check
   ```

5. **Submit Pull Request**
   - Include comprehensive tests
   - Update documentation
   - Follow code style guidelines

### Code Style Guidelines

- Use type hints for all functions
- Follow PEP 8 style guide
- Include comprehensive docstrings
- Add unit tests for new features
- Maintain >90% test coverage

### Testing Requirements

- Unit tests for all new functions
- Integration tests for workflows
- Performance benchmarks for algorithms
- Security validation for defense mechanisms

## Troubleshooting

### Common Issues

#### CUDA/GPU Issues
```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Force CPU usage
export CUDA_VISIBLE_DEVICES=""
```

#### Memory Issues
```bash
# Reduce batch size in training
export ML01_BATCH_SIZE=32

# Enable memory optimization
export ML01_MEMORY_OPTIMIZE=true
```

#### Dataset Issues
```bash
# Re-download MNIST
make data-clean
make data-download

# Verify data integrity
make data-verify
```

#### Import Issues
```bash
# Check Python path
python -c "import sys; print(sys.path)"

# Reinstall in development mode
pip install -e .
```

### Performance Optimization

#### CPU Optimization
- Use appropriate number of workers for data loading
- Enable PyTorch CPU optimizations
- Consider Intel MKL optimizations

#### GPU Optimization
- Use appropriate batch sizes
- Enable mixed precision training
- Optimize memory usage patterns

#### Memory Management
- Use gradient checkpointing for large models
- Implement proper data loading strategies
- Monitor memory usage patterns

### Debugging Tools

#### Logging
```python
# Enable debug logging
import logging
logging.getLogger().setLevel(logging.DEBUG)

# Use logging decorators
@log_execution_time
def debug_function():
    pass
```

#### Profiling
```bash
# Profile execution
make profile

# Memory profiling
make profile-memory

# GPU profiling
make profile-gpu
```

#### Visualization
```python
# Visualize model architecture
model.summary()

# Plot training curves
trainer.plot_training_curves()

# Visualize attacks
attacker.visualize_attack(...)
```

## Advanced Usage

### Custom Attack Implementation

```python
class CustomAttacker:
    def __init__(self, model, device):
        self.model = model
        self.device = device
    
    def custom_attack(self, data, labels, **kwargs):
        # Implement custom attack logic
        pass
```

### Custom Defense Implementation

```python
class CustomDefense:
    def __init__(self, model, device):
        self.model = model
        self.device = device
    
    def detect_adversarial(self, data, **kwargs):
        # Implement custom detection logic
        pass
```

### Batch Processing

```python
# Large-scale evaluation
results = []
for batch in dataloader:
    batch_results = evaluate_batch(batch)
    results.append(batch_results)

# Parallel processing
from multiprocessing import Pool
with Pool() as pool:
    results = pool.map(process_batch, batches)
```

## License

This project is licensed under the MIT License. See LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{ai_security_labs,
  title={AI Security Labs: Adversarial Machine Learning Platform},
  author={Your Name},
  year={2024},
  url={https://github.com/your-repo/ai-security-labs}
}
```

## Support

For support and questions:
- GitHub Issues: <repository-url>/issues
- Documentation: <documentation-url>
- Email: your-email@domain.com
