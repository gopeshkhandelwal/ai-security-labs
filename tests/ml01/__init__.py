"""
ML01 Test Package

Test suite for the ML01 Input Manipulation lab, covering:
- SimpleCNN model testing
- FGSM attack validation
- Adversarial defense evaluation
- Integration workflow testing
"""

__version__ = "1.0.0"

# ML01 specific test configuration
ML01_TEST_CONFIG = {
    'model_params': {
        'input_channels': 1,
        'num_classes': 10
    },
    'attack_params': {
        'epsilon_range': [0.1, 0.25, 0.3, 0.5],
        'max_iterations': 10
    },
    'defense_params': {
        'gradient_threshold': 5.0,
        'confidence_threshold': 0.5
    },
    'performance_thresholds': {
        'min_attack_success_rate': 0.1,
        'min_defense_accuracy': 0.6,
        'max_false_positive_rate': 0.3
    }
}

__all__ = ['ML01_TEST_CONFIG']
