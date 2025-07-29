"""
OWASP ML01: Input Manipulation Attack Lab.

This lab demonstrates adversarial input manipulation attacks and defenses using 
the Fast Gradient Sign Method (FGSM) on MNIST digit classification, following
OWASP ML01 guidelines.

Components:
- model: SimpleCNN architecture optimized for adversarial analysis
- attack_fgsm: FGSM attack implementation with comprehensive analysis  
- defense_fgsm: Multi-method adversarial defense system
- train_model: Professional model training pipeline

Attack Techniques:
- Fast Gradient Sign Method (FGSM)
- Targeted and untargeted attacks
- Epsilon scaling analysis
- Success rate evaluation

Defense Mechanisms:
- Gradient norm detection
- Prediction confidence analysis
- Combined detection strategies
- ROC curve optimization

Real-world Impact:
- Image classification bypass
- Autonomous vehicle attacks
- Medical diagnosis manipulation
- Security system evasion

Usage:
    # Train vulnerable model
    from src.owasp.ml01_input_manipulation.train_model import ModelTrainer
    trainer = ModelTrainer()
    model = trainer.train()
    
    # Execute FGSM attack
    from src.owasp.ml01_input_manipulation.attack_fgsm import FGSMAttacker
    attacker = FGSMAttacker(model, device)
    adversarial_data, attack_info = attacker.fgsm_attack(data, labels, epsilon=0.25)
    
    # Implement defense
    from src.owasp.ml01_input_manipulation.defense_fgsm import AdversarialDefense
    defense = AdversarialDefense(model, device)
    is_detected, detection_info = defense.detect_adversarial_combined(data, labels)
"""

__version__ = "1.0.0"

# Lab configuration
ML01_CONFIG = {
    'attack_type': 'Input Manipulation',
    'algorithm': 'Fast Gradient Sign Method (FGSM)', 
    'target_model': 'SimpleCNN',
    'dataset': 'MNIST',
    'severity': 'High',
    'complexity': 'Medium',
    'detection_methods': ['gradient_norm', 'confidence_analysis'],
    'mitigation_strategies': ['adversarial_training', 'input_preprocessing', 'detection_systems']
}

# Import main components
from .model import SimpleCNN
from .attack_fgsm import FGSMAttacker  
from .defense_fgsm import AdversarialDefense
from .train_model import ModelTrainer

__all__ = [
    'ML01_CONFIG',
    'SimpleCNN', 
    'FGSMAttacker',
    'AdversarialDefense', 
    'ModelTrainer'
]
