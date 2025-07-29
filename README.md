# AI Security Labs

## ⚡ Quick Start

```bash
# Setup and run complete ML01 lab
git clone <repository-url>
cd ai-security-labs
make setup && make ml01-full
```

## 🎯 Features

### 🔬 **OWASP ML01: Input Manipulation Lab**
- **FGSM Attack Implementation**: Fast Gradient Sign Method with batch processing
- **Multi-Method Defense System**: Gradient norm + confidence analysis with ROC curves  
- **Professional CNN Model**: Enhanced SimpleCNN with validation and debugging
- **Comprehensive Evaluation**: Attack success rates, defense metrics, visualization
- **OWASP Compliance**: Follows OWASP ML Security Top 10 guidelines


## 🚀 Commands

```bash
# Complete setup
make setup

# Run ML01 lab components
make ml01-train     # Train OWASP ML01 model
make ml01-attack    # Generate adversarial examples 
make ml01-defense   # Evaluate defenses
make ml01-full      # Complete OWASP ML01 pipeline

# Development workflow
make lint           # Code quality checks
make test           # Run test suite
make clean          # Clean generated files
make monitor        # Monitor system performance
```

## � Core Components

### **SimpleCNN Model** (`src/owasp/ml01_input_manipulation/model.py`)
```python
from src.owasp.ml01_input_manipulation.model import SimpleCNN

model = SimpleCNN()
predictions, confidence = model.predict(data)
gradients = model.get_gradients(data, labels)
```

### **FGSM Attacker** (`src/owasp/ml01_input_manipulation/attack_fgsm.py`)
```python
from src.owasp.ml01_input_manipulation.attack_fgsm import FGSMAttacker

attacker = FGSMAttacker(model, device)
adversarial_data, attack_info = attacker.fgsm_attack(data, labels, epsilon=0.25)
```

### **Adversarial Defense** (`src/owasp/ml01_input_manipulation/defense_fgsm.py`)
```python
from src.owasp.ml01_input_manipulation.defense_fgsm import AdversarialDefense

defense = AdversarialDefense(model, device)
is_adversarial, detection_info = defense.detect_adversarial_combined(data, labels)
```

## 📦 Requirements

- **Python**: 3.8+ with virtual environment support
- **PyTorch**: 1.9+ with optional CUDA acceleration
- **Development Tools**: Included in `requirements-dev.txt`

## 🤝 Contributing

1. Fork and clone repository
2. Setup development environment: `make setup`
3. Create feature branch
4. Implement changes with tests
5. Run quality checks: `make lint test`
6. Submit pull request

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

**Author**: Gopesh Khandelwal <gopeshkhandelwal@gmail.com>

**🔒 Build secure, robust ML systems with hands-on adversarial learning!**
