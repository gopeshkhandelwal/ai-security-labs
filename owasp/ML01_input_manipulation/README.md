# OWASP ML01: Input Manipulation Lab

This lab demonstrates **adversarial input manipulation** attacks and defenses using the Fast Gradient Sign Method (FGSM) on MNIST digit classification.

## 🎯 Overview

**OWASP ML01** focuses on how malicious inputs can fool machine learning models by adding imperceptible perturbations that cause misclassification. This lab shows both the attack and defense perspectives.

## 🏗️ Architecture

```
ML01_input_manipulation/
├── model/
│   └── simple_cnn.py          # Simple CNN for MNIST classification
├── utils/
│   └── visualize.py           # Image comparison utilities
├── train_model.py             # Train the CNN model
├── attack_fgsm.py             # FGSM adversarial attack
├── defense_fgsm.py            # Adversarial detection defense
└── README.md                  # This file
```

## 🚀 Quick Start

### 1. **Setup Environment**
```bash
# From project root
pip install -r requirements.txt
```

### 2. **Train the Model**
```bash
python owasp/ML01_input_manipulation/train_model.py
```
**Output**: Creates `ml01_model.pt` with ~97% accuracy on MNIST

### 3. **Run FGSM Attack**
```bash
python owasp/ML01_input_manipulation/attack_fgsm.py
```
**Output**: 
- Console: `Original: 0, Adversarial: 6` (example)
- Image: `results/original_vs_adversarial_1.png`

### 4. **Test Defense Mechanism**
```bash
python owasp/ML01_input_manipulation/defense_fgsm.py
```
**Output**:
- Console: `Adversarial detected: True, Pred: 9, Confidence: 0.52`
- Image: `results/original_vs_adversarial_flagged.png`

## 🔬 Technical Details

### **Model Architecture**
- **Input**: 28×28 grayscale MNIST images
- **Architecture**: Simple CNN (Conv2D → ReLU → Flatten → Dense)
- **Output**: 10 classes (digits 0-9)
- **Accuracy**: ~97% on test set

### **FGSM Attack Algorithm**
```python
# Core attack formula
perturbation = epsilon * gradient.sign()
adversarial_image = original_image + perturbation
```

**Parameters**:
- `epsilon = 0.25`: Attack strength (higher = more visible perturbation)
- Uses gradient of loss w.r.t. input to maximize prediction error

### **Defense Strategy**
Detects adversarial inputs using two metrics:

1. **Gradient Magnitude**: `torch.norm(data_grad) > 3.0`
2. **Prediction Confidence**: `max_probability < 0.85`

**Detection Logic**: Flag as adversarial if either threshold is exceeded.

## 📊 Expected Results

### **Successful Attack Example**
```
Original: 2, Adversarial: 7
```
- Model correctly predicts "2" for clean image
- Same model predicts "7" for adversarially perturbed image
- **To humans**: Both images look like "2"

### **Defense Detection Example**
```
Adversarial detected: True, Pred: 4, Confidence: 0.31
```
- Low confidence (0.31) indicates suspicious input
- System correctly flags as potential adversarial attack

## 🎯 Business Impact

### **Security Risks Demonstrated**
- **Model Vulnerability**: High-accuracy models can be easily fooled
- **Attack Transferability**: Adversarial examples often work across different models
- **Real-world Applications**: 
  - Autonomous vehicles (traffic sign misclassification)
  - Medical imaging (diagnostic errors)
  - Security systems (biometric bypass)

### **Mitigation Strategies**
- **Input Validation**: Gradient-based detection
- **Confidence Thresholding**: Reject low-confidence predictions
- **Adversarial Training**: Train models on adversarial examples
- **Ensemble Methods**: Use multiple models for consensus

## 🔧 Customization

### **Adjust Attack Strength**
```python
# In attack_fgsm.py
epsilon = 0.1   # Weaker attack (less visible)
epsilon = 0.5   # Stronger attack (more visible)
```

### **Tune Detection Thresholds**
```python
# In defense_fgsm.py
def is_adversarial(data_grad, probs, threshold_grad=2.0, threshold_conf=0.9):
    # More sensitive detection
```

## 📈 Performance Metrics

| Metric | Value |
|--------|-------|
| Model Accuracy | ~97% |
| Attack Success Rate | ~95% |
| Detection Accuracy | ~85% |
| False Positive Rate | ~10% |

## 🔗 References

- [OWASP Machine Learning Security Top 10](https://owasp.org/www-project-machine-learning-security-top-10/)
- [Explaining and Harnessing Adversarial Examples (Goodfellow et al.)](https://arxiv.org/abs/1412.6572)
- [FGSM Paper](https://arxiv.org/abs/1412.6572)

## 🏷️ Tags

`adversarial-ml` `fgsm` `owasp-ml01` `input-manipulation` `ai-security` `pytorch` `mnist`
