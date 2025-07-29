"""
OWASP ML Security Labs Package.

This package implements hands-on labs for the OWASP Machine Learning Security Top 10,
providing practical demonstrations of ML vulnerabilities and defense mechanisms.

Available Labs:
- ML01: Input Manipulation - Adversarial examples using FGSM
- ML02: Data Poisoning (Coming Soon)
- ML03: Model Inversion (Coming Soon)
- ML04: Membership Inference (Coming Soon)
- ML05: Model Theft (Coming Soon)
- ML06: AI Supply Chain (Coming Soon)
- ML07: Transfer Learning (Coming Soon)  
- ML08: Model Skewing (Coming Soon)
- ML09: Output Integrity (Coming Soon)
- ML10: Model Poisoning (Coming Soon)

Each lab follows the OWASP structure:
- Attack demonstration
- Defense implementation  
- Real-world impact analysis
- Mitigation strategies

Usage:
    from src.owasp.ml01_input_manipulation import FGSMAttacker, AdversarialDefense
    from src.owasp.ml01_input_manipulation.model import SimpleCNN
"""

__version__ = "1.0.0"
__author__ = "AI Security Labs Team"

# OWASP ML Top 10 reference
OWASP_ML_TOP_10 = {
    "ML01": "Input Manipulation Attack",
    "ML02": "Data Poisoning Attack", 
    "ML03": "Model Inversion Attack",
    "ML04": "Membership Inference Attack",
    "ML05": "Model Theft",
    "ML06": "AI Supply Chain Attacks",
    "ML07": "Transfer Learning Attack",
    "ML08": "Model Skewing",
    "ML09": "Output Integrity Attack", 
    "ML10": "Model Poisoning"
}

# Available labs
AVAILABLE_LABS = ["ML01"]

__all__ = [
    "OWASP_ML_TOP_10",
    "AVAILABLE_LABS"
]
