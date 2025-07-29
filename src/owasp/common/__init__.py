"""
OWASP ML Security Labs - Common Utilities.

This module contains OWASP-specific utilities, configurations, and helper functions
used across different OWASP ML security lab modules.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional

from ...common import get_logger, ProjectPaths

logger = get_logger(__name__)


class OWASPConfig:
    """OWASP ML labs configuration management."""
    
    def __init__(self, config_file: Optional[str] = None):
        """Initialize OWASP configuration."""
        self.config_file = config_file
        self._config = self._load_default_owasp_config()
        
        if config_file and Path(config_file).exists():
            self._load_config_file(config_file)
    
    def _load_default_owasp_config(self) -> Dict[str, Any]:
        """Load default OWASP ML labs configuration."""
        return {
            'ml01_input_manipulation': {
                'model': {
                    'architecture': 'SimpleCNN',
                    'input_channels': 1,
                    'num_classes': 10,
                    'learning_rate': 0.001,
                    'batch_size': 64,
                    'epochs': 2
                },
                'attack': {
                    'method': 'FGSM',
                    'epsilon_values': [0.1, 0.25, 0.3],
                    'default_epsilon': 0.25,
                    'targeted': False,
                    'max_iterations': 100
                },
                'defense': {
                    'gradient_threshold': 0.025,
                    'confidence_threshold': 0.8,
                    'detection_methods': ['gradient_norm', 'confidence']
                }
            },
            'ml02_data_poisoning': {
                'enabled': False,
                'description': 'Coming Soon - Data Poisoning Attacks'
            },
            'ml03_model_inversion': {
                'enabled': False,
                'description': 'Coming Soon - Model Inversion Attacks'
            },
            'data': {
                'dataset': 'MNIST',
                'normalize': True,
                'download': True,
                'train_split': 0.8,
                'test_batch_size': 1000
            },
            'logging': {
                'level': 'INFO',
                'format': 'detailed',
                'max_file_size': '10MB',
                'backup_count': 5
            },
            'output': {
                'save_models': True,
                'save_plots': True,
                'generate_reports': True,
                'plot_format': 'png',
                'plot_dpi': 300,
                'results_structure': 'owasp/{lab_name}'
            },
            'security': {
                'enable_validation': True,
                'log_attacks': True,
                'sanitize_inputs': True
            }
        }
    
    def _load_config_file(self, config_file: str) -> None:
        """Load configuration from file."""
        try:
            with open(config_file, 'r') as f:
                file_config = json.load(f)
            
            # Merge configurations
            self._deep_update(self._config, file_config)
            logger.info(f"OWASP configuration loaded from {config_file}")
            
        except Exception as e:
            logger.warning(f"Failed to load OWASP config file {config_file}: {e}")
    
    def _deep_update(self, base_dict: Dict, update_dict: Dict) -> None:
        """Deep update dictionary."""
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get OWASP configuration value using dot notation."""
        keys = key.split('.')
        value = self._config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any) -> None:
        """Set OWASP configuration value using dot notation."""
        keys = key.split('.')
        config = self._config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def save(self, filename: str) -> None:
        """Save OWASP configuration to file."""
        with open(filename, 'w') as f:
            json.dump(self._config, f, indent=2)
        logger.info(f"OWASP configuration saved to {filename}")
    
    def get_lab_config(self, lab_name: str) -> Dict[str, Any]:
        """Get configuration for specific OWASP lab."""
        return self.get(lab_name, {})
    
    def is_lab_enabled(self, lab_name: str) -> bool:
        """Check if specific OWASP lab is enabled."""
        return self.get(f"{lab_name}.enabled", True)


# OWASP ML Top 10 reference mapping
OWASP_ML_TOP_10_MAPPING = {
    "ML01": {
        "name": "Input Manipulation Attack",
        "module": "ml01_input_manipulation",
        "description": "Adversarial examples that cause misclassification",
        "severity": "High",
        "implemented": True
    },
    "ML02": {
        "name": "Data Poisoning Attack", 
        "module": "ml02_data_poisoning",
        "description": "Malicious training data injection",
        "severity": "High",
        "implemented": False
    },
    "ML03": {
        "name": "Model Inversion Attack",
        "module": "ml03_model_inversion", 
        "description": "Extracting sensitive training data from models",
        "severity": "Medium",
        "implemented": False
    },
    "ML04": {
        "name": "Membership Inference Attack",
        "module": "ml04_membership_inference",
        "description": "Determining if data was used in training",
        "severity": "Medium", 
        "implemented": False
    },
    "ML05": {
        "name": "Model Theft",
        "module": "ml05_model_theft",
        "description": "Stealing model functionality and parameters",
        "severity": "High",
        "implemented": False
    },
    "ML06": {
        "name": "AI Supply Chain Attacks",
        "module": "ml06_supply_chain",
        "description": "Compromised models, datasets, or libraries",
        "severity": "Critical",
        "implemented": False
    },
    "ML07": {
        "name": "Transfer Learning Attack",
        "module": "ml07_transfer_learning",
        "description": "Backdoors in pre-trained models",
        "severity": "High",
        "implemented": False
    },
    "ML08": {
        "name": "Model Skewing", 
        "module": "ml08_model_skewing",
        "description": "Gradual model performance degradation",
        "severity": "Medium",
        "implemented": False
    },
    "ML09": {
        "name": "Output Integrity Attack",
        "module": "ml09_output_integrity", 
        "description": "Compromising model output reliability",
        "severity": "High",
        "implemented": False
    },
    "ML10": {
        "name": "Model Poisoning",
        "module": "ml10_model_poisoning",
        "description": "Direct manipulation of model parameters",
        "severity": "Critical",
        "implemented": False
    }
}


def get_owasp_lab_info(lab_id: str) -> Dict[str, Any]:
    """Get information about specific OWASP ML lab."""
    return OWASP_ML_TOP_10_MAPPING.get(lab_id.upper(), {})


def list_available_labs() -> Dict[str, Dict[str, Any]]:
    """List all available OWASP ML labs."""
    return {k: v for k, v in OWASP_ML_TOP_10_MAPPING.items() if v.get('implemented', False)}


def validate_owasp_lab_structure(lab_path: Path) -> bool:
    """Validate OWASP lab directory structure."""
    required_files = ['__init__.py', 'model.py', 'train_model.py']
    
    if not lab_path.is_dir():
        return False
    
    for file in required_files:
        if not (lab_path / file).exists():
            logger.warning(f"Missing required file in {lab_path}: {file}")
            return False
    
    return True


# Export OWASP-specific utilities
__all__ = [
    'OWASPConfig',
    'ProjectPaths',
    'OWASP_ML_TOP_10_MAPPING', 
    'get_owasp_lab_info',
    'list_available_labs',
    'validate_owasp_lab_structure'
]
