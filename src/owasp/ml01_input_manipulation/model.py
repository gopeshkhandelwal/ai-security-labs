"""
OWASP ML01: SimpleCNN Model for Input Manipulation Lab.

Enhanced CNN architecture optimized for adversarial analysis and security research.
Includes comprehensive validation, gradient computation, and debugging capabilities
for studying input manipulation attacks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional

from ...common import get_logger, get_device

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional

from ...common import get_logger, get_device


logger = get_logger(__name__)


class SimpleCNN(nn.Module):
    """
    Enhanced SimpleCNN for adversarial robustness analysis.
    
    Features:
    - Configurable architecture
    - Gradient tracking for adversarial analysis
    - Model statistics computation
    - Device management
    """
    
    def __init__(
        self, 
        num_classes: int = 10,
        dropout_rate: float = 0.5,
        use_batch_norm: bool = True,
        device: Optional[str] = None
    ):
        super(SimpleCNN, self).__init__()
        
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        self.device = device or get_device()
        
        logger.info(f"Initializing SimpleCNN with {num_classes} classes, dropout={dropout_rate}, batch_norm={use_batch_norm}")
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        
        # Batch normalization layers (optional)
        if self.use_batch_norm:
            self.bn1 = nn.BatchNorm2d(32)
            self.bn2 = nn.BatchNorm2d(64)
            self.bn3 = nn.BatchNorm2d(128)
        
        # Pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        
        # Dropout layer
        self.dropout = nn.Dropout(self.dropout_rate)
        
        # Calculate the size of flattened features after convolution layers
        # For MNIST (28x28): after 3 conv+pool layers: 128 * 3 * 3 = 1152
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)
        
        # Move model to device
        self.to(self.device)
        
        # Log model info
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"Model created with {total_params:,} total parameters ({trainable_params:,} trainable)")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        # Ensure input is on correct device
        x = x.to(self.device)
        
        # First convolutional block
        x = self.conv1(x)
        if self.use_batch_norm:
            x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)
        
        # Second convolutional block
        x = self.conv2(x)
        if self.use_batch_norm:
            x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)
        
        # Third convolutional block
        x = self.conv3(x)
        if self.use_batch_norm:
            x = self.bn3(x)
        x = F.relu(x)
        x = self.pool(x)
        
        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)
        
        # Fully connected layers with dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        
        x = self.fc3(x)
        
        return x
    
    def get_model_stats(self) -> Dict[str, Any]:
        """Get comprehensive model statistics."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        # Calculate model size in MB
        param_size = 0
        buffer_size = 0
        
        for param in self.parameters():
            param_size += param.numel() * param.element_size()
        
        for buffer in self.buffers():
            buffer_size += buffer.numel() * buffer.element_size()
        
        model_size_mb = (param_size + buffer_size) / 1024 / 1024
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': model_size_mb,
            'num_classes': self.num_classes,
            'dropout_rate': self.dropout_rate,
            'use_batch_norm': self.use_batch_norm,
            'device': str(self.device)
        }
    
    def enable_gradient_computation(self):
        """Enable gradient computation for adversarial analysis."""
        for param in self.parameters():
            param.requires_grad_(True)
        logger.debug("Enabled gradient computation for all parameters")
    
    def disable_gradient_computation(self):
        """Disable gradient computation to save memory during inference."""
        for param in self.parameters():
            param.requires_grad_(False)
        logger.debug("Disabled gradient computation for all parameters")
    
    def get_feature_maps(self, x: torch.Tensor, layer_name: str = 'conv1') -> torch.Tensor:
        """
        Extract feature maps from a specific layer for visualization.
        
        Args:
            x: Input tensor
            layer_name: Name of layer to extract features from ('conv1', 'conv2', 'conv3')
            
        Returns:
            Feature maps from the specified layer
        """
        x = x.to(self.device)
        
        if layer_name == 'conv1':
            x = self.conv1(x)
            if self.use_batch_norm:
                x = self.bn1(x)
            return F.relu(x)
        
        # Pass through conv1
        x = self.conv1(x)
        if self.use_batch_norm:
            x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)
        
        if layer_name == 'conv2':
            x = self.conv2(x)
            if self.use_batch_norm:
                x = self.bn2(x)
            return F.relu(x)
        
        # Pass through conv2
        x = self.conv2(x)
        if self.use_batch_norm:
            x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)
        
        if layer_name == 'conv3':
            x = self.conv3(x)
            if self.use_batch_norm:
                x = self.bn3(x)
            return F.relu(x)
        
        raise ValueError(f"Unknown layer name: {layer_name}")


def create_model(config: Dict[str, Any]) -> SimpleCNN:
    """
    Factory function to create a SimpleCNN model from configuration.
    
    Args:
        config: Configuration dictionary containing model parameters
        
    Returns:
        Configured SimpleCNN model
    """
    model_config = config.get('model', {})
    
    model = SimpleCNN(
        num_classes=model_config.get('num_classes', 10),
        dropout_rate=model_config.get('dropout_rate', 0.5),
        use_batch_norm=model_config.get('use_batch_norm', True),
        device=config.get('device')
    )
    
    logger.info("Created SimpleCNN model from configuration")
    return model
