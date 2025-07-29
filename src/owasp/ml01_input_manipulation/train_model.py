"""
OWASP ML01: Model Training Pipeline.

Professional training pipeline for SimpleCNN model with comprehensive logging,
progress tracking, and model validation optimized for adversarial research.

Author: Gopesh Khandelwal <gopeshkhandelwal@gmail.com>
License: CC BY-NC 4.0
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from typing import Dict, Any, Tuple, Optional
from pathlib import Path
import json
import argparse
import sys
import time

from ...common import (
    get_logger, get_device, set_reproducible_seed, 
    save_json, get_timestamp, ProgressTracker
)
from ..common import OWASPConfig, ProjectPaths
from .model import SimpleCNN

logger = get_logger(__name__)


def format_duration(seconds: float) -> str:
    """Format duration in seconds to human readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


class ModelTrainer:
    """
    Comprehensive model trainer with logging and monitoring.
    
    This class encapsulates the entire training pipeline including
    data loading, model training, evaluation, and result saving.
    """
    
    def __init__(self, config: Optional[OWASPConfig] = None, paths: Optional[ProjectPaths] = None, device: Optional[torch.device] = None):
        """
        Initialize ModelTrainer.
        
        Args:
            config: OWASP configuration object
            paths: Project paths object
            device: PyTorch device for training
        """
        self.config = config or OWASPConfig()
        self.paths = paths or ProjectPaths.from_root(Path(__file__).parent.parent.parent.parent)
        self.device = device or get_device()
        self.model = None
        self.train_loader = None
        self.test_loader = None
        self.optimizer = None
        self.criterion = nn.CrossEntropyLoss()        
        self.training_history = {
            'train_loss': [],
            'train_accuracy': [],
            'test_loss': [],
            'test_accuracy': [],
            'epochs': [],
            'learning_rate': []
        }
        
        logger.info(f"ModelTrainer initialized on device: {self.device}")
    
    def setup_data(self) -> None:
        """
        Setup data loaders for training and testing.
        
        Creates MNIST data loaders with appropriate transforms and
        logs dataset information.
        """
        logger.info("Setting up data loaders...")
        
        # Define transforms
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        
        if self.config.get('data.normalize', True):
            transform.transforms.append(
                transforms.Normalize((0.1307,), (0.3081,))
            )
            logger.debug("Added normalization to transforms")
        
        # Load datasets
        try:
            train_dataset = datasets.MNIST(
                root=str(self.paths.data),
                train=True,
                transform=transform,
                download=self.config.get('data.download', True)
            )
            
            test_dataset = datasets.MNIST(
                root=str(self.paths.data),
                train=False,
                transform=transform,
                download=self.config.get('data.download', True)
            )
            
            logger.info(f"MNIST dataset loaded - Train: {len(train_dataset)}, Test: {len(test_dataset)}")
            
        except Exception as e:
            logger.error(f"Failed to load MNIST dataset: {e}")
            raise
        
        # Create data loaders
        batch_size = self.config.get('ml01_input_manipulation.model.batch_size', 64)
        test_batch_size = self.config.get('data.test_batch_size', 1000)
        
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )
        
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=test_batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
        
        logger.info(
            f"Data loaders created - "
            f"Train batches: {len(self.train_loader)}, "
            f"Test batches: {len(self.test_loader)}"
        )
    
    def setup_model(self) -> None:
        """
        Setup model, optimizer, and loss criterion.
        
        Initializes the SimpleCNN model and moves it to the appropriate device.
        """
        logger.info("Setting up model...")
        
        # Create model
        input_channels = self.config.get('ml01_input_manipulation.model.input_channels', 1)
        num_classes = self.config.get('ml01_input_manipulation.model.num_classes', 10)
        
        self.model = SimpleCNN(
            num_classes=num_classes,
            device=self.device
        )
        
        # Setup optimizer
        learning_rate = self.config.get('ml01_input_manipulation.model.learning_rate', 0.001)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Setup loss criterion
        self.criterion = nn.CrossEntropyLoss()
        
        logger.info(f"Model setup complete - Device: {self.device}")
        logger.info(f"Optimizer: Adam (lr={learning_rate})")
        model_stats = self.model.get_model_stats()
        logger.debug(f"Model stats: {model_stats}")
    
    def train_epoch(self, epoch: int) -> Tuple[float, float]:
        """
        Train the model for one epoch.
        
        Args:
            epoch: Current epoch number (0-indexed)
            
        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.train()
        
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        # Progress tracking
        progress = ProgressTracker(
            total=len(self.train_loader),
            description=f"Epoch {epoch + 1}"
        )
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            # Move data to device
            data, target = data.to(self.device), target.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            predictions = output.argmax(dim=1)
            correct_predictions += (predictions == target).sum().item()
            total_samples += target.size(0)
            
            # Log progress
            if batch_idx % 100 == 0:
                current_accuracy = 100.0 * correct_predictions / total_samples
                logger.debug(
                    f"Epoch {epoch + 1}, Batch {batch_idx}/{len(self.train_loader)}, "
                    f"Loss: {loss.item():.6f}, Acc: {current_accuracy:.2f}%"
                )
            
            progress.update()
        
        progress.finish()
        
        # Calculate epoch metrics
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100.0 * correct_predictions / total_samples
        
        logger.info(
            f"Epoch {epoch + 1} training complete - "
            f"Loss: {avg_loss:.6f}, Accuracy: {accuracy:.2f}%"
        )
        
        return avg_loss, accuracy
    
    def evaluate_model(self) -> Tuple[float, float]:
        """
        Evaluate the model on test set.
        
        Returns:
            Tuple of (test_loss, test_accuracy)
        """
        logger.info("Evaluating model on test set...")
        
        self.model.eval()
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                predictions = output.argmax(dim=1)
                correct_predictions += (predictions == target).sum().item()
                total_samples += target.size(0)
        
        avg_loss = total_loss / len(self.test_loader)
        accuracy = 100.0 * correct_predictions / total_samples
        
        logger.info(
            f"Test evaluation complete - "
            f"Loss: {avg_loss:.6f}, Accuracy: {accuracy:.2f}%"
        )
        
        return avg_loss, accuracy
    
    def save_model(self, filename: str = None) -> Path:
        """
        Save the trained model with metadata.
        
        Args:
            filename: Optional custom filename
            
        Returns:
            Path to saved model file
        """
        if filename is None:
            timestamp = get_timestamp()
            filename = f"ml01_model_{timestamp}.pt"
        
        # Create OWASP subdirectory in models
        owasp_models_dir = self.paths.models / 'owasp'
        owasp_models_dir.mkdir(exist_ok=True, parents=True)
        
        model_path = owasp_models_dir / filename
        
        try:
            # Save model state dict
            torch.save(self.model.state_dict(), model_path)
            
            # Save training metadata
            metadata = {
                'model_class': 'SimpleCNN',
                'model_config': {
                    'input_channels': self.config.get('ml01_input_manipulation.model.input_channels', 1),
                    'num_classes': self.config.get('ml01_input_manipulation.model.num_classes', 10)
                },
                'training_config': {
                    'epochs': self.config.get('ml01_input_manipulation.model.epochs', 2),
                    'batch_size': self.config.get('ml01_input_manipulation.model.batch_size', 64),
                    'learning_rate': self.config.get('ml01_input_manipulation.model.learning_rate', 0.001)
                },
                'training_history': self.training_history,
                'final_performance': {
                    'train_accuracy': self.training_history['train_accuracy'][-1] if self.training_history['train_accuracy'] else 0,
                    'test_accuracy': self.training_history['test_accuracy'][-1] if self.training_history['test_accuracy'] else 0
                },
                'timestamp': get_timestamp(),
                'device': str(self.device)
            }
            
            metadata_path = model_path.with_suffix('.json')
            save_json(metadata, metadata_path)
            
            logger.info(f"Model saved to {model_path}")
            logger.info(f"Metadata saved to {metadata_path}")
            
            return model_path
            
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            raise
    
    def train(self, epochs: int = None) -> Dict[str, Any]:
        """
        Execute the complete training pipeline.
        
        Args:
            epochs: Number of training epochs (overrides config)
            
        Returns:
            Dictionary containing training results
        """
        if epochs is None:
            epochs = self.config.get('ml01_input_manipulation.model.epochs', 2)
        
        logger.info(f"Starting training pipeline for {epochs} epochs")
        training_start_time = time.time()
        
        # Setup
        self.setup_data()
        self.setup_model()
        
        # Training loop
        for epoch in range(epochs):
            epoch_start_time = time.time()
            
            # Train epoch
            train_loss, train_accuracy = self.train_epoch(epoch)
            
            # Evaluate model
            test_loss, test_accuracy = self.evaluate_model()
            
            # Update history
            self.training_history['epochs'].append(epoch + 1)
            self.training_history['train_loss'].append(train_loss)
            self.training_history['train_accuracy'].append(train_accuracy)
            self.training_history['test_loss'].append(test_loss)
            self.training_history['test_accuracy'].append(test_accuracy)
            self.training_history['learning_rate'].append(
                self.optimizer.param_groups[0]['lr']
            )
            
            epoch_time = time.time() - epoch_start_time
            logger.info(f"Epoch {epoch + 1} completed in {format_duration(epoch_time)}")
        
        # Save model
        model_path = self.save_model("ml01_model.pt")  # Standard name for compatibility
        
        # Calculate total training time
        total_training_time = time.time() - training_start_time
        
        # Prepare results
        results = {
            'success': True,
            'epochs_completed': epochs,
            'training_time': total_training_time,
            'model_path': str(model_path),
            'final_metrics': {
                'train_accuracy': self.training_history['train_accuracy'][-1],
                'test_accuracy': self.training_history['test_accuracy'][-1],
                'train_loss': self.training_history['train_loss'][-1],
                'test_loss': self.training_history['test_loss'][-1]
            },
            'training_history': self.training_history
        }
        
        logger.info(
            f"Training pipeline completed successfully in {format_duration(total_training_time)}"
        )
        logger.info(
            f"Final performance - "
            f"Train: {results['final_metrics']['train_accuracy']:.2f}%, "
            f"Test: {results['final_metrics']['test_accuracy']:.2f}%"
        )
        
        return results


def main():
    """Main entry point for model training."""
    parser = argparse.ArgumentParser(description="Train ML01 SimpleCNN model")
    
    parser.add_argument(
        '--epochs', type=int, default=None,
        help='Number of training epochs (default: from config)'
    )
    parser.add_argument(
        '--batch-size', type=int, default=None,
        help='Training batch size (default: from config)'
    )
    parser.add_argument(
        '--learning-rate', type=float, default=None,
        help='Learning rate (default: from config)'
    )
    parser.add_argument(
        '--log-level', type=str, default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--config', type=str, default=None,
        help='Path to configuration file'
    )
    
    args = parser.parse_args()
    
    try:
        # Setup logging
        # Logging is already configured
        
        # Set reproducible seed
        set_reproducible_seed(args.seed)
        
        # Setup paths from project root (4 levels up from this file) and config
        paths = ProjectPaths.from_root(Path(__file__).parent.parent.parent.parent)
        paths.ensure_dirs()
        
        config = OWASPConfig(args.config)
        
        # Override config with command line arguments
        if args.epochs is not None:
            config.set('ml01_input_manipulation.model.epochs', args.epochs)
        if args.batch_size is not None:
            config.set('ml01_input_manipulation.model.batch_size', args.batch_size)
        if args.learning_rate is not None:
            config.set('ml01_input_manipulation.model.learning_rate', args.learning_rate)
        
        # Create trainer and run training
        trainer = ModelTrainer(config=config, paths=paths)
        results = trainer.train()
        
        # Save results
        results_file = paths.results / 'owasp' / 'ml01' / f"training_results_{get_timestamp()}.json"
        save_json(results, results_file)
        
        logger.info(f"Training results saved to {results_file}")
        
        return 0
        
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
