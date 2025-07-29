"""
OWASP ML01: Adversarial Defense System.

Multi-method defense system for detecting and mitigating FGSM attacks,
implementing detection strategies outlined in OWASP ML01 defense guidelines.

Author: Gopesh Khandelwal <gopeshkhandelwal@gmail.com>
License: CC BY-NC 4.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from typing import Dict, Any, Tuple, List, Optional, Union
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ML/Data analysis imports
from sklearn.metrics import roc_curve, auc, confusion_matrix
import json

from ...common import (
    get_logger, get_device, set_reproducible_seed,
    save_json, get_timestamp, setup_logging
)
from ..common import OWASPConfig, ProjectPaths
from ...common.utils import validate_model_file, tensor_info
from .model import SimpleCNN

import argparse
import sys
import time
from pathlib import Path
from typing import Tuple, Dict, List, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc, confusion_matrix

from .model import SimpleCNN
from .attack_fgsm import FGSMAttacker
from ...common import (
    get_logger, get_device, set_reproducible_seed,
    save_json, get_timestamp
)
from ..common import OWASPConfig, ProjectPaths
from .model import SimpleCNN
from .attack_fgsm import FGSMAttacker

logger = get_logger(__name__)


class ProgressTracker:
    """Simple progress tracker for optimization."""
    
    def __init__(self, total: int, description: str = "Progress"):
        self.total = total
        self.current = 0
        self.description = description
        
    def update(self):
        """Update progress by one step."""
        self.current += 1
        if self.current % max(1, self.total // 20) == 0:  # Log every 5%
            percent = (self.current / self.total) * 100
            logger.info(f"{self.description}: {self.current}/{self.total} ({percent:.1f}%)")
    
    def finish(self):
        """Complete the progress tracking."""
        logger.info(f"{self.description} completed: {self.total}/{self.total} (100.0%)")


class AdversarialDefense:
    """
    Comprehensive adversarial defense system.
    
    This class implements multiple detection methods for identifying
    adversarial examples with configurable thresholds and performance
    evaluation capabilities.
    """
    
    def __init__(self, model: nn.Module, device: torch.device, config: OWASPConfig = None):
        """
        Initialize the defense system.
        
        Args:
            model: Target model to protect
            device: Device for computations
            config: OWASP configuration (if None, will create default)
        """
        self.model = model
        self.device = device
        self.model.eval()
        
        # Get configuration from OWASP config (single source of truth)
        if config is None:
            config = OWASPConfig()
        
        defense_config = config.get_lab_config('ml01_input_manipulation')['defense']
        self.gradient_threshold = defense_config['gradient_threshold']
        self.confidence_threshold = defense_config['confidence_threshold']
        
        # Detection statistics
        self.detection_stats = {
            'total_samples': 0,
            'detected_adversarial': 0,
            'missed_adversarial': 0,
            'false_positives': 0,
            'true_negatives': 0
        }
        
        logger.info("Adversarial Defense system initialized")
    
    def compute_gradient_norm(self, data: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute gradient norm for input data.
        
        High gradient norms often indicate adversarial examples as they
        require large gradients to fool the model.
        
        Args:
            data: Input tensor with requires_grad=True
            target: Target labels
            
        Returns:
            Gradient norms for each sample
        """
        if not data.requires_grad:
            data = data.clone().detach().requires_grad_(True)
        
        # Forward pass
        output = self.model(data)
        
        # Compute loss
        loss = F.cross_entropy(output, target)
        
        # Backward pass
        self.model.zero_grad()
        loss.backward()
        
        # Get gradients and compute norms
        gradients = data.grad.data
        gradient_norms = torch.norm(gradients.view(data.size(0), -1), p=2, dim=1)
        
        logger.debug(f"Gradient norms - Mean: {gradient_norms.mean().item():.3f}, Max: {gradient_norms.max().item():.3f}")
        
        return gradient_norms
    
    def compute_prediction_confidence(self, data: torch.Tensor) -> torch.Tensor:
        """
        Compute prediction confidence scores.
        
        Adversarial examples often result in lower confidence predictions
        as the model becomes uncertain about the manipulated input.
        
        Args:
            data: Input tensor
            
        Returns:
            Confidence scores for each sample
        """
        with torch.no_grad():
            output = self.model(data)
            probabilities = F.softmax(output, dim=1)
            confidence_scores = probabilities.max(dim=1)[0]
        
        logger.debug(f"Confidence scores - Mean: {confidence_scores.mean().item():.3f}, Min: {confidence_scores.min().item():.3f}")
        
        return confidence_scores
    
    def detect_adversarial_gradient(
        self, 
        data: torch.Tensor, 
        target: torch.Tensor,
        threshold: Optional[float] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Detect adversarial examples using gradient norm analysis.
        
        Args:
            data: Input tensor
            target: Target labels
            threshold: Gradient norm threshold (uses default if None)
            
        Returns:
            Tuple of (detection_flags, gradient_norms)
        """
        if threshold is None:
            threshold = self.gradient_threshold
        
        gradient_norms = self.compute_gradient_norm(data, target)
        detection_flags = gradient_norms > threshold
        
        detected_count = detection_flags.sum().item()
        total_count = data.size(0)
        
        logger.debug(
            f"Gradient-based detection - Threshold: {threshold:.2f}, "
            f"Detected: {detected_count}/{total_count} ({100*detected_count/total_count:.1f}%)"
        )
        
        return detection_flags, gradient_norms
    
    def detect_adversarial_confidence(
        self, 
        data: torch.Tensor,
        threshold: Optional[float] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Detect adversarial examples using confidence analysis.
        
        Adversarial examples often result in HIGH confidence predictions
        as the attack forces the model to be overconfident about wrong classifications.
        
        Args:
            data: Input tensor
            threshold: Confidence threshold (uses default if None)
            
        Returns:
            Tuple of (detection_flags, confidence_scores)
        """
        if threshold is None:
            threshold = self.confidence_threshold
        
        confidence_scores = self.compute_prediction_confidence(data)
        # Changed logic: detect samples with HIGH confidence as potentially adversarial
        detection_flags = confidence_scores > threshold
        
        detected_count = detection_flags.sum().item()
        total_count = data.size(0)
        
        logger.debug(
            f"Confidence-based detection - Threshold: {threshold:.2f}, "
            f"Detected: {detected_count}/{total_count} ({100*detected_count/total_count:.1f}%)"
        )
        
        return detection_flags, confidence_scores
    
    def detect_adversarial_combined(
        self, 
        data: torch.Tensor, 
        target: torch.Tensor,
        grad_threshold: Optional[float] = None,
        conf_threshold: Optional[float] = None,
        combination_method: str = 'or'
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Detect adversarial examples using combined methods.
        
        Args:
            data: Input tensor
            target: Target labels
            grad_threshold: Gradient threshold
            conf_threshold: Confidence threshold
            combination_method: How to combine detections ('or', 'and')
            
        Returns:
            Tuple of (detection_flags, detection_info)
        """
        # Get individual detections
        grad_flags, grad_norms = self.detect_adversarial_gradient(data, target, grad_threshold)
        conf_flags, conf_scores = self.detect_adversarial_confidence(data, conf_threshold)
        
        # Combine detections
        if combination_method == 'or':
            combined_flags = grad_flags | conf_flags
        elif combination_method == 'and':
            combined_flags = grad_flags & conf_flags
        else:
            raise ValueError(f"Unknown combination method: {combination_method}")
        
        detection_info = {
            'gradient_flags': grad_flags,
            'confidence_flags': conf_flags,
            'gradient_norms': grad_norms,
            'confidence_scores': conf_scores,
            'combination_method': combination_method
        }
        
        detected_count = combined_flags.sum().item()
        total_count = data.size(0)
        
        logger.info(
            f"Combined detection ({combination_method}) - "
            f"Detected: {detected_count}/{total_count} ({100*detected_count/total_count:.1f}%)"
        )
        
        return combined_flags, detection_info
    
    def evaluate_defense_performance(
        self,
        clean_data: torch.Tensor,
        adversarial_data: torch.Tensor,
        labels: torch.Tensor,
        grad_threshold: Optional[float] = None,
        conf_threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Evaluate defense performance on clean and adversarial data.
        
        Args:
            clean_data: Clean input samples
            adversarial_data: Adversarial input samples
            labels: True labels for both datasets
            grad_threshold: Gradient threshold
            conf_threshold: Confidence threshold
            
        Returns:
            Comprehensive performance metrics
        """
        logger.info("Evaluating defense performance...")
        
        if grad_threshold is None:
            grad_threshold = self.gradient_threshold
        if conf_threshold is None:
            conf_threshold = self.confidence_threshold
        
        # Test on clean data (should not be detected as adversarial)
        clean_flags, clean_info = self.detect_adversarial_combined(
            clean_data, labels, grad_threshold, conf_threshold
        )
        
        # Test on adversarial data (should be detected as adversarial)
        adv_flags, adv_info = self.detect_adversarial_combined(
            adversarial_data, labels, grad_threshold, conf_threshold
        )
        
        # Calculate confusion matrix components
        # Clean data: True label = 0 (not adversarial)
        # Adversarial data: True label = 1 (adversarial)
        
        true_negatives = (~clean_flags).sum().item()  # Clean correctly identified as clean
        false_positives = clean_flags.sum().item()    # Clean incorrectly flagged as adversarial
        
        true_positives = adv_flags.sum().item()       # Adversarial correctly detected
        false_negatives = (~adv_flags).sum().item()   # Adversarial missed
        
        # Calculate performance metrics
        total_samples = len(clean_data) + len(adversarial_data)
        accuracy = (true_positives + true_negatives) / total_samples
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Prepare data for ROC curve
        y_true = torch.cat([
            torch.zeros(len(clean_data)),  # Clean samples (label 0)
            torch.ones(len(adversarial_data))  # Adversarial samples (label 1)
        ]).cpu().numpy()
        
        # Use gradient norm as score for ROC
        clean_grad_norms = clean_info['gradient_norms'].cpu().numpy()
        adv_grad_norms = adv_info['gradient_norms'].cpu().numpy()
        y_scores = np.concatenate([clean_grad_norms, adv_grad_norms])
        
        # Calculate ROC curve
        fpr, tpr, roc_thresholds = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        # Compile results
        results = {
            'thresholds': {
                'gradient_threshold': grad_threshold,
                'confidence_threshold': conf_threshold
            },
            'confusion_matrix': {
                'true_positives': true_positives,
                'false_positives': false_positives,
                'true_negatives': true_negatives,
                'false_negatives': false_negatives
            },
            'performance_metrics': {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score
            },
            'detection_rates': {
                'clean_detection_rate': false_positives / len(clean_data),
                'adversarial_detection_rate': true_positives / len(adversarial_data)
            },
            'roc_analysis': {
                'fpr': fpr.tolist(),
                'tpr': tpr.tolist(),
                'thresholds': roc_thresholds.tolist(),
                'roc_auc': roc_auc
            },
            'sample_statistics': {
                'clean_samples': len(clean_data),
                'adversarial_samples': len(adversarial_data),
                'total_samples': total_samples
            }
        }
        
        logger.info(
            f"Defense evaluation complete - "
            f"Accuracy: {accuracy:.3f}, Precision: {precision:.3f}, "
            f"Recall: {recall:.3f}, F1: {f1_score:.3f}, AUC: {roc_auc:.3f}"
        )
        
        return results
    
    def optimize_thresholds(
        self,
        clean_data: torch.Tensor,
        adversarial_data: torch.Tensor,
        labels: torch.Tensor,
        grad_threshold_range: Tuple[float, float] = (1.0, 10.0),
        conf_threshold_range: Tuple[float, float] = (0.5, 0.95),
        num_steps: int = 20
    ) -> Dict[str, Any]:
        """
        Optimize detection thresholds using grid search.
        
        Args:
            clean_data: Clean input samples
            adversarial_data: Adversarial input samples
            labels: True labels
            grad_threshold_range: Range for gradient threshold
            conf_threshold_range: Range for confidence threshold
            num_steps: Number of steps in grid search
            
        Returns:
            Optimization results with best thresholds
        """
        logger.info("Optimizing detection thresholds...")
        
        # Create threshold grids
        grad_thresholds = np.linspace(grad_threshold_range[0], grad_threshold_range[1], num_steps)
        conf_thresholds = np.linspace(conf_threshold_range[0], conf_threshold_range[1], num_steps)
        
        best_f1 = 0.0
        best_thresholds = None
        optimization_results = []
        
        total_combinations = len(grad_thresholds) * len(conf_thresholds)
        progress = ProgressTracker(total_combinations, "Threshold optimization")
        
        for grad_thresh in grad_thresholds:
            for conf_thresh in conf_thresholds:
                # Evaluate performance with these thresholds
                results = self.evaluate_defense_performance(
                    clean_data, adversarial_data, labels,
                    grad_thresh, conf_thresh
                )
                
                f1_score = results['performance_metrics']['f1_score']
                
                optimization_results.append({
                    'gradient_threshold': grad_thresh,
                    'confidence_threshold': conf_thresh,
                    'f1_score': f1_score,
                    'accuracy': results['performance_metrics']['accuracy'],
                    'precision': results['performance_metrics']['precision'],
                    'recall': results['performance_metrics']['recall']
                })
                
                if f1_score > best_f1:
                    best_f1 = f1_score
                    best_thresholds = {
                        'gradient_threshold': grad_thresh,
                        'confidence_threshold': conf_thresh,
                        'performance': results['performance_metrics']
                    }
                
                progress.update()
        
        progress.finish()
        
        # Update thresholds with best values
        if best_thresholds:
            self.gradient_threshold = best_thresholds['gradient_threshold']
            self.confidence_threshold = best_thresholds['confidence_threshold']
        
        logger.info(
            f"Threshold optimization complete - "
            f"Best F1: {best_f1:.3f}, "
            f"Best thresholds: grad={best_thresholds['gradient_threshold']:.2f}, "
            f"conf={best_thresholds['confidence_threshold']:.2f}"
        )
        
        return {
            'best_thresholds': best_thresholds,
            'optimization_history': optimization_results,
            'search_parameters': {
                'grad_threshold_range': grad_threshold_range,
                'conf_threshold_range': conf_threshold_range,
                'num_steps': num_steps
            }
        }
    
    def visualize_roc_curve(self, evaluation_results: Dict[str, Any], save_path: Optional[Path] = None) -> plt.Figure:
        """
        Create ROC curve visualization.
        
        Args:
            evaluation_results: Results from evaluate_defense_performance
            save_path: Optional path to save the plot
            
        Returns:
            Matplotlib figure
        """
        logger.debug("Creating ROC curve visualization...")
        
        fpr = evaluation_results['roc_analysis']['fpr']
        tpr = evaluation_results['roc_analysis']['tpr']
        roc_auc = evaluation_results['roc_analysis']['roc_auc']
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Plot ROC curve
        ax.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.3f})')
        
        # Plot random classifier line
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                label='Random classifier')
        
        # Add current operating point
        tn = evaluation_results['confusion_matrix']['true_negatives']
        fp = evaluation_results['confusion_matrix']['false_positives']
        fn = evaluation_results['confusion_matrix']['false_negatives']
        tp = evaluation_results['confusion_matrix']['true_positives']
        
        current_fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        current_tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        ax.plot(current_fpr, current_tpr, 'ro', markersize=8, 
                label=f'Current thresholds (FPR={current_fpr:.3f}, TPR={current_tpr:.3f})')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve - Adversarial Detection Performance')
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"ROC curve saved to {save_path}")
        
        return fig


def main():
    """Main entry point for defense evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate defense against FGSM attacks")
    
    parser.add_argument(
        '--model-path', type=str, default=None,
        help='Path to trained model (default: models/ml01_model.pt)'
    )
    parser.add_argument(
        '--epsilon', type=float, default=0.25,
        help='Epsilon for generating adversarial examples (default: 0.25)'
    )
    parser.add_argument(
        '--num-samples', type=int, default=1000,
        help='Number of samples to test (default: 1000)'
    )
    parser.add_argument(
        '--optimize-thresholds', action='store_true',
        help='Optimize detection thresholds'
    )
    parser.add_argument(
        '--visualize', action='store_true',
        help='Create visualizations'
    )
    parser.add_argument(
        '--log-level', type=str, default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )
    parser.add_argument(
        '--output-dir', type=str, default=None,
        help='Output directory for results (default: results/ml01)'
    )
    
    args = parser.parse_args()
    
    try:
        # Setup logging
        # Logging is already configured
        setup_logging(log_level=args.log_level)
        
        # Setup configuration (single source of truth)
        config = OWASPConfig()
        defense_config = config.get_lab_config('ml01_input_manipulation')['defense']
        
        logger.info(f"Using defense configuration from OWASP config:")
        logger.info(f"  Gradient threshold: {defense_config['gradient_threshold']}")
        logger.info(f"  Confidence threshold: {defense_config['confidence_threshold']}")
        
        # Setup paths from project root (4 levels up from this file)
        paths = ProjectPaths.from_root(Path(__file__).parent.parent.parent.parent)
        paths.ensure_dirs()
        
        # Determine output directory
        if args.output_dir:
            output_dir = Path(args.output_dir)
        else:
            output_dir = paths.results / 'owasp' / 'ml01'
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Setup device
        device = get_device()
        
        # Load model
        if args.model_path:
            model_path = Path(args.model_path)
        else:
            model_path = paths.models / 'owasp' / 'ml01_model.pt'
        
        if not validate_model_file(model_path):
            logger.error(f"Invalid or missing model file: {model_path}")
            return 1
        
        # Initialize model
        model = SimpleCNN().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        
        logger.info(f"Model loaded from {model_path}")
        
        # Setup data
        transform = transforms.ToTensor()
        test_dataset = datasets.MNIST(
            root=str(paths.data),
            train=False,
            transform=transform,
            download=True
        )
        
        # Create subset
        indices = torch.randperm(len(test_dataset))[:args.num_samples]
        test_subset = torch.utils.data.Subset(test_dataset, indices)
        
        test_loader = DataLoader(test_subset, batch_size=args.num_samples, shuffle=False)
        
        # Get test data
        data, labels = next(iter(test_loader))
        data, labels = data.to(device), labels.to(device)
        
        logger.info(f"Test data prepared - {len(data)} samples")
        
        # Generate adversarial examples
        logger.info(f"Generating adversarial examples with ε={args.epsilon}")
        attacker = FGSMAttacker(model, device)
        adversarial_data, attack_info = attacker.fgsm_attack(data, labels, args.epsilon)
        
        logger.info(f"Adversarial examples generated - Success rate: {attack_info['success_rate']:.3f}")
        
        # Initialize defense with config (single source of truth)
        defense = AdversarialDefense(model, device, config)
        
        logger.info(f"Defense initialized with thresholds:")
        logger.info(f"  Gradient threshold: {defense.gradient_threshold}")
        logger.info(f"  Confidence threshold: {defense.confidence_threshold}")
        
        # Optimize thresholds if requested
        if args.optimize_thresholds:
            logger.info("Optimizing detection thresholds...")
            optimization_results = defense.optimize_thresholds(data, adversarial_data, labels)
            
            # Save optimization results
            opt_results_file = output_dir / f"threshold_optimization_{get_timestamp()}.json"
            save_json(optimization_results, opt_results_file)
            logger.info(f"Optimization results saved to {opt_results_file}")
        
        # Evaluate defense performance
        evaluation_results = defense.evaluate_defense_performance(
            data, adversarial_data, labels,
            defense.gradient_threshold, defense.confidence_threshold
        )
        
        # Save evaluation results
        eval_results_file = output_dir / f"defense_evaluation_{get_timestamp()}.json"
        save_json(evaluation_results, eval_results_file)
        
        # Create visualizations if requested
        if args.visualize:
            logger.info("Creating defense visualizations...")
            
            # ROC curve
            roc_path = output_dir / f"defense_roc_curve_{get_timestamp()}.png"
            fig = defense.visualize_roc_curve(evaluation_results, roc_path)
            plt.close(fig)
        
        logger.info("Defense evaluation completed successfully!")
        logger.info(f"Results saved to {eval_results_file}")
        
        # Print summary
        metrics = evaluation_results['performance_metrics']
        print(f"\nDefense Performance Summary:")
        print(f"Accuracy:  {metrics['accuracy']:.3f}")
        print(f"Precision: {metrics['precision']:.3f}")
        print(f"Recall:    {metrics['recall']:.3f}")
        print(f"F1-Score:  {metrics['f1_score']:.3f}")
        print(f"ROC AUC:   {evaluation_results['roc_analysis']['roc_auc']:.3f}")
        
        return 0
        
    except KeyboardInterrupt:
        logger.warning("Defense evaluation interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Defense evaluation failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
