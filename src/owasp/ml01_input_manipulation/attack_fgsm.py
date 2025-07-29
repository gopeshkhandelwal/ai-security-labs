"""
OWASP ML01: Fast Gradient Sign Method (FGSM) Attack Implementation.

Enterprise-grade adversarial attack implementation demonstrating input manipulation
vulnerabilities as outlined in OWASP ML01 guidelines.

Author: Gopesh Khandelwal <gopeshkhandelwal@gmail.com>
License: CC BY-NC 4.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from typing import Dict, Any, Tuple, List, Optional
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from typing import Dict, Any, Tuple, List, Optional
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse
import sys
import time

from ...common import (
    get_logger, get_device, set_reproducible_seed,
    save_json, get_timestamp, ProgressTracker, setup_logging
)
from ..common import OWASPConfig, ProjectPaths
from ...common.utils import validate_model_file, tensor_info
from .model import SimpleCNN

logger = get_logger(__name__)


class FGSMAttacker:
    """
    Fast Gradient Sign Method (FGSM) adversarial attack implementation.
    
    This class provides comprehensive FGSM attack capabilities including
    single and batch attacks, success rate analysis, and visualization.
    """
    
    def __init__(self, model: nn.Module, device: torch.device):
        """
        Initialize FGSM attacker.
        
        Args:
            model: Target model to attack
            device: Device for computations
        """
        self.model = model
        self.device = device
        self.model.eval()  # Set to evaluation mode
        
        # Attack statistics
        self.attack_stats = {
            'total_samples': 0,
            'successful_attacks': 0,
            'failed_attacks': 0,
            'epsilon_values': [],
            'perturbation_norms': []
        }
        
        logger.info("FGSM Attacker initialized")
    
    def fgsm_attack(
        self, 
        data: torch.Tensor, 
        target: torch.Tensor, 
        epsilon: float = 0.25
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Perform FGSM attack on input data.
        
        The FGSM attack generates adversarial examples using:
        x_adv = x + ε * sign(∇_x J(θ, x, y))
        
        where:
        - x: original input
        - ε: perturbation magnitude (epsilon)
        - ∇_x J: gradient of loss w.r.t. input
        - θ: model parameters
        - y: true label
        
        Args:
            data: Input tensor with shape (batch_size, channels, height, width)
            target: Target labels with shape (batch_size,)
            epsilon: Perturbation magnitude
            
        Returns:
            Tuple of (adversarial_examples, attack_info)
        """
        # Validate inputs
        if not isinstance(epsilon, (int, float)) or epsilon < 0:
            raise ValueError(f"Epsilon must be non-negative number, got {epsilon}")
        
        if data.shape[0] != target.shape[0]:
            raise ValueError(
                f"Batch size mismatch: data {data.shape[0]}, target {target.shape[0]}"
            )
        
        logger.debug(f"Starting FGSM attack with ε={epsilon}")
        logger.debug(f"Input data info: {tensor_info(data)}")
        
        # Ensure data requires gradients
        data = data.clone().detach().requires_grad_(True)
        
        # Forward pass to get initial predictions
        initial_output = self.model(data)
        initial_predictions = initial_output.argmax(dim=1)
        initial_probabilities = F.softmax(initial_output, dim=1)
        initial_confidence = initial_probabilities.max(dim=1)[0]
        
        # Calculate loss
        loss = F.cross_entropy(initial_output, target)
        
        # Backward pass to get gradients
        self.model.zero_grad()
        loss.backward()
        
        # Get gradients and create perturbation
        data_gradients = data.grad.data
        gradient_norm = torch.norm(data_gradients, p=2)
        
        # FGSM perturbation: ε * sign(gradient)
        perturbation = epsilon * data_gradients.sign()
        
        # Create adversarial examples
        adversarial_data = data + perturbation
        
        # Clamp to valid pixel range [0, 1]
        adversarial_data = torch.clamp(adversarial_data, 0, 1)
        
        # Forward pass with adversarial examples
        with torch.no_grad():
            adversarial_output = self.model(adversarial_data)
            adversarial_predictions = adversarial_output.argmax(dim=1)
            adversarial_probabilities = F.softmax(adversarial_output, dim=1)
            adversarial_confidence = adversarial_probabilities.max(dim=1)[0]
        
        # Calculate attack success
        successful_attacks = (adversarial_predictions != target).sum().item()
        total_samples = data.shape[0]
        success_rate = successful_attacks / total_samples
        
        # Calculate perturbation statistics
        perturbation_l2_norm = torch.norm(perturbation.view(total_samples, -1), p=2, dim=1)
        perturbation_linf_norm = torch.norm(perturbation.view(total_samples, -1), p=float('inf'), dim=1)
        
        # Compile attack information
        attack_info = {
            'epsilon': epsilon,
            'success_rate': success_rate,
            'successful_attacks': successful_attacks,
            'total_samples': total_samples,
            'gradient_norm': gradient_norm.item(),
            'perturbation_stats': {
                'l2_norm_mean': perturbation_l2_norm.mean().item(),
                'l2_norm_std': perturbation_l2_norm.std().item(),
                'l2_norm_max': perturbation_l2_norm.max().item(),
                'linf_norm_mean': perturbation_linf_norm.mean().item(),
                'linf_norm_max': perturbation_linf_norm.max().item()
            },
            'confidence_stats': {
                'initial_confidence_mean': initial_confidence.mean().item(),
                'adversarial_confidence_mean': adversarial_confidence.mean().item(),
                'confidence_drop': (initial_confidence - adversarial_confidence).mean().item()
            },
            'predictions': {
                'initial': initial_predictions.cpu().tolist(),
                'adversarial': adversarial_predictions.cpu().tolist(),
                'targets': target.cpu().tolist()
            }
        }
        
        # Update attack statistics
        self.attack_stats['total_samples'] += total_samples
        self.attack_stats['successful_attacks'] += successful_attacks
        self.attack_stats['failed_attacks'] += (total_samples - successful_attacks)
        self.attack_stats['epsilon_values'].append(epsilon)
        self.attack_stats['perturbation_norms'].extend(perturbation_l2_norm.cpu().tolist())
        
        logger.info(
            f"FGSM attack completed - ε={epsilon:.3f}, "
            f"Success rate: {success_rate:.3f} ({successful_attacks}/{total_samples})"
        )
        
        return adversarial_data.detach(), attack_info
    
    def batch_attack(
        self, 
        data_loader: DataLoader, 
        epsilon_values: List[float] = None,
        max_batches: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Perform FGSM attacks on multiple batches with different epsilon values.
        
        Args:
            data_loader: DataLoader containing test data
            epsilon_values: List of epsilon values to test
            max_batches: Maximum number of batches to process
            
        Returns:
            Comprehensive attack results
        """
        if epsilon_values is None:
            epsilon_values = [0.1, 0.25, 0.3]
        
        logger.info(f"Starting batch FGSM attack with ε values: {epsilon_values}")
        
        batch_results = {}
        total_samples_processed = 0
        
        # Process each epsilon value
        for epsilon in epsilon_values:
            logger.info(f"Testing epsilon = {epsilon}")
            
            epsilon_results = {
                'epsilon': epsilon,
                'batch_attacks': [],
                'total_samples': 0,
                'total_successful': 0,
                'overall_success_rate': 0.0
            }
            
            progress = ProgressTracker(
                total=min(len(data_loader), max_batches or len(data_loader)),
                description=f"FGSM ε={epsilon}"
            )
            
            for batch_idx, (data, target) in enumerate(data_loader):
                if max_batches and batch_idx >= max_batches:
                    break
                
                # Move to device
                data, target = data.to(self.device), target.to(self.device)
                
                # Perform attack
                try:
                    adversarial_data, attack_info = self.fgsm_attack(data, target, epsilon)
                    
                    epsilon_results['batch_attacks'].append(attack_info)
                    epsilon_results['total_samples'] += attack_info['total_samples']
                    epsilon_results['total_successful'] += attack_info['successful_attacks']
                    
                    total_samples_processed += attack_info['total_samples']
                    
                except Exception as e:
                    logger.error(f"Attack failed for batch {batch_idx}, ε={epsilon}: {e}")
                    continue
                
                progress.update()
            
            progress.finish()
            
            # Calculate overall success rate for this epsilon
            if epsilon_results['total_samples'] > 0:
                epsilon_results['overall_success_rate'] = (
                    epsilon_results['total_successful'] / epsilon_results['total_samples']
                )
            
            batch_results[f"epsilon_{epsilon}"] = epsilon_results
            
            logger.info(
                f"Epsilon {epsilon} complete - "
                f"Success rate: {epsilon_results['overall_success_rate']:.3f} "
                f"({epsilon_results['total_successful']}/{epsilon_results['total_samples']})"
            )
        
        # Compile overall results
        overall_results = {
            'attack_type': 'FGSM',
            'epsilon_values': epsilon_values,
            'total_samples_processed': total_samples_processed,
            'epsilon_results': batch_results,
            'attack_statistics': self.attack_stats.copy(),
            'timestamp': get_timestamp()
        }
        
        logger.info(f"Batch attack completed - Processed {total_samples_processed} samples")
        
        return overall_results
    
    def visualize_attack(
        self, 
        original: torch.Tensor, 
        adversarial: torch.Tensor,
        original_pred: int,
        adversarial_pred: int,
        epsilon: float,
        save_path: Optional[Path] = None
    ) -> plt.Figure:
        """
        Create visualization comparing original and adversarial examples.
        
        Args:
            original: Original image tensor
            adversarial: Adversarial image tensor
            original_pred: Original prediction
            adversarial_pred: Adversarial prediction
            epsilon: Epsilon value used
            save_path: Optional path to save the plot
            
        Returns:
            Matplotlib figure
        """
        logger.debug("Creating attack visualization...")
        
        # Convert tensors to numpy arrays
        orig_img = original.squeeze().detach().cpu().numpy()
        adv_img = adversarial.squeeze().detach().cpu().numpy()
        diff_img = adv_img - orig_img
        
        # Create figure with subplots
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f'FGSM Attack Visualization (ε={epsilon})', fontsize=16, fontweight='bold')
        
        # Original image
        axes[0].imshow(orig_img, cmap='gray')
        axes[0].set_title(f'Original\nPrediction: {original_pred}', fontsize=12)
        axes[0].axis('off')
        
        # Adversarial image
        color = 'red' if adversarial_pred != original_pred else 'green'
        axes[1].imshow(adv_img, cmap='gray')
        axes[1].set_title(f'Adversarial\nPrediction: {adversarial_pred}', 
                         fontsize=12, color=color)
        axes[1].axis('off')
        
        # Difference (amplified for visibility)
        diff_amplified = diff_img * 50  # Amplify for visualization
        im = axes[2].imshow(diff_amplified, cmap='RdBu', vmin=-1, vmax=1)
        axes[2].set_title('Perturbation\n(50x amplified)', fontsize=12)
        axes[2].axis('off')
        
        # Add colorbar
        plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Attack visualization saved to {save_path}")
        
        return fig
    
    def generate_attack_report(self, results: Dict[str, Any], output_dir: Path) -> Path:
        """
        Generate comprehensive attack report.
        
        Args:
            results: Attack results from batch_attack
            output_dir: Directory to save the report
            
        Returns:
            Path to the generated report
        """
        logger.info("Generating attack report...")
        
        # Create report content
        report = {
            'attack_summary': {
                'attack_type': results['attack_type'],
                'timestamp': results['timestamp'],
                'total_samples': results['total_samples_processed'],
                'epsilon_values_tested': results['epsilon_values']
            },
            'epsilon_analysis': {},
            'overall_statistics': results['attack_statistics'],
            'recommendations': []
        }
        
        # Analyze each epsilon value
        for epsilon_key, epsilon_data in results['epsilon_results'].items():
            epsilon = epsilon_data['epsilon']
            success_rate = epsilon_data['overall_success_rate']
            
            report['epsilon_analysis'][str(epsilon)] = {
                'success_rate': success_rate,
                'samples_tested': epsilon_data['total_samples'],
                'successful_attacks': epsilon_data['total_successful'],
                'vulnerability_level': self._assess_vulnerability(success_rate)
            }
        
        # Generate recommendations
        report['recommendations'] = self._generate_recommendations(report['epsilon_analysis'])
        
        # Save report
        report_path = output_dir / f"fgsm_attack_report_{get_timestamp()}.json"
        save_json(report, report_path)
        
        logger.info(f"Attack report saved to {report_path}")
        
        return report_path
    
    def _assess_vulnerability(self, success_rate: float) -> str:
        """Assess vulnerability level based on attack success rate."""
        if success_rate >= 0.8:
            return "CRITICAL"
        elif success_rate >= 0.6:
            return "HIGH"
        elif success_rate >= 0.4:
            return "MEDIUM"
        elif success_rate >= 0.2:
            return "LOW"
        else:
            return "MINIMAL"
    
    def _generate_recommendations(self, epsilon_analysis: Dict) -> List[str]:
        """Generate security recommendations based on attack results."""
        recommendations = []
        
        high_vulnerability_epsilons = [
            eps for eps, data in epsilon_analysis.items()
            if data['vulnerability_level'] in ['CRITICAL', 'HIGH']
        ]
        
        if high_vulnerability_epsilons:
            recommendations.extend([
                "Implement adversarial training with FGSM examples",
                "Add input preprocessing defenses (e.g., compression, smoothing)",
                "Consider ensemble methods for improved robustness",
                "Implement gradient masking techniques",
                "Add detection mechanisms for adversarial inputs"
            ])
        else:
            recommendations.append("Model shows good robustness against FGSM attacks")
        
        return recommendations


def main():
    """Main entry point for FGSM attack."""
    parser = argparse.ArgumentParser(description="Run FGSM adversarial attack on ML01 model")
    
    parser.add_argument(
        '--model-path', type=str, default='models/owasp/ml01_model.pt',
        help='Path to trained model (default: modelsowasp/ml01_model.pt)'
    )
    parser.add_argument(
        '--epsilon', type=float, nargs='+', default=[0.1, 0.25, 0.3],
        help='Epsilon values for attack (default: [0.1, 0.25, 0.3])'
    )
    parser.add_argument(
        '--num-samples', type=int, default=None,
        help='Number of samples to attack (default: all test set)'
    )
    parser.add_argument(
        '--batch-size', type=int, default=64,
        help='Batch size for attack (default: 64)'
    )
    parser.add_argument(
        '--visualize', action='store_true',
        help='Create visualization of attacks'
    )
    parser.add_argument(
        '--log-level', type=str, default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )
    parser.add_argument(
        '--output-dir', type=str, default='owasp/results/ml01',
        help='Output directory for results (default: owasp/results/ml01)'
    )
    
    args = parser.parse_args()
    
    try:
        # Setup logging
        # Logging is already configured
        setup_logging(log_level=args.log_level)
        
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
        
        # Create subset if requested
        if args.num_samples:
            indices = torch.randperm(len(test_dataset))[:args.num_samples]
            test_dataset = torch.utils.data.Subset(test_dataset, indices)
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False
        )
        
        logger.info(f"Test data loaded - {len(test_dataset)} samples")
        
        # Initialize attacker
        attacker = FGSMAttacker(model, device)
        
        # Run attack
        results = attacker.batch_attack(
            test_loader,
            epsilon_values=args.epsilon,
            max_batches=None
        )
        
        # Save results
        results_file = output_dir / f"fgsm_attack_results_{get_timestamp()}.json"
        save_json(results, results_file)
        
        # Generate report
        report_path = attacker.generate_attack_report(results, output_dir)
        
        # Create visualization if requested
        if args.visualize:
            logger.info("Creating attack visualizations...")
            
            # Get a sample for visualization
            data_iter = iter(test_loader)
            sample_data, sample_target = next(data_iter)
            sample_data, sample_target = sample_data[:1].to(device), sample_target[:1].to(device)
            
            for epsilon in args.epsilon:
                adversarial_data, attack_info = attacker.fgsm_attack(
                    sample_data, sample_target, epsilon
                )
                
                # Create visualization
                viz_path = output_dir / f"fgsm_visualization_eps_{epsilon}_{get_timestamp()}.png"
                fig = attacker.visualize_attack(
                    sample_data[0], adversarial_data[0],
                    attack_info['predictions']['targets'][0],
                    attack_info['predictions']['adversarial'][0],
                    epsilon, viz_path
                )
                plt.close(fig)
        
        logger.info("FGSM attack completed successfully!")
        logger.info(f"Results saved to {results_file}")
        logger.info(f"Report saved to {report_path}")
        
        return 0
        
    except KeyboardInterrupt:
        logger.warning("Attack interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Attack failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
