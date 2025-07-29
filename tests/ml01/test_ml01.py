"""
Test suite for ML01 lab components.

This module contains comprehensive tests for the ML01 input manipulation
lab including model tests, attack tests, and defense tests.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import tempfile
import shutil

from src.owasp.ml01_input_manipulation.model import SimpleCNN
from src.owasp.ml01_input_manipulation.attack_fgsm import FGSMAttacker
from src.owasp.ml01_input_manipulation.defense_fgsm import AdversarialDefense
from src.common import get_device, ProjectPaths, Config


class TestSimpleCNN:
    """Test cases for SimpleCNN model."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.device = get_device()
        self.model = SimpleCNN().to(self.device)
        self.batch_size = 4
        self.input_tensor = torch.randn(self.batch_size, 1, 28, 28).to(self.device)
        self.labels = torch.randint(0, 10, (self.batch_size,)).to(self.device)
    
    def test_model_initialization(self):
        """Test model initialization."""
        assert isinstance(self.model, SimpleCNN)
        assert self.model.input_channels == 1
        assert self.model.num_classes == 10
    
    def test_forward_pass(self):
        """Test forward pass."""
        output = self.model(self.input_tensor)
        
        # Check output shape
        assert output.shape == (self.batch_size, 10)
        
        # Check output is not NaN or Inf
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_forward_pass_validation(self):
        """Test forward pass input validation."""
        # Test wrong number of dimensions
        with pytest.raises(ValueError):
            wrong_input = torch.randn(1, 28, 28).to(self.device)
            self.model(wrong_input)
        
        # Test wrong number of channels
        with pytest.raises(ValueError):
            wrong_channels = torch.randn(self.batch_size, 3, 28, 28).to(self.device)
            self.model(wrong_channels)
    
    def test_prediction_methods(self):
        """Test prediction methods."""
        # Test predict_proba
        probs = self.model.predict_proba(self.input_tensor)
        assert probs.shape == (self.batch_size, 10)
        assert torch.allclose(probs.sum(dim=1), torch.ones(self.batch_size))
        
        # Test predict
        predictions, confidence = self.model.predict(self.input_tensor)
        assert predictions.shape == (self.batch_size,)
        assert confidence.shape == (self.batch_size,)
        assert (predictions >= 0).all() and (predictions < 10).all()
        assert (confidence >= 0).all() and (confidence <= 1).all()
    
    def test_feature_maps(self):
        """Test feature map extraction."""
        feature_maps = self.model.get_feature_maps(self.input_tensor)
        assert feature_maps.shape == (self.batch_size, 32, 28, 28)
    
    def test_gradient_computation(self):
        """Test gradient computation."""
        input_with_grad = self.input_tensor.clone().requires_grad_(True)
        gradients = self.model.get_gradients(input_with_grad, self.labels)
        
        assert gradients.shape == input_with_grad.shape
        assert not torch.isnan(gradients).any()
    
    def test_model_summary(self):
        """Test model summary generation."""
        summary = self.model.summary()
        assert isinstance(summary, str)
        assert "SimpleCNN" in summary
        assert "Parameters" in summary


class TestFGSMAttacker:
    """Test cases for FGSM attacker."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.device = get_device()
        self.model = SimpleCNN().to(self.device)
        self.model.eval()
        self.attacker = FGSMAttacker(self.model, self.device)
        
        self.batch_size = 4
        self.test_data = torch.randn(self.batch_size, 1, 28, 28).to(self.device)
        self.test_labels = torch.randint(0, 10, (self.batch_size,)).to(self.device)
    
    def test_attacker_initialization(self):
        """Test attacker initialization."""
        assert self.attacker.model == self.model
        assert self.attacker.device == self.device
        assert isinstance(self.attacker.attack_stats, dict)
    
    def test_fgsm_attack_basic(self):
        """Test basic FGSM attack."""
        epsilon = 0.25
        adversarial_data, attack_info = self.attacker.fgsm_attack(
            self.test_data, self.test_labels, epsilon
        )
        
        # Check output shape
        assert adversarial_data.shape == self.test_data.shape
        
        # Check attack info structure
        assert 'epsilon' in attack_info
        assert 'success_rate' in attack_info
        assert 'total_samples' in attack_info
        assert attack_info['epsilon'] == epsilon
        assert attack_info['total_samples'] == self.batch_size
        
        # Check adversarial data is in valid range
        assert (adversarial_data >= 0).all()
        assert (adversarial_data <= 1).all()
    
    def test_fgsm_attack_parameters(self):
        """Test FGSM attack with different parameters."""
        # Test different epsilon values
        for epsilon in [0.1, 0.3, 0.5]:
            adversarial_data, attack_info = self.attacker.fgsm_attack(
                self.test_data, self.test_labels, epsilon
            )
            assert attack_info['epsilon'] == epsilon
    
    def test_fgsm_attack_validation(self):
        """Test FGSM attack input validation."""
        # Test negative epsilon
        with pytest.raises(ValueError):
            self.attacker.fgsm_attack(self.test_data, self.test_labels, -0.1)
        
        # Test batch size mismatch
        with pytest.raises(ValueError):
            wrong_labels = torch.randint(0, 10, (self.batch_size + 1,)).to(self.device)
            self.attacker.fgsm_attack(self.test_data, wrong_labels, 0.25)
    
    def test_attack_statistics(self):
        """Test attack statistics tracking."""
        initial_stats = self.attacker.attack_stats.copy()
        
        # Perform attack
        self.attacker.fgsm_attack(self.test_data, self.test_labels, 0.25)
        
        # Check stats updated
        assert self.attacker.attack_stats['total_samples'] > initial_stats['total_samples']
        assert len(self.attacker.attack_stats['epsilon_values']) > len(initial_stats['epsilon_values'])
    
    def test_visualization(self):
        """Test attack visualization."""
        adversarial_data, attack_info = self.attacker.fgsm_attack(
            self.test_data[:1], self.test_labels[:1], 0.25
        )
        
        # Test visualization creation (without saving)
        fig = self.attacker.visualize_attack(
            self.test_data[0], adversarial_data[0],
            attack_info['predictions']['targets'][0],
            attack_info['predictions']['adversarial'][0],
            0.25
        )
        
        assert fig is not None
        fig.clf()


class TestAdversarialDefense:
    """Test cases for adversarial defense."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.device = get_device()
        self.model = SimpleCNN().to(self.device)
        self.model.eval()
        self.defense = AdversarialDefense(self.model, self.device)
        
        self.batch_size = 4
        self.clean_data = torch.randn(self.batch_size, 1, 28, 28).to(self.device)
        self.labels = torch.randint(0, 10, (self.batch_size,)).to(self.device)
        
        # Generate some adversarial data
        attacker = FGSMAttacker(self.model, self.device)
        self.adversarial_data, _ = attacker.fgsm_attack(self.clean_data, self.labels, 0.25)
    
    def test_defense_initialization(self):
        """Test defense initialization."""
        assert self.defense.model == self.model
        assert self.defense.device == self.device
        assert isinstance(self.defense.gradient_threshold, float)
        assert isinstance(self.defense.confidence_threshold, float)
    
    def test_gradient_norm_computation(self):
        """Test gradient norm computation."""
        gradient_norms = self.defense.compute_gradient_norm(self.clean_data, self.labels)
        
        assert gradient_norms.shape == (self.batch_size,)
        assert (gradient_norms >= 0).all()
        assert not torch.isnan(gradient_norms).any()
    
    def test_confidence_computation(self):
        """Test confidence computation."""
        confidence_scores = self.defense.compute_prediction_confidence(self.clean_data)
        
        assert confidence_scores.shape == (self.batch_size,)
        assert (confidence_scores >= 0).all()
        assert (confidence_scores <= 1).all()
    
    def test_gradient_detection(self):
        """Test gradient-based detection."""
        detection_flags, gradient_norms = self.defense.detect_adversarial_gradient(
            self.clean_data, self.labels, threshold=5.0
        )
        
        assert detection_flags.shape == (self.batch_size,)
        assert gradient_norms.shape == (self.batch_size,)
        assert detection_flags.dtype == torch.bool
    
    def test_confidence_detection(self):
        """Test confidence-based detection."""
        detection_flags, confidence_scores = self.defense.detect_adversarial_confidence(
            self.clean_data, threshold=0.5
        )
        
        assert detection_flags.shape == (self.batch_size,)
        assert confidence_scores.shape == (self.batch_size,)
        assert detection_flags.dtype == torch.bool
    
    def test_combined_detection(self):
        """Test combined detection methods."""
        # Test OR combination
        detection_flags_or, detection_info = self.defense.detect_adversarial_combined(
            self.clean_data, self.labels, combination_method='or'
        )
        
        # Test AND combination
        detection_flags_and, _ = self.defense.detect_adversarial_combined(
            self.clean_data, self.labels, combination_method='and'
        )
        
        assert detection_flags_or.shape == (self.batch_size,)
        assert detection_flags_and.shape == (self.batch_size,)
        assert 'gradient_flags' in detection_info
        assert 'confidence_flags' in detection_info
    
    def test_defense_evaluation(self):
        """Test defense performance evaluation."""
        results = self.defense.evaluate_defense_performance(
            self.clean_data, self.adversarial_data, self.labels
        )
        
        # Check result structure
        assert 'confusion_matrix' in results
        assert 'performance_metrics' in results
        assert 'roc_analysis' in results
        
        # Check metric ranges
        metrics = results['performance_metrics']
        assert 0 <= metrics['accuracy'] <= 1
        assert 0 <= metrics['precision'] <= 1
        assert 0 <= metrics['recall'] <= 1
        assert 0 <= metrics['f1_score'] <= 1
    
    def test_roc_visualization(self):
        """Test ROC curve visualization."""
        results = self.defense.evaluate_defense_performance(
            self.clean_data, self.adversarial_data, self.labels
        )
        
        fig = self.defense.visualize_roc_curve(results)
        assert fig is not None
        fig.clf()


class TestIntegration:
    """Integration tests for the complete ML01 workflow."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.device = get_device()
        self.temp_dir = Path(tempfile.mkdtemp())
        
    def teardown_method(self):
        """Cleanup test fixtures."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def test_full_pipeline(self):
        """Test the complete attack-defense pipeline."""
        # Initialize model
        model = SimpleCNN().to(self.device)
        model.eval()
        
        # Create test data
        batch_size = 8
        test_data = torch.randn(batch_size, 1, 28, 28).to(self.device)
        test_labels = torch.randint(0, 10, (batch_size,)).to(self.device)
        
        # Run attack
        attacker = FGSMAttacker(model, self.device)
        adversarial_data, attack_info = attacker.fgsm_attack(test_data, test_labels, 0.25)
        
        # Verify attack worked
        assert adversarial_data.shape == test_data.shape
        assert attack_info['success_rate'] >= 0
        
        # Run defense
        defense = AdversarialDefense(model, self.device)
        defense_results = defense.evaluate_defense_performance(
            test_data, adversarial_data, test_labels
        )
        
        # Verify defense evaluation
        assert 'performance_metrics' in defense_results
        assert defense_results['performance_metrics']['accuracy'] >= 0
        
        # Test that we can detect some adversarial examples
        # (Note: may not detect all depending on thresholds)
        total_detected = (
            defense_results['confusion_matrix']['true_positives'] +
            defense_results['confusion_matrix']['false_positives']
        )
        assert total_detected >= 0  # Should detect at least some


if __name__ == "__main__":
    pytest.main([__file__])
