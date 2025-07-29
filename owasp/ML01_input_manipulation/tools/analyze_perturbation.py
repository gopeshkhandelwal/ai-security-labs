import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from model.simple_cnn import SimpleCNN
import matplotlib.pyplot as plt
import numpy as np
import os

def visualize_perturbation():
    """Visualize what perturbation actually looks like."""
    
    # Setup (same as your attack script)
    os.makedirs("results", exist_ok=True)
    transform = transforms.ToTensor()
    test_data = datasets.MNIST(root="./data", train=False, transform=transform, download=True)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=True)
    
    model = SimpleCNN()
    model.load_state_dict(torch.load("ml01_model.pt"))
    model.eval()
    
    # Get a sample
    image, label = next(iter(test_loader))
    image.requires_grad = True
    output = model(image)
    pred = output.argmax(dim=1)
    
    # Calculate gradient (same as your code)
    loss = nn.CrossEntropyLoss()(output, label)
    model.zero_grad()
    loss.backward()
    data_grad = image.grad.data
    
    # Generate perturbations with different epsilon values
    epsilons = [0.1, 0.25, 0.5, 1.0]
    
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    fig.suptitle('Understanding Perturbation: What Gets Added to Fool the Model', fontsize=16, fontweight='bold')
    
    for i, eps in enumerate(epsilons):
        # Calculate perturbation
        perturbation = eps * data_grad.sign()
        perturbed_image = torch.clamp(image + perturbation, 0, 1)
        
        # Get new prediction
        with torch.no_grad():
            adv_output = model(perturbed_image)
            adv_pred = adv_output.argmax(dim=1)
            adv_conf = torch.softmax(adv_output, dim=1).max()
        
        # Row 1: Original vs Perturbed
        axes[0, i].imshow(image.squeeze().detach().numpy(), cmap='gray')
        axes[0, i].set_title(f'Original\nPred: {pred.item()}', fontweight='bold')
        axes[0, i].axis('off')
        
        # Row 2: Pure Perturbation (amplified for visibility)
        pert_visual = perturbation.squeeze().detach().numpy()
        axes[1, i].imshow(pert_visual, cmap='RdBu', vmin=-eps, vmax=eps)
        axes[1, i].set_title(f'Perturbation (ε={eps})\nMagnitude: ±{eps}', fontweight='bold')
        axes[1, i].axis('off')
        
        # Row 3: Adversarial Result
        axes[2, i].imshow(perturbed_image.squeeze().detach().numpy(), cmap='gray')
        axes[2, i].set_title(f'Adversarial\nPred: {adv_pred.item()} ({adv_conf:.2f})', 
                           fontweight='bold', 
                           color='red' if adv_pred != pred else 'green')
        axes[2, i].axis('off')
    
    # Add row labels
    fig.text(0.02, 0.75, 'Original\nImage', fontsize=14, fontweight='bold', rotation=90, va='center')
    fig.text(0.02, 0.5, 'Perturbation\n(Noise Added)', fontsize=14, fontweight='bold', rotation=90, va='center')
    fig.text(0.02, 0.25, 'Result\n(Original + Noise)', fontsize=14, fontweight='bold', rotation=90, va='center')
    
    # Add explanation
    explanation = [
        "🔍 Key Observations:",
        "• Perturbation is structured noise (not random)",
        "• Higher ε = more visible perturbation",
        "• Even small ε can fool the model",
        "• Perturbation targets model's decision boundaries"
    ]
    
    fig.text(0.02, 0.15, '\n'.join(explanation), fontsize=12,
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8),
             verticalalignment='top')
    
    plt.tight_layout()
    plt.subplots_adjust(left=0.08, bottom=0.3)
    plt.savefig('results/perturbation_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Perturbation analysis saved to: results/perturbation_analysis.png")
    print(f"📊 Original prediction: {pred.item()}")
    
    # Show perturbation statistics
    for eps in epsilons:
        pert = eps * data_grad.sign()
        print(f"ε={eps}: Perturbation range [{pert.min():.3f}, {pert.max():.3f}], Mean magnitude: {pert.abs().mean():.3f}")

def explain_perturbation_properties():
    """Explain the mathematical properties of perturbation."""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Perturbation Properties: Why It Works', fontsize=16, fontweight='bold')
    
    # Property 1: L∞ Bound
    ax1.set_title('Property 1: Bounded Magnitude (L∞ ≤ ε)', fontweight='bold')
    x = np.linspace(-1, 1, 100)
    eps_vals = [0.1, 0.25, 0.5]
    colors = ['blue', 'orange', 'red']
    
    for eps, color in zip(eps_vals, colors):
        y = np.where(np.abs(x) <= eps, 1, 0)
        ax1.plot(x, y, label=f'ε = {eps}', linewidth=3, color=color)
    
    ax1.set_xlabel('Perturbation Value')
    ax1.set_ylabel('Allowed (1) / Not Allowed (0)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.text(0, 0.5, 'All perturbation values\nmust be within ±ε', 
             ha='center', bbox=dict(boxstyle="round", facecolor="wheat"))
    
    # Property 2: Gradient Direction
    ax2.set_title('Property 2: Follows Gradient Direction', fontweight='bold')
    
    # Create a simple loss landscape
    x = np.linspace(-2, 2, 20)
    y = np.linspace(-2, 2, 20)
    X, Y = np.meshgrid(x, y)
    Z = X**2 + Y**2  # Simple quadratic loss
    
    # Plot contours
    contours = ax2.contour(X, Y, Z, levels=10, colors='gray', alpha=0.5)
    ax2.clabel(contours, inline=True, fontsize=8)
    
    # Show gradient direction at a point
    point_x, point_y = 0.5, 0.5
    grad_x, grad_y = 2*point_x, 2*point_y  # Gradient of x² + y²
    
    ax2.arrow(point_x, point_y, grad_x*0.3, grad_y*0.3, 
              head_width=0.1, head_length=0.1, fc='red', ec='red', linewidth=3)
    ax2.plot(point_x, point_y, 'ro', markersize=10, label='Current point')
    ax2.text(point_x + grad_x*0.4, point_y + grad_y*0.4, 'Gradient\n(steepest ascent)', 
             color='red', fontweight='bold')
    
    ax2.set_xlabel('Input Dimension 1')
    ax2.set_ylabel('Input Dimension 2')
    ax2.legend()
    ax2.set_title('Perturbation follows gradient\nto increase loss', fontweight='bold')
    
    # Property 3: Sign Function
    ax3.set_title('Property 3: Sign Function Simplification', fontweight='bold')
    
    gradient_vals = np.linspace(-5, 5, 1000)
    sign_vals = np.sign(gradient_vals)
    
    ax3.plot(gradient_vals, gradient_vals, label='Original Gradient', alpha=0.7, linewidth=2)
    ax3.plot(gradient_vals, sign_vals, label='sign(Gradient)', linewidth=3, color='red')
    ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax3.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    
    ax3.set_xlabel('Gradient Value')
    ax3.set_ylabel('Output')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.text(0, 3, 'sign() function:\n• +1 if gradient > 0\n• -1 if gradient < 0\n• 0 if gradient = 0', 
             bbox=dict(boxstyle="round", facecolor="lightblue"))
    
    # Property 4: Effect Visualization
    ax4.set_title('Property 4: Cumulative Effect', fontweight='bold')
    
    # Show how small perturbations add up
    pixels = np.arange(10)
    original = np.ones(10) * 0.5  # Gray pixels
    perturbation = np.array([0.1, -0.1, 0.1, -0.1, 0.1, -0.1, 0.1, -0.1, 0.1, -0.1])
    adversarial = original + perturbation
    
    width = 0.25
    ax4.bar(pixels - width, original, width, label='Original', alpha=0.7, color='gray')
    ax4.bar(pixels, perturbation, width, label='Perturbation', alpha=0.7, color='red')
    ax4.bar(pixels + width, adversarial, width, label='Adversarial', alpha=0.7, color='orange')
    
    ax4.set_xlabel('Pixel Index')
    ax4.set_ylabel('Pixel Value')
    ax4.legend()
    ax4.set_ylim(0, 1)
    ax4.text(5, 0.8, 'Small perturbations\nacross many pixels\ncreate large effect', 
             ha='center', bbox=dict(boxstyle="round", facecolor="lightgreen"))
    
    plt.tight_layout()
    plt.savefig('results/perturbation_properties.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Perturbation properties diagram saved to: results/perturbation_properties.png")

if __name__ == "__main__":
    print("🔬 Analyzing perturbation in FGSM attack...")
    visualize_perturbation()
    explain_perturbation_properties()
    print("\n📊 Perturbation Analysis Complete!")
    print("📁 Generated files:")
    print("   • results/perturbation_analysis.png")
    print("   • results/perturbation_properties.png")
