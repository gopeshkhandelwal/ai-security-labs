import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def create_fgsm_attack_diagram():
    """Create a comprehensive diagram explaining the FGSM attack process."""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('FGSM (Fast Gradient Sign Method) Attack Explained', fontsize=20, fontweight='bold')
    
    # Step 1: Original Image and Prediction
    ax1.set_title('Step 1: Original Image Classification', fontsize=14, fontweight='bold')
    
    # Simulate original MNIST digit (create a simple "2" pattern)
    original_img = np.zeros((28, 28))
    # Draw a rough "2" shape
    original_img[8:12, 5:20] = 0.8  # top horizontal
    original_img[12:16, 15:20] = 0.8  # right vertical
    original_img[16:20, 5:20] = 0.8  # middle horizontal
    original_img[20:24, 5:10] = 0.8  # left vertical
    original_img[24:28, 5:20] = 0.8  # bottom horizontal
    
    ax1.imshow(original_img, cmap='gray', vmin=0, vmax=1)
    ax1.set_xlabel('Model Prediction: "2" (Confidence: 98%)', fontsize=12, fontweight='bold', color='green')
    ax1.set_xticks([])
    ax1.set_yticks([])
    
    # Add arrow and label
    ax1.annotate('Original Input', xy=(14, 5), xytext=(14, -3),
                arrowprops=dict(arrowstyle='->', lw=2, color='blue'),
                ha='center', fontsize=12, fontweight='bold')
    
    # Step 2: Gradient Calculation
    ax2.set_title('Step 2: Calculate Loss Gradient', fontsize=14, fontweight='bold')
    
    # Simulate gradient (random noise pattern for visualization)
    np.random.seed(42)
    gradient = np.random.randn(28, 28) * 0.3
    gradient_sign = np.sign(gradient)
    
    im2 = ax2.imshow(gradient_sign, cmap='RdBu', vmin=-1, vmax=1)
    ax2.set_xlabel('∇Loss = ∂Loss/∂Input\n(Direction of steepest increase)', fontsize=12, fontweight='bold')
    ax2.set_xticks([])
    ax2.set_yticks([])
    
    # Add colorbar for gradient
    cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.7)
    cbar2.set_label('Gradient Sign', rotation=270, labelpad=15)
    
    # Step 3: Add Perturbation
    ax3.set_title('Step 3: Add Adversarial Perturbation', fontsize=14, fontweight='bold')
    
    # Create adversarial example
    epsilon = 0.25
    perturbation = epsilon * gradient_sign
    adversarial_img = np.clip(original_img + perturbation, 0, 1)
    
    ax3.imshow(adversarial_img, cmap='gray', vmin=0, vmax=1)
    ax3.set_xlabel(f'Adversarial = Original + ε×sign(∇Loss)\nε = {epsilon}', 
                  fontsize=12, fontweight='bold')
    ax3.set_xticks([])
    ax3.set_yticks([])
    
    # Add formula annotation
    ax3.text(14, 30, 'x_adv = x + ε × sign(∇_x L(θ, x, y))', 
             ha='center', fontsize=11, bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    
    # Step 4: Fooled Prediction
    ax4.set_title('Step 4: Model is Fooled!', fontsize=14, fontweight='bold')
    ax4.imshow(adversarial_img, cmap='gray', vmin=0, vmax=1)
    ax4.set_xlabel('Model Prediction: "7" (Confidence: 87%)', fontsize=12, fontweight='bold', color='red')
    ax4.set_xticks([])
    ax4.set_yticks([])
    
    # Add comparison text
    ax4.annotate('Looks like "2" to humans\nBut model sees "7"!', 
                xy=(14, 5), xytext=(14, -8),
                arrowprops=dict(arrowstyle='->', lw=2, color='red'),
                ha='center', fontsize=12, fontweight='bold', color='red')
    
    # Add process flow arrows between subplots
    fig.text(0.32, 0.75, '→', fontsize=30, ha='center', va='center', color='blue', fontweight='bold')
    fig.text(0.68, 0.75, '→', fontsize=30, ha='center', va='center', color='blue', fontweight='bold')
    fig.text(0.5, 0.52, '↓', fontsize=30, ha='center', va='center', color='blue', fontweight='bold')
    
    # Add key insights box
    insights = [
        "🎯 Key Insights:",
        "• Perturbation is IMPERCEPTIBLE to humans",
        "• Changes are strategically placed to maximize model error",
        "• Attack exploits model's decision boundaries",
        "• Same technique works across different model architectures"
    ]
    
    fig.text(0.02, 0.15, '\n'.join(insights), fontsize=12, 
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8),
             verticalalignment='top')
    
    # Add algorithm box
    algorithm = [
        "🔬 FGSM Algorithm:",
        "1. Forward pass: get model prediction",
        "2. Calculate loss L(θ, x, y)",
        "3. Compute gradient: ∇_x L(θ, x, y)",
        "4. Generate perturbation: ε × sign(∇_x L)",
        "5. Create adversarial example: x_adv = x + perturbation"
    ]
    
    fig.text(0.52, 0.15, '\n'.join(algorithm), fontsize=12,
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8),
             verticalalignment='top')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9, bottom=0.3)
    plt.savefig('results/fgsm_attack_diagram.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ FGSM Attack Diagram saved to: results/fgsm_attack_diagram.png")

def create_attack_vs_defense_flowchart():
    """Create a flowchart showing attack vs defense strategies."""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 10))
    fig.suptitle('Attack vs Defense: Complete Workflow', fontsize=18, fontweight='bold')
    
    # Attack Flow (Left side)
    ax1.set_title('🗡️ ATTACKER Workflow', fontsize=16, fontweight='bold', color='red')
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.axis('off')
    
    # Attack flow boxes
    attack_boxes = [
        (5, 9, "1. Load Target Model"),
        (5, 7.5, "2. Select Input Image"),
        (5, 6, "3. Calculate Loss Gradient"),
        (5, 4.5, "4. Generate Perturbation"),
        (5, 3, "5. Create Adversarial Example"),
        (5, 1.5, "🎯 SUCCESS: Model Fooled!")
    ]
    
    for x, y, text in attack_boxes:
        if "SUCCESS" in text:
            color = 'red'
            alpha = 0.3
        else:
            color = 'orange'
            alpha = 0.2
            
        rect = patches.Rectangle((x-2, y-0.4), 4, 0.8, 
                               linewidth=2, edgecolor='red', 
                               facecolor=color, alpha=alpha)
        ax1.add_patch(rect)
        ax1.text(x, y, text, ha='center', va='center', fontweight='bold', fontsize=11)
        
        # Add arrows between boxes
        if y > 1.5:
            ax1.arrow(x, y-0.5, 0, -0.5, head_width=0.2, head_length=0.1, 
                     fc='red', ec='red', linewidth=2)
    
    # Defense Flow (Right side)
    ax2.set_title('🛡️ DEFENDER Workflow', fontsize=16, fontweight='bold', color='blue')
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.axis('off')
    
    # Defense flow boxes
    defense_boxes = [
        (5, 9, "1. Receive Input Image"),
        (5, 7.5, "2. Calculate Input Gradients"),
        (5, 6, "3. Measure Gradient Magnitude"),
        (5, 4.5, "4. Check Prediction Confidence"),
        (5, 3, "5. Apply Detection Thresholds"),
        (5, 1.5, "🛡️ DEFENSE: Attack Detected!")
    ]
    
    for x, y, text in defense_boxes:
        if "DEFENSE" in text:
            color = 'blue'
            alpha = 0.3
        else:
            color = 'lightblue'
            alpha = 0.2
            
        rect = patches.Rectangle((x-2, y-0.4), 4, 0.8, 
                               linewidth=2, edgecolor='blue', 
                               facecolor=color, alpha=alpha)
        ax2.add_patch(rect)
        ax2.text(x, y, text, ha='center', va='center', fontweight='bold', fontsize=11)
        
        # Add arrows between boxes
        if y > 1.5:
            ax2.arrow(x, y-0.5, 0, -0.5, head_width=0.2, head_length=0.1, 
                     fc='blue', ec='blue', linewidth=2)
    
    # Add detection criteria
    criteria_text = [
        "Detection Criteria:",
        "• Gradient Norm > 3.0",
        "• Confidence < 0.85",
        "• Either condition → FLAGGED"
    ]
    
    ax2.text(8.5, 3, '\n'.join(criteria_text), fontsize=10,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
             verticalalignment='center')
    
    plt.tight_layout()
    plt.savefig('results/attack_vs_defense_flowchart.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Attack vs Defense Flowchart saved to: results/attack_vs_defense_flowchart.png")

if __name__ == "__main__":
    # Create results directory
    import os
    os.makedirs("results", exist_ok=True)
    
    # Generate both diagrams
    create_fgsm_attack_diagram()
    create_attack_vs_defense_flowchart()
    
    print("\n🎨 All diagrams generated successfully!")
    print("📁 Check the results/ directory for:")
    print("   • fgsm_attack_diagram.png")
    print("   • attack_vs_defense_flowchart.png")
