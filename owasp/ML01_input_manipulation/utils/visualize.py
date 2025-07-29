import matplotlib.pyplot as plt

def compare_images(original, adversarial, orig_label, adv_label, filepath):
    plt.figure(figsize=(6, 3))
    plt.subplot(1, 2, 1)
    plt.title(f"Original: {orig_label}")
    plt.imshow(original.squeeze().detach().numpy(), cmap="gray")

    plt.subplot(1, 2, 2)
    plt.title(f"Adversarial: {adv_label}")
    plt.imshow(adversarial.squeeze().detach().numpy(), cmap="gray")

    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()