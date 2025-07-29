import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from model.simple_cnn import SimpleCNN
from utils.visualize import compare_images
import os

# Create results directory if it doesn't exist
os.makedirs("results", exist_ok=True)

transform = transforms.ToTensor()
test_data = datasets.MNIST(root="./data", train=False, transform=transform, download=True)
test_loader = DataLoader(test_data, batch_size=1, shuffle=True)

model = SimpleCNN()
model.load_state_dict(torch.load("ml01_model.pt"))
model.eval()

def is_adversarial(data_grad, probs, threshold_grad=3.0, threshold_conf=0.85):
    grad_norm = torch.norm(data_grad).item()
    confidence = probs.max().item()
    return grad_norm > threshold_grad or confidence < threshold_conf

image, label = next(iter(test_loader))
image.requires_grad = True
output = model(image)
init_pred = output.argmax(dim=1)
loss = nn.CrossEntropyLoss()(output, label)
model.zero_grad()
loss.backward()
data_grad = image.grad.data

epsilon = 0.25
perturbed = torch.clamp(image + epsilon * data_grad.sign(), 0, 1)
probs = F.softmax(model(perturbed), dim=1)
final_pred = probs.argmax(dim=1)

flagged = is_adversarial(data_grad, probs)
compare_images(image, perturbed, init_pred.item(), final_pred.item(), "results/original_vs_adversarial_flagged.png")
print(f"Adversarial detected: {flagged}, Pred: {final_pred.item()}, Confidence: {probs.max().item():.2f}")