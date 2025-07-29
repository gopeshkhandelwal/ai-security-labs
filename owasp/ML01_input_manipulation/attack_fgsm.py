import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from model.simple_cnn import SimpleCNN
from utils.visualize import compare_images

transform = transforms.ToTensor()
test_data = datasets.MNIST(root="./data", train=False, transform=transform, download=True)
test_loader = DataLoader(test_data, batch_size=1, shuffle=True)

model = SimpleCNN()
model.load_state_dict(torch.load("ml01_model.pt"))
model.eval()

image, label = next(iter(test_loader))
image.requires_grad = True
output = model(image)
pred = output.argmax(dim=1)

loss = nn.CrossEntropyLoss()(output, label)
model.zero_grad()
loss.backward()

data_grad = image.grad.data
epsilon = 0.25
perturbed = torch.clamp(image + epsilon * data_grad.sign(), 0, 1)
adv_output = model(perturbed)
adv_pred = adv_output.argmax(dim=1)

compare_images(image, perturbed, pred.item(), adv_pred.item(), "results/original_vs_adversarial_1.png")
print(f"Original: {pred.item()}, Adversarial: {adv_pred.item()}")