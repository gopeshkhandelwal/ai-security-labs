import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from model.simple_cnn import SimpleCNN

def train_model():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Data loading
    transform = transforms.ToTensor()
    train_data = datasets.MNIST(root="../../data", train=True, transform=transform, download=True)
    test_data = datasets.MNIST(root="../../data", train=False, transform=transform, download=True)
    
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=1000, shuffle=False)
    
    # Model, loss, optimizer
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training - quick 2 epochs
    print("Starting training...")
    model.train()
    for epoch in range(2):
        running_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if batch_idx % 300 == 0:
                print(f'Epoch {epoch+1}/2, Batch {batch_idx}, Loss: {loss.item():.6f}')
    
    # Test the model
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')
    
    # Save the model
    torch.save(model.cpu().state_dict(), "ml01_model.pt")
    print("Model saved as ml01_model.pt")

if __name__ == "__main__":
    train_model()
