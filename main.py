# main.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

# Tensor operatifons
def tensor_operations():
    print("Tensor Operations:")
    x = torch.tensor([1, 2, 3])
    print("Simple tensor:", x)
    random_tensor = torch.rand(3, 3)
    print("Random tensor:\n", random_tensor)

# Neural Network definition
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Dataset creation
class MyDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# Training function
def train_model(model, dataloader, num_epochs):
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    for epoch in range(num_epochs):
        total_loss = 0
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss:.4f}")

# Main function
def main():
    # Tensor operations
    tensor_operations()

    # Create model
    model = SimpleNet()
    print("\nModel structure:")
    print(model)

    # Create dummy data
    np.random.seed(42)
    data = torch.from_numpy(np.random.rand(100, 10).astype(np.float32))
    labels = torch.from_numpy(np.random.rand(100, 1).astype(np.float32))

    # Create dataset and dataloader
    dataset = MyDataset(data, labels)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Train the model
    print("\nTraining the model:")
    train_model(model, dataloader, num_epochs=5)

    # Save the model
    torch.save(model.state_dict(), 'model.pth')
    print("\nModel saved as 'model.pth'")

    # Load the model
    loaded_model = SimpleNet()
    loaded_model.load_state_dict(torch.load('model.pth', weights_only=True))
    loaded_model.eval()
    print("Model loaded successfully")

    # Make a prediction
    with torch.no_grad():
        random_input = torch.rand(1, 10)
        prediction = loaded_model(random_input)
        print(f"\nPrediction for random input: {prediction.item():.4f}")

if __name__ == "__main__":
    main()
