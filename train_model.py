import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
import torch.utils.data
import cv2
import numpy as np
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset
import platform
import os
def get_device():
    """Determine and return the optimal device for PyTorch computations."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

device = get_device()
print(f"System: {platform.system().lower()}")
print(f"Using device: {device}")

print(torch.__version__)  
print(torch.version.cuda) 
print(torch.cuda.is_available())  
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))  
else:
    print("CUDA is not available. Using CPU.")

class ChestXRayDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        categories = ["normal", "pneumonia"]
        label_map = {"normal": 0, "pneumonia": 1}

        for category in categories:
            category_path = os.path.join(root_dir, category)
            if not os.path.exists(category_path):
                print(f"Warning: Directory not found: {category_path}")
                continue
            for file_name in os.listdir(category_path):
                if file_name.lower().endswith('.jpeg'):
                    self.image_paths.append(os.path.join(category_path, file_name))
                    self.labels.append(label_map[category])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load the image
        image_path = self.image_paths[idx]
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Image not found at path: {image_path}")
        
        # Convert grayscale to RGB by duplicating the channels
        image = np.stack([image, image, image], axis=-1)
        image = image.astype(np.float32) / 255.0

        # Apply transforms if provided
        if self.transform:
            image = self.transform(image)
        
        label = self.labels[idx]
        return image, label
    
    # Define transforms
transform = transforms.Compose([
    transforms.ToTensor(),  # Converts H,W,C NumPy array to C,H,W tensor
    transforms.Normalize(mean=[0.485, 0.485, 0.485], std=[0.229, 0.229, 0.229])
])

train_dir = os.path.join(os.path.dirname(__file__), "..", "processed_dataset", "train")
test_dir = os.path.join(os.path.dirname(__file__), "..", "processed_dataset", "test")

train_dataset = ChestXRayDataset(root_dir=train_dir, transform=transform)
test_dataset = ChestXRayDataset(root_dir=test_dir, transform=transform)

# Create data loaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

# Test the data loader
images, labels = next(iter(train_loader))
print(f"Image batch shape: {images.shape}")  
print(f"Label batch shape: {labels.shape}") 

# Task 1: Load Pretrained MobileNetV2
model = models.mobilenet_v2(weights="IMAGENET1K_V1")  # Use weights instead of pretrained

# Task 2: Modify Classifier to Output 2 Classes (normal, pneumonia)
model.classifier[1] = torch.nn.Linear(in_features=model.classifier[1].in_features, out_features=2)

# Task 3: Move Model to Device
model = model.to(device)

# Task 4: Test - Print Classifier to Confirm Modification
print("Modified Classifier Structure:")
print(model.classifier)

# Classification
criterion = nn.CrossEntropyLoss()  # Fixed: Removed redundant L1Loss definition, kept CrossEntropyLoss for binary classification

# Using Adam Optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Fixed: Corrected 'paramters' to 'parameters'

# Example training step (corrected from the incorrect snippet)
model.train()  # Set model to training mode
optimizer.zero_grad()  # Clear gradients
inputs, labels = next(iter(train_loader))  # Get a batch of data
inputs, labels = inputs.to(device), labels.to(device)  # Move data to device
outputs = model(inputs)  # Forward pass
loss = criterion(outputs, labels)  # Compute loss
loss.backward()  
optimizer.step() 

print("Loss function and optimizer set up successfully.")

# Training loop
for epoch in range(5):
    model.train()
    running_loss = 0.0
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    avg_loss = running_loss / len(train_loader)
    print(f'Epoch {epoch + 1}/5, Training Loss: {avg_loss:.4f}')

# Validation loop
model.eval()
val_loss = 0.0
correct = 0
total = 0
with torch.no_grad():
    for data, target in test_loader:  
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = criterion(output, target)
        val_loss += loss.item()
        _, predicted = torch.max(output, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

avg_val_loss = val_loss / len(test_loader)
val_accuracy = 100 * correct / total
print(f'Validation Loss: {avg_val_loss:.4f}, Accuracy: {val_accuracy:.2f}%')

# Saving Path
save_dir = os.path.join(os.path.dirname(__file__), "..")  
save_path = os.path.join(save_dir, "pneumonia_model.pth")

# Save the model's state dictionary
torch.save(model.state_dict(), save_path)
print(f"Model saved to {save_path}")