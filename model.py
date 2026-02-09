"""
CNN Model for CIFAR-10 Image Classification
Architecture: 3 Convolutional blocks + 2 Fully Connected layers
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    """
    Convolutional Neural Network for CIFAR-10 classification.
    Input: 32x32x3 RGB images
    Output: 10 class probabilities
    """
    
    # CIFAR-10 class labels
    CLASSES = [
        'airplane', 'automobile', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck'
    ]
    
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()
        
        # Convolutional Block 1
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        # Convolutional Block 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Convolutional Block 3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Fully Connected Layers
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)
        
        # Max Pooling
        self.pool = nn.MaxPool2d(2, 2)
        
    def forward(self, x):
        # Block 1: Conv -> BatchNorm -> ReLU -> Pool
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        
        # Block 2: Conv -> BatchNorm -> ReLU -> Pool
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        
        # Block 3: Conv -> BatchNorm -> ReLU -> Pool
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        # Flatten
        x = x.view(-1, 128 * 4 * 4)
        
        # Fully Connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
    
    def predict(self, x):
        """Get class probabilities using softmax."""
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = F.softmax(logits, dim=1)
        return probabilities


def get_model(pretrained_path=None, device='cpu'):
    """
    Create and optionally load a pretrained CNN model.
    
    Args:
        pretrained_path: Path to saved model weights
        device: Device to load model on ('cpu' or 'cuda')
    
    Returns:
        model: CNN model instance
    """
    model = CNN()
    
    if pretrained_path:
        model.load_state_dict(torch.load(pretrained_path, map_location=device))
        model.eval()
    
    return model.to(device)
