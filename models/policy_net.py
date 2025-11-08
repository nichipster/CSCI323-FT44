"""
Policy Network Architecture for 9x9 Go

Based on AlphaGo paper (Silver et al., 2016)
Simplified for 9x9 board size
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyNetwork(nn.Module):
    """
    Convolutional Neural Network for predicting move probabilities
    
    Input: 9x9x4 tensor (board state + history)
    Output: 82-dimensional vector (81 moves + pass)
    """
    
    def __init__(self, input_channels=4, num_filters=48):
        """
        Args:
            input_channels: Number of input feature planes (default: 4)
            num_filters: Number of filters in convolutional layers
        """
        super(PolicyNetwork, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, num_filters, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(num_filters, num_filters*2, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(num_filters*2, num_filters*2, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(num_filters*2, num_filters*2, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(num_filters*2, 1, kernel_size=1)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.bn2 = nn.BatchNorm2d(num_filters*2)
        self.bn3 = nn.BatchNorm2d(num_filters*2)
        self.bn4 = nn.BatchNorm2d(num_filters*2)
        
        # Fully connected layer
        self.fc = nn.Linear(81, 82)  # 81 board positions + 1 pass move
        
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, 4, 9, 9)
            
        Returns:
            Move probabilities of shape (batch_size, 82)
        """
        # Convolutional layers with ReLU and batch norm
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.conv5(x))
        
        # Flatten
        x = x.view(-1, 81)
        
        # Fully connected layer
        x = self.fc(x)
        
        # Softmax for probability distribution
        return F.softmax(x, dim=1)
    
    def predict_move(self, board_state, legal_moves_mask=None):
        """
        Predict best move for a given board state
        
        Args:
            board_state: Tensor of shape (4, 9, 9)
            legal_moves_mask: Binary mask for legal moves (optional)
            
        Returns:
            move_idx: Index of predicted move
            probability: Probability of that move
        """
        self.eval()
        with torch.no_grad():
            # Add batch dimension
            x = board_state.unsqueeze(0)
            
            # Get probabilities
            probs = self.forward(x).squeeze(0)
            
            # Mask illegal moves if provided
            if legal_moves_mask is not None:
                probs = probs * legal_moves_mask
                probs = probs / probs.sum()  # Renormalize
            
            # Get best move
            move_idx = torch.argmax(probs).item()
            probability = probs[move_idx].item()
            
        return move_idx, probability
    
    def save(self, filepath):
        """Save model checkpoint"""
        torch.save({
            'model_state_dict': self.state_dict(),
            'architecture': {
                'input_channels': self.conv1.in_channels,
                'num_filters': self.conv1.out_channels
            }
        }, filepath)
    
    @classmethod
    def load(cls, filepath, device='cpu'):
        """Load model from checkpoint"""
        checkpoint = torch.load(filepath, map_location=device)
        model = cls(**checkpoint['architecture'])
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        return model


if __name__ == "__main__":
    # Test the network
    model = PolicyNetwork(input_channels=4, num_filters=48)
    
    # Create dummy input
    dummy_input = torch.randn(1, 4, 9, 9)
    
    # Forward pass
    output = model(dummy_input)
    
    print(f"Model architecture:\n{model}")
    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output sum (should be ~1.0): {output.sum().item():.4f}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
