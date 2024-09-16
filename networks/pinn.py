
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 1)
        )

    def forward(self, x, t):
        inputs = torch.cat([x, t], dim=1)
        u = self.net(inputs)
        return u

    def extract_features(self, x, t):
        """
        This method extracts the features right before the last layer.
        """
        inputs = torch.cat([x, t], dim=1)
        features = self.net[:-1](inputs)  # Extract features before the last layer (the final Linear layer)
        return features
