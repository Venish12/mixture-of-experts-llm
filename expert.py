import torch
import torch.nn as nn

class Expert(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()

        # First linear layer
        self.fc1 = nn.Linear(input_dim, hidden_dim)

        # Activation
        self.relu = nn.ReLU()

        # Second linear layer (back to input_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


