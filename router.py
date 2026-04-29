import torch
import torch.nn as nn
import torch.nn.functional as F

class Router(nn.Module):
    def __init__(self, input_dim, num_experts):
        super().__init__()

        self.num_experts = num_experts

        # Linear layer to compute scores for each expert
        self.gate = nn.Linear(input_dim, num_experts)

    def forward(self, x):
        # x shape: (batch_size, input_dim)

        # Compute scores
        logits = self.gate(x)

        # Convert to probabilities
        probs = F.softmax(logits, dim=-1)

        # Select top-1 expert for each token
        top_expert = torch.argmax(probs, dim=-1)

        return top_expert, probs


