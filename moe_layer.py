import torch
import torch.nn as nn
from expert import Expert
from router import Router

class MoELayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_experts):
        super().__init__()

        self.num_experts = num_experts

        # Create experts
        self.experts = nn.ModuleList([
            Expert(input_dim, hidden_dim) for _ in range(num_experts)
        ])

        # Create router
        self.router = Router(input_dim, num_experts)

    def forward(self, x):
        # x shape: (batch_size, input_dim)

        expert_idx, _ = self.router(x)

        output = torch.zeros_like(x)

        # Route each token
        for i in range(x.size(0)):
            idx = expert_idx[i].item()
            output[i] = self.experts[idx](x[i])

        return output


