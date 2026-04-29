import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from moe_layer import MoELayer

def main():
    input_dim = 16
    hidden_dim = 32
    num_experts = 4

    model = MoELayer(input_dim, hidden_dim, num_experts)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    losses = []  # store loss values

    for epoch in range(20):
        x = torch.randn(32, input_dim)

        output = model(x)
        loss = criterion(output, x)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

    # 📊 Plot loss
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.show()


if __name__ == "__main__":
    main()