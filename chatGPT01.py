import torch

# Example 1D tensor
parent = torch.randn(32)  # Shape (32,)

# Add a new dimension with unsqueeze and repeat across the new dimension
children = parent.unsqueeze(1).repeat(1, 40)  # Shape (32, 40)

print(children.shape)  # Output: torch.Size([32, 40])
