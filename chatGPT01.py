import torch
# Create a tensor
x = torch.zeros(5, 5, dtype=torch.long)

# Indices to update (with duplicates)
indices = (torch.tensor([0, 1, 1]), torch.tensor([1, 3, 3]))
values = torch.tensor([10, 20, 30])

# Use index_put_ with accumulate=True
x.index_put_(indices, values, accumulate=True)

print("After index_put_ with accumulate=True:\n", x)
