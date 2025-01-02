import torch

# Example where 4 is missing in the table
table = torch.tensor([0, 1, 2, 3, 1, 2, 2, 3, 0], dtype=torch.long)

# Use torch.bincount
count = torch.bincount(table, minlength=5)  # Ensures output has length 5

print("Input table:", table)
print("Count tensor:", count)
