import torch


def unique(x, dim=None):
    """Unique elements of x and indices of those unique elements
    https://github.com/pytorch/pytorch/issues/36748#issuecomment-619514810

    e.g.

    unique(tensor([
        [1, 2, 3],
        [1, 2, 4],
        [1, 2, 3],
        [1, 2, 5]
    ]), dim=0)
    => (tensor([[1, 2, 3],
                [1, 2, 4],
                [1, 2, 5]]),
        tensor([0, 1, 3]))
    """
    uniq, inverse = torch.unique(
        x, sorted=True, return_inverse=True, dim=dim)
    perm = torch.arange(inverse.size(0), dtype=inverse.dtype,
                        device=inverse.device)
    inverse, perm = inverse.flip([0]), perm.flip([0])
    return uniq, inverse.new_empty(uniq.size(0)).scatter_(0, inverse, perm)


# Example input tensors
tables = torch.tensor([0, 1, 1, 2, 0, 2, 1, 0], dtype=torch.long)
nodes = torch.tensor([5, 6, 6, 7, 5, 7, 6, 8], dtype=torch.long)
values = torch.tensor([10.5, 20.0, 25.0, 30.0, 11.0, 35.0, 22.0, 12.5], dtype=torch.float32)

# Combine `tables` and `nodes` into a single tensor of shape (N, 2)
combined = torch.stack([tables, nodes], dim=1)  # Shape: (N, 2)
unique_combined, unique_indices = unique(combined, dim=0)

# Use these indices to select unique values
unique_values = values[unique_indices]

# Output
print("Original Combined Tensor (tables, nodes):")
print(combined)

print("\nUnique Combined Tensor (tables, nodes):")
print(unique_combined)

print("\nIndices of First Occurrences:")
print(unique_indices)

print("\nUnique Values (from `values` tensor):")
print(unique_values)
