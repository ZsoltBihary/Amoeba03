import torch

# Example tensors
table = torch.tensor([1, 2, 1, 3, 2, 1], dtype=torch.long)  # Shape (6,)
node = torch.tensor([4, 5, 4, 6, 5, 7], dtype=torch.long)   # Shape (6,)
position = torch.zeros((6, 3), dtype=torch.int32)
position[:, 0] = table
position[:, 1] = node
position[:, 2] = node - table
# Combine table and node into tuples
tuples = torch.stack((table, node), dim=1)  # Shape (6, 2)
# Find unique tuples and their indices
unique_tuples, unique_inverse_indices = torch.unique(tuples, dim=0, return_inverse=True, return_counts=False)
unique_n = unique_tuples.shape[0]
all_unique_indices = torch.arange(unique_n)
unique_position = torch.zeros((unique_n, 3), dtype=torch.int32)
unique_position[unique_inverse_indices, :] = position[unique_inverse_indices, :]
# # Filter out repeated tuples
# unique_tuples1 = tuples[unique_indices]
# filtered_table = table[unique_indices]
# filtered_node = node[unique_indices]

# Print results
print("Filtered Table:", filtered_table)
print("Filtered Node:", filtered_node)

#
# # Example tensors
# table = torch.tensor([1, 2, 1, 3, 2, 1], dtype=torch.long)  # Shape (6,)
# node = torch.tensor([4, 5, 4, 6, 5, 7], dtype=torch.long)   # Shape (6,)
#
# # Define max_table (assumed known or computed as the maximum value in table + 1)
# max_table = table.max().item() + 1
#
# # Encode tuples as unique numbers
# code = node * max_table + table  # Shape (6,)
#
# # Find unique codes and their indices
# unique_codes, unique_indices = torch.unique(code, return_inverse=True, return_counts=False, sorted=False)
#
# # Filter the original table and node using unique indices
# filtered_table = table[unique_indices]
# filtered_node = node[unique_indices]
#
# # Print results
# print("Filtered Table:", filtered_table)
# print("Filtered Node:", filtered_node)
