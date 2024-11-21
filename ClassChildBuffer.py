import torch
from line_profiler_pycharm import profile


class ChildBuffer:
    def __init__(self, buffer_size):
        self.size = buffer_size
        self.next_idx = 0
        self.table = torch.zeros(self.size, dtype=torch.long)
        self.parent = torch.zeros(self.size, dtype=torch.long)
        self.child = torch.zeros(self.size, dtype=torch.long)
        self.parent_player = torch.zeros(self.size, dtype=torch.int32)

    def empty(self):
        self.next_idx = 0
        return

    def resize(self, new_size):
        # Create new tensors with the increased size
        new_table = torch.zeros(new_size, dtype=torch.long)
        new_parent = torch.zeros(new_size, dtype=torch.long)
        new_child = torch.zeros(new_size, dtype=torch.long)
        new_parent_player = torch.zeros(new_size, dtype=torch.long)
        # Copy existing data into the new tensors
        new_table[:self.next_idx] = self.table[:self.next_idx]
        new_parent[:self.next_idx] = self.parent[:self.next_idx]
        new_child[:self.next_idx] = self.child[:self.next_idx]
        new_parent_player[:self.next_idx] = self.parent_player[:self.next_idx]
        # Update references to the new buffers
        self.table = new_table
        self.parent = new_parent
        self.child = new_child
        self.parent_player = new_parent_player
        # Update the buffer size
        self.size = new_size
        return

    def add(self, tables, parents, children, parent_players):
        num_new_entries = tables.shape[0]
        end_idx = self.next_idx + num_new_entries
        # Check if buffer size needs to be increased
        if end_idx > self.size:
            # Double the size to accommodate growth
            new_size = max(self.size * 2, end_idx)
            self.resize(new_size)
        # Add the new data
        self.table[self.next_idx: end_idx] = tables
        self.parent[self.next_idx: end_idx] = parents
        self.child[self.next_idx: end_idx] = children
        self.parent_player[self.next_idx: end_idx] = parent_players
        # Update the next index
        self.next_idx = end_idx
        return

    def get_data(self):
        return (self.table[: self.next_idx],
                self.parent[: self.next_idx],
                self.child[: self.next_idx],
                self.parent_player[: self.next_idx])

        # self.ucb = -99999.9 * torch.ones((self.num_table, self.num_node), dtype=torch.float32)

        # self.ucb[children] = (parent_player * child_q +
        #                               2.0 * child_prior * torch.sqrt(parent_count + 1) / (child_count + 1))
