import torch
from helper_functions import unique  # , duplicate_indices
from line_profiler_pycharm import profile


class LeafBuffer:
    def __init__(self, buffer_size, action_size, max_depth):
        self.size = buffer_size
        self.next_idx = 0

        self.table = torch.zeros(self.size, dtype=torch.long)
        self.node = torch.zeros(self.size, dtype=torch.long)
        self.player = torch.zeros(self.size, dtype=torch.int32)
        self.position = torch.zeros((self.size, action_size), dtype=torch.int32)
        self.path = torch.zeros((self.size, max_depth), dtype=torch.long)
        self.depth = torch.zeros(self.size, dtype=torch.long)
        self.logit = torch.zeros((self.size, action_size), dtype=torch.float32)
        self.value = torch.zeros(self.size, dtype=torch.float32)
        self.is_terminal = torch.zeros(self.size, dtype=torch.bool)

        self.multi = torch.zeros(self.size, dtype=torch.int32)

    def empty(self):
        self.next_idx = 0

    def add_leaves(self, tables, nodes, players, positions, paths, depths):
        end_idx = self.next_idx + tables.shape[0]
        self.table[self.next_idx: end_idx] = tables
        self.node[self.next_idx: end_idx] = nodes
        self.player[self.next_idx: end_idx] = players
        self.position[self.next_idx: end_idx, :] = positions
        self.path[self.next_idx: end_idx, :] = paths
        self.depth[self.next_idx: end_idx] = depths

        self.next_idx = end_idx
        return

    def get_states(self):
        # players = self.player[: self.next_idx].view(-1, 1)
        # positions = self.position[: self.next_idx]
        states = self.player[: self.next_idx].view(-1, 1) * self.position[: self.next_idx]
        return states

    @profile
    def filter_unique(self):
        tables = self.table[: self.next_idx]
        nodes = self.node[: self.next_idx]
        players = self.player[: self.next_idx]
        positions = self.position[: self.next_idx, :]
        paths = self.path[: self.next_idx, :]
        depths = self.depth[: self.next_idx]
        # Combine tables and nodes into a single tensor of shape (N, 2)
        combined = torch.stack([tables, nodes], dim=1)
        uni_combined, uni_count, uni_index = unique(combined, dim=0)
        # Use these indices to select unique values
        self.empty()
        self.add_leaves(tables[uni_index],
                        nodes[uni_index],
                        players[uni_index],
                        positions[uni_index, :],
                        paths[uni_index, :],
                        depths[uni_index])
        self.multi[: self.next_idx] = uni_count
        return

    def add_eval_results(self, logits, values, are_terminal):
        self.logit[: self.next_idx, :] = logits
        self.value[: self.next_idx] = values * self.player[: self.next_idx]
        self.is_terminal[: self.next_idx] = are_terminal
        return

    def get_leaf_data(self):
        return (self.table[: self.next_idx],
                self.node[: self.next_idx],
                self.player[: self.next_idx],
                # self.position[: self.next_idx, :],
                self.logit[: self.next_idx, :],
                # self.value[: self.next_idx],
                self.is_terminal[: self.next_idx])
        # self.multi[: self.next_idx])

    def get_path_data(self):
        # Truncate path tensors based on max depth ...
        maximal_depth = torch.max(self.depth[: self.next_idx])
        paths = self.path[: self.next_idx, :maximal_depth]
        # Flatten the path tensor to get nodes
        nodes = paths.flatten()
        # Repeat the table, value and multi tensors
        tables = self.table[: self.next_idx].repeat_interleave(maximal_depth)
        values = self.value[: self.next_idx].repeat_interleave(maximal_depth)
        multis = self.multi[: self.next_idx].repeat_interleave(maximal_depth)
        # Filter out zero nodes
        node_filter = (nodes > 0)
        return tables[node_filter], nodes[node_filter], values[node_filter], multis[node_filter]


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
        new_parent_player = torch.zeros(new_size, dtype=torch.int32)
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

    @profile
    def filter_unique(self):
        tables, parents, children, players = self.get_data()
        #  Combine data into a single tensor of shape (N, 4)
        combined = torch.stack([tables, parents, children, players], dim=1)
        uni_combined = torch.unique(combined, dim=0)
        uni_tables = uni_combined[:, 0]
        uni_parents = uni_combined[:, 1]
        uni_children = uni_combined[:, 2]
        uni_players = uni_combined[:, 3]
        self.empty()
        self.add(uni_tables, uni_parents, uni_children, uni_players)
        return


class BufferManager:
    def __init__(self, leaf_buffer_size, child_buffer_size, min_batch_size, action_size, max_depth):

        # self.leaf_buffer_size = leaf_buffer_size
        # self.child_buffer_size = child_buffer_size
        self.min_batch_size = min_batch_size
        self.action_size = action_size
        self.max_depth = max_depth

        self.leaf_collect = LeafBuffer(leaf_buffer_size, action_size, max_depth)
        self.leaf_eval = LeafBuffer(leaf_buffer_size, action_size, max_depth)
        self.child_collect = ChildBuffer(child_buffer_size)
        self.child_eval = ChildBuffer(child_buffer_size)
        self.batch_full = False

    def reset(self):
        self.leaf_collect.empty()
        self.leaf_eval.empty()
        self.child_collect.empty()
        self.child_eval.empty()
        self.batch_full = False

    def add_leaves(self, tables, nodes, players, positions, paths, depths):
        self.leaf_collect.add_leaves(tables, nodes, players, positions, paths, depths)
        if self.leaf_collect.next_idx > self.min_batch_size:
            self.batch_full = True

    def add_children(self, tables, parents, children, parent_players, collect):
        if collect:
            self.child_collect.add(tables, parents, children, parent_players)
        else:
            self.child_eval.add(tables, parents, children, parent_players)

    def get_states(self):
        return self.leaf_eval.get_states()

    def add_eval_results(self, logits, values, are_terminal):
        self.leaf_eval.add_eval_results(logits, values, are_terminal)

    def get_expand_data(self):
        return self.leaf_eval.get_leaf_data()

    def get_propagate_data(self):
        return self.leaf_eval.get_path_data()

    def get_ucb_data(self):
        return self.child_eval.get_data()

    @profile
    def post_process(self):
        self.leaf_collect.filter_unique()
        # self.child_collect.filter_unique()
        return

    def swap_buffers(self):
        self.leaf_collect, self.leaf_eval = self.leaf_eval, self.leaf_collect
        self.leaf_collect.empty()
        self.child_collect, self.child_eval = self.child_eval, self.child_collect
        self.child_collect.empty()
        self.batch_full = False
