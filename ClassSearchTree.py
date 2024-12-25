import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from typing import Tuple
# from ClassBufferManager import BufferManager
from helper_functions import duplicate_indices
# from line_profiler_pycharm import profile


class SearchTree:
    def __init__(self, num_table, num_node, num_child):
        self.num_table = num_table
        self.num_child = num_child
        self.num_node = num_node
        # Set up tree attributes
        self.next_node = 2 * torch.ones(self.num_table, dtype=torch.long)
        self.is_leaf = torch.ones((self.num_table, self.num_node), dtype=torch.bool)
        self.is_terminal = torch.zeros((self.num_table, self.num_node), dtype=torch.bool)
        self.player = torch.ones((self.num_table, self.num_node), dtype=torch.int32)
        self.count = torch.zeros((self.num_table, self.num_node), dtype=torch.int32)
        self.value_sum = torch.zeros((self.num_table, self.num_node), dtype=torch.float32)
        self.value = torch.zeros((self.num_table, self.num_node), dtype=torch.float32)
        self.start_child = 2 * torch.ones((self.num_table, self.num_node), dtype=torch.long)
        self.parent = torch.ones((self.num_table, self.num_node), dtype=torch.long)
        self.action = torch.zeros((self.num_table, self.num_node), dtype=torch.long)
        self.prior = torch.zeros((self.num_table, self.num_node), dtype=torch.float32)
        self.ucb = -99999.9 * torch.ones((self.num_table, self.num_node), dtype=torch.float32)
        # self.search_count = torch.zeros(self.num_table, dtype=torch.int32)

    def reset(self):
        self.next_node[:] = 2
        self.is_leaf[:, :] = True
        self.is_terminal[:, :] = False
        self.player[:, :] = 0
        self.count[:, :] = 0
        self.value_sum[:, :] = 0.0
        self.value[:, :] = 0.0
        self.start_child[:, :] = 2
        self.parent[:, :] = 1
        self.action[:, :] = 0
        self.prior[:, :] = 0
        self.ucb[:, :] = -99999.9
        # self.search_count[:] = 0
        return

    def get_children(self, parent_table, parent_node):
        child_table = parent_table.unsqueeze(1).repeat(1, self.num_child)
        # Add the start index to the offsets to get the actual indices ... relying on broadcasting here ...
        start_node = self.start_child[parent_table, parent_node].reshape(-1, 1)
        node_offset = torch.arange(self.num_child, dtype=torch.long).reshape(1, -1)
        child_node = start_node + node_offset
        return child_table, child_node, node_offset

    def update_ucb(self, table, parent, child, parent_player):
        # table, parent, child, parent_player = self.child_buffer.get_data()
        # Is this so simple? Perhaps we need to keep fresh results ...
        child_q = self.value[table, child]
        child_prior = self.prior[table, child]
        parent_count = self.count[table, parent]
        child_count = self.count[table, child]
        self.ucb[table, child] = (parent_player * child_q +
                                  2.0 * child_prior * torch.sqrt(parent_count + 1) / (child_count + 1))
        return

    def update_tree(self):

        # # EXPAND
        # # TODO:
        # (table,
        #  node,
        #  player,
        #  position,
        #  logit,
        #  value,
        #  is_terminal) = self.leaf_buffer.get_eval_data()
        # # select unique leaves ...
        #
        # combined = torch.stack([table, node], dim=1)  # Shape: (N, 2)
        # uni_combined, uni_indices = unique(combined, dim=0)
        # uni_table = uni_combined[:, 0]
        # uni_node = uni_combined[:, 1]
        # # Use these indices to select unique values, etc.
        # uni_player = player[uni_indices]
        # uni_policy = logit[uni_indices, :]
        # uni_value = value[uni_indices]
        # uni_is_terminal = is_terminal[uni_indices]
        #
        # self.value[uni_table, uni_node] = uni_value
        # self.is_terminal[uni_table, uni_node] = uni_is_terminal
        #
        # block_offset = duplicate_indices(uni_table)
        # self.start_child[uni_table, uni_node] = self.next_node[uni_table] + block_offset * self.num_child
        #
        # # # UPDATE counts and values in the trees ...
        # # # Scatter-add to handle repeated indices
        # # # Prepare the indices
        # # indices = torch.stack([tables, nodes], dim=0)  # Shape (2, 10)
        # # # Perform scatter-add
        # # self.value_sum.scatter_add_(0, indices, values)
        #
        # # UPDATE ucb in the trees ...
        # self.update_ucb()
        return

    def calc_priors(self, logit):
        top_values, top_action = torch.topk(logit, self.num_child, dim=1)
        top_prior = torch.softmax(top_values, dim=1)
        return top_action, top_prior

    def expand(self, table, node, player, position, logit, value):
        # Here we assume that (table, node) tuples are unique ...
        # We also assume that the nodes are not terminal, so we expand all in the list ...
        block_offset = duplicate_indices(table)
        begin_child = self.next_node[table] + block_offset * self.num_child
        self.start_child[table, node] = begin_child
        end_child = begin_child + self.num_child

        action, prior = self.calc_priors(logit)

        # self.prior[table, begin_child: end_child] = prior[:]

        # Count multiplicity of tables, adjust self.next_node accordingly.
        value_counts = torch.bincount(table)
        table_count = value_counts[table]
        self.next_node[table] += self.num_child * table_count[table]

        return
