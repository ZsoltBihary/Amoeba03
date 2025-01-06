import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from typing import Tuple
# from ClassBufferManager import BufferManager
from helper_functions import duplicate_indices
from line_profiler_pycharm import profile


class SearchTree:
    def __init__(self, num_table, num_node, num_child):
        self.num_table = num_table
        self.num_child = num_child
        self.num_node = num_node
        # Set up tree attributes
        # self.next_node = 2 * torch.ones(self.num_table, dtype=torch.long)
        self.next_node = torch.zeros(self.num_table, dtype=torch.long)
        self.is_leaf = torch.zeros((self.num_table, self.num_node), dtype=torch.bool)
        self.is_terminal = torch.zeros((self.num_table, self.num_node), dtype=torch.bool)
        # self.player = torch.ones((self.num_table, self.num_node), dtype=torch.int32)
        self.count = torch.zeros((self.num_table, self.num_node), dtype=torch.int32)
        self.value_sum = torch.zeros((self.num_table, self.num_node), dtype=torch.float32)
        self.value = torch.zeros((self.num_table, self.num_node), dtype=torch.float32)
        self.start_child = torch.zeros((self.num_table, self.num_node), dtype=torch.long)
        # self.parent = torch.ones((self.num_table, self.num_node), dtype=torch.long)
        self.action = torch.zeros((self.num_table, self.num_node), dtype=torch.long)
        self.prior = torch.zeros((self.num_table, self.num_node), dtype=torch.float32)
        self.ucb = torch.zeros((self.num_table, self.num_node), dtype=torch.float32)

    def __str__(self):
        # User-friendly string representation
        return f"MyClass(next_node={self.next_node.tolist()}, start_child={self.start_child[:, :20].tolist()})"

    def reset(self):
        # self.next_node[:] = 2
        self.next_node[:] = 2
        self.is_leaf[:, :] = True
        # TODO: just for testing ... begin
        # self.is_leaf[:, 1] = False
        # TODO: just for testing ... end
        self.is_terminal[:, :] = False
        # self.player[:, :] = 0
        self.count[:, :] = 0
        self.value_sum[:, :] = 0.0
        # self.value[:, :] = 0.0
        # self.start_child[:, :] = 2
        # self.start_child[:, :] = 0
        # self.parent[:, :] = 1
        # self.action[:, :] = 0
        # self.prior[:, :] = 0.0
        # self.ucb[:, :] = -9999.9
        return

    def get_children(self, parent_table, parent_node):
        # child_table = parent_table.unsqueeze(1).repeat(1, self.num_child)
        # Add the start index to the offsets to get the actual indices ... relying on broadcasting here ...
        start_node = self.start_child[parent_table, parent_node].reshape(-1, 1)
        node_offset = torch.arange(self.num_child, dtype=torch.long).reshape(1, -1)
        child_node = start_node + node_offset
        # return child_table, child_node, node_offset
        return child_node

    def calc_priors(self, logit):
        top_values, top_actions = torch.topk(logit, self.num_child, dim=1)
        top_prior = torch.softmax(top_values, dim=1)
        return top_actions, top_prior

    @profile
    def expand(self, table, parent, logit):
        # ***** These are modified within expand
        #       self.next_node
        #       self.is_leaf
        #       self.start_child
        #       self.action
        #       self.prior
        # Here we assume that (table, node) tuples are unique ...
        # We also assume that the nodes are not terminal, so we expand all in the list ...
        # Deal with the parents ...
        self.is_leaf[table, parent] = False
        block_offset = duplicate_indices(table)
        begin_child = self.next_node[table] + block_offset * self.num_child
        self.start_child[table, parent] = begin_child
        # Deal with the children ...
        children = self.get_children(table, parent)
        actions, priors = self.calc_priors(logit)
        self.action[table.view(-1, 1), children] = actions
        self.prior[table.view(-1, 1), children] = priors
        # Count multiplicity of tables, adjust self.next_node accordingly ...
        table_count = torch.bincount(table, minlength=self.num_table)
        self.next_node[:] += self.num_child * table_count[:]
        # Return children as we need them for ucb update ...
        return children

    def back_propagate(self, table, node, value, multi):
        # TODO: ***** These are modified within propagation
        #       self.count[:, :]
        #       self.value_sum[:, :]
        #       self.value[:, :]
        self.count.index_put_((table, node), multi, accumulate=True)
        self.value_sum.index_put_((table, node), multi * value, accumulate=True)
        self.value[table, node] = self.value_sum[table, node] / self.count[table, node]
        return

    def update_ucb(self, table, parent, child, parent_player):
        child_q = self.value[table, child]
        child_prior = self.prior[table, child]
        parent_count = self.count[table, parent]
        child_count = self.count[table, child]
        # self.ucb[table, child] = (parent_player * child_q +
        #                           2.0 * child_prior * torch.sqrt(parent_count + 1) / (child_count + 1))
        #
        # ucb_value = (parent_player * child_q +
        #                           2.0 * child_prior * torch.sqrt(parent_count + 1) / (child_count + 1))

        # # Calculate UCB normally
        # ucb_values = (parent_player * child_q +
        #               2.0 * child_prior * torch.sqrt(parent_count + 1) / (child_count + 1))
        #
        # # Assign a very low value where child_prior < 0.001
        # self.ucb[table, child] = torch.where(child_prior < 0.001,
        #                                      -9999.9,
        #                                      ucb_values)

        # Assign a very low value where child_prior < 0.001
        self.ucb[table, child] = torch.where(child_prior < 0.001,
                                             -9999.9,
                                             (parent_player * child_q +
                                              2.0 * child_prior * torch.sqrt(parent_count + 1) / (child_count + 1)))

        return
