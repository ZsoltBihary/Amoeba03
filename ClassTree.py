import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from AmoebaClass import Amoeba
# from typing import Tuple
# from ClassEvaluator import EvaluationBuffer
# from ClassModel import evaluate01
from line_profiler_pycharm import profile


class Tree:
    # def __init__(self, args: dict, game, terminal_check, model):
    def __init__(self, num_table, num_node, num_child):
        # ***** These are constant parameters that class methods never modify *****
        self.num_table = num_table
        self.num_node = num_node
        self.num_child = num_child
        self.root_node = 1

        # ***** These are variables that class methods will modify *****
        # Set up tree
        self.next_node = 2 * torch.ones(self.num_table, dtype=torch.long)
        self.is_leaf = torch.ones((self.num_table, self.num_node), dtype=torch.bool)
        self.is_terminal = torch.zeros((self.num_table, self.num_node), dtype=torch.bool)
        self.player = torch.zeros((self.num_table, self.num_node), dtype=torch.int32)
        self.count = torch.zeros((self.num_table, self.num_node), dtype=torch.int32)
        self.value_sum = torch.zeros((self.num_table, self.num_node), dtype=torch.float32)
        self.value = torch.zeros((self.num_table, self.num_node), dtype=torch.float32)
        self.start_child = 2 * torch.ones((self.num_table, self.num_node), dtype=torch.long)
        self.parent = torch.ones((self.num_table, self.num_node), dtype=torch.long)
        self.action = torch.zeros((self.num_table, self.num_node), dtype=torch.long)
        self.prior = torch.zeros((self.num_table, self.num_node), dtype=torch.float32)
        self.ucb = -99999.9 * torch.ones((self.num_table, self.num_node), dtype=torch.float32)

    def reset(self):
        # ***** These are variables that class methods will modify *****
        # Set up tree
        self.next_node = 2 * torch.ones(self.num_table, dtype=torch.long)
        self.is_leaf = torch.ones((self.num_table, self.num_node), dtype=torch.bool)
        self.is_terminal = torch.zeros((self.num_table, self.num_node), dtype=torch.bool)
        self.player = torch.zeros((self.num_table, self.num_node), dtype=torch.int32)
        self.count = torch.zeros((self.num_table, self.num_node), dtype=torch.int32)
        self.value_sum = torch.zeros((self.num_table, self.num_node), dtype=torch.float32)
        self.value = torch.zeros((self.num_table, self.num_node), dtype=torch.float32)
        self.start_child = 2 * torch.ones((self.num_table, self.num_node), dtype=torch.long)
        # self.num_child = torch.zeros((self.num_table, self.num_node), dtype=torch.long)
        self.parent = torch.ones((self.num_table, self.num_node), dtype=torch.long)
        self.action = torch.zeros((self.num_table, self.num_node), dtype=torch.long)
        self.prior = torch.zeros((self.num_table, self.num_node), dtype=torch.float32)
        self.ucb = -99999.9 * torch.ones((self.num_table, self.num_node), dtype=torch.float32)

    def get_children(self, parent_table, parent_node):
        child_table = parent_table.unsqueeze(1).repeat(1, self.num_child)
        # Add the start index to the offsets to get the actual indices ... relying on broadcasting here ...
        start_node = self.start_child[parent_table, parent_node].reshape(-1, 1)
        node_offset = torch.arange(self.num_child, dtype=torch.long).reshape(1, -1)
        child_node = start_node + node_offset
        return child_table, child_node, node_offset

    def best_child(self, parent_table, parent_node):
        child_table, child_node, node_offset = self.get_children(parent_table, parent_node)
        best_idx = torch.argmax(self.ucb[child_table, child_node], dim=1)
        # best_idx = torch.argmax(ucb1, dim=1)
        best_child_node = child_node[torch.arange(best_idx.shape[0]), best_idx]
        return child_node[torch.arange(best_idx.shape[0]), best_idx]

    def allocate_child_block(self, parent_table, parent_node):
        self.start_child[parent_table, parent_node] = self.next_node[parent_table]
        self.next_node[parent_table] += self.num_child
        return

    @profile
    def expand(self):
        # ***** Evaluate *****
        self.leaf_is_terminal, self.leaf_logit, self.leaf_value = (
            self.evaluate(self.leaf_player, self.leaf_position))

        # ***** Expand *****
        legal_mask = (torch.abs(self.leaf_position) < 0.5)
        all_actions = torch.arange(0, self.action_size)
        for i in range(self.num_leaf):
            parent_i = self.leaf_node[i]
            self.player[parent_i] = self.leaf_player[i]
            self.is_terminal[parent_i] = self.leaf_is_terminal[i]
            to_expand = self.is_leaf[parent_i] & ~self.leaf_is_terminal[i]
            if to_expand:
                self.is_leaf[parent_i] = False
                actions = all_actions[legal_mask[i]]
                policy_logit = self.leaf_logit[i, legal_mask[i]]

                # Determine the number of children to select (use min to avoid selecting more than available)
                n_child = min(self.max_child, policy_logit.shape[0])
                # Get the top n_child values and their indices from policy_logit
                top_values, top_indices = torch.topk(policy_logit, n_child)
                # Use the indices to select the corresponding actions
                selected_policy_logit = top_values
                selected_actions = actions[top_indices]

                policy = torch.nn.functional.softmax(selected_policy_logit, dim=0)
                start_children = self.next_node
                end_children = start_children + n_child

                self.start_child[parent_i] = start_children
                self.num_child[parent_i] = n_child
                self.action[start_children: end_children] = selected_actions
                self.prior[start_children: end_children] = policy
                self.parent[start_children: end_children] = parent_i
                self.next_node += n_child

    @profile
    def back_propagate(self):
        for leaf in range(self.num_leaf):
            self.count[self.leaf_path[leaf, :]] += 1
            self.value_sum[self.leaf_path[leaf, :]] += self.leaf_value[leaf]
        self.value[self.leaf_path] = self.value_sum[self.leaf_path] / self.count[self.leaf_path]
        children = self.children_buffer[:self.children_next_idx]
        self.children_next_idx = 0
        parents = self.parent[children]
        parent_player = self.player[parents]
        parent_count = self.count[parents]
        child_count = self.count[children]
        child_q = self.value[children]
        child_prior = self.prior[children]

        self.ucb[children] = (parent_player * child_q +
                              2.0 * child_prior * torch.sqrt(parent_count + 1) / (child_count + 1))

    #     # 'node_idx': torch.zeros((self.num_leaf, self.max_depth), dtype=torch.long),
    #     path_idx = self.leaf_buffer['node_idx'][:]
    #     # print(path_idx)
    #     for i in range(self.num_leaf):
    #         path = path_idx[i]
    #         path = path[path > 0]
    #         self.tree['count'][path] += 1
    #         self.tree['value_sum'][path] += self.leaf_buffer['value'][i]
    #     for i in range(self.num_leaf):
    #         path = path_idx[i]
    #         path = path[path > 0]
    #         self.tree['value'][path] = self.tree['value_sum'][path] / self.tree['count'][path]

    @profile
    def analyze(self, player, position):
        self.reset_tree()
        for i_MC in range(self.num_MC):
            # print('i_MC = ', i_MC)
            self.search_leaves(player, position)
            self.update_tree()
            self.back_propagate()

        # root_idx = 1
        root_value = self.value[self.root_node]
        # start_idx = self.tree['start_child_idx'][root_idx].item()
        # end_idx = start_idx + self.tree['num_child'][root_idx].item()
        # root_children_idx = torch.arange(start_idx, end_idx)
        root_children = self.get_children(self.root_node)
        children_actions = self.action[root_children]
        action_count = torch.zeros(self.action_size, dtype=torch.int32)
        action_count[children_actions] = self.count[root_children]
        action_weight = action_count.to(dtype=torch.float32)
        action_policy = action_weight / torch.sum(action_weight)

        # action_value = torch.zeros(self.action_size, dtype=torch.float32)
        # action_value[children_actions] = self.tree['value'][root_children_idx]
        # self.game.print_board(position)
        # print('move count: \n', action_count.view(self.game.board_size, -1))
        # print('move weight: \n', action_weight.view(self.game.board_size, -1))
        # print('value = ', root_value)
        # print('move policy: \n', action_policy.view(self.game.board_size, -1))

        return action_policy, root_value

    # ****** THIS IS THE OLDER VERSION ************************
    # @profile
    # def search_one_leaf(self, leaf, depth, node_idx, player, position):
    #     while not self.tree['is_leaf'][node_idx]:
    #         self.leaf_buffer['node_idx'][leaf, depth] = node_idx
    #         ucb = self.calc_ucb(node_idx, player)
    #         best_offset = torch.argmax(ucb)
    #         best_child_idx = self.tree['start_child_idx'][node_idx] + best_offset
    #         action = self.tree['action'][best_child_idx]
    #         position = self.game.get_new_position(position, player, action)
    #         player = -player
    #         depth += 1
    #         node_idx = best_child_idx
    #     # Leaf found !!!
    #     self.leaf_buffer['node_idx'][leaf, depth] = node_idx
    #     self.add_leaf(leaf, leaf+1, node_idx, player, position)
    #     return
    # ****** THIS IS THE OLDER VERSION ************************

    # ****** THIS IS THE OLDER VERSION ************************
    # @profile
    # def search_leaves(self, depth, node_idx, player, position):
    #     # print('   - depth, node_idx: ', depth, node_idx)
    #     if self.i_leaf == self.num_leaf:
    #         return
    #     self.leaf_buffer['node_idx'][self.i_leaf:, depth] = node_idx
    #     if self.tree['is_leaf'][node_idx]:
    #         if depth+1 < self.max_depth:
    #             self.leaf_buffer['node_idx'][self.i_leaf, depth+1:] = 0
    #         self.add_leaf(node_idx, player, position)
    #     else:
    #         new_player = -player
    #         ucb = self.calc_ucb(node_idx, player)
    #         for i_branch in range(self.num_branch):
    #             if self.i_leaf == self.num_leaf:
    #                 break
    #             best_offset = torch.argmax(ucb)
    #             ucb[best_offset] -= 0.2
    #             best_child_idx = self.tree['start_child_idx'][node_idx] + best_offset
    #             action = self.tree['action'][best_child_idx]
    #             new_position = self.game.get_new_position(position, player, action)
    #             self.search_leaves(depth+1, best_child_idx, new_player, new_position)
    # ****** THIS IS THE OLDER VERSION ************************

    # ****** THIS IS THE OLDER VERSION ************************
    # def calc_ucb(self, parent, parent_player):
    #
    #     # Let us calculate the ucb for all children ...
    #     # parent_player = self.node_player[parent_table].view(-1, 1)
    #     start_children_idx = self.tree['start_child_idx'][node_idx].item()
    #     end_children_idx = start_children_idx + self.tree['num_child'][node_idx].item()
    #     child_nodes = torch.arange(start_children_idx, end_children_idx)
    #     parent_count = self.tree['count'][node_idx]
    #     child_count = self.tree['count'][child_nodes]
    #     child_q = self.tree['value'][child_nodes]
    #     child_prior = self.tree['prior'][child_nodes]
    #     ucb = (parent_player * child_q +
    #            2.0 * child_prior * torch.sqrt(parent_count + 1) / (child_count + 1))
    #
    #     return ucb
    # ****** THIS IS THE OLDER VERSION ************************
