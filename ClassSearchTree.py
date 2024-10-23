import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from AmoebaClass import Amoeba
# from typing import Tuple
# from ClassEvaluator import EvaluationBuffer
# from ClassModel import evaluate01
from line_profiler_pycharm import profile


class SearchTree:
    def __init__(self, args: dict, game, terminal_check, model):
        # ***** These are constant parameters that class methods never modify *****
        self.args = args
        self.game = game
        self.terminal_check = terminal_check
        self.model = model
        # Often used parameters
        self.action_size = game.action_size
        self.num_MC = args.get('num_MC')          
        self.CUDA_device = args.get('CUDA_device')
        # Parameters for tree and leaf buffer
        self.num_leaf = args.get('num_leaf')        
        self.num_node = self.num_MC * self.num_leaf * self.action_size + 10
        self.max_depth = self.action_size + 1
        self.root_node = 1

        # ***** These are variables that class methods will modify *****
        # Set up tree
        self.next_node = 2
        self.is_leaf = torch.ones(self.num_node, dtype=torch.bool)
        self.is_terminal = torch.zeros(self.num_node, dtype=torch.bool)
        self.player = torch.zeros(self.num_node, dtype=torch.int32)
        self.count = torch.zeros(self.num_node, dtype=torch.int32)
        self.value_sum = torch.zeros(self.num_node, dtype=torch.float32)
        self.value = torch.zeros(self.num_node, dtype=torch.float32)
        self.start_child = 2 * torch.ones(self.num_node, dtype=torch.long)
        self.num_child = torch.zeros(self.num_node, dtype=torch.long)
        self.parent = torch.ones(self.num_node, dtype=torch.long)
        self.action = torch.zeros(self.num_node, dtype=torch.long)
        self.prior = torch.zeros(self.num_node, dtype=torch.float32)
        self.ucb = torch.zeros(self.num_node, dtype=torch.float32)       
        # Set up leaf buffer
        self.leaf_path = torch.zeros((self.num_leaf, self.max_depth), dtype=torch.long)
        self.leaf_node = torch.zeros(self.num_leaf, dtype=torch.long)
        self.leaf_player = torch.ones(self.num_leaf, dtype=torch.int32)
        self.leaf_position = torch.zeros((self.num_leaf, self.action_size), dtype=torch.int32)
        self.leaf_is_terminal = torch.zeros(self.num_leaf, dtype=torch.bool)
        self.leaf_logit = torch.zeros((self.num_leaf, self.action_size), dtype=torch.float32)
        self.leaf_value = torch.zeros(self.num_leaf, dtype=torch.float32)
        # Set up children buffer
        self.children_next_idx = 0
        self.children_buffer = torch.zeros(self.num_leaf*self.max_depth*self.action_size, dtype=torch.long)

    def reset_tree(self):
        # ***** These are variables that class methods will modify *****
        # Set up tree
        self.next_node = 2
        self.is_leaf = torch.ones(self.num_node, dtype=torch.bool)
        self.is_terminal = torch.zeros(self.num_node, dtype=torch.bool)
        self.player = torch.zeros(self.num_node, dtype=torch.int32)
        self.count = torch.zeros(self.num_node, dtype=torch.int32)
        self.value_sum = torch.zeros(self.num_node, dtype=torch.float32)
        self.value = torch.zeros(self.num_node, dtype=torch.float32)
        self.start_child = 2 * torch.ones(self.num_node, dtype=torch.long)
        self.num_child = torch.zeros(self.num_node, dtype=torch.long)
        self.parent = torch.ones(self.num_node, dtype=torch.long)
        self.action = torch.zeros(self.num_node, dtype=torch.long)
        self.prior = torch.zeros(self.num_node, dtype=torch.float32)
        self.ucb = torch.zeros(self.num_node, dtype=torch.float32)
        # Set up leaf buffer
        self.leaf_path = torch.zeros((self.num_leaf, self.max_depth), dtype=torch.long)
        self.leaf_node = torch.zeros(self.num_leaf, dtype=torch.long)
        self.leaf_player = torch.ones(self.num_leaf, dtype=torch.int32)
        self.leaf_position = torch.zeros((self.num_leaf, self.action_size), dtype=torch.int32)
        self.leaf_is_terminal = torch.zeros(self.num_leaf, dtype=torch.bool)
        self.leaf_logit = torch.zeros((self.num_leaf, self.action_size), dtype=torch.float32)
        self.leaf_value = torch.zeros(self.num_leaf, dtype=torch.float32)
        # Set up children buffer
        self.children_next_idx = 0
        self.children_buffer = torch.zeros(self.num_leaf * self.max_depth * self.action_size, dtype=torch.long)

    def get_children(self, parent):
        start_ch = self.start_child[parent].item()
        end_ch = start_ch + self.num_child[parent].item()
        return torch.arange(start_ch, end_ch)

    def add_children_to_buffer(self, children):
        children_n = children.shape[0]
        children_end_idx = self.children_next_idx + children_n
        self.children_buffer[self.children_next_idx: children_end_idx] = children
        self.children_next_idx = children_end_idx

    def add_leaf_to_buffer(self, leaf, node, player, position):
        # pass
        # print('   i_leaf = ', self.i_leaf)
        # self.game.print_board(position)
        self.leaf_node[leaf] = node
        self.leaf_player[leaf] = player
        self.leaf_position[leaf, :] = position

    @profile
    def search_one_leaf(self, leaf, player, position):
        # def search_one_leaf(self, leaf, depth, node_idx, player, position):
        depth = 0
        node = self.root_node
        while not self.is_leaf[node]:
            self.leaf_path[leaf, depth] = node
            children = self.get_children(node)
            self.add_children_to_buffer(children)
            # ucb = self.calc_ucb(node_idx, player)
            best_offset = torch.argmax(self.ucb[children])
            best_child = self.start_child[node] + best_offset
            self.ucb[best_child] -= 0.01
            best_action = self.action[best_child]
            position = self.game.get_new_position(position, player, best_action)
            player = -player
            depth += 1
            node = best_child
        # Leaf found !!!
        self.leaf_path[leaf, depth] = node
        self.add_leaf_to_buffer(leaf, node, player, position)
        return

    @profile
    def search_leaves(self, player, position):
        for leaf in torch.arange(0, self.num_leaf):
            self.search_one_leaf(leaf, player, position)
        return

    @profile
    def check_EOG(self, players, positions):

        states = players[:, None] * positions
        states_CUDA = states.to(device=self.CUDA_device, dtype=torch.float32, non_blocking=True)
        # result_CUDA = self.model(states_CUDA)
        term_indicator = self.terminal_check(states_CUDA).to(device='cpu', non_blocking=False)
        # logit = result_CUDA[0].to(device='cpu', non_blocking=False)
        # value = result_CUDA[1].to(device='cpu', non_blocking=False)
        # Interpret result ...
        dir_max = term_indicator[:, 0]
        dir_min = term_indicator[:, 1]
        sum_abs = term_indicator[:, 2]
        plus_mask = (dir_max + 0.1 > self.game.win_length)
        minus_mask = (dir_min - 0.1 < -self.game.win_length)
        draw_mask = (sum_abs + 0.1 > self.action_size)
        value = torch.zeros(players.shape[0], dtype=torch.float32)
        value[draw_mask] = 0.0
        value[plus_mask] = 1.0
        value[minus_mask] = -1.0
        value = players * value
        terminal_mask = plus_mask | minus_mask | draw_mask

        return terminal_mask, value

    @profile
    def evaluate(self, players, positions):

        states = players[:, None] * positions
        states_CUDA = states.to(device=self.CUDA_device, dtype=torch.float32, non_blocking=True)
        result_CUDA = self.model(states_CUDA)
        term_indicator = self.terminal_check(states_CUDA).to(device='cpu', non_blocking=False)
        logit = result_CUDA[0].to(device='cpu', non_blocking=False)
        value = result_CUDA[1].to(device='cpu', non_blocking=False)
        # Interpret result ...
        dir_max = term_indicator[:, 0]
        dir_min = term_indicator[:, 1]
        sum_abs = term_indicator[:, 2]
        plus_mask = (dir_max + 0.1 > self.game.win_length)
        minus_mask = (dir_min - 0.1 < -self.game.win_length)
        draw_mask = (sum_abs + 0.1 > self.action_size)
        value[draw_mask] = 0.0
        value[plus_mask] = 1.0
        value[minus_mask] = -1.0
        value = players * value
        is_terminal = plus_mask | minus_mask | draw_mask

        return is_terminal, logit, value

    @profile
    def update_tree(self):
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
                policy = torch.nn.functional.softmax(policy_logit, dim=0)
                n_child = actions.shape[0]
                start_children = self.next_node
                end_children = start_children + n_child

                self.start_child[parent_i] = start_children
                self.num_child[parent_i] = n_child
                self.action[start_children: end_children] = actions
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
        print('value = ', root_value)
        print('move policy: \n', action_policy.view(self.game.board_size, -1))

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
