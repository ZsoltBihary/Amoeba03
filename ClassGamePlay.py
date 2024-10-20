import torch
# import torch.nn as nn
import torch.nn.functional as F
# from AmoebaClass import Amoeba
# from typing import Tuple
from ClassEvaluator import EvaluationBuffer
# from ClassModel import evaluate01
from line_profiler_pycharm import profile


class GamePLay:
    def __init__(self, args: dict, game, terminal_check, model):
        self.args = args
        self.game = game
        self.terminal_check = terminal_check
        self.model = model
        # Often used parameters
        self.action_size = game.action_size
        self.num_MC = args.get('num_MC')
        self.num_branch = args.get('num_branch')
        self.num_leaf = args.get('num_leaf')
        self.num_node = self.num_MC * self.num_leaf * self.action_size + 10
        self.CUDA_device = args.get('CUDA_device')

        # Set up search tree
        self.tree = {
            'is_leaf': torch.ones(self.num_node, dtype=torch.bool),
            'is_terminal': torch.zeros(self.num_node, dtype=torch.bool),
            'count': torch.zeros(self.num_node, dtype=torch.int32),
            'value_sum': torch.zeros(self.num_node, dtype=torch.float32),
            'value': torch.zeros(self.num_node, dtype=torch.float32),
            # 'parent_idx': -torch.ones(self.num_node, dtype=torch.long),
            'start_child_idx': 2 * torch.ones(self.num_node, dtype=torch.long),
            'num_child': torch.zeros(self.num_node, dtype=torch.long),
            'action': torch.zeros(self.num_node, dtype=torch.long),
            'prior': torch.zeros(self.num_node, dtype=torch.float32),
            'next_node_idx': 2
        }

        # Set up leaf buffer
        self.i_leaf = 0
        # self.n_leaf = 0
        self.max_depth = self.action_size + 1
        self.leaf_buffer = {
            'node_idx': torch.zeros((self.num_leaf, self.max_depth), dtype=torch.long),
            'leaf_idx': torch.zeros(self.num_leaf, dtype=torch.long),
            'player': torch.ones(self.num_leaf, dtype=torch.int32),
            'position': torch.zeros((self.num_leaf, self.action_size), dtype=torch.int32),
            'is_terminal': torch.zeros(self.num_leaf, dtype=torch.bool),
            'policy': torch.zeros((self.num_leaf, self.action_size), dtype=torch.float32),
            'value': torch.zeros(self.num_leaf, dtype=torch.float32),
        }

    def reset_tree(self):
        pass

    def add_leaf(self, node_idx, player, position):
        # print('   i_leaf = ', self.i_leaf)
        # self.game.print_board(position)
        self.leaf_buffer['leaf_idx'][self.i_leaf] = node_idx
        self.leaf_buffer['player'][self.i_leaf] = player
        self.leaf_buffer['position'][self.i_leaf, :] = position
        self.i_leaf += 1

    def calc_ucb(self, node_idx, parent_player):

        # Let us calculate the ucb for all children ...
        # parent_player = self.node_player[parent_table].view(-1, 1)
        start_children_idx = self.tree['start_child_idx'][node_idx].item()
        end_children_idx = start_children_idx + self.tree['num_child'][node_idx].item()
        child_nodes = torch.arange(start_children_idx, end_children_idx)
        parent_count = self.tree['count'][node_idx]
        child_count = self.tree['count'][child_nodes]
        child_q = self.tree['value'][child_nodes]
        child_prior = self.tree['prior'][child_nodes]
        ucb = (parent_player * child_q +
               2.0 * child_prior * torch.sqrt(parent_count + 1) / (child_count + 1))

        return ucb

    @profile
    def search_leaves(self, depth, node_idx, player, position):
        # print('   - depth, node_idx: ', depth, node_idx)
        if self.i_leaf == self.num_leaf:
            return
        self.leaf_buffer['node_idx'][self.i_leaf:, depth] = node_idx
        if self.tree['is_leaf'][node_idx]:
            if depth+1 < self.max_depth:
                self.leaf_buffer['node_idx'][self.i_leaf, depth+1:] = 0
            self.add_leaf(node_idx, player, position)
        else:
            new_player = -player
            ucb = self.calc_ucb(node_idx, player)
            for i_branch in range(self.num_branch):
                if self.i_leaf == self.num_leaf:
                    break
                best_offset = torch.argmax(ucb)
                ucb[best_offset] -= 0.2
                best_child_idx = self.tree['start_child_idx'][node_idx] + best_offset
                action = self.tree['action'][best_child_idx]
                new_position = self.game.get_new_position(position, player, action)
                self.search_leaves(depth+1, best_child_idx, new_player, new_position)

    @profile
    def update_tree(self):
        # ***** Evaluate *****
        players = self.leaf_buffer['player'][:self.i_leaf]
        positions = self.leaf_buffer['position'][:self.i_leaf, :]
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
        value[minus_mask] = -1.0
        value[plus_mask] = 1.0
        self.leaf_buffer['value'][:self.i_leaf] = players * value

        # ***** EXPAND *****
        terminal_mask = plus_mask | minus_mask | draw_mask
        expand_mask = ~terminal_mask
        all_i_leaf = torch.arange(0, self.i_leaf)
        parent_i = all_i_leaf[expand_mask]
        all_leaf_idx = self.leaf_buffer['leaf_idx'][:self.i_leaf]
        parent_idx = all_leaf_idx[expand_mask]
        legal_mask = (torch.abs(positions) < 0.5)
        all_actions = torch.arange(0, self.action_size)

        self.tree['is_terminal'][all_leaf_idx] = terminal_mask
        self.tree['is_leaf'][parent_idx] = False
        for i in parent_i:
            actions = all_actions[legal_mask[i]]
            policy_logit = logit[i, legal_mask[i]]
            policy = torch.nn.functional.softmax(policy_logit, dim=0)
            n_child = actions.shape[0]
            start_children = self.tree['next_node_idx']
            end_children = start_children + n_child
            parent = self.leaf_buffer['leaf_idx'][i]
            self.tree['start_child_idx'][parent] = start_children
            self.tree['num_child'][parent] = n_child
            self.tree['action'][start_children: end_children] = actions
            self.tree['prior'][start_children: end_children] = policy
            self.tree['next_node_idx'] += n_child

    @profile
    def back_propagate(self):
        # 'node_idx': torch.zeros((self.num_leaf, self.max_depth), dtype=torch.long),
        path_idx = self.leaf_buffer['node_idx'][:self.i_leaf]
        # print(path_idx)
        for i in range(self.i_leaf):
            path = path_idx[i]
            path = path[path > 0]
            self.tree['count'][path] += 1
            self.tree['value_sum'][path] += self.leaf_buffer['value'][i]
        for i in range(self.i_leaf):
            path = path_idx[i]
            path = path[path > 0]
            self.tree['value'][path] = self.tree['value_sum'][path] / self.tree['count'][path]

    @profile
    def analyze(self, player, position):

        for i_MC in range(self.num_MC):
            print('i_MC = ', i_MC)
            self.i_leaf = 0
            self.search_leaves(0, 1, player, position)
            self.update_tree()
            self.back_propagate()

        root_idx = 1
        root_value = self.tree['value'][root_idx]
        start_idx = self.tree['start_child_idx'][root_idx].item()
        end_idx = start_idx + self.tree['num_child'][root_idx].item()
        root_children_idx = torch.arange(start_idx, end_idx)
        children_actions = self.tree['action'][root_children_idx]
        action_count = torch.zeros(self.action_size, dtype=torch.int32)
        action_count[children_actions] = self.tree['count'][root_children_idx]
        action_weight = action_count.to(dtype=torch.float32)
        action_policy = action_weight / torch.sum(action_weight)

        return action_policy, root_value

        # action_value = torch.zeros(self.action_size, dtype=torch.float32)
        # action_value[children_actions] = self.tree['value'][root_children_idx]

        # self.game.print_board(position)
        # print('move count: \n', action_count.view(self.game.board_size, -1))
        # print('move weight: \n', action_weight.view(self.game.board_size, -1))
        # print('move policy: \n', action_policy.view(self.game.board_size, -1))
