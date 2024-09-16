import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from AmoebaClass import Amoeba
# import ModelClass as mo
from typing import Tuple
from line_profiler_pycharm import profile


class Analyzer:
    def __init__(self, args: dict, evaluator, current_player: torch.Tensor, current_state: torch.Tensor):
        self.args = args
        self.evaluator = evaluator
        self.current_player = current_player.detach().clone()
        self.current_state = current_state.detach().clone()

        self.evaluate = evaluator.evaluate
        self.check_EOG = evaluator.check_EOG
        self.model = evaluator.model
        self.board_size = args.get('board_size')
        self.action_size = self.board_size * self.board_size
        self.num_child = args.get('num_child')
        self.num_table = args.get('num_table')
        self.num_MC = args.get('num_MC')
        self.num_node = self.num_MC * self.num_child + 100
        # self.inv_temp = args.get('inv_temp')
        self.device = self.args.get('CPU_device')
        # Set up search tree ...
        self.tree = {
            'player': torch.zeros((self.num_table, self.num_node), dtype=torch.int32, device=self.device),
            # 'is_leaf': torch.zeros((self.num_table, self.num_node), dtype=torch.bool, device=self.device),
            'is_leaf': torch.ones((self.num_table, self.num_node), dtype=torch.bool, device=self.device),
            'is_terminal': torch.zeros((self.num_table, self.num_node), dtype=torch.bool, device=self.device),
            'count': torch.zeros((self.num_table, self.num_node), dtype=torch.int32, device=self.device),
            'value_sum': torch.zeros((self.num_table, self.num_node), dtype=torch.float32, device=self.device),
            'value': torch.zeros((self.num_table, self.num_node), dtype=torch.float32, device=self.device),
            'parent': -torch.ones((self.num_table, self.num_node), dtype=torch.long, device=self.device),
            'start_child': -torch.ones((self.num_table, self.num_node), dtype=torch.long, device=self.device),
            'action': torch.zeros((self.num_table, self.num_node), dtype=torch.long, device=self.device),
            'prior': torch.zeros((self.num_table, self.num_node), dtype=torch.float32, device=self.device),

            'next_node': torch.ones(self.num_table, dtype=torch.long, device=self.device)
        }

    def reset(self, current_player: torch.Tensor, current_state: torch.Tensor):
        self.current_player = current_player.detach().clone()
        self.current_state = current_state.detach().clone()

        self.tree['player'][:, :] = 0
        self.tree['is_leaf'][:, :] = True
        self.tree['is_terminal'][:, :] = False
        self.tree['count'][:, :] = 0
        self.tree['value_sum'][:, :] = 0.0
        self.tree['value'][:, :] = 0.0
        self.tree['parent'][:, :] = -1
        self.tree['start_child'][:, :] = -1
        self.tree['action'][:, :] = 0
        self.tree['prior'][:, :] = 0.0
        self.tree['next_node'][:] = 1

    def find_children(self, parent_table: torch.Tensor, parent_node: torch.Tensor) \
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        child_table = parent_table.reshape(-1, 1)
        # Add the start index to the offsets to get the actual indices ... relying on broadcasting here ...
        start_node = self.tree['start_child'][parent_table, parent_node].reshape(-1, 1)
        node_offset = torch.arange(self.num_child, dtype=torch.long, device=self.device).reshape(1, -1)
        child_node = start_node + node_offset
        return child_table, child_node, node_offset

    def best_child(self, parent_table: torch.Tensor, parent_node: torch.Tensor) \
            -> torch.Tensor:

        child_table, child_node, node_offset = self.find_children(parent_table, parent_node)
        # Let us calculate the ucb for all children ...
        parent_player = self.tree['player'][parent_table, parent_node].reshape(-1, 1)
        parent_count = self.tree['count'][parent_table, parent_node].reshape(-1, 1)
        child_count = self.tree['count'][child_table, child_node]
        child_q = self.tree['value'][child_table, child_node]
        child_prior = self.tree['prior'][child_table, child_node]
        ucb = (parent_player * child_q +
               2.0 * child_prior * torch.sqrt(parent_count + 1) / (child_count + 1))
        best_offset = torch.argmax(ucb, dim=1)
        best_child_node = self.tree['start_child'][parent_table, parent_node] + best_offset

        return best_child_node

    @profile
    def search_for_leaf(self):
        # I am also collecting table and node indices for efficient propagation ...
        # DONE: construct states for the leaf nodes ...
        all_table = torch.arange(self.num_table, dtype=torch.long, device=self.device)
        leaf_node = torch.zeros(self.num_table, dtype=torch.long, device=self.device)
        player_swap = torch.ones(self.num_table, dtype=torch.int32, device=self.device)
        all_moves = torch.zeros((self.num_table, self.action_size), dtype=torch.int32, device=self.device)
        table = all_table[~self.tree['is_leaf'][all_table, leaf_node]]

        while table.numel() > 0:
            # print("still searching ...")
            node = leaf_node[table]
            child_node = self.best_child(table, node)

            action = self.tree['action'][table, child_node]
            all_moves[table, action] += self.tree['player'][table, node]
            player_swap[table] = -1 * player_swap[table]
            leaf_node[table] = child_node
            table = table[~self.tree['is_leaf'][table, child_node]]

        leaf_player = self.current_player * player_swap
        leaf_state = self.current_state + all_moves

        return leaf_node, leaf_player, leaf_state

    @profile
    def expand(self, leaf_node, leaf_player, action, prob, is_terminal):

        all_table = torch.arange(self.num_table, dtype=torch.long, device=self.device)

        self.tree['player'][all_table, leaf_node] = leaf_player
        self.tree['is_terminal'][all_table, leaf_node] = is_terminal

        # Collect table indices where is_terminal = False, we will expand these ...
        # First deal with the parent leaf nodes ...

        parent_table = torch.masked_select(all_table, ~is_terminal)
        parent_node = leaf_node[parent_table]
        self.tree['is_leaf'][parent_table, parent_node] = False
        self.tree['start_child'][parent_table, parent_node] = self.tree['next_node'][parent_table]

        self.tree['is_leaf'][parent_table, parent_node] = False

        self.tree['start_child'][parent_table, parent_node] = self.tree['next_node'][parent_table]

        # Now deal with the children nodes ...
        child_table, child_node, node_offset = self.find_children(parent_table, parent_node)
        self.tree['parent'][child_table, child_node] = parent_node.reshape(-1, 1)
        self.tree['action'][child_table, child_node] = action[child_table, node_offset]
        self.tree['prior'][child_table, child_node] = prob[child_table, node_offset]

        self.tree['next_node'][parent_table] += self.num_child

        return 0  # space-holder ...

    @profile
    def back_propagate(self, leaf_node, value):
        table = torch.arange(self.num_table, dtype=torch.long, device=self.device)
        node = leaf_node[table]

        while table.numel() > 0:
            # back-propagate ...
            self.tree['count'][table, node] += 1
            self.tree['value_sum'][table, node] += value[table]
            self.tree['value'][table, node] = (self.tree['value_sum'][table, node] /
                                               self.tree['count'][table, node])
            node = self.tree['parent'][table, node]
            table = table[node >= 0]
            node = node[node >= 0]

        return 0

    @profile
    def run_MC_path(self):
        leaf_node, leaf_player, leaf_state = self.search_for_leaf()
        action, prob, value, is_terminal = self.evaluate(leaf_player, leaf_state)
        self.expand(leaf_node, leaf_player, action, prob, is_terminal)
        self.back_propagate(leaf_node, value)

        return 0

    @profile
    def analyze(self, player, state, inv_temp):

        self.reset(player, state)
        for _ in range(self.num_MC):
            self.run_MC_path()

        value = self.tree['value'][:, 0]
        # Calculate probabilities for root child actions ...
        all_table = torch.arange(self.num_table, dtype=torch.long, device=self.device)
        root_node = torch.zeros(self.num_table, dtype=torch.long, device=self.device)
        child_table, child_node, node_offset = self.find_children(all_table, root_node)
        action = self.tree['action'][child_table, child_node]
        count = self.tree['count'][child_table, child_node]
        prob = count / torch.sum(count, dim=1, keepdim=True)
        # Sharpen up probabilities with inv_temp ...
        prob = prob ** (inv_temp.reshape(-1, 1))
        probability = prob / torch.sum(prob, dim=1, keepdim=True)

        return action, probability, value

    def select_move(self, action, probability):
        move = torch.zeros(self.num_table, dtype=torch.long, device=self.device)
        # Loop through each table to select an action based on probabilities
        for i_table in range(self.num_table):
            move[i_table] = action[i_table][torch.multinomial(probability[i_table], 1)]

        return move

    def make_move(self, move):

        new_state = self.current_state.detach().clone()
        all_table = torch.arange(self.num_table, dtype=torch.long, device=self.device)
        new_state[all_table, move[all_table]] = self.current_player
        new_player = -1 * self.current_player

        return new_player, new_state

# class PathManager:
#     def __init__(self, max_size):
#         self.max_size = max_size
#         self.current_size = 0
#         self.path_table = np.zeros(max_size, dtype=np.int32)
#         self.path_node = np.zeros(max_size, dtype=np.int32)
#
#     def append(self, new_table, new_node):
#         new_length = len(new_table)  # should be the same as len(new_node)
#         self.path_table[self.current_size:self.current_size + new_length] = np.copy(new_table)
#         self.path_node[self.current_size:self.current_size + new_length] = np.copy(new_node)
#         self.current_size += new_length
#         # print(self.current_size, " / ", self.max_size)
#
#     def take(self):
#         return self.path_table[:self.current_size], self.path_node[:self.current_size]
#
#     def empty(self):
#         self.current_size = 0
