import torch
# import torch.nn as nn
import torch.nn.functional as F
# from AmoebaClass import Amoeba
# from typing import Tuple
from ClassEvaluator import EvaluationBuffer
# from ClassModel import evaluate01
from line_profiler_pycharm import profile


class AlphaZero:
    def __init__(self, args: dict, game, terminal_check, model):
        self.args = args
        self.game = game
        self.terminal_check = terminal_check
        self.model = model
        # Often used parameters
        self.num_table = args.get('num_table')
        self.num_child = args.get('num_child')
        self.num_MC = args.get('num_MC')
        self.num_moves = args.get('num_moves')
        self.num_node = (self.num_MC + 100) * self.num_child
        self.action_size = game.action_size
        self.CUDA_device = args.get('CUDA_device')
        # Circular buffer to collect and batch leaf positions
        self.eval_buffer = EvaluationBuffer(args)
        # Keep track of tables under search, current node for search, search count
        self.search_table_idx = torch.arange(self.num_table, dtype=torch.long)
        self.search_node_idx = torch.zeros(self.num_table, dtype=torch.long)
        self.search_count = torch.zeros(self.num_table, dtype=torch.int32)
        self.move_count = torch.zeros(self.num_table, dtype=torch.int32)
        # Keep track of tables under evaluation, and evaluation status
        self.evaluated_table_idx = torch.arange(args.get('num_table'), dtype=torch.long)
        self.evaluation_is_on = False
        # Keep track of root player, node player, and current position, node position on all tables
        self.root_player = torch.ones(self.num_table, dtype=torch.int32)
        self.root_position = torch.zeros((self.num_table, self.action_size), dtype=torch.int32)
        self.node_player = torch.ones(self.num_table, dtype=torch.int32)
        self.node_position = torch.zeros((self.num_table, self.action_size), dtype=torch.int32)

        # Set up search tree
        self.tree = {
            'player': torch.zeros((self.num_table, self.num_node), dtype=torch.int32),
            'is_leaf': torch.ones((self.num_table, self.num_node), dtype=torch.bool),
            'is_terminal': torch.zeros((self.num_table, self.num_node), dtype=torch.bool),
            'count': torch.zeros((self.num_table, self.num_node), dtype=torch.int32),
            'value_sum': torch.zeros((self.num_table, self.num_node), dtype=torch.float32),
            'value': torch.zeros((self.num_table, self.num_node), dtype=torch.float32),
            'parent_idx': -torch.ones((self.num_table, self.num_node), dtype=torch.long),
            'start_child_idx': -torch.ones((self.num_table, self.num_node), dtype=torch.long),
            'action': torch.zeros((self.num_table, self.num_node), dtype=torch.long),
            'prior': torch.zeros((self.num_table, self.num_node), dtype=torch.float32),
            'next_node_idx': torch.ones(self.num_table, dtype=torch.long)
        }

    @profile
    def evaluate(self):

        state_data, self.evaluated_table_idx = self.eval_buffer.get_states()
        state_CUDA = state_data.to(self.CUDA_device, non_blocking=True)

        terminal_signal = self.terminal_check(state_CUDA)
        with torch.no_grad():
            logit, value = self.model(state_CUDA)
            logit = logit - torch.abs(state_CUDA) * 999999.9
            policy = F.softmax(logit, dim=1)

        result_CUDA = torch.cat([policy, value, terminal_signal], dim=1)
        return result_CUDA

    @profile
    def find_children(self, parent_table, parent_node):
        child_table = parent_table.unsqueeze(1).repeat(1, self.num_child)
        # child_table = parent_table.reshape(-1, 1)
        # Add the start index to the offsets to get the actual indices ... relying on broadcasting here ...
        start_node = self.tree['start_child_idx'][parent_table, parent_node].reshape(-1, 1)
        node_offset = torch.arange(self.num_child, dtype=torch.long).reshape(1, -1)
        child_node = start_node + node_offset
        return child_table, child_node, node_offset

    def best_child(self):
        parent_table = self.search_table_idx
        parent_node = self.search_node_idx[parent_table]
        child_table, child_node, node_offset = self.find_children(parent_table, parent_node)
        # Let us calculate the ucb for all children ...
        parent_player = self.node_player[parent_table].view(-1, 1)
        parent_count = self.tree['count'][parent_table, parent_node].reshape(-1, 1)
        child_count = self.tree['count'][child_table, child_node]
        child_q = self.tree['value'][child_table, child_node]
        child_prior = self.tree['prior'][child_table, child_node]
        ucb = (parent_player * child_q +
               2.0 * child_prior * torch.sqrt(parent_count + 1) / (child_count + 1))
        best_offset = torch.argmax(ucb, dim=1)
        best_child_node = self.tree['start_child_idx'][parent_table, parent_node] + best_offset

        return best_child_node

    @profile
    def search_tree(self):
        # collect leaf positions + index metadata
        for i_search in range(4):
            # One step of search for leaf nodes ...

            leaf_mask = self.tree['is_leaf'][self.search_table_idx, self.search_node_idx[self.search_table_idx]]
            leaf_table_idx = self.search_table_idx[leaf_mask]
            # Prepare leaf states for evaluation and add them to evaluation buffer
            num_leaf = leaf_table_idx.shape[0]
            if num_leaf > 0:
                new_state = self.node_player[leaf_table_idx].reshape(-1, 1) * self.node_position[leaf_table_idx]
                self.eval_buffer.add_states(new_state, leaf_table_idx)

            # Continue search on other tables
            self.search_table_idx = self.search_table_idx[~leaf_mask]
            num_search = self.search_table_idx.shape[0]
            if num_search > 0:
                # Find best child
                best_node_idx = self.best_child()

                # best_node_idx = self.search_node_idx[self.search_table_idx] + 1  # just for testing ...

                self.search_node_idx[self.search_table_idx] = best_node_idx
                action = self.tree['action'][self.search_table_idx, best_node_idx]
                self.node_position[self.search_table_idx, action] = self.node_player[self.search_table_idx]
                self.node_player[self.search_table_idx] *= -1

            else:  # if num_search = 0:
                break
            if self.eval_buffer.num_data > self.eval_buffer.batch_size:
                break

    @profile
    def format_act_prob(self, policy):
        prob, act = torch.topk(policy, k=self.num_child, dim=1)
        # Change very low probabilities to 0.0 ...
        mask = prob < 0.0001
        prob[mask] = 0.0
        # Normalize only the rows where the maximum prob is greater than 0.0001 ...
        max_prob, _ = prob.max(dim=1)
        max_mask = max_prob > 0.0001
        prob = torch.where(
            max_mask.unsqueeze(1),
            F.normalize(prob, p=1, dim=1),
            prob
        )
        # Make very low probability actions prohibitive ...
        prob[mask] = -1000000.0
        # and the actions invalid ... This should produce errors
        act[mask] = 1000

        return act, prob

    @profile
    def back_propagate(self, eval_value):
        table = self.evaluated_table_idx
        node = self.search_node_idx[table]
        new_value = eval_value

        while table.numel() > 0:
            # back-propagate ...
            self.tree['count'][table, node] += 1
            self.tree['value_sum'][table, node] += new_value
            self.tree['value'][table, node] = (self.tree['value_sum'][table, node] /
                                               self.tree['count'][table, node])
            node = self.tree['parent_idx'][table, node]
            table = table[node >= 0]
            new_value = new_value[node >= 0]
            node = node[node >= 0]

    @profile
    def update_tree(self, result):

        # Interpret result ...
        value = result[:, -4]
        dir_max = result[:, -3]
        dir_min = result[:, -2]
        sum_abs = result[:, -1]
        plus_mask = (dir_max > 4.9)
        minus_mask = (dir_min < -4.9)
        draw_mask = (sum_abs > self.action_size)
        value[draw_mask] = 0.0
        value[minus_mask] = -1.0
        value[plus_mask] = 1.0
        value *= self.node_player[self.evaluated_table_idx]
        # ***** EXPAND *****
        terminal_mask = plus_mask | minus_mask | draw_mask
        expand_mask = ~terminal_mask

        if torch.any(expand_mask):

            expand_table_idx = self.evaluated_table_idx[expand_mask]
            expand_node_idx = self.search_node_idx[expand_table_idx]
            # self.tree['is_terminal'][all_table, leaf_node] = is_terminal    DO WE NEED THIS???
            # self.tree['player'][all_table, leaf_node] = leaf_player    DO WE NEED THIS???

            # First deal with the parent leaf nodes ...
            self.tree['is_leaf'][expand_table_idx, expand_node_idx] = False
            self.tree['start_child_idx'][expand_table_idx, expand_node_idx] = (
                self.tree)['next_node_idx'][expand_table_idx]

            # Now deal with the children nodes ...
            child_table_idx, child_node_idx, node_offset = self.find_children(expand_table_idx, expand_node_idx)
            self.tree['parent_idx'][child_table_idx, child_node_idx] = expand_node_idx.reshape(-1, 1)

            policy = result[expand_mask, : -4]
            action, prob = self.format_act_prob(policy)
            self.tree['action'][child_table_idx, child_node_idx] = action
            self.tree['prior'][child_table_idx, child_node_idx] = prob

            self.tree['next_node_idx'][expand_table_idx] += self.num_child

        # ***** BACK-PROPAGATE ******
        self.back_propagate(value)

        # Update search count for tables just evaluated
        self.search_count[self.evaluated_table_idx] += 1
        self.move_count[self.search_count >= self.num_MC] += 1
        self.search_count[self.search_count >= self.num_MC] = 0

        # Add back the evaluated tables for leaf search
        self.search_table_idx = torch.cat((self.search_table_idx, self.evaluated_table_idx), dim=0)

    @profile
    def run(self):
        i_move = 1
        while True:
            # Coarse-grained loop starts here
            for i_step in range(100):
                # Fine-grained loop starts here
                if self.eval_buffer.num_data > self.eval_buffer.batch_size:
                    self.evaluation_is_on = True
                    result_CUDA = self.evaluate()
                self.search_tree()

                if self.evaluation_is_on:
                    self.evaluation_is_on = False
                    result = result_CUDA.to('cpu', non_blocking=False)
                    self.update_tree(result)

            # Coarse-grained operations start here
            mean_move_count = torch.mean(self.move_count.float()).item()

            max_next_node_idx = torch.max(self.tree['next_node_idx'])
            print('max_next_node_idx', max_next_node_idx)

            if mean_move_count > 0.99 * i_move:
                print('mean_move_count = ', i_move)
                i_move += 1
            if i_move > self.num_moves:
                break

#     def reset(self, current_player: torch.Tensor, current_state: torch.Tensor):
#         self.current_player = current_player.detach().clone()
#         self.current_state = current_state.detach().clone()
#
#         self.tree['player'][:, :] = 0
#         self.tree['is_leaf'][:, :] = True
#         self.tree['is_terminal'][:, :] = False
#         self.tree['count'][:, :] = 0
#         self.tree['value_sum'][:, :] = 0.0
#         self.tree['value'][:, :] = 0.0
#         self.tree['parent'][:, :] = -1
#         self.tree['start_child'][:, :] = -1
#         self.tree['action'][:, :] = 0
#         self.tree['prior'][:, :] = 0.0
#         self.tree['next_node'][:] = 1
#
#     def find_children(self, parent_table: torch.Tensor, parent_node: torch.Tensor) \
#             -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#         child_table = parent_table.reshape(-1, 1)
#         # Add the start index to the offsets to get the actual indices ... relying on broadcasting here ...
#         start_node = self.tree['start_child'][parent_table, parent_node].reshape(-1, 1)
#         node_offset = torch.arange(self.num_child, dtype=torch.long, device=self.device).reshape(1, -1)
#         child_node = start_node + node_offset
#         return child_table, child_node, node_offset
#
#     def best_child(self, parent_table: torch.Tensor, parent_node: torch.Tensor) \
#             -> torch.Tensor:
#
#         child_table, child_node, node_offset = self.find_children(parent_table, parent_node)
#         # Let us calculate the ucb for all children ...
#         parent_player = self.tree['player'][parent_table, parent_node].reshape(-1, 1)
#         parent_count = self.tree['count'][parent_table, parent_node].reshape(-1, 1)
#         child_count = self.tree['count'][child_table, child_node]
#         child_q = self.tree['value'][child_table, child_node]
#         child_prior = self.tree['prior'][child_table, child_node]
#         ucb = (parent_player * child_q +
#                2.0 * child_prior * torch.sqrt(parent_count + 1) / (child_count + 1))
#         best_offset = torch.argmax(ucb, dim=1)
#         best_child_node = self.tree['start_child'][parent_table, parent_node] + best_offset
#
#         return best_child_node
#
#     @profile
#     def search_for_leaf(self):
#         # I am also collecting table and node indices for efficient propagation ...
#         # DONE: construct states for the leaf nodes ...
#         all_table = torch.arange(self.num_table, dtype=torch.long, device=self.device)
#         leaf_node = torch.zeros(self.num_table, dtype=torch.long, device=self.device)
#         player_swap = torch.ones(self.num_table, dtype=torch.int32, device=self.device)
#         all_moves = torch.zeros((self.num_table, self.action_size), dtype=torch.int32, device=self.device)
#         table = all_table[~self.tree['is_leaf'][all_table, leaf_node]]
#
#         while table.numel() > 0:
#             # print("still searching ...")
#             node = leaf_node[table]
#             child_node = self.best_child(table, node)
#
#             action = self.tree['action'][table, child_node]
#             all_moves[table, action] += self.tree['player'][table, node]
#             player_swap[table] = -1 * player_swap[table]
#             leaf_node[table] = child_node
#             table = table[~self.tree['is_leaf'][table, child_node]]
#
#         leaf_player = self.current_player * player_swap
#         leaf_state = self.current_state + all_moves
#
#         return leaf_node, leaf_player, leaf_state
#
#     @profile
#     def run_MC_path(self):
#         leaf_node, leaf_player, leaf_state = self.search_for_leaf()
#         action, prob, value, is_terminal = self.evaluate(leaf_player, leaf_state)
#         self.expand(leaf_node, leaf_player, action, prob, is_terminal)
#         self.back_propagate(leaf_node, value)
#
#         return 0
#
#     @profile
#     def analyze(self, player, state, inv_temp):
#
#         self.reset(player, state)
#         for _ in range(self.num_MC):
#             self.run_MC_path()
#
#         value = self.tree['value'][:, 0]
#         # Calculate probabilities for root child actions ...
#         all_table = torch.arange(self.num_table, dtype=torch.long, device=self.device)
#         root_node = torch.zeros(self.num_table, dtype=torch.long, device=self.device)
#         child_table, child_node, node_offset = self.find_children(all_table, root_node)
#         action = self.tree['action'][child_table, child_node]
#         count = self.tree['count'][child_table, child_node]
#         prob = count / torch.sum(count, dim=1, keepdim=True)
#         # Sharpen up probabilities with inv_temp ...
#         prob = prob ** (inv_temp.reshape(-1, 1))
#         probability = prob / torch.sum(prob, dim=1, keepdim=True)
#
#         return action, probability, value
#
#     def select_move(self, action, probability):
#         move = torch.zeros(self.num_table, dtype=torch.long, device=self.device)
#         # Loop through each table to select an action based on probabilities
#         for i_table in range(self.num_table):
#             move[i_table] = action[i_table][torch.multinomial(probability[i_table], 1)]
#
#         return move
#
#     def make_move(self, move):
#
#         new_state = self.current_state.detach().clone()
#         all_table = torch.arange(self.num_table, dtype=torch.long, device=self.device)
#         new_state[all_table, move[all_table]] = self.current_player
#         new_player = -1 * self.current_player
#
#         return new_player, new_state
#
# # class PathManager:
# #     def __init__(self, max_size):
# #         self.max_size = max_size
# #         self.current_size = 0
# #         self.path_table = np.zeros(max_size, dtype=np.int32)
# #         self.path_node = np.zeros(max_size, dtype=np.int32)
# #
# #     def append(self, new_table, new_node):
# #         new_length = len(new_table)  # should be the same as len(new_node)
# #         self.path_table[self.current_size:self.current_size + new_length] = np.copy(new_table)
# #         self.path_node[self.current_size:self.current_size + new_length] = np.copy(new_node)
# #         self.current_size += new_length
# #         # print(self.current_size, " / ", self.max_size)
# #
# #     def take(self):
# #         return self.path_table[:self.current_size], self.path_node[:self.current_size]
# #
# #     def empty(self):
# #         self.current_size = 0
