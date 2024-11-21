import torch
from line_profiler_pycharm import profile


class LeafBuffer:
    def __init__(self, game, buffer_size, min_batch_size):

        self.buffer_size = buffer_size
        self.min_batch_size = min_batch_size
        self.action_size = game.action_size
        self.max_depth = game.action_size + 1

        self.table = torch.zeros(self.buffer_size, dtype=torch.long)
        self.node = torch.zeros(self.buffer_size, dtype=torch.long)
        self.player = torch.zeros(self.buffer_size, dtype=torch.int32)
        self.position = torch.zeros((self.buffer_size, self.action_size), dtype=torch.int32)
        self.path = torch.zeros((self.buffer_size, self.max_depth), dtype=torch.long)
        self.depth = torch.zeros(self.buffer_size, dtype=torch.long)

        self.policy = torch.zeros((self.buffer_size, self.action_size), dtype=torch.float32)
        self.value = torch.zeros(self.buffer_size, dtype=torch.float32)
        self.is_terminal = torch.zeros(self.buffer_size, dtype=torch.bool)

        self.next_idx = 0
        self.batch_size = 0
        # self.batch_full = False

    def add_leaves(self, tables, nodes,  players, positions, paths, depths):
        end_idx = self.next_idx + tables.shape[0]
        self.table[self.next_idx: end_idx] = tables
        self.node[self.next_idx: end_idx] = nodes
        self.player[self.next_idx: end_idx] = players
        self.position[self.next_idx: end_idx, :] = positions
        self.path[self.next_idx: end_idx, :] = paths
        self.depth[self.next_idx: end_idx, :] = depths

        self.next_idx = end_idx
        return

    def get_states(self):
        states = self.player[: self.next_idx] * self.position[: self.next_idx]
        self.batch_size = self.next_idx
        # self.batch_full = True
        return states

    def add_eval_results(self, policies, values, are_terminal):
        self.policy[: self.batch_size, :] = policies
        self.value[: self.batch_size] = values
        self.is_terminal[: self.batch_size] = are_terminal
        return

    def get_eval_data(self):
        return (self.table[: self.batch_size],
                self.node[: self.batch_size],
                self.player[: self.batch_size],
                self.position[: self.batch_size],
                self.policy[: self.batch_size, :],
                self.value[: self.batch_size],
                self.is_terminal[: self.batch_size])

    def get_path_data(self):
        # self.table = torch.zeros(self.buffer_size, dtype=torch.long)
        # self.path = torch.zeros((self.buffer_size, self.max_depth), dtype=torch.long)
        # self.depth = torch.zeros(self.buffer_size, dtype=torch.long)
        # self.value = torch.zeros(self.buffer_size, dtype=torch.float32)

        # truncate path tensors based on max depth ...
        maximal_depth = torch.max(self.depth)
        paths = self.path[:, :maximal_depth]

        return


    #
    # def get_results(self):
    #     tables = self.table[: self.batch_size]
    #     nodes = self.node[: self.batch_size]
    #     paths = self.path[: self.batch_size, :]
    #
    #     players = self.player[: self.batch_size]
    #     positions = self.position[: self.batch_size, :]
    #
    #     policies = self.policy[: self.batch_size, :]
    #     values = self.value[: self.batch_size]
    #     are_terminal = self.is_terminal[: self.batch_size]
    #
    #     self.batch_full = False
    #     # Now move leaf data to beginning of buffer
    #     size_to_move = self.next_idx - self.batch_size
    #     if size_to_move > 0:
    #         self.table[:size_to_move] = self.table[self.batch_size:self.next_idx].clone()
    #         self.node[:size_to_move] = self.node[self.batch_size:self.next_idx].clone()
    #         self.path[:size_to_move, :] = self.path[self.batch_size:self.next_idx, :].clone()
    #         self.player[:size_to_move] = self.player[self.batch_size:self.next_idx].clone()
    #         self.position[:size_to_move, :] = self.position[self.batch_size:self.next_idx, :].clone()
    #
    #     self.next_idx = size_to_move
    #
    #     return tables, nodes, paths, players, positions, policies, values, are_terminal
