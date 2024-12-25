import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from typing import Tuple
# from ClassBufferManager import BufferManager
# from line_profiler_pycharm import profile


class AgentManager:
    # def __init__(self, args: dict, game: Amoeba, root_player, root_position):
    def __init__(self, num_agent, action_size, max_depth, buffer_mgr):

        self.num_agent = num_agent
        self.action_size = action_size
        self.max_depth = max_depth
        self.buffer_mgr = buffer_mgr
        self.all_idx = torch.arange(self.num_agent)

        self.active = torch.zeros(self.num_agent, dtype=torch.bool)

        self.table = torch.zeros(self.num_agent, dtype=torch.long)
        self.node = torch.zeros(self.num_agent, dtype=torch.long)
        self.depth = torch.zeros(self.num_agent, dtype=torch.long)
        self.player = torch.zeros(self.num_agent, dtype=torch.int32)
        self.position = torch.zeros((self.num_agent, self.action_size), dtype=torch.int32)
        self.path = torch.zeros((self.num_agent, self.max_depth), dtype=torch.long)

    def reset(self):
        self.active[:] = False
        return

    def get_indices(self):
        return (self.table[self.active],
                self.node[self.active])

    def update(self, new_node, action, is_leaf):
        # self.table = torch.zeros(self.num_agent, dtype=torch.long)
        self.node[self.active] = new_node
        self.depth[self.active] += 1
        self.position[self.active, action] = self.player[self.active]
        self.player[self.active] *= -1
        self.path[self.active, self.depth[self.active]] = new_node
        if is_leaf.any():
            active_idx = self.all_idx[self.active]
            leaf_idx = active_idx[is_leaf]
            self.save_leaves(leaf_idx)

        return

    def save_leaves(self, leaf_idx):
        self.buffer_mgr.add_leaves(self.table[leaf_idx],
                                   self.node[leaf_idx],
                                   self.player[leaf_idx],
                                   self.position[leaf_idx, :],
                                   self.path[leaf_idx, :],
                                   self.depth[leaf_idx])
        # free up agents that have found leaves
        self.active[leaf_idx] = False
        return

    def activate_agents(self):
        # TODO: This is just a proxy for now ...
        self.active[:] = True
        return
