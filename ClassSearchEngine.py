import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from typing import Tuple
from ClassBufferManager import BufferManager
from ClassSearchTree import SearchTree
# from ClassModel import evaluate01
from line_profiler_pycharm import profile
# from helper_functions import unique, duplicate_indices


class SearchEngine:
    def __init__(self, args: dict, game, terminal_check, model):
        self.args = args
        self.game = game
        self.terminal_check = terminal_check
        self.model = model

        # self.evaluator = evaluator

        # Set up parameters
        self.num_table = args.get('num_table')
        self.num_child = args.get('num_child')
        self.num_MC = args.get('num_MC')
        self.num_agent = args.get('num_agent')
        self.num_node = (self.num_MC + 100) * self.num_child
        self.action_size = game.action_size
        self.max_depth = game.action_size + 1
        self.CUDA_device = args.get('CUDA_device')
        # Set up buffer manager
        self.buffer_mgr = BufferManager(leaf_buffer_size=args.get('leaf_buffer_size'),
                                        child_buffer_size=args.get('leaf_buffer_size') * self.num_child * 10,
                                        min_batch_size=args.get('eval_batch_size'),
                                        action_size=self.action_size,
                                        max_depth=self.max_depth)
        # Set up search tree
        self.tree = SearchTree(num_table=self.num_table,
                               num_node=self.num_node,
                               num_child=self.num_child)
        # Set up root attributes
        self.root_player = torch.zeros(self.num_table, dtype=torch.int32)
        self.root_position = torch.zeros((self.num_table, self.action_size), dtype=torch.int32)
        # Set up search all_agents attributes
        self.all_agents = torch.arange(self.num_agent)
        self.active = torch.zeros(self.num_agent, dtype=torch.bool)
        self.table = torch.zeros(self.num_agent, dtype=torch.long)
        self.node = torch.zeros(self.num_agent, dtype=torch.long)
        self.depth = torch.zeros(self.num_agent, dtype=torch.long)
        self.player = torch.zeros(self.num_agent, dtype=torch.int32)
        self.position = torch.zeros((self.num_agent, self.action_size), dtype=torch.int32)
        self.path = torch.zeros((self.num_agent, self.max_depth), dtype=torch.long)

    def reset(self, root_player, root_position):
        self.buffer_mgr.reset()
        self.tree.reset()
        self.root_player[:] = root_player
        self.root_position[:, :] = root_position
        self.active[:] = False
        return

    def get_active_indices(self):
        return (self.all_agents[self.active],
                self.table[self.active],
                self.node[self.active])

    def save_leaves(self):
        agent, table, node = self.get_active_indices()
        leaf_agent = agent[self.tree.is_leaf[table, node]]
        if leaf_agent.shape[0] > 0:
            self.buffer_mgr.add_leaves(self.table[leaf_agent],
                                       self.node[leaf_agent],
                                       self.player[leaf_agent],
                                       self.position[leaf_agent, :],
                                       self.path[leaf_agent, :],
                                       self.depth[leaf_agent])
            # free up agents that have found leaves
            self.active[leaf_agent] = False
        return

    def activate_agents(self):
        passive_agents = self.all_agents[~self.active]
        num_new = min(self.num_table, passive_agents.shape[0])
        # TODO: This is just a proxy for now ...
        new_tables = torch.arange(num_new)
        new_agents = passive_agents[: num_new]
        self.active[new_agents] = True
        self.table[new_agents] = new_tables
        self.node[new_agents] = 1
        self.depth[new_agents] = 0
        self.player[new_agents] = self.root_player[new_tables]
        self.position[new_agents, :] = self.root_position[new_tables, :]
        self.path[new_agents, 0] = 1
        return

    def save_children(self, table, parent_node, child_node, parent_player):
        # Step 1: Replicate table, parent_node and parent_player to 2d shape
        table_expanded = table.unsqueeze(1).repeat(1, self.num_child)
        parent_node_expanded = parent_node.unsqueeze(1).repeat(1, self.num_child)
        parent_player_expanded = parent_player.unsqueeze(1).repeat(1, self.num_child)
        # Step 2: Flatten all tensors
        tables = table_expanded.flatten()
        parents = parent_node_expanded.flatten()
        children = child_node.flatten()
        players = parent_player_expanded.flatten()
        self.buffer_mgr.add_children(tables, parents, children, players)
        return

    def update_agents(self):
        if not self.active.any():
            # print('No active agents ...')
            return
        # All agents hold nodes already expanded at this point ...
        agent, table, parent_node = self.get_active_indices()
        child_node = self.tree.get_children(table, parent_node)
        # save all the child information to the child buffer, we will use this info to update ucb ...
        self.save_children(table, parent_node, child_node, self.player[agent])
        # find the best child node based on current ucb ...
        ucb_tensor = self.tree.ucb[table.view(-1, 1), child_node]
        best_idx = torch.argmax(ucb_tensor, dim=1)
        new_node = child_node[torch.arange(best_idx.shape[0]), best_idx]
        # lower ucb for best child to facilitate branching for consecutive paths ...
        self.tree.ucb[table, new_node] -= 0.1

        self.node[agent] = new_node
        self.depth[agent] += 1
        # TODO: Make this general, based on the Amoeba class ...
        new_action = self.tree.action[table, new_node]
        self.position[agent, new_action] = self.player[agent]
        self.player[agent] *= -1
        self.path[agent, self.depth[agent]] = new_node
        return

    def collect_leaves(self):
        while not self.buffer_mgr.batch_full:
            self.activate_agents()
            self.save_leaves()
            self.update_agents()
        # TODO: post-process leaf buffer ...
        return

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
        terminal_mask = plus_mask | minus_mask | draw_mask

        return terminal_mask, logit, value

    @profile
    def analyze(self, player, position):

        self.reset(player, position)
        self.collect_leaves()
        return

        # self.root_expand()
        # for i_MC in range(self.num_MC):
        #     # print('i_MC = ', i_MC)
        #     # self.i_leaf = 0
        #     # self.search_leaves(0, 1, player, position)
        #     self.search_leaves(0, self.num_leaf, 0, 1, player, position)
        #     self.update_tree()
        #     self.back_propagate()
        #
        # root_idx = 1
        # root_value = self.tree['value'][root_idx]
        # start_idx = self.tree['start_child_idx'][root_idx].item()
        # end_idx = start_idx + self.tree['num_child'][root_idx].item()
        # root_children_idx = torch.arange(start_idx, end_idx)
        # children_actions = self.tree['action'][root_children_idx]
        # action_count = torch.zeros(self.action_size, dtype=torch.int32)
        # action_count[children_actions] = self.tree['count'][root_children_idx]
        # action_weight = action_count.to(dtype=torch.float32)
        # action_policy = action_weight / torch.sum(action_weight)
        #
        # # action_value = torch.zeros(self.action_size, dtype=torch.float32)
        # # action_value[children_actions] = self.tree['value'][root_children_idx]
        # # self.game.print_board(position)
        # # print('move count: \n', action_count.view(self.game.board_size, -1))
        # # print('move weight: \n', action_weight.view(self.game.board_size, -1))
        # print('move logit: \n', action_policy.view(self.game.board_size, -1))
        # return action_policy, root_value

    # @profile
    # def search_one_step(self):
    #     # save leaves if any have been found (and deactivate agents that found leaves) ...
    #     self.save_leaves()
    #     # activate new agents to start new paths ...
    #     self.activate_agents()
    #     # get active indexes ...
    #     all_agents, table, node = self.get_active_indices()
    #     # find the best child node (and save all children involved) ...
    #     best_node = self.explore_children(table, node)
    #     # update state of the search agents
    #     self.update_agents(all_agents, table, best_node)

    # def root_expand(self):
    #     # evaluate
    #     terminal_mask, logit, value = self.evaluate(self.root_player, self.root_position)
    #     # expand
    #     table = torch.arange(self.num_table)
    #     node = self.next_node[table]
    #     self.expand(table, node, self.root_player, self.root_position, logit, value)
    #
    #     return

    # def explore_children(self, all_agents, parent_table, parent_node):
    #     # get the child node indexes on the parent tables ...
    #     child_table, child_node = self.tree.get_children(parent_table, parent_node)
    #     #
    #     # child_table = parent_table.unsqueeze(1).repeat(1, self.num_child)
    #     # # add the start index to the offsets to get the actual indices ... relying on broadcasting here ...
    #     # start_node = self.start_child[parent_table, parent_node].reshape(-1, 1)
    #     # node_offset = torch.arange(self.num_child, dtype=torch.long).reshape(1, -1)
    #     # child_node = start_node + node_offset
    #     # save all the child information to the child buffer, we will use this info to update ucb ...
    #     self.save_children(child_table, parent_node, child_node, self.player[parent_table, parent_node])
    #     # find the best child node based on current ucb ...
    #     best_idx = torch.argmax(self.ucb[child_table, child_node], dim=1)
    #     best_child_node = child_node[torch.arange(best_idx.shape[0]), best_idx]
    #     # lower ucb for best child to facilitate branching for consecutive paths ...
    #     self.ucb[parent_table, best_child_node] -= 0.1
    #
    #     return best_child_node
