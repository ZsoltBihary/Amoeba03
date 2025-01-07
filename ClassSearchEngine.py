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
        self.num_node = (self.num_MC + 200) * self.num_child
        self.action_size = game.action_size
        self.max_depth = game.action_size + 1
        self.CUDA_device = args.get('CUDA_device')
        # self.max_num_branch = args.get('num_branch')
        # self.num_branch = 1
        # Set up buffer manager
        self.buffer_mgr = BufferManager(leaf_buffer_size=args.get('leaf_buffer_size'),
                                        child_buffer_size=args.get('leaf_buffer_size') * self.num_child * 2,
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
        # Set up helper attributes
        self.table_order = torch.zeros(self.num_table, dtype=torch.long)
        self.ucb_penalty = 0.1
        # self.branch_penalty = 0.05
        self.av_num_agent = 0.0

    def reset(self, root_player, root_position):
        self.buffer_mgr.reset()
        self.tree.reset()
        self.root_player[:] = root_player
        self.root_position[:, :] = root_position
        self.active[:] = False
        self.table_order = torch.arange(self.num_table)
        self.av_num_agent = 1.0
        # num_legal = torch.sum(position == 0)
        # self.num_branch = min(self.max_num_branch, num_legal.item())
        return

    @profile
    def get_active_indices(self):
        return (self.all_agents[self.active],
                self.table[self.active],
                self.node[self.active])

    @profile
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

    @profile
    def activate_agents(self):
        passive_agents = self.all_agents[~self.active]
        num_new = min(self.num_table, passive_agents.shape[0])
        # TODO: This is just a proxy for now ... max 1 agent / table
        new_tables = self.table_order[: num_new]
        new_agents = passive_agents[: num_new]
        self.active[new_agents] = True
        self.table[new_agents] = new_tables
        self.node[new_agents] = 1

        self.player[new_agents] = self.root_player[new_tables]
        self.position[new_agents, :] = self.root_position[new_tables, :]
        self.path[new_agents, 0] = 1
        self.depth[new_agents] = 1
        return

    @profile
    def save_children(self, table, parent_node, child_node, parent_player, collect):
        # Step 1: Replicate table, parent_node and parent_player to 2d shape
        table_expanded = table.unsqueeze(1).repeat(1, self.num_child)
        parent_node_expanded = parent_node.unsqueeze(1).repeat(1, self.num_child)
        parent_player_expanded = parent_player.unsqueeze(1).repeat(1, self.num_child)
        # Step 2: Flatten all tensors
        tables = table_expanded.flatten()
        parents = parent_node_expanded.flatten()
        children = child_node.flatten()
        players = parent_player_expanded.flatten()
        self.buffer_mgr.add_children(tables, parents, children, players, collect)
        return

    def split_agents(self, agent, table, child_node, depth_max):

        passive_agent = self.all_agents[~self.active]
        old_idx = torch.arange(agent.shape[0])[self.depth[agent] <= depth_max]
        old_agent = agent[old_idx]
        old_table = table[old_idx]
        old_child_node = child_node[old_idx, :]
        if passive_agent.shape[0] >= old_agent.shape[0]:
            new_agent = passive_agent[: old_agent.shape[0]]
            self.active[new_agent] = True
            self.table[new_agent] = old_table

            ucb_tensor = self.tree.ucb[old_table.view(-1, 1), old_child_node]
            best_idx = torch.argmax(ucb_tensor, dim=1)
            new_node2 = old_child_node[torch.arange(best_idx.shape[0]), best_idx]
            # Lower ucb for best child to facilitate branching for consecutive paths ...
            self.tree.ucb[old_table, new_node2] -= self.ucb_penalty

            self.node[new_agent] = new_node2
            # TODO: Make this general, based on the Amoeba class ...
            new_action = self.tree.action[old_table, new_node2]
            self.position[new_agent, :] = self.position[old_agent, :]
            self.position[new_agent, new_action] = self.player[old_agent]
            self.player[new_agent] = -self.player[old_agent]
            self.path[new_agent, :] = self.path[old_agent, :]
            self.path[new_agent, self.depth[old_agent]] = new_node2
            self.depth[new_agent] = self.depth[old_agent] + 1

        return

    @profile
    def update_agents(self):
        if not self.active.any():
            # print('No active agents ...')
            return
        # All agents hold nodes already expanded at this point ...
        agent, table, parent_node = self.get_active_indices()
        n_agent = agent.shape[0]
        self.av_num_agent = self.av_num_agent * 0.9 + n_agent * 0.1
        child_node = self.tree.get_children(table, parent_node)
        # Save all the child information to the child buffer, we will use this info to update ucb ...
        self.save_children(table, parent_node, child_node, self.player[agent], collect=True)
        # Find the best child node based on current ucb ...
        ucb_tensor = self.tree.ucb[table.view(-1, 1), child_node]
        best_idx = torch.argmax(ucb_tensor, dim=1)
        new_node = child_node[torch.arange(best_idx.shape[0]), best_idx]
        # Lower ucb for best child to facilitate branching for consecutive paths ...
        self.tree.ucb[table, new_node] -= self.ucb_penalty

        # self.split_agents(agent, table, child_node, depth_max=2)
        # self.split_agents(agent, table, child_node, depth_max=4)
        # self.split_agents(agent, table, child_node, depth_max=6)
        # self.split_agents(agent, table, child_node, depth_max=10)

        # # Let us do some branching if num_table = 1 ...
        # if self.num_table == 1:
        #
        #     passive_agent = self.all_agents[~self.active]
        #     if passive_agent.shape[0] >= agent.shape[0]:
        #         agent2 = passive_agent[: agent.shape[0]]
        #         self.active[agent2] = True
        #         self.table[agent2] = 0
        #
        #         ucb_tensor = self.tree.ucb[table.view(-1, 1), child_node]
        #         best_idx = torch.argmax(ucb_tensor, dim=1)
        #         new_node2 = child_node[torch.arange(best_idx.shape[0]), best_idx]
        #         # Lower ucb for best child to facilitate branching for consecutive paths ...
        #         self.tree.ucb[table, new_node2] -= self.branch_penalty
        #
        #         self.node[agent2] = new_node2
        #         # TODO: Make this general, based on the Amoeba class ...
        #         new_action = self.tree.action[table, new_node2]
        #         self.position[agent2, :] = self.position[agent, :]
        #         self.position[agent2, new_action] = self.player[agent]
        #         self.player[agent2] = -self.player[agent]
        #         self.path[agent2, :] = self.path[agent, :]
        #         self.path[agent2, self.depth[agent]] = new_node2
        #         self.depth[agent2] = self.depth[agent] + 1
        #
        #         passive_agent = self.all_agents[~self.active]
        #         if passive_agent.shape[0] >= agent.shape[0]:
        #             agent2 = passive_agent[: agent.shape[0]]
        #             self.active[agent2] = True
        #             self.table[agent2] = 0
        #
        #             ucb_tensor = self.tree.ucb[table.view(-1, 1), child_node]
        #             best_idx = torch.argmax(ucb_tensor, dim=1)
        #             new_node2 = child_node[torch.arange(best_idx.shape[0]), best_idx]
        #             # Lower ucb for best child to facilitate branching for consecutive paths ...
        #             self.tree.ucb[table, new_node2] -= self.branch_penalty
        #
        #             self.node[agent2] = new_node2
        #             # TODO: Make this general, based on the Amoeba class ...
        #             new_action = self.tree.action[table, new_node2]
        #             self.position[agent2, :] = self.position[agent, :]
        #             self.position[agent2, new_action] = self.player[agent]
        #             self.player[agent2] = -self.player[agent]
        #             self.path[agent2, :] = self.path[agent, :]
        #             self.path[agent2, self.depth[agent]] = new_node2
        #             self.depth[agent2] = self.depth[agent] + 1
        #
        #             passive_agent = self.all_agents[~self.active]
        #             if passive_agent.shape[0] >= agent.shape[0]:
        #                 agent2 = passive_agent[: agent.shape[0]]
        #                 self.active[agent2] = True
        #                 self.table[agent2] = 0
        #
        #                 ucb_tensor = self.tree.ucb[table.view(-1, 1), child_node]
        #                 best_idx = torch.argmax(ucb_tensor, dim=1)
        #                 new_node2 = child_node[torch.arange(best_idx.shape[0]), best_idx]
        #                 # Lower ucb for best child to facilitate branching for consecutive paths ...
        #                 self.tree.ucb[table, new_node2] -= self.ucb_penalty
        #
        #                 self.node[agent2] = new_node2
        #                 # TODO: Make this general, based on the Amoeba class ...
        #                 new_action = self.tree.action[table, new_node2]
        #                 self.position[agent2, :] = self.position[agent, :]
        #                 self.position[agent2, new_action] = self.player[agent]
        #                 self.player[agent2] = -self.player[agent]
        #                 self.path[agent2, :] = self.path[agent, :]
        #                 self.path[agent2, self.depth[agent]] = new_node2
        #                 self.depth[agent2] = self.depth[agent] + 1
        #             else:
        #                 self.tree.ucb[table, new_node2] += (self.branch_penalty - self.ucb_penalty)
        #
        #         else:
        #             self.tree.ucb[table, new_node2] += (self.branch_penalty - self.ucb_penalty)

        # Update agent attributes ...
        # self.tree.ucb[table, new_node] += (self.branch_penalty - self.ucb_penalty)
        self.node[agent] = new_node
        # TODO: Make this general, based on the Amoeba class ...
        new_action = self.tree.action[table, new_node]
        self.position[agent, new_action] = self.player[agent]
        # TODO: What happens if this move happens to go beyond a terminal position?????
        self.player[agent] *= -1
        self.path[agent, self.depth[agent]] = new_node
        self.depth[agent] += 1
        return

    @profile
    def collect_leaves(self):
        self.table_order[:] = torch.argsort(self.tree.count[:, 1])
        while not self.buffer_mgr.batch_full:
            self.activate_agents()
            self.save_leaves()
            self.update_agents()
            # TODO: Wait a minute ... What if we step beyond a terminal node??????
        # Post-process leaf buffer ...
        self.buffer_mgr.post_process()
        return

    @profile
    def start_evaluation(self):
        self.buffer_mgr.swap_buffers()
        states = self.buffer_mgr.get_states()
        states_CUDA = states.to(device=self.CUDA_device, dtype=torch.float32, non_blocking=True)
        with torch.no_grad():
            term_indicator_CUDA = self.terminal_check(states_CUDA)
            result_CUDA = self.model(states_CUDA)
        return term_indicator_CUDA, result_CUDA

    @profile
    def end_evaluation(self, term_indicator_CUDA, result_CUDA):
        term_indicator = term_indicator_CUDA.to(device='cpu', non_blocking=False)
        logit = result_CUDA[0].to(device='cpu', non_blocking=False)
        value = result_CUDA[1].to(device='cpu', non_blocking=False)
        # term_indicator = term_indicator_CUDA.to(device='cpu', non_blocking=True)
        # logit = result_CUDA[0].to(device='cpu', non_blocking=True)
        # value = result_CUDA[1].to(device='cpu', non_blocking=True)
        # Interpret result ...
        dir_max = term_indicator[:, 0]
        dir_min = term_indicator[:, 1]
        sum_abs = term_indicator[:, 2]
        plus_mask = (dir_max + 0.1 > self.game.win_length)
        minus_mask = (dir_min - 0.1 < -self.game.win_length)
        draw_mask = (sum_abs + 0.1 > self.action_size)
        value[draw_mask] = 0.0
        value[plus_mask] = 1.05
        value[minus_mask] = -1.05
        # value = players * value
        terminal_mask = plus_mask | minus_mask | draw_mask

        self.buffer_mgr.add_eval_results(logit, value, terminal_mask)
        return

    @profile
    def expand_tree(self):
        # ***** These are modified within expand_tree
        #       tree.is_terminal
        #       tree.next_node
        #       tree.is_leaf
        #       tree.start_child
        #       tree.action
        #       tree.prior
        table, node, player, logit, is_term = self.buffer_mgr.get_expand_data()
        self.tree.is_terminal[table, node] = is_term
        # Only expand non-terminal leaves ...
        to_expand = ~is_term
        exp_table, exp_node = table[to_expand], node[to_expand]
        exp_player, exp_logit = player[to_expand], logit[to_expand]
        exp_children = self.tree.expand(table[to_expand], node[to_expand], logit[to_expand])
        # Save leaf_children data to the EVAL children buffer ...
        self.save_children(exp_table, exp_node, exp_children, exp_player, False)
        return

    @profile
    def back_propagate(self):
        # ***** These are modified within propagation
        #       tree.count
        #       tree.value_sum
        #       tree.value
        table, node, value, multi = self.buffer_mgr.get_propagate_data()
        self.tree.back_propagate(table, node, value, multi)
        return

    @profile
    def update_ucb(self):
        # ***** This is modified within ucb update
        #       tree.ucb
        table, parent, child, parent_player = self.buffer_mgr.get_ucb_data()
        self.tree.update_ucb(table, parent, child, parent_player)

        return

    @profile
    def analyze(self, player, position):
        self.reset(player, position)
        # self.collect_leaves()
        while True:
            # In the meantime, collect new leaf information ...
            self.collect_leaves()
            # Send states to CUDA and start evaluation on GPU ...
            term_indicator_CUDA, result_CUDA = self.start_evaluation()
            # Send CUDA results back to CPU, and process results ...
            self.end_evaluation(term_indicator_CUDA, result_CUDA)
            # Update search tree ...
            self.expand_tree()
            # print(self.tree)
            self.back_propagate()
            self.update_ucb()

            min_MC = torch.min(self.tree.count[:, 1])
            # print(min_MC.item())
            # print(round(self.av_num_agent))
            if min_MC > self.num_MC:
                break

        # Formulate output ...
        table = torch.arange(self.num_table)
        root = torch.ones(self.num_table, dtype=torch.long)
        position_value = self.tree.value[table, root]
        root_children = self.tree.get_children(table, root)
        counts = self.tree.count[table.view(-1, 1), root_children]
        actions = self.tree.action[table.view(-1, 1), root_children]
        probs = counts / torch.sum(counts)
        move_policy = torch.zeros((self.num_table, self.action_size), dtype=torch.float32)
        move_policy[table.view(-1, 1), actions] = probs
        return move_policy, position_value
