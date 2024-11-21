import torch
# import torch.nn as nn
# import torch.nn.functional as F
from ClassAmoeba import Amoeba
# from typing import Tuple
from ClassLeafBuffer import LeafBuffer
from ClassChildBuffer import ChildBuffer
# from ClassPathManager import PathManager
# from ClassTree import Tree
# from ClassModel import evaluate01
from line_profiler_pycharm import profile


class SearchTree:
    def __init__(self, args: dict, game: Amoeba):
        self.args = args
        self.game = game
        # self.evaluator = evaluator
        # self.terminal_check = terminal_check
        # self.model = model

        # Set up parameters
        self.num_table = args.get('num_table')
        self.num_child = args.get('num_child')
        self.num_MC = args.get('num_MC')
        self.num_agent = args.get('num_agent')
        self.num_node = (self.num_MC + 100) * self.num_child
        self.action_size = game.action_size
        self.max_depth = game.action_size + 1
        self.CUDA_device = args.get('CUDA_device')
        # Set up subclasses
        self.leaf_buffer = LeafBuffer(game, args.get('leaf_buffer_size'), args.get('eval_batch_size'))
        self.child_buffer = ChildBuffer(self.num_agent * 20 * self.num_child)
        # Set up tree attributes
        self.next_node = 2 * torch.ones(self.num_table, dtype=torch.long)
        self.is_leaf = torch.ones((self.num_table, self.num_node), dtype=torch.bool)
        self.is_terminal = torch.zeros((self.num_table, self.num_node), dtype=torch.bool)
        self.player = torch.ones((self.num_table, self.num_node), dtype=torch.int32)
        self.count = torch.zeros((self.num_table, self.num_node), dtype=torch.int32)
        self.value_sum = torch.zeros((self.num_table, self.num_node), dtype=torch.float32)
        self.value = torch.zeros((self.num_table, self.num_node), dtype=torch.float32)
        self.start_child = 2 * torch.ones((self.num_table, self.num_node), dtype=torch.long)
        self.parent = torch.ones((self.num_table, self.num_node), dtype=torch.long)
        self.action = torch.zeros((self.num_table, self.num_node), dtype=torch.long)
        self.prior = torch.zeros((self.num_table, self.num_node), dtype=torch.float32)
        self.ucb = -99999.9 * torch.ones((self.num_table, self.num_node), dtype=torch.float32)
        # Set up path agent attributes
        self.all_agents = torch.arange(self.num_agent)
        self.agent_active = torch.ones(self.num_agent, dtype=torch.bool)
        self.agent_table = torch.zeros(self.num_agent, dtype=torch.long)
        self.agent_node = torch.zeros(self.num_agent, dtype=torch.long)
        self.path_depth = torch.zeros(self.num_agent, dtype=torch.long)
        self.agent_player = torch.zeros(self.num_agent, dtype=torch.int32)
        self.agent_position = torch.zeros((self.num_agent, self.action_size), dtype=torch.int32)
        self.agent_path = torch.zeros((self.num_agent, self.max_depth), dtype=torch.long)
        # Keep track of search count, root player, root position on each table
        self.search_count = torch.zeros(self.num_table, dtype=torch.int32)
        self.root_player = torch.ones(self.num_table, dtype=torch.int32)
        self.root_position = torch.zeros((self.num_table, self.action_size), dtype=torch.int32)
        # self.move_count = torch.zeros(self.num_table, dtype=torch.int32)
        # # Keep track of tables under evaluation, and evaluation status
        # self.evaluated_table_idx = torch.arange(args.get('num_table'), dtype=torch.long)
        # self.evaluation_is_on = False
        # Keep track of root player, node player, and current position, node position on all tables

        # self.node_player = torch.ones(self.num_table, dtype=torch.int32)
        # self.node_position = torch.zeros((self.num_table, self.action_size), dtype=torch.int32)

    def get_active_indices(self):
        a = self.all_agents[self.agent_active]
        return (self.all_agents[self.agent_active],
                self.agent_table[self.agent_active],
                self.agent_node[self.agent_active])

    def save_leaves(self):
        # get active agent indices ... I assume there is always at least one active agent ...
        agent, table, node = self.get_active_indices()
        # detect leaves ...
        leaf_idx = agent[self.is_leaf[table, node]]
        # if no leaf found, do nothing ...
        if leaf_idx.shape[0] == 0:
            return
        # else get data from active agents that found leaves ...
        leaf_table = self.agent_table[leaf_idx]
        leaf_node = self.agent_node[leaf_idx]
        leaf_player = self.agent_player[leaf_idx]
        leaf_position = self.agent_position[leaf_idx]
        leaf_path = self.agent_path[leaf_idx]
        leaf_depth = self.path_depth[leaf_idx]
        # add leaf data to leaf buffer ...
        self.leaf_buffer.add_leaves(leaf_table, leaf_node, leaf_player, leaf_position, leaf_path, leaf_depth)
        # free up agents that have found leaves
        self.agent_active[leaf_idx] = False
        # TODO: def allocate_child_block(self, parent_table, parent_node):
        #     self.start_child[parent_table, parent_node] = self.next_node[parent_table]
        #     self.next_node[parent_table] += self.num_child
        return

    def activate_agents(self):
        # TODO: This is just a proxy for now ...
        self.agent_active[:] = True
        return self.all_agents[self.agent_active]

    def save_all_children(self, child_table, parent_node, child_node, parent_player):
        # Step 1: Replicate parent_node and parent_player to 2d shape
        parent_node_expanded = parent_node.unsqueeze(1).repeat(1, self.num_child)  # Shape: (6, 20)
        parent_player_expanded = parent_player.unsqueeze(1).repeat(1, self.num_child)  # Shape: (6, 20)
        # Step 2: Flatten all tensors
        tables = child_table.flatten()
        parents = parent_node_expanded.flatten()
        children = child_node.flatten()
        players = parent_player_expanded.flatten()

        self.child_buffer.add(tables, parents, children, players)
        return

    def explore_children(self, parent_table, parent_node):
        # get the child node indexes on the parent tables ...
        child_table = parent_table.unsqueeze(1).repeat(1, self.num_child)
        # add the start index to the offsets to get the actual indices ... relying on broadcasting here ...
        start_node = self.start_child[parent_table, parent_node].reshape(-1, 1)
        node_offset = torch.arange(self.num_child, dtype=torch.long).reshape(1, -1)
        child_node = start_node + node_offset
        # save all the child information to the child buffer, we will use this info to update ucb ...
        self.save_all_children(child_table, parent_node, child_node, self.player[parent_table, parent_node])
        # find the best child node based on current ucb ...
        best_idx = torch.argmax(self.ucb[child_table, child_node], dim=1)
        best_child_node = child_node[torch.arange(best_idx.shape[0]), best_idx]
        # lower ucb for best child to facilitate branching for consecutive paths ...
        self.ucb[parent_table, best_child_node] -= 0.1

        return best_child_node

    def update_agents(self, agent, table, new_node):
        # TODO: comment
        self.agent_node[agent] = new_node
        self.agent_position[agent, self.action[table, new_node]] = self.agent_player[agent]
        self.agent_player[agent] *= -1
        self.path_depth[agent] += 1
        self.agent_path[agent, self.path_depth[agent]] = new_node
        return

    def update_ucb(self):

        table, parent, child, parent_player = self.child_buffer.get_data()
        # TODO: Is this so simple? Perhaps we need to keep fresh results ...
        self.child_buffer.empty()
        child_q = self.value[table, child]
        child_prior = self.prior[table, child]
        parent_count = self.count[table, parent]
        child_count = self.count[table, child]
        self.ucb[table, child] = (parent_player * child_q +
                                  2.0 * child_prior * torch.sqrt(parent_count + 1) / (child_count + 1))

        return

    @profile
    def search_one_step(self):
        # save leaves if any have been found (and deactivate agents that found leaves) ...
        self.save_leaves()
        # activate new agents to start new paths ...
        self.activate_agents()
        # get active indexes ...
        agent, table, node = self.get_active_indices()
        # find the best child node (and save all children involved) ...
        best_node = self.explore_children(table, node)
        # update state of the search agents
        self.update_agents(agent, table, best_node)


    def update_tree(self):

        # EXPAND
        # get leaf info from buffer ...
        (table,
         node,
         player,
         position,
         policy,
         value,
         is_terminal) = self.leaf_buffer.get_eval_data()

        # # UPDATE counts and values in the trees ...
        # # Scatter-add to handle repeated indices
        # # Prepare the indices
        # indices = torch.stack([tables, nodes], dim=0)  # Shape (2, 10)
        # # Perform scatter-add
        # self.value_sum.scatter_add_(0, indices, values)

        # UPDATE ucb in the trees ...
        self.update_ucb()
        return

    # @profile
    # def expand(self):
    #     # def get_eval_data(self):
    #     (table,
    #      node,
    #      player,
    #      position,
    #      policy,
    #      value,
    #      is_terminal) = self.leaf_buffer.get_eval_data()
    #
    #     return

        # ***** Evaluate *****
        # self.leaf_is_terminal, self.leaf_logit, self.leaf_value = (
        #     self.evaluate(self.leaf_player, self.leaf_position))

        # ***** Expand *****
        # legal_mask = (torch.abs(self.leaf_position) < 0.5)
        # all_actions = torch.arange(0, self.action_size)
        # for i in range(self.num_leaf):
        #     parent_i = self.leaf_node[i]
        #     self.player[parent_i] = self.leaf_player[i]
        #     self.is_terminal[parent_i] = self.leaf_is_terminal[i]
        #     to_expand = self.is_leaf[parent_i] & ~self.leaf_is_terminal[i]
        #     if to_expand:
        #         self.is_leaf[parent_i] = False
        #         actions = all_actions[legal_mask[i]]
        #         policy_logit = self.leaf_logit[i, legal_mask[i]]
        #
        #         # Determine the number of children to select (use min to avoid selecting more than available)
        #         n_child = min(self.max_child, policy_logit.shape[0])
        #         # Get the top n_child values and their indices from policy_logit
        #         top_values, top_indices = torch.topk(policy_logit, n_child)
        #         # Use the indices to select the corresponding actions
        #         selected_policy_logit = top_values
        #         selected_actions = actions[top_indices]
        #
        #         policy = torch.nn.functional.softmax(selected_policy_logit, dim=0)
        #         start_children = self.next_node
        #         end_children = start_children + n_child
        #
        #         self.start_child[parent_i] = start_children
        #         self.num_child[parent_i] = n_child
        #         self.action[start_children: end_children] = selected_actions
        #         self.prior[start_children: end_children] = policy
        #         self.parent[start_children: end_children] = parent_i
        #         self.next_node += n_child


    # @profile
    # def run(self):
    #     i_move = 1
    #     while True:
    #         # Coarse-grained loop starts here
    #         for i_step in range(100):
    #             # Fine-grained loop starts here
    #             if self.eval_buffer.num_data > self.eval_buffer.batch_size:
    #                 self.evaluation_is_on = True
    #                 result_CUDA = self.evaluate()
    #             self.search_tree()
    #
    #             if self.evaluation_is_on:
    #                 self.evaluation_is_on = False
    #                 result = result_CUDA.to('cpu', non_blocking=False)
    #                 self.update_tree(result)
    #
    #         # Coarse-grained operations start here
    #         mean_move_count = torch.mean(self.move_count.float()).item()
    #
    #         max_next_node_idx = torch.max(self.tree['next_node_idx'])
    #         print('max_next_node_idx', max_next_node_idx)
    #
    #         if mean_move_count > 0.99 * i_move:
    #             print('mean_move_count = ', i_move)
    #             i_move += 1
    #         if i_move > self.num_moves:
    #             break

#
# #
# # class Tree:
# #     # def __init__(self, args: dict, game, terminal_check, model):
# #     def __init__(self, num_table, num_node, num_child):
# #         # ***** These are constant parameters that class methods never modify *****
# #         self.num_table = num_table
# #         self.num_node = num_node
# #         self.num_child = num_child
# #         self.root_node = 1
# #
# #         # ***** These are variables that class methods will modify *****
# #         # Set up tree
# #         self.next_node = 2 * torch.ones(self.num_table, dtype=torch.long)
# #         self.is_leaf = torch.ones((self.num_table, self.num_node), dtype=torch.bool)
# #         self.is_terminal = torch.zeros((self.num_table, self.num_node), dtype=torch.bool)
# #         self.player = torch.zeros((self.num_table, self.num_node), dtype=torch.int32)
# #         self.count = torch.zeros((self.num_table, self.num_node), dtype=torch.int32)
# #         self.value_sum = torch.zeros((self.num_table, self.num_node), dtype=torch.float32)
# #         self.value = torch.zeros((self.num_table, self.num_node), dtype=torch.float32)
# #         self.start_child = 2 * torch.ones((self.num_table, self.num_node), dtype=torch.long)
# #         self.parent = torch.ones((self.num_table, self.num_node), dtype=torch.long)
# #         self.action = torch.zeros((self.num_table, self.num_node), dtype=torch.long)
# #         self.prior = torch.zeros((self.num_table, self.num_node), dtype=torch.float32)
# #         self.ucb = -99999.9 * torch.ones((self.num_table, self.num_node), dtype=torch.float32)
# #
# #     def reset(self):
# #         # ***** These are variables that class methods will modify *****
# #         # Set up tree
# #         self.next_node = 2 * torch.ones(self.num_table, dtype=torch.long)
# #         self.is_leaf = torch.ones((self.num_table, self.num_node), dtype=torch.bool)
# #         self.is_terminal = torch.zeros((self.num_table, self.num_node), dtype=torch.bool)
# #         self.player = torch.zeros((self.num_table, self.num_node), dtype=torch.int32)
# #         self.count = torch.zeros((self.num_table, self.num_node), dtype=torch.int32)
# #         self.value_sum = torch.zeros((self.num_table, self.num_node), dtype=torch.float32)
# #         self.value = torch.zeros((self.num_table, self.num_node), dtype=torch.float32)
# #         self.start_child = 2 * torch.ones((self.num_table, self.num_node), dtype=torch.long)
# #         # self.num_child = torch.zeros((self.num_table, self.num_node), dtype=torch.long)
# #         self.parent = torch.ones((self.num_table, self.num_node), dtype=torch.long)
# #         self.action = torch.zeros((self.num_table, self.num_node), dtype=torch.long)
# #         self.prior = torch.zeros((self.num_table, self.num_node), dtype=torch.float32)
# #         self.ucb = -99999.9 * torch.ones((self.num_table, self.num_node), dtype=torch.float32)
# #
# #     def get_children(self, parent_table, parent_node):
# #         child_table = parent_table.unsqueeze(1).repeat(1, self.num_child)
# #         # Add the start index to the offsets to get the actual indices ... relying on broadcasting here ...
# #         start_node = self.start_child[parent_table, parent_node].reshape(-1, 1)
# #         node_offset = torch.arange(self.num_child, dtype=torch.long).reshape(1, -1)
# #         child_node = start_node + node_offset
# #         return child_table, child_node, node_offset
# #
# #     def best_child(self, parent_table, parent_node):
# #         child_table, child_node, node_offset = self.get_children(parent_table, parent_node)
# #         best_idx = torch.argmax(self.ucb[child_table, child_node], dim=1)
# #         # best_idx = torch.argmax(ucb1, dim=1)
# #         best_child_node = child_node[torch.arange(best_idx.shape[0]), best_idx]
# #         return child_node[torch.arange(best_idx.shape[0]), best_idx]
# #
# #     def allocate_child_block(self, parent_table, parent_node):
# #         self.start_child[parent_table, parent_node] = self.next_node[parent_table]
# #         self.next_node[parent_table] += self.num_child
# #         return
# #

# #     @profile
# #     def back_propagate(self):
# #         for leaf in range(self.num_leaf):
# #             self.count[self.leaf_path[leaf, :]] += 1
# #             self.value_sum[self.leaf_path[leaf, :]] += self.leaf_value[leaf]
# #         self.value[self.leaf_path] = self.value_sum[self.leaf_path] / self.count[self.leaf_path]
# #         children = self.children_buffer[:self.children_next_idx]
# #         self.children_next_idx = 0
# #         parents = self.parent[children]
# #         parent_player = self.player[parents]
# #         parent_count = self.count[parents]
# #         child_count = self.count[children]
# #         child_q = self.value[children]
# #         child_prior = self.prior[children]
# #
# #         self.ucb[children] = (parent_player * child_q +
# #                               2.0 * child_prior * torch.sqrt(parent_count + 1) / (child_count + 1))
# #
# #     #     # 'node_idx': torch.zeros((self.num_leaf, self.max_depth), dtype=torch.long),
# #     #     path_idx = self.leaf_buffer['node_idx'][:]
# #     #     # print(path_idx)
# #     #     for i in range(self.num_leaf):
# #     #         path = path_idx[i]
# #     #         path = path[path > 0]
# #     #         self.tree['count'][path] += 1
# #     #         self.tree['value_sum'][path] += self.leaf_buffer['value'][i]
# #     #     for i in range(self.num_leaf):
# #     #         path = path_idx[i]
# #     #         path = path[path > 0]
# #     #         self.tree['value'][path] = self.tree['value_sum'][path] / self.tree['count'][path]
# #
# #     @profile
# #     def analyze(self, player, position):
# #         self.reset_tree()
# #         for i_MC in range(self.num_MC):
# #             # print('i_MC = ', i_MC)
# #             self.search_leaves(player, position)
# #             self.update_tree()
# #             self.back_propagate()
# #
# #         # root_idx = 1
# #         root_value = self.value[self.root_node]
# #         # start_idx = self.tree['start_child_idx'][root_idx].item()
# #         # end_idx = start_idx + self.tree['num_child'][root_idx].item()
# #         # root_children_idx = torch.arange(start_idx, end_idx)
# #         root_children = self.get_children(self.root_node)
# #         children_actions = self.action[root_children]
# #         action_count = torch.zeros(self.action_size, dtype=torch.int32)
# #         action_count[children_actions] = self.count[root_children]
# #         action_weight = action_count.to(dtype=torch.float32)
# #         action_policy = action_weight / torch.sum(action_weight)
# #
# #         # action_value = torch.zeros(self.action_size, dtype=torch.float32)
# #         # action_value[children_actions] = self.tree['value'][root_children_idx]
# #         # self.game.print_board(position)
# #         # print('move count: \n', action_count.view(self.game.board_size, -1))
# #         # print('move weight: \n', action_weight.view(self.game.board_size, -1))
# #         # print('value = ', root_value)
# #         # print('move policy: \n', action_policy.view(self.game.board_size, -1))
# #
# #         return action_policy, root_value
# #
# #     # ****** THIS IS THE OLDER VERSION ************************
# #     # @profile
# #     # def search_one_leaf(self, leaf, depth, node_idx, player, position):
# #     #     while not self.tree['is_leaf'][node_idx]:
# #     #         self.leaf_buffer['node_idx'][leaf, depth] = node_idx
# #     #         ucb = self.calc_ucb(node_idx, player)
# #     #         best_offset = torch.argmax(ucb)
# #     #         best_child_idx = self.tree['start_child_idx'][node_idx] + best_offset
# #     #         action = self.tree['action'][best_child_idx]
# #     #         position = self.game.get_new_position(position, player, action)
# #     #         player = -player
# #     #         depth += 1
# #     #         node_idx = best_child_idx
# #     #     # Leaf found !!!
# #     #     self.leaf_buffer['node_idx'][leaf, depth] = node_idx
# #     #     self.add_leaf(leaf, leaf+1, node_idx, player, position)
# #     #     return
# #     # ****** THIS IS THE OLDER VERSION ************************
# #
# #     # ****** THIS IS THE OLDER VERSION ************************
# #     # @profile
# #     # def search_leaves(self, depth, node_idx, player, position):
# #     #     # print('   - depth, node_idx: ', depth, node_idx)
# #     #     if self.i_leaf == self.num_leaf:
# #     #         return
# #     #     self.leaf_buffer['node_idx'][self.i_leaf:, depth] = node_idx
# #     #     if self.tree['is_leaf'][node_idx]:
# #     #         if depth+1 < self.max_depth:
# #     #             self.leaf_buffer['node_idx'][self.i_leaf, depth+1:] = 0
# #     #         self.add_leaf(node_idx, player, position)
# #     #     else:
# #     #         new_player = -player
# #     #         ucb = self.calc_ucb(node_idx, player)
# #     #         for i_branch in range(self.num_branch):
# #     #             if self.i_leaf == self.num_leaf:
# #     #                 break
# #     #             best_offset = torch.argmax(ucb)
# #     #             ucb[best_offset] -= 0.2
# #     #             best_child_idx = self.tree['start_child_idx'][node_idx] + best_offset
# #     #             action = self.tree['action'][best_child_idx]
# #     #             new_position = self.game.get_new_position(position, player, action)
# #     #             self.search_leaves(depth+1, best_child_idx, new_player, new_position)
# #     # ****** THIS IS THE OLDER VERSION ************************
# #
# #     # ****** THIS IS THE OLDER VERSION ************************
# #     # def calc_ucb(self, parent, parent_player):
# #     #
# #     #     # Let us calculate the ucb for all children ...
# #     #     # parent_player = self.node_player[parent_table].view(-1, 1)
# #     #     start_children_idx = self.tree['start_child_idx'][node_idx].item()
# #     #     end_children_idx = start_children_idx + self.tree['num_child'][node_idx].item()
# #     #     child_nodes = torch.arange(start_children_idx, end_children_idx)
# #     #     parent_count = self.tree['count'][node_idx]
# #     #     child_count = self.tree['count'][child_nodes]
# #     #     child_q = self.tree['value'][child_nodes]
# #     #     child_prior = self.tree['prior'][child_nodes]
# #     #     ucb = (parent_player * child_q +
# #     #            2.0 * child_prior * torch.sqrt(parent_count + 1) / (child_count + 1))
# #     #
# #     #     return ucb
# #     # ****** THIS IS THE OLDER VERSION ************************
