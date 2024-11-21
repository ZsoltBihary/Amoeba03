import torch
import numpy as np
from ClassAmoeba import Amoeba
# from ClassModel import TerminalCheck01, TrivialModel01, TrivialModel02, SimpleModel01
from ClassSearchTree import SearchTree
# from ClassGamePlay import GamePLay
# from ClassAlphaZero import AlphaZero
# from ClassEvaluator import EvaluationBuffer
# from torchinfo import summary
# from line_profiler_pycharm import profile
# from ClassTree import Tree
import time

# Collect parameters in a dictionary
args = {
    'board_size': 6,
    'win_length': 5,
    'CUDA_device': 'cuda' if torch.cuda.is_available() else 'cpu',
    # 'CUDA_device': 'cpu',
    'num_leaf': 8,
    # 'num_branch': 2,
    'num_MC': 200,
    'num_child': 20,
    'num_table': 2,
    'num_agent': 6,
    # 'num_moves': 5,
    'leaf_buffer_size': 128,
    'eval_batch_size': 32,
    'res_channels': 32,
    'hid_channels': 16,
    'num_res': 4,
    'policy_hid_channels': 32,
    'value_hid_dim': 64
}

game = Amoeba(args)
stree = SearchTree(args, game)
stree.search_one_step()
# stree.save_leaves()

a = 42

# tree = Tree(3, 10, 4)
#
# tree.reset()
# parent_table = torch.zeros(2, dtype=torch.long)
# parent_table[1] = 2
# parent_node = torch.zeros(2, dtype=torch.long)
# parent_node[0] = 2
# parent_node[1] = 4
#
# tree.get_children(parent_table, parent_node)
# tree.best_child(parent_table, parent_node)
# tree.allocate_child_block(parent_table, parent_node)
# print(tree)

#
# terminal_check = TerminalCheck01(args)
# model = TrivialModel02(args)
# model = SimpleModel01(args)
# model.eval()
# tree = SearchTree(args, game, terminal_check, model)
# player = 1
# position = game.get_empty_position()
# position[46] = 1
# position[62] = -1
# # position[72] = 1

