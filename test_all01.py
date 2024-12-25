import torch
# import numpy as np
from ClassAmoeba import Amoeba
from ClassModel import TerminalCheck01, TrivialModel01, TrivialModel02, SimpleModel01
# from ClassSearchTree import SearchTree
from ClassSearchEngine import SearchEngine
# from ClassGamePlay import GamePLay
# from ClassEvaluator import EvaluationBuffer
# from torchinfo import summary
# from line_profiler_pycharm import profile
# from ClassTree import Tree
# import time

# Collect parameters in a dictionary
args = {
    'board_size': 6,
    'win_length': 5,
    'CUDA_device': 'cuda' if torch.cuda.is_available() else 'cpu',
    # 'CUDA_device': 'cpu',
    'num_leaf': 8,
    # 'num_branch': 2,
    'num_MC': 200,
    'num_child': 5,
    'num_table': 2,
    'num_agent': 6,
    # 'num_moves': 5,
    'leaf_buffer_size': 32,
    'eval_batch_size': 16
    # 'res_channels': 32,
    # 'hid_channels': 16,
    # 'num_res': 4,
    # 'policy_hid_channels': 32,
    # 'value_hid_dim': 64
}

game = Amoeba(args)

terminal_check = TerminalCheck01(args)
# model = TrivialModel01(args)
model = SimpleModel01(args)
model.eval()

root_player = torch.ones(args.get('num_table'), dtype=torch.int32)
root_player[1] = -1
root_position = game.get_empty_positions(args.get('num_table'))
root_position[1, 0] = 1

engine = SearchEngine(args, game, terminal_check, model)
engine.reset(root_player, root_position)

# stree = SearchTree(args, game, terminal_check, model)
# stree.reset(root_player, root_position)
# stree.analyze(root_player, root_position)

# active = stree.activate_agents()

a = 42

# stree.search_one_step()
# stree.save_leaves()

# terminal_check = TerminalCheck01(args)
# model = TrivialModel02(args)
# model = SimpleModel01(args)
# model.eval()
# tree = SearchTree(args, game, terminal_check, model)
