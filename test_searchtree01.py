import torch
import numpy as np
from ClassAmoeba import Amoeba
from ClassModel import TerminalCheck01, TrivialModel01, DeepMindModel01
from ClassSearchTree import SearchTree
# from ClassGamePlay import GamePLay
# from ClassAlphaZero import AlphaZero
# from ClassEvaluator import EvaluationBuffer
# from torchinfo import summary
# from line_profiler_pycharm import profile
import time

# Collect parameters in a dictionary
args = {
    'board_size': 11,
    'win_length': 5,
    'CUDA_device': 'cuda' if torch.cuda.is_available() else 'cpu',
    # 'CUDA_device': 'cpu',
    'num_leaf': 32,
    # 'num_branch': 2,
    'num_MC': 200,
    # 'num_child': 9,
    # 'num_table': 2,
    # 'num_moves': 5,
    # 'eval_batch_size': 128,
    'res_channels': 32,
    'hid_channels': 16,
    'num_res': 4,
    'policy_hid_channels': 32,
    'value_hid_dim': 64
}

game = Amoeba(args)
terminal_check = TerminalCheck01(args)
model = TrivialModel01(args)
# model = DeepMindModel01(args)
model.eval()
tree = SearchTree(args, game, terminal_check, model)
# gplay = GamePLay(args, game, terminal_check, model)
# n_states = 2

player = 1
position = game.get_empty_position()
# position[20] = -1
# position[4] = -1
# position[16] = 1

game.print_board(position)
# action_policy, root_value = tree.analyze(player, position)

start = time.time()

i_move = 0

while True:
    players = player * torch.ones(1, dtype=torch.float32)
    positions = position[None, :]
    terminal_mask, value = tree.check_EOG(players, positions)
    if terminal_mask[0]:
        print('Game ended. Result = ', value[0])
        break
    # tree.analyze(player, position)
    # tree.reset_tree()
    action_policy, root_value = tree.analyze(player, position)
    action = torch.argmax(action_policy)
    print('Value = ', root_value)
    i_move += 1
    print('i_move = ', i_move)
    position = game.get_new_position(position, player, action)
    player *= -1
    game.print_board(position)
    # if i_move > 20:
    #     break

elapsed_time = (time.time() - start) / 60.0
print(f"Elapsed time: {elapsed_time:.2f} minutes")
