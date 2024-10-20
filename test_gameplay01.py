import torch
import numpy as np
from ClassAmoeba import Amoeba
from ClassModel import TerminalCheck01, TrivialModel01, DeepMindModel01
from ClassGamePlay import GamePLay
# from ClassAlphaZero import AlphaZero
# from ClassEvaluator import EvaluationBuffer
# from torchinfo import summary
# from line_profiler_pycharm import profile
import time

# Collect parameters in a dictionary
args = {
    'board_size': 3,
    'win_length': 3,
    'CUDA_device': 'cuda' if torch.cuda.is_available() else 'cpu',
    # 'CUDA_device': 'cpu',
    'num_branch': 2,
    'num_leaf': 32,
    # 'num_child': 9,
    # 'num_table': 2,
    'num_MC': 200,
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
gplay = GamePLay(args, game, terminal_check, model)
# n_states = 2

player = -1
position = game.get_empty_position()
position[0] = 1
# position[24] = -1
# position[9] = 1
# position[20] = -1
# position[1] = 1
# position[2] = -1

start = time.time()

i_move = 0
game.print_board(position)
while True:
    i_move += 1
    print('i_move = ', i_move)
    action_policy, root_value = gplay.analyze(player, position)





elapsed_time = (time.time() - start) / 60.0
print(f"Elapsed time: {elapsed_time:.2f} minutes")
