import torch
import numpy as np
from ClassAmoeba import Amoeba
from ClassModel import TerminalCheck01, DeepMindModel01, SimpleModel01
from ClassAlphaZero import AlphaZero
# from ClassEvaluator import EvaluationBuffer
from torchinfo import summary
from line_profiler_pycharm import profile
import time

# Collect parameters in a dictionary
args = {
    'board_size': 7,
    'win_length': 5,
    'CUDA_device': 'cuda' if torch.cuda.is_available() else 'cpu',
    # 'CUDA_device': 'cpu',
    'num_child': 40,
    'num_table': 200,
    'num_MC': 100,
    'num_moves': 5,
    'eval_batch_size': 128,
    'res_channels': 32,
    'hid_channels': 16,
    'num_res': 4,
    'policy_hid_channels': 32,
    'value_hid_dim': 64
}

game = Amoeba(args)
terminal_check = TerminalCheck01(args)
# model = DeepMindModel01(args)
model = SimpleModel01(args)
model.eval()

positions = game.get_random_positions(2, 6, 4).to(dtype=torch.float32)
position_CUDA = positions.cuda()
game.print_board(positions[0, :])

policy, value = model(position_CUDA)
policy_int = (1 * policy[0].reshape(-1, args.get('board_size'))).to(device='cpu').detach().to(dtype=torch.int32).numpy()
print(policy_int)
print(value[0])

# summary(model, input_data=position_CUDA, verbose=1)
