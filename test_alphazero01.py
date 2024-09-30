import torch
import numpy as np
from ClassAmoeba import Amoeba
from ClassModel import TerminalCheck01, DeepMindModel01
from ClassAlphaZero import AlphaZero
# from ClassEvaluator import EvaluationBuffer
from torchinfo import summary
from line_profiler_pycharm import profile
import time

# Collect parameters in a dictionary
args = {
    'board_size': 15,
    'win_length': 5,
    'CUDA_device': 'cuda' if torch.cuda.is_available() else 'cpu',
    # 'CUDA_device': 'cpu',
    'num_child': 50,
    'num_table': 400,
    'num_MC': 200,
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
model = DeepMindModel01(args)
model.eval()
alpha = AlphaZero(args, game, terminal_check, model)
print('AlphaZero set up done')

start = time.time()
alpha.run()

elapsed_time = (time.time() - start) / 60.0
data_gen = args.get('num_moves') * args.get('num_table')
data_per_minute = round(data_gen / elapsed_time)
print(f"Elapsed time: {elapsed_time:.2f} minutes")
print("data generated = ", data_gen)
print("data_per_minute = ", data_per_minute)

# summary(model, input_data=position_CUDA, verbose=2)
