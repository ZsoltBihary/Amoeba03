import torch
import numpy as np
from ClassAmoeba import Amoeba
# from EncoderClass import SimpleEncoder01, BiharyEncoder01
# from ModelClass import SimpleModel01, SimpleModel02, BiharyModel01, BiharyModel02
from ClassModel import TestModel01
# from EvaluatorClass import Evaluator
from torchinfo import summary
from line_profiler_pycharm import profile
import time


@profile
def eval_cycle(args: dict, batch_s):

    for i_MC in range(args.get('num_MC')):
        for node_idx in range(8):
            new_position = torch.ones((batch_s, game.action_size),
                                      dtype=torch.float32, device=args.get('CPU_device'))

            pinned_position[:, :] = position[:, :]

            position_CUDA = pinned_position.to(args.get('CUDA_device'), non_blocking=True)
            with torch.no_grad():
                result_CUDA = model(position_CUDA)

            result = result_CUDA.to(args.get('CPU_device'), non_blocking=True)
            # result = model(position_CUDA).to(args.get('CPU_device'))


# Collect parameters in a dictionary
args = {
    'board_size': 15,
    'win_length': 5,
    'CPU_device': 'cpu',
    'CUDA_device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'num_child': 50,
    'num_table': 280,
    'num_MC': 200,
    'max_moves': 10,
    'buffer_size': 50000
}
# print(torch.cuda.is_available())

batch_size = args.get('num_table') // 8
game = Amoeba(args)
model = TestModel01(args, 96)
model.eval()
position = torch.ones((batch_size, game.action_size),
                      dtype=torch.float32, device=args.get('CPU_device'))
pinned_position = position.pin_memory()

start = time.time()

for i in range(args.get('max_moves')):
    print(i)
    eval_cycle(args, batch_size)

elapsed_time = (time.time() - start) / 60.0
data_gen = args.get('max_moves') * args.get('num_table')
data_per_minute = round(data_gen / elapsed_time)

print(f"Elapsed time: {elapsed_time:.2f} minutes")
print("data generated = ", data_gen)
print("data_per_minute = ", data_per_minute)

# summary(model, input_data=position_CUDA, verbose=2)
