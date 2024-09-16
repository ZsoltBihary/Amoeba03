import torch
import numpy as np
from ClassAmoeba import Amoeba
# from EncoderClass import SimpleEncoder01, BiharyEncoder01
# from ModelClass import SimpleModel01, SimpleModel02, BiharyModel01, BiharyModel02
from ClassModel import TestModel01
# from EvaluatorClass import Evaluator
from torchinfo import summary
from line_profiler_pycharm import profile


@profile
def eval_cycle(args: dict):
    active_table_idx = torch.arange(args.get('num_table'),
                                    dtype=torch.long, device=args.get('CPU_device'))
    for i_MC in range(args.get('num_MC')):
        for node_idx in range(5):
            # Probability of selecting each integer
            selection_prob = 0.2
            # Generate random probabilities for each element in the tensor
            random_probs = torch.rand(active_table_idx.shape, device=active_table_idx.device)
            # Create a mask where the probability is less than the selection probability (0.1)
            mask = random_probs < selection_prob
            # Select elements that are selected (True in the mask)
            eval_idx = active_table_idx[mask]
            # Select elements that are not selected (False in the mask)
            active_table_idx = active_table_idx[~mask]

            position = eval_idx.reshape(-1, 1) * torch.ones((1, game.action_size),
                                                            dtype=torch.float32, device=args.get('CPU_device'))

            pinned_position = position.pin_memory()

            position_CUDA = pinned_position.to(args.get('CUDA_device'), non_blocking=True)

            result_CUDA = model(position_CUDA)

            result = result_CUDA.to(args.get('CPU_device'), non_blocking=True)
            active_table_idx = torch.cat((active_table_idx, eval_idx), dim=0)
            # result = model(position_CUDA).to(args.get('CPU_device'))


# Collect parameters in a dictionary
args = {
    'board_size': 15,
    'win_length': 5,
    'CPU_device': 'cpu',
    'CUDA_device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'num_child': 50,
    'num_table': 16,
    'num_MC': 100,
    'max_moves': 30,
    'buffer_size': 50000
}
# print(torch.cuda.is_available())

game = Amoeba(args)
model = TestModel01(args, 50)

for i in range(10):
    print(i)
    eval_cycle(args)


# summary(model, input_data=position_CUDA, verbose=2)
