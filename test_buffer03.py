import torch
import numpy as np
from ClassAmoeba import Amoeba
# from EncoderClass import SimpleEncoder01, BiharyEncoder01
# from ModelClass import SimpleModel01, SimpleModel02, BiharyModel01, BiharyModel02
from ClassModel import TestModel01
from ClassEvaluator import EvalBuffer
from torchinfo import summary
from line_profiler_pycharm import profile
import time


@profile
def move_cycle(args: dict):

    selection_prob = 0.2
    all_table_idx = torch.arange(args.get('num_table'),
                                 dtype=torch.long, device=args.get('CPU_device'))
    leaf_idx = torch.zeros(args.get('num_table'),
                           dtype=torch.long, device=args.get('CPU_device'))
    active_table_idx = torch.arange(args.get('num_table'),
                                    dtype=torch.long, device=args.get('CPU_device'))
    count_eval = torch.zeros(args.get('num_table'),
                             dtype=torch.int32, device=args.get('CPU_device'))
    buffer.empty()
    # In principle, we want to organize a loop that simultaneously performs the MC searches
    # on all the tables ...
    # But for testing, we break it up to two nested loops ...

    while True:

        for i_search in range(100):

            # collect leaf positions + index metadata
            leaf_idx[active_table_idx] += 1
            random_probs = torch.rand(active_table_idx.shape, device=active_table_idx.device)
            mask = random_probs < selection_prob
            eval_table_idx = active_table_idx[mask]
            batch_size = eval_table_idx.shape[0]
            active_table_idx = active_table_idx[~mask]

            if batch_size > 0:
                new_position = eval_table_idx.reshape(-1, 1) * torch.ones((1, game.action_size),
                                                                          dtype=torch.float32,
                                                                          device=args.get('CPU_device'))
                new_idx = torch.zeros((batch_size, 2),
                                      dtype=torch.long, device=args.get('CPU_device'))
                new_idx[:, 0] = eval_table_idx
                new_idx[:, 1] = leaf_idx[eval_table_idx]
                buffer.add_positions(new_position, new_idx)

            if buffer.num_data >= buffer.min_batch_size:

                position_data, idx_data = buffer.get_positions()
                # position_data = position_data.pin_memory()
                position_CUDA = position_data.to(args.get('CUDA_device'), non_blocking=True)
                evaluated_table_idx = idx_data[:, 0]
                count_eval[evaluated_table_idx] += 1
                with torch.no_grad():
                    result_CUDA = model(position_CUDA)

                result = result_CUDA.to(args.get('CPU_device'), non_blocking=True)
                # update_tree(result, idx_data)
                new_tree = torch.sum(result) + torch.sum(idx_data)  # just something for testing

                # Add back the evaluated tables that still need more MC runs to active tables
                active_table_idx = torch.cat((active_table_idx, evaluated_table_idx), dim=0)

        # Monitor MC search progress ...
        min_count_eval = torch.min(count_eval).item()
        mean_count_eval = torch.sum(count_eval).item() // args.get('num_table')
        max_count_eval = torch.max(count_eval).item()
        # Let's check if we want to stop the MC searches ...
        if mean_count_eval > args.get('num_MC'):
            break

    print('min, mean, max count_eval = ', min_count_eval, ',', mean_count_eval, ',', max_count_eval)


# Collect parameters in a dictionary
args = {
    'board_size': 15,
    'win_length': 5,
    'CPU_device': 'cpu',
    'CUDA_device': 'cuda' if torch.cuda.is_available() else 'cpu',
    # 'CUDA_device': 'cpu',
    'num_child': 50,
    'num_table': 300,
    'num_MC': 500,
    'max_moves': 5,
    'buffer_size': 50000
}
game = Amoeba(args)
model = TestModel01(args, 32)
model.eval()
buffer = EvalBuffer(args, 128, 128)

start = time.time()

for i in range(args.get('max_moves')):
    print('move = ', i)
    move_cycle(args)

elapsed_time = (time.time() - start) / 60.0
data_gen = args.get('max_moves') * args.get('num_table')
data_per_minute = round(data_gen / elapsed_time)

print(f"Elapsed time: {elapsed_time:.2f} minutes")
print("data generated = ", data_gen)
print("data_per_minute = ", data_per_minute)

# summary(model, input_data=position_CUDA, verbose=2)
