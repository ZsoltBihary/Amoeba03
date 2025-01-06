import torch
from ClassAmoeba import Amoeba
from ClassModel import TerminalCheck01, TrivialModel01, TrivialModel02, SimpleModel01, DeepMindModel01
from ClassSearchEngine import SearchEngine
# from ClassGamePlay import GamePLay
# from ClassEvaluator import EvaluationBuffer
# from torchinfo import summary
# from line_profiler_pycharm import profile
# from ClassTree import Tree
# import time

# Collect parameters in a dictionary
args = {
    'board_size': 11,
    'win_length': 5,
    'CUDA_device': 'cuda' if torch.cuda.is_available() else 'cpu',
    # 'CUDA_device': 'cpu',
    # 'num_leaf': 8,
    # 'num_branch': 2,
    'num_MC': 100000,
    'num_child': 20,
    'num_table': 2,
    'num_agent': 2000,
    # 'num_moves': 5,
    'leaf_buffer_size': 20000,
    'eval_batch_size': 400,
    'res_channels': 64,
    'hid_channels': 32,
    'num_res': 8,
    'policy_hid_channels': 64,
    'value_hid_dim': 128
}

game = Amoeba(args)

terminal_check = TerminalCheck01(args)
# model = TrivialModel01(args)
model = TrivialModel02(args)
# model = SimpleModel01(args)
# model = DeepMindModel01(args)
model.eval()
engine = SearchEngine(args, game, terminal_check, model)

player = -torch.ones(args.get('num_table'), dtype=torch.int32)
position = game.get_random_positions(n_state=args.get('num_table'), n_plus=1, n_minus=0)
table = torch.arange(args.get('num_table'))

for i in range(40):
    print(i)
    game.print_board(position[0])
    move_policy, position_value = engine.analyze(player, position)

    print("position value = ", position_value[0])
    # print("move policy:\n", torch.round(100 * move_policy[0, :].view(game.board_size, -1)))
    move = torch.argmax(move_policy, dim=1)
    position[table, move] = player
    player *= -1
    a = 42
game.print_board(position[0])

a = 42
