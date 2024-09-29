import torch
import torch.nn as nn
import torch.nn.functional as F
from line_profiler_pycharm import profile


class CustomConvLayer(nn.Module):
    def __init__(self, kernel, padding):
        super(CustomConvLayer, self).__init__()
        # Register the kernel as a buffer (non-trainable)
        self.register_buffer('kernel', kernel)
        self.padding = padding

    def forward(self, x):
        # Apply the convolution using the kernel
        # Since this is a predefined kernel, we do not have bias
        return F.conv2d(x, self.kernel, padding=self.padding)  # padding=2 to keep input size


class TerminalCheck01(nn.Module):
    # def __init__(self, args: dict, input_sim_shape, output_sim_shape):
    def __init__(self, args: dict):
        super(TerminalCheck01, self).__init__()
        self.board_size = args.get('board_size')
        self.device = args.get('CUDA_device')

        kernel = torch.tensor([
            [[[0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0],
              [1, 1, 1, 1, 1],
              [0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0]]],

            [[[0, 0, 1, 0, 0],
              [0, 0, 1, 0, 0],
              [0, 0, 1, 0, 0],
              [0, 0, 1, 0, 0],
              [0, 0, 1, 0, 0]]],

            [[[1, 0, 0, 0, 0],
              [0, 1, 0, 0, 0],
              [0, 0, 1, 0, 0],
              [0, 0, 0, 1, 0],
              [0, 0, 0, 0, 1]]],

            [[[0, 0, 0, 0, 1],
              [0, 0, 0, 1, 0],
              [0, 0, 1, 0, 0],
              [0, 1, 0, 0, 0],
              [1, 0, 0, 0, 0]]]
        ], dtype=torch.float32)

        self.dir_conv = CustomConvLayer(kernel, 2)
        self.to(self.device)

    @profile
    def forward(self, state_CUDA):
        x = state_CUDA.view(state_CUDA.shape[0], 1, self.board_size, self.board_size)
        dir_sum = self.dir_conv(x)
        dir_max = torch.amax(dir_sum, dim=(1, 2, 3))
        dir_min = torch.amin(dir_sum, dim=(1, 2, 3))
        sum_abs = torch.sum(torch.abs(state_CUDA), dim=1) + 0.1

        return torch.stack([dir_max, dir_min, sum_abs], dim=1)


class InputBlock(nn.Module):
    def __init__(self, res_channels):
        super(InputBlock, self).__init__()

        self.conv = nn.Conv2d(
            in_channels=3, out_channels=res_channels,
            kernel_size=5, stride=1, padding=2, bias=False
        )
        self.batch_norm = nn.BatchNorm2d(res_channels)
        self.relu = nn.ReLU()

    @profile
    def forward(self, x):

        out = self.conv(x)
        out = self.batch_norm(out)  # BatchNorm after first Conv
        out = self.relu(out)
        return out


class ResBlock(nn.Module):
    def __init__(self, res_channels, hid_channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=res_channels, out_channels=hid_channels,
            kernel_size=3, stride=1, padding=1, bias=False
        )
        self.conv2 = nn.Conv2d(
            in_channels=hid_channels, out_channels=res_channels,
            kernel_size=3, stride=1, padding=1, bias=False
        )
        self.batch_norm1 = nn.BatchNorm2d(hid_channels)
        self.batch_norm2 = nn.BatchNorm2d(res_channels)
        self.relu = nn.ReLU()

    @profile
    def forward(self, x):
        identity = x  # Shortcut connection
        out = self.conv1(x)
        out = self.batch_norm1(out)  # BatchNorm after first Conv
        out = self.relu(out)
        out = self.conv2(out)
        out = self.batch_norm2(out)  # BatchNorm after second Conv
        out += identity  # Residual addition
        out = self.relu(out)
        return out


class PolicyHead01(nn.Module):
    def __init__(self, res_channels, hid_channels):
        super(PolicyHead01, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=res_channels, out_channels=hid_channels,
            kernel_size=5, stride=1, padding=2, bias=False
        )
        self.conv2 = nn.Conv2d(
            in_channels=hid_channels, out_channels=1,
            kernel_size=5, stride=1, padding=2, bias=False
        )
        self.batch_norm1 = nn.BatchNorm2d(hid_channels)
        # self.batch_norm2 = nn.BatchNorm2d(res_channels)
        self.relu = nn.ReLU()

    @profile
    def forward(self, x):
        x = self.conv1(x)
        x = self.batch_norm1(x)  # BatchNorm after first Conv
        x = self.relu(x)
        x = self.conv2(x)
        logit = x.squeeze(1).view(x.shape[0], -1)
        # These are the logit probabilities
        return logit


class ValueHead01(nn.Module):
    def __init__(self, res_channels, state_size, hid_dim):
        super(ValueHead01, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=res_channels, out_channels=1,
            kernel_size=1, stride=1, padding=0, bias=False
        )
        self.batch_norm1 = nn.BatchNorm2d(1)
        self.relu = nn.ReLU()
        self.lin1 = nn.Linear(in_features=state_size, out_features=hid_dim,
                              bias=True)
        # self.relu = nn.ReLU()
        self.lin2 = nn.Linear(in_features=hid_dim, out_features=1,
                              bias=True)
        self.tanh = nn.Tanh()

    @profile
    def forward(self, x):
        x = self.conv1(x)
        x = self.batch_norm1(x)  # BatchNorm after first Conv
        x = self.relu(x)
        # board = state_CUDA.view(state_CUDA.shape[0], self.board_size, self.board_size)
        x = self.lin1(x.view(x.shape[0], -1))
        x = self.relu(x)
        x = self.lin2(x)
        value = self.tanh(x)
        return value


class DeepMindModel01(nn.Module):
    def __init__(self, args: dict):
        super(DeepMindModel01, self).__init__()
        self.board_size = args.get('board_size')
        self.res_channels = args.get('res_channels')
        self.hid_channels = args.get('hid_channels')
        self.num_res = args.get('num_res')
        self.policy_hid_channels = args.get('policy_hid_channels')
        self.value_hid_dim = args.get('value_hid_dim')
        self.device = args.get('CUDA_device')
        # First convolution (input layer)
        self.input_conv = InputBlock(self.res_channels)
        # Tower of residual blocks
        self.res_tower = nn.ModuleList([ResBlock(self.res_channels, self.hid_channels)
                                        for _ in range(self.num_res)])
        self.policy_head = PolicyHead01(self.res_channels, self.policy_hid_channels)
        self.value_head = ValueHead01(self.res_channels,
                                      self.board_size ** 2,
                                      self.value_hid_dim)
        self.to(self.device)

    @profile
    def forward(self, state_CUDA):
        # reshape and one-hot-encode the input
        board = state_CUDA.view(state_CUDA.shape[0], self.board_size, self.board_size)
        board_plus = torch.clamp(board, min=0, max=1)
        board_minus = -torch.clamp(board, min=-1, max=0)
        board_zero = 1 - board_plus - board_minus
        x = torch.stack([board_zero, board_plus, board_minus], dim=1)
        # convolution on the encoded input
        x = self.input_conv(x)
        # residual tower
        for res_block in self.res_tower:
            x = res_block(x)
        # policy head
        logit = self.policy_head(x)
        # value head
        value = self.value_head(x)

        return logit, value


# *****************************************************************************************
# LEGACY EXAMPLE CODE *********************************************************************
# *****************************************************************************************
# class ResTower(nn.Module):
#     def __init__(self, in_channels, intermediate_channels, n_blocks):
#         super(ResTower, self).__init__()
#         self.in_channels = in_channels
#         self.intermediate_channels = intermediate_channels
#         self.n_blocks = n_blocks
#
#         # Create a list of residual blocks
#         self.blocks = nn.ModuleList([
#             ResBlock(in_channels, intermediate_channels) for _ in range(n_blocks)
#         ])
#
#     @profile
#     def forward(self, x):
#         for block in self.blocks:
#             x = block(x)
#         return x
#
#
# class SpatiallySymmetricBiasLayer(nn.Module):
#     def __init__(self, board_size: int, num_feat: int):
#         super(SpatiallySymmetricBiasLayer, self).__init__()
#
#         self.board_size = board_size
#         self.num_feat = num_feat
#
#         # Calculate the number of rings (num_mask) based on the board size
#         self.num_mask = (board_size - 1) // 2
#
#         # Trainable parameter
#         self.masked_bias = nn.Parameter(torch.zeros(self.num_mask, num_feat))
#
#         # Precompute the masks (shape: (num_mask, board_size, board_size))
#         self.masks = self.create_masks()
#         print(self.masks)
#
#     def create_masks(self) -> torch.Tensor:
#         """
#         This function creates and returns the 3D tensor of masks with shape
#         (num_mask, board_size, board_size). The masks represent concentric
#         rectangular "rings" around the center of the board.
#         """
#         full_masks = torch.zeros(self.num_mask + 1, self.board_size, self.board_size)
#         masks = torch.zeros(self.num_mask, self.board_size, self.board_size)
#
#         # Generate full squares for each ring level
#         for i in range(self.num_mask + 1):
#             size = self.board_size - 2 * i
#             if size > 0:
#                 full_square = torch.ones(size, size)
#                 full_masks[i, i:self.board_size - i, i:self.board_size - i] = full_square
#
#         # Calculate the difference between consecutive squares to get rings
#         for i in range(self.num_mask):
#
#             masks[i] = full_masks[self.num_mask-i-1] - full_masks[self.num_mask-i]
#
#         return masks
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         x: Input tensor of shape (batch_size, num_feat, board_size, board_size)
#         """
#         batch_size = x.size(0)
#
#         # Compute the bias using einsum
#         bias = torch.einsum('mf,mbw->fbw', self.masked_bias, self.masks)
#         bias = bias.unsqueeze(0)  # Add batch dimension (1, num_feat, board_size, board_size)
#         bias = bias.expand(batch_size, -1, -1, -1)  # Expand to match the batch size
#
#         # Add the bias to the input tensor
#         x = x + bias
#
# #         return x#

#
# @profile
# def encode01(args: dict, state_CUDA):
#
#     device_CUDA = args.get('CUDA_device')
#     board_size = args.get('board_size')
#
#     # board = state_CUDA.view(-1, board_size, board_size).to(torch.long)
#     board = state_CUDA.reshape(-1, board_size, board_size)   # already of dtype torch.long
#     board_plus = torch.clamp(board, min=0, max=1)
#     board_minus = torch.clamp(board, min=-1, max=0)
#     board_zero = 1 - board_plus + board_minus
#     sum_plus = torch.ones((board.shape[0], 4, board_size, board_size),
#                           dtype=torch.long, device=device_CUDA)
#     sum_minus = -torch.ones((board.shape[0], 4, board_size, board_size),
#                             dtype=torch.long, device=device_CUDA)
#     # horizontal ********************************
#     sum_plus[:, 0, :, 2:-2] = (board_plus[:, :, :-4] + board_plus[:, :, 1:-3] +
#                                board_plus[:, :, 2:-2] +
#                                board_plus[:, :, 3:-1] + board_plus[:, :, 4:])
#     sum_minus[:, 0, :, 2:-2] = (board_minus[:, :, :-4] + board_minus[:, :, 1:-3] +
#                                 board_minus[:, :, 2:-2] +
#                                 board_minus[:, :, 3:-1] + board_minus[:, :, 4:])
#     # vertical ********************************
#     sum_plus[:, 1, 2:-2, :] = (board_plus[:, :-4, :] + board_plus[:, 1:-3, :] +
#                                board_plus[:, 2:-2, :] +
#                                board_plus[:, 3:-1, :] + board_plus[:, 4:, :])
#     sum_minus[:, 1, 2:-2, :] = (board_minus[:, :-4, :] + board_minus[:, 1:-3, :] +
#                                 board_minus[:, 2:-2, :] +
#                                 board_minus[:, 3:-1, :] + board_minus[:, 4:, :])
#     # diagonal1 ********************************
#     sum_plus[:, 2, 2:-2, 2:-2] = (board_plus[:, :-4, :-4] + board_plus[:, 1:-3, 1:-3] +
#                                   board_plus[:, 2:-2, 2:-2] +
#                                   board_plus[:, 3:-1, 3:-1] + board_plus[:, 4:, 4:])
#     sum_minus[:, 2, 2:-2, 2:-2] = (board_minus[:, :-4, :-4] + board_minus[:, 1:-3, 1:-3] +
#                                    board_minus[:, 2:-2, 2:-2] +
#                                    board_minus[:, 3:-1, 3:-1] + board_minus[:, 4:, 4:])
#     # diagonal2 ********************************
#     sum_plus[:, 3, 2:-2, 2:-2] = (board_plus[:, :-4, 4:] + board_plus[:, 1:-3, 3:-1] +
#                                   board_plus[:, 2:-2, 2:-2] +
#                                   board_plus[:, 3:-1, 1:-3] + board_plus[:, 4:, :-4])
#     sum_minus[:, 3, 2:-2, 2:-2] = (board_minus[:, :-4, 4:] + board_minus[:, 1:-3, 3:-1] +
#                                    board_minus[:, 2:-2, 2:-2] +
#                                    board_minus[:, 3:-1, 1:-3] + board_minus[:, 4:, :-4])
#
#     alive = (sum_plus * sum_minus >= 0).to(torch.long)
#     sum_index = (sum_plus + sum_minus + 6) * alive
#     line_type = F.one_hot(sum_index, num_classes=12)
#     board_type = torch.stack([board_zero, board_plus, -board_minus], dim=3)
#
#     return board_type, line_type

