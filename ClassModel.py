import torch
import torch.nn as nn
import torch.nn.functional as F
from line_profiler_pycharm import profile


class TestModel01(nn.Module):
    # def __init__(self, args: dict, input_sim_shape, output_sim_shape):
    def __init__(self, args: dict, hidden_dim):
        super(TestModel01, self).__init__()
        self.board_size = args.get('board_size')
        self.device = args.get('CUDA_device')

        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=hidden_dim,
            kernel_size=5, stride=1, padding=2, bias=False
        )

        self.conv2 = nn.Conv2d(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            kernel_size=3, stride=1, padding=1, bias=False
        )

        self.conv3 = nn.Conv2d(
            in_channels=hidden_dim,
            out_channels=1,
            kernel_size=3, stride=1, padding=1, bias=False
        )

        self.to(self.device)

    @profile
    def forward(self, position):
        x = position.view(position.shape[0], 1, self.board_size, self.board_size)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv2(x)
        x = self.conv2(x)
        x = self.conv2(x)
        x = self.conv2(x)
        x = self.conv2(x)
        x = self.conv2(x)
        x = self.conv2(x)
        x = self.conv2(x)
        x = self.conv2(x)
        x = self.conv2(x)
        x = self.conv2(x)
        x = self.conv2(x)
        x = self.conv2(x)
        x = self.conv2(x)
        x = self.conv2(x)
        out = self.conv3(x)

        return out

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
# # # class ResBlock(nn.Module):
# # #     def __init__(self, in_channels, intermediate_channels):
# # #         super(ResBlock, self).__init__()
# # #         self.in_channels = in_channels
# # #         self.intermediate_channels = intermediate_channels
# # #
# # #         self.conv1 = SymmetricConv(in_channels, intermediate_channels)
# # #         self.conv2 = SymmetricConv(intermediate_channels, in_channels)
# # #
# # #         # Adding BatchNorm after each SymmetricConv
# # #         self.batch_norm1 = nn.BatchNorm2d(intermediate_channels)
# # #         self.batch_norm2 = nn.BatchNorm2d(in_channels)
# # #
# # #         self.relu = nn.ReLU()
# # #
# # #     @profile
# # #     def forward(self, x):
# # #         identity = x  # Shortcut connection
# # #
# # #         out = self.conv1(x)
# # #         out = self.batch_norm1(out)  # BatchNorm after first SymmetricConv
# # #         out = self.relu(out)
# # #
# # #         out = self.conv2(out)
# # #         out = self.batch_norm2(out)  # BatchNorm after second SymmetricConv
# # #
# # #         out = identity + 0.1 * out  # Residual addition
# # #         out = self.relu(out)
# # #
# # #         return out
