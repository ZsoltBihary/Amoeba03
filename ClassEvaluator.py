import torch
from line_profiler_pycharm import profile


class EvalBuffer:
    def __init__(self, args: dict, min_batch_size, max_batch_size):

        self.args = args
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.buffer_size = args.get('num_table')
        self.position_size = args.get('board_size') ** 2
        self.device = args.get('CPU_device')

        # Position buffer (holds positions that need evaluation)
        self.position_buffer = torch.zeros((self.buffer_size, self.position_size),
                                           dtype=torch.float32, device=self.device).pin_memory()
        # Metadata index buffer (holds (table_idx, node_idx) metadata for stored positions)
        self.idx_buffer = torch.zeros((self.buffer_size, 2),
                                      dtype=torch.long, device=self.device).pin_memory()

        # It is a circular buffer, let us define the starting and ending indexes ...
        self.start_idx = 0
        self.end_idx = 0
        self.num_data = 0

    @profile
    def add_positions(self, position_data, idx_data):
        # Number of positions to add
        num_positions = position_data.size(0)
        # Calculate available space based on current data in the buffer
        available_space = self.buffer_size - self.num_data
        # If not enough space, move the start index to make room
        if num_positions > available_space:
            self.start_idx = (self.start_idx + num_positions - available_space) % self.buffer_size
        # Now add the new positions, respecting the circular buffer logic
        # Calculate where to start inserting positions
        first_chunk_size = min(num_positions, self.buffer_size - self.end_idx)
        # Add the first chunk of positions
        self.position_buffer[self.end_idx:self.end_idx + first_chunk_size] = position_data[:first_chunk_size]
        self.idx_buffer[self.end_idx:self.end_idx + first_chunk_size] = idx_data[:first_chunk_size]
        # If needed, wrap around and insert the second chunk at the beginning of the buffer
        if first_chunk_size < num_positions:
            second_chunk_size = num_positions - first_chunk_size
            self.position_buffer[0:second_chunk_size] = position_data[first_chunk_size:]
            self.idx_buffer[0:second_chunk_size] = idx_data[first_chunk_size:]
        # Update the end index
        self.end_idx = (self.end_idx + num_positions) % self.buffer_size
        # Update self.num_data to reflect the number of rows the buffer is currently holding
        if num_positions > available_space:
            self.num_data = self.buffer_size  # Buffer is full, set num_data to buffer size
        else:
            self.num_data += num_positions  # Increase the count of data in the buffer

    @profile
    def get_positions(self):
        # Check if there is enough data in the buffer to return a batch
        if self.num_data < self.min_batch_size:
            return None, None
        # Determine the batch size, which should be at most self.max_batch_size
        batch_size = min(self.max_batch_size, self.num_data)
        # Initialize position_data and idx_data to store the batch we are going to return
        position_data = torch.empty((batch_size, self.position_size), dtype=torch.float32, device=self.device)
        idx_data = torch.empty((batch_size, 2), dtype=torch.long, device=self.device)
        # Calculate how much data we can copy without wrapping around
        first_chunk_size = min(batch_size, self.buffer_size - self.start_idx)
        # Get the first chunk of data from the buffer
        position_data[:first_chunk_size] = self.position_buffer[self.start_idx:self.start_idx + first_chunk_size]
        idx_data[:first_chunk_size] = self.idx_buffer[self.start_idx:self.start_idx + first_chunk_size]
        # If the batch wraps around, get the second chunk from the start of the buffer
        if first_chunk_size < batch_size:
            second_chunk_size = batch_size - first_chunk_size
            position_data[first_chunk_size:] = self.position_buffer[0:second_chunk_size]
            idx_data[first_chunk_size:] = self.idx_buffer[0:second_chunk_size]
        # Update the start_idx and num_data to reflect the consumption of the batch
        self.start_idx = (self.start_idx + batch_size) % self.buffer_size
        self.num_data -= batch_size

        # Return the batch of position data and index data
        return position_data, idx_data

    def empty(self):
        # Reset the start and end indices, and the number of elements in the buffer
        self.start_idx = 0
        self.end_idx = 0
        self.num_data = 0

    # def add_positions(self, position_data, idx_data):
    #     # Number of positions to add
    #     num_positions = position_data.size(0)
    #
    #     # Ensure we have space in the buffer by checking the capacity
    #     # Calculate how many positions are currently in the buffer
    #     if self.end_idx >= self.start_idx:
    #         current_size = self.end_idx - self.start_idx
    #     else:
    #         current_size = self.buffer_size - self.start_idx + self.end_idx
    #
    #     # Calculate available space and how many entries we would overwrite
    #     available_space = self.buffer_size - current_size
    #     if num_positions > available_space:
    #         # If there's not enough space, move the start index forward to make space
    #         self.start_idx = (self.start_idx + num_positions - available_space) % self.buffer_size
    #
    #     # Now add the new positions, respecting the circular buffer logic
    #     # Calculate where to start inserting positions
    #     first_chunk_size = min(num_positions, self.buffer_size - self.end_idx)
    #
    #     # Add the first chunk of positions
    #     self.position_buffer[self.end_idx:self.end_idx + first_chunk_size] = position_data[:first_chunk_size]
    #     self.idx_buffer[self.end_idx:self.end_idx + first_chunk_size] = idx_data[:first_chunk_size]
    #
    #     # If needed, wrap around and insert the second chunk at the beginning of the buffer
    #     if first_chunk_size < num_positions:
    #         second_chunk_size = num_positions - first_chunk_size
    #         self.position_buffer[0:second_chunk_size] = position_data[first_chunk_size:]
    #         self.idx_buffer[0:second_chunk_size] = idx_data[first_chunk_size:]
    #
    #     # Update the end index
    #     self.end_idx = (self.end_idx + num_positions) % self.buffer_size
    #
    #     # If we wrapped around and overwrote old data, the start index should be updated
    #     if num_positions >= available_space:
    #         self.start_idx = self.end_idx  # Fully overwrite means start == end

#         batch_size = positions.size(0)
#         if self.position_size + batch_size > self.max_size:
#             raise RuntimeError("Buffer overflow: not enough space to add new positions")
#
#         # Add positions to the position buffer
#         self.position_buffer[self.position_size:self.position_size + batch_size] = positions.to(self.device)
#         self.table_indices[self.position_size:self.position_size + batch_size] = table_idx
#         self.node_indices[self.position_size:self.position_size + batch_size] = node_idx
#
#         self.position_size += batch_size
#
#     def evaluate_positions(self, model):
#         """
#         Evaluate positions using the provided model if enough positions are available.
#
#         :param model: The neural network model for evaluation.
#         """
#         if self.position_size < self.threshold:
#             return  # Not enough data to evaluate according to the threshold
#
#         # Determine the number of positions to evaluate in this batch
#         eval_batch_size = min(self.batch_size, self.position_size - self.result_size)
#
#         # Prepare batch of positions for evaluation
#         batch_positions = self.position_buffer[self.result_size:self.result_size + eval_batch_size].to(self.device)
#
#         # Evaluate using the model
#         with torch.no_grad():
#             values, policies = model(batch_positions)  # Assuming the model returns (value, policy)
#
#         # Store results in the result buffer
#         self.value_buffer[self.result_size:self.result_size + eval_batch_size] = values.cpu()
#         self.policy_buffer[self.result_size:self.result_size + eval_batch_size] = policies.cpu()
#
#         # Update result size to reflect the newly evaluated positions
#         self.result_size += eval_batch_size
#
#     def get_results(self):
#         """
#         Retrieve the results (values, policies, and metadata) from the buffer.
#         This method returns the current results and shifts the buffers to remove them.
#
#         :return: Tuple of (values, policies, table_indices, node_indices).
#         """
#         if self.result_size == 0:
#             return None, None, None, None  # No results to return
#
#         # Get current batch of results
#         batch_values = self.value_buffer[:self.result_size]
#         batch_policies = self.policy_buffer[:self.result_size]
#         batch_table_idx = self.table_indices[:self.result_size]
#         batch_node_idx = self.node_indices[:self.result_size]
#
#         # Shift the buffers to remove the processed results
#         self._shift_buffers(self.result_size)
#
#         return batch_values, batch_policies, batch_table_idx, batch_node_idx
#
#     def _shift_buffers(self, shift_size):
#         """
#         Shift the buffers to discard processed entries and make room for new ones.
#         Shift both positions and results together to maintain alignment.
#
#         :param shift_size: Number of entries to shift.
#         """
#         remaining_size = self.position_size - shift_size
#         self.position_buffer[:remaining_size] = self.position_buffer[shift_size:self.position_size]
#         self.value_buffer[:remaining_size] = self.value_buffer[shift_size:self.position_size]
#         self.policy_buffer[:remaining_size] = self.policy_buffer[shift_size:self.position_size]
#         self.table_indices[:remaining_size] = self.table_indices[shift_size:self.position_size]
#         self.node_indices[:remaining_size] = self.node_indices[shift_size:self.position_size]
#
#         self.position_size -= shift_size
#         self.result_size -= shift_size
#
#     def clear(self):
#         """
#         Clear the buffer.
#         """
#         self.position_size = 0
#         self.result_size = 0
#
#     def get_current_data(self):
#         """
#         Get all current data in the buffer without clearing it.
#
#         :return: Tensors containing all current positions, values, policies, and metadata.
#         """
#         return (self.position_buffer[:self.position_size],
#                 self.value_buffer[:self.result_size],
#                 self.policy_buffer[:self.result_size],
#                 self.table_indices[:self.result_size],
#                 self.node_indices[:self.result_size])
#
# # Example Usage
# if __name__ == "__main__":
#     # Assume the model returns (values, policies) for input positions
#     class DummyModel(torch.nn.Module):
#         def forward(self, x):
#             return torch.randn(x.size(0), 1), torch.randn(x.size(0), 5)
#
#     model = DummyModel().to('cuda')
#     buffer_engine = BufferEngine(
#         max_size=800,
#         position_shape=(3,),  # Example shape for position tensor
#         value_shape=(1,),     # Example shape for value tensor
#         policy_shape=(5,),    # Example shape for policy tensor
#         threshold=200,
#         batch_size=300,
#         device='cpu'
#     )
