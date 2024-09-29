import torch
from line_profiler_pycharm import profile


class EvaluationBuffer:
    def __init__(self, args: dict):

        self.args = args
        self.buffer_size = args.get('num_table')
        self.batch_size = args.get('eval_batch_size')
        self.state_size = args.get('board_size') ** 2

        self.state_buffer = torch.zeros((self.buffer_size, self.state_size), dtype=torch.float32)
        self.idx_buffer = torch.zeros(self.buffer_size, dtype=torch.long)

        # It is a circular buffer, let us define the starting and ending indexes and the amount of data held
        self.start_idx = 0
        self.end_idx = 0
        self.num_data = 0

    @profile
    def add_states(self, state_data, idx_data):
        # Number of states to add
        num_states = state_data.size(0)
        # Calculate available space based on current data in the buffer
        available_space = self.buffer_size - self.num_data
        # If not enough space, move the start index to make room
        if num_states > available_space:
            self.start_idx = (self.start_idx + num_states - available_space) % self.buffer_size
        # Now add the new states, respecting the circular buffer logic
        # Calculate where to start inserting states
        first_chunk_size = min(num_states, self.buffer_size - self.end_idx)
        # Add the first chunk of states
        self.state_buffer[self.end_idx:self.end_idx + first_chunk_size] = state_data[:first_chunk_size]
        self.idx_buffer[self.end_idx:self.end_idx + first_chunk_size] = idx_data[:first_chunk_size]
        # If needed, wrap around and insert the second chunk at the beginning of the buffer
        if first_chunk_size < num_states:
            second_chunk_size = num_states - first_chunk_size
            self.state_buffer[0:second_chunk_size] = state_data[first_chunk_size:]
            self.idx_buffer[0:second_chunk_size] = idx_data[first_chunk_size:]
        # Update the end index
        self.end_idx = (self.end_idx + num_states) % self.buffer_size
        # Update self.num_data to reflect the number of rows the buffer is currently holding
        if num_states > available_space:
            self.num_data = self.buffer_size  # Buffer is full, set num_data to buffer size
        else:
            self.num_data += num_states  # Increase the count of data in the buffer

    @profile
    def get_states(self):
        # Check if there is enough data in the buffer to return a batch
        if self.num_data < self.batch_size:
            return None, None
        # Determine the batch size, which should be at most self.max_batch_size
        batch_size = min(self.batch_size, self.num_data)
        # Initialize state_data and idx_data to store the batch we are going to return
        state_data = torch.empty((batch_size, self.state_size), dtype=torch.float32)
        idx_data = torch.empty(batch_size, dtype=torch.long)
        # Calculate how much data we can copy without wrapping around
        first_chunk_size = min(batch_size, self.buffer_size - self.start_idx)
        # Get the first chunk of data from the buffer
        state_data[:first_chunk_size] = self.state_buffer[self.start_idx:self.start_idx + first_chunk_size]
        idx_data[:first_chunk_size] = self.idx_buffer[self.start_idx:self.start_idx + first_chunk_size]
        # If the batch wraps around, get the second chunk from the start of the buffer
        if first_chunk_size < batch_size:
            second_chunk_size = batch_size - first_chunk_size
            state_data[first_chunk_size:] = self.state_buffer[0:second_chunk_size]
            idx_data[first_chunk_size:] = self.idx_buffer[0:second_chunk_size]
        # Update the start_idx and num_data to reflect the consumption of the batch
        self.start_idx = (self.start_idx + batch_size) % self.buffer_size
        self.num_data -= batch_size

        # Return the batch of state data and index data
        return state_data, idx_data

    def empty(self):
        # Reset the start and end indices, and the number of elements in the buffer
        self.start_idx = 0
        self.end_idx = 0
        self.num_data = 0
