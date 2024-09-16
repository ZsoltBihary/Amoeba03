import torch


class BufferEngine:
    def __init__(self, max_size, position_shape, value_shape, policy_shape, threshold, batch_size, device='cpu'):
        """
        Initialize the BufferEngine with configurations for positions, values, policies, and associated metadata.

        :param max_size: Maximum number of entries the buffer can hold.
        :param position_shape: Shape of each position tensor (e.g., (state_size,)).
        :param value_shape: Shape of each value tensor (e.g., (1,)).
        :param policy_shape: Shape of each policy tensor (e.g., (action_size,)).
        :param threshold: The minimum number of entries before processing is triggered.
        :param batch_size: Number of entries to process at a time.
        :param device: Device where the position tensors are stored ('cpu' or 'cuda').
        """
        self.max_size = max_size
        self.threshold = threshold
        self.batch_size = batch_size
        self.device = device

        # Position buffer (holds positions that need evaluation)
        self.position_buffer = torch.zeros((max_size,) + position_shape, device=device)

        # Result buffer (temporarily holds evaluation results: values and policies)
        self.value_buffer = torch.zeros((max_size,) + value_shape, device=device)
        self.policy_buffer = torch.zeros((max_size,) + policy_shape, device=device)

        # Metadata to track table and node indices (stored on CPU for minimal transfer)
        self.table_indices = torch.zeros(max_size, dtype=torch.long)
        self.node_indices = torch.zeros(max_size, dtype=torch.long)

        # Sizes of buffers
        self.position_size = 0  # Number of positions currently in the buffer
        self.result_size = 0    # Number of results currently in the buffer

    def add_positions(self, positions, table_idx, node_idx):
        """
        Add new positions to the buffer along with their metadata.

        :param positions: Tensor of shape (batch_size, position_shape).
        :param table_idx: Tensor of shape (batch_size,) indicating table indices.
        :param node_idx: Tensor of shape (batch_size,) indicating node indices.
        """
        batch_size = positions.size(0)
        if self.position_size + batch_size > self.max_size:
            raise RuntimeError("Buffer overflow: not enough space to add new positions")

        # Add positions to the position buffer
        self.position_buffer[self.position_size:self.position_size + batch_size] = positions.to(self.device)
        self.table_indices[self.position_size:self.position_size + batch_size] = table_idx
        self.node_indices[self.position_size:self.position_size + batch_size] = node_idx

        self.position_size += batch_size

    def evaluate_positions(self, model):
        """
        Evaluate positions using the provided model if enough positions are available.

        :param model: The neural network model for evaluation.
        """
        if self.position_size < self.threshold:
            return  # Not enough data to evaluate according to the threshold

        # Determine the number of positions to evaluate in this batch
        eval_batch_size = min(self.batch_size, self.position_size - self.result_size)

        # Prepare batch of positions for evaluation
        batch_positions = self.position_buffer[self.result_size:self.result_size + eval_batch_size].to(self.device)

        # Evaluate using the model
        with torch.no_grad():
            values, policies = model(batch_positions)  # Assuming the model returns (value, policy)

        # Store results in the result buffer
        self.value_buffer[self.result_size:self.result_size + eval_batch_size] = values.cpu()
        self.policy_buffer[self.result_size:self.result_size + eval_batch_size] = policies.cpu()

        # Update result size to reflect the newly evaluated positions
        self.result_size += eval_batch_size

    def get_results(self):
        """
        Retrieve the results (values, policies, and metadata) from the buffer.
        This method returns the current results and shifts the buffers to remove them.

        :return: Tuple of (values, policies, table_indices, node_indices).
        """
        if self.result_size == 0:
            return None, None, None, None  # No results to return

        # Get current batch of results
        batch_values = self.value_buffer[:self.result_size]
        batch_policies = self.policy_buffer[:self.result_size]
        batch_table_idx = self.table_indices[:self.result_size]
        batch_node_idx = self.node_indices[:self.result_size]

        # Shift the buffers to remove the processed results
        self._shift_buffers(self.result_size)

        return batch_values, batch_policies, batch_table_idx, batch_node_idx

    def _shift_buffers(self, shift_size):
        """
        Shift the buffers to discard processed entries and make room for new ones.
        Shift both positions and results together to maintain alignment.

        :param shift_size: Number of entries to shift.
        """
        remaining_size = self.position_size - shift_size
        self.position_buffer[:remaining_size] = self.position_buffer[shift_size:self.position_size]
        self.value_buffer[:remaining_size] = self.value_buffer[shift_size:self.position_size]
        self.policy_buffer[:remaining_size] = self.policy_buffer[shift_size:self.position_size]
        self.table_indices[:remaining_size] = self.table_indices[shift_size:self.position_size]
        self.node_indices[:remaining_size] = self.node_indices[shift_size:self.position_size]

        self.position_size -= shift_size
        self.result_size -= shift_size

    def clear(self):
        """
        Clear the buffer.
        """
        self.position_size = 0
        self.result_size = 0

    def get_current_data(self):
        """
        Get all current data in the buffer without clearing it.

        :return: Tensors containing all current positions, values, policies, and metadata.
        """
        return (self.position_buffer[:self.position_size],
                self.value_buffer[:self.result_size],
                self.policy_buffer[:self.result_size],
                self.table_indices[:self.result_size],
                self.node_indices[:self.result_size])

# Example Usage
if __name__ == "__main__":
    # Assume the model returns (values, policies) for input positions
    class DummyModel(torch.nn.Module):
        def forward(self, x):
            return torch.randn(x.size(0), 1), torch.randn(x.size(0), 5)

    model = DummyModel().to('cuda')
    buffer_engine = BufferEngine(
        max_size=800,
        position_shape=(3,),  # Example shape for position tensor
        value_shape=(1,),     # Example shape for value tensor
        policy_shape=(5,),    # Example shape for policy tensor
        threshold=200,
        batch_size=300,
        device='cpu'
    )
