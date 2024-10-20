import torch


class Amoeba:
    def __init__(self, args: dict):
        """Encapsulates the rules of a general Amoeba-type game"""
        self.args = args
        self.board_size = self.args.get('board_size')
        self.win_length = self.args.get('win_length')
        self.action_size = self.board_size * self.board_size  # this is also the state size

        self.symmetry_index, self.inv_symmetry_index = self.calculate_symmetry()

    def get_empty_position(self):
        return torch.zeros(self.action_size, dtype=torch.int32)

    def get_new_position(self, position, player, action):
        new_position = position.clone()
        if new_position[action] == 0:  # Assuming 0 means the position is unoccupied
            new_position[action] = player
        else:
            raise ValueError("Invalid action: The position is already occupied.")
        return new_position

    def get_empty_positions(self, n_state: int):
        return torch.zeros((n_state, self.action_size), dtype=torch.int32)

    def get_random_positions(self, n_state: int, n_plus: int, n_minus: int) -> torch.Tensor:

        state = self.get_empty_positions(n_state)
        for i in range(n_state):
            for _ in range(n_plus):
                valid_indices = torch.nonzero(state[i] == 0)
                if valid_indices.numel() > 0:
                    # Randomly select one of these indices
                    random_index = valid_indices[torch.randint(len(valid_indices), (1,)).item()]
                    state[i, random_index] = 1
            for _ in range(n_minus):
                valid_indices = torch.nonzero(state[i] == 0)
                if valid_indices.numel() > 0:
                    # Randomly select one of these indices
                    random_index = valid_indices[torch.randint(len(valid_indices), (1,)).item()]
                    state[i, random_index] = -1
        return state

    def print_board(self, state: torch.Tensor):
        value_to_char = {-1: 'O', 0: '-', 1: 'X'}
        board = state.reshape(self.board_size, self.board_size)
        horizontal = " "
        for _ in range(self.board_size * 3):
            horizontal = horizontal + "-"
        print(horizontal)
        # Iterate through each row of the board
        for row in board:
            row_chars = [value_to_char[value.item()] for value in row]
            row_string = '  '.join(row_chars)
            print("| " + row_string + " |")
        print(horizontal)

# FROM LEGACY CODE ***************************************************************
#
    def calculate_symmetry(self):
        # set up the symmetry transformations in terms of reindexing state vectors
        # there are 8 transformations of a square board
        flip45_index = torch.zeros(self.action_size, dtype=torch.long,
                                   device=self.args.get('device'))
        flip_col_index = torch.zeros(self.action_size, dtype=torch.long,
                                     device=self.args.get('device'))
        rot90_index = torch.zeros(self.action_size, dtype=torch.long,
                                  device=self.args.get('device'))

        for i in range(self.action_size):
            row = i // self.board_size
            col = i % self.board_size
            flip45_index[i] = self.board_size * col + row
            flip_col_index[i] = self.board_size * row + (self.board_size - 1 - col)
        for i in range(self.action_size):
            rot90_index[i] = flip45_index[flip_col_index[i]]

        symmetry_index = torch.zeros((8, self.action_size), dtype=torch.long,
                                     device=self.args.get('device'))
        symmetry_index[0, :] = torch.arange(self.action_size)
        symmetry_index[1, :] = rot90_index
        for i in range(self.action_size):
            symmetry_index[2, i] = rot90_index[symmetry_index[1, i]]
        for i in range(self.action_size):
            symmetry_index[3, i] = rot90_index[symmetry_index[2, i]]
        symmetry_index[4, :] = flip45_index
        for i in range(self.action_size):
            symmetry_index[5, i] = rot90_index[symmetry_index[4, i]]
        for i in range(self.action_size):
            symmetry_index[6, i] = rot90_index[symmetry_index[5, i]]
        for i in range(self.action_size):
            symmetry_index[7, i] = rot90_index[symmetry_index[6, i]]
        # most transformations happen to be equal to their inverse, except two
        inv_symmetry_index = symmetry_index.clone()
        inv_symmetry_index[1, :] = symmetry_index[3, :]
        inv_symmetry_index[3, :] = symmetry_index[1, :]

        return symmetry_index, inv_symmetry_index

    def get_symmetry_states(self, state):
        sym_states = state[..., self.symmetry_index[0, :]]
        for i in range(1, 8):
            sym_states = torch.cat((sym_states, state[..., self.symmetry_index[i, :]]), dim=0)
            # sym_states.append(state[..., self.symmetry_index[i, :]])
        #
        # for sym in self.symmetry_index:
        #     sym_states.append(copy(state[sym]))
        return sym_states
