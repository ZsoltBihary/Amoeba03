import torch
from line_profiler_pycharm import profile


class PathManager:
    def __init__(self, num_agent, max_depth, action_size):

        self.num_agent = num_agent
        self.max_depth = max_depth
        self.action_size = action_size

        self.is_active = torch.ones(self.num_agent, dtype=torch.bool)

        self.table_idx = torch.zeros(self.num_agent, dtype=torch.long)
        self.node_idx = torch.zeros(self.num_agent, dtype=torch.long)
        self.depth = torch.zeros(self.num_agent, dtype=torch.long)

        self.player = torch.zeros(self.num_agent, dtype=torch.int32)
        self.position = torch.zeros((self.num_agent, self.action_size), dtype=torch.int32)
        self.path = torch.zeros((self.num_agent, self.max_depth), dtype=torch.long)

    def get_active_idx(self):
        return self.table_idx[self.is_active], self.node_idx[self.is_active]

        # # Keep track of tables under search, current node for search, search count
        # self.search_table_idx = torch.arange(self.num_table, dtype=torch.long)
        # self.search_node_idx = torch.zeros(self.num_table, dtype=torch.long)
        # self.search_count = torch.zeros(self.num_table, dtype=torch.int32)
        # self.move_count = torch.zeros(self.num_table, dtype=torch.int32)
        # # Keep track of tables under evaluation, and evaluation status
        # self.evaluated_table_idx = torch.arange(args.get('num_table'), dtype=torch.long)
        # self.evaluation_is_on = False
        # # Keep track of root player, node player, and current position, node position on all tables
        # self.player = torch.ones(self.num_table, dtype=torch.int32)
        # self.position = torch.zeros((self.num_table, self.action_size), dtype=torch.int32)
        # self.node_player = torch.ones(self.num_table, dtype=torch.int32)
        # self.node_position = torch.zeros((self.num_table, self.action_size), dtype=torch.int32)
