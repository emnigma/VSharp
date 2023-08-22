import torch
from torch.nn import Linear
from torch_geometric.nn import TAGConv, GraphConv, SAGEConv


class StateModelEncoder(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = TAGConv(5, hidden_channels, 2)
        self.conv2 = TAGConv(hidden_channels, hidden_channels, 3)  # TAGConv
        self.conv3 = GraphConv((-1, -1), hidden_channels)  # SAGEConv
        self.conv4 = SAGEConv((-1, -1), hidden_channels)
        self.lin = Linear(hidden_channels, out_channels)

    def convert_to_single_tensor(
        self, x_dict, edge_index_dict, edge_attr, fixed_size_tensor=False
    ):
        self.sizes_dict = {}
        self.vertices_n = x_dict["game_vertex"].size()[0]
        self.states_n = x_dict["state_vertex"].size()[0]
        vertex_to_vertex_n = edge_index_dict["game_vertex to game_vertex"].size()[1]
        state_to_state_n = edge_index_dict[
            "state_vertex parent_of state_vertex"
        ].size()[1]
        vertex_state_history_n = edge_attr["game_vertex history state_vertex"].size()[0]
        vertex_state_in_n = edge_index_dict["game_vertex in state_vertex"].size()[1]

        # fixed rows number
        rows_n = x_dict["game_vertex"].size()[1] + x_dict["state_vertex"].size()[1] + 14
        # calculate final tensor sizes
        if fixed_size_tensor:
            columns_n = 10000
        else:
            columns_n = max(
                self.states_n,
                self.vertices_n,
                vertex_to_vertex_n,
                state_to_state_n,
                vertex_state_history_n,
                vertex_state_in_n,
            )

        final_tensor = torch.zeros(rows_n, columns_n)
        game_param_n = x_dict["game_vertex"].size()[1]
        self.sizes_dict["game_vertex"] = (0, game_param_n)
        row_start = 0
        final_tensor[row_start:game_param_n, : self.vertices_n] = x_dict[
            "game_vertex"
        ].transpose(0, 1)
        row_start += game_param_n

        state_param_n = x_dict["state_vertex"].size()[1]
        self.sizes_dict["state_vertex"] = (row_start, state_param_n)
        final_tensor[row_start : row_start + state_param_n, : self.states_n] = x_dict[
            "state_vertex"
        ].transpose(0, 1)
        row_start += state_param_n

        self.sizes_dict["game_vertex history state_vertex"] = (
            row_start,
            vertex_state_history_n,
        )
        final_tensor[
            row_start : row_start + 2, :vertex_state_history_n
        ] = edge_index_dict["game_vertex history state_vertex"]
        row_start += 2

        self.sizes_dict["attr game_vertex history state_vertex"] = (
            row_start,
            vertex_state_history_n,
        )
        final_tensor[row_start : row_start + 1, :vertex_state_history_n] = edge_attr[
            "game_vertex history state_vertex"
        ].transpose(0, 1)
        row_start += 1

        self.sizes_dict["state_vertex history game_vertex"] = (
            row_start,
            vertex_state_history_n,
        )
        final_tensor[
            row_start : row_start + 2, :vertex_state_history_n
        ] = edge_index_dict["state_vertex history game_vertex"]
        row_start += 2

        self.sizes_dict["attr state_vertex history game_vertex"] = (
            row_start,
            vertex_state_history_n,
        )
        final_tensor[row_start : row_start + 1, :vertex_state_history_n] = edge_attr[
            "state_vertex history game_vertex"
        ].transpose(0, 1)
        row_start += 1

        self.sizes_dict["game_vertex in state_vertex"] = (row_start, vertex_state_in_n)
        final_tensor[row_start : row_start + 2, :vertex_state_in_n] = edge_index_dict[
            "game_vertex in state_vertex"
        ]
        row_start += 2

        self.sizes_dict["state_vertex in game_vertex"] = (row_start, vertex_state_in_n)
        final_tensor[row_start : row_start + 2, :vertex_state_in_n] = edge_index_dict[
            "state_vertex in game_vertex"
        ]
        row_start += 2

        self.sizes_dict["game_vertex to game_vertex"] = (row_start, vertex_to_vertex_n)
        final_tensor[row_start : row_start + 2, :vertex_to_vertex_n] = edge_index_dict[
            "game_vertex to game_vertex"
        ]
        row_start += 2

        self.sizes_dict["state_vertex parent_of state_vertex"] = (
            row_start,
            state_to_state_n,
        )
        final_tensor[row_start : row_start + 2, :state_to_state_n] = edge_index_dict[
            "state_vertex parent_of state_vertex"
        ]
        return final_tensor

    def forward(self, x):
        game_x = x[
            self.sizes_dict["game_vertex"][0] : self.sizes_dict["game_vertex"][1],
            : self.vertices_n,
        ].transpose(0, 1)
        start = self.sizes_dict["game_vertex to game_vertex"][0]
        vertex_to_vertex = x[
            start : start + 2, : self.sizes_dict["game_vertex to game_vertex"][1]
        ].to(torch.int64)

        game_x = self.conv1(
            game_x,
            vertex_to_vertex,
        ).relu()

        start = self.sizes_dict["state_vertex"][0]

        state_x = x[
            start : start + self.sizes_dict["state_vertex"][1], : self.states_n
        ].transpose(0, 1)

        start = self.sizes_dict["game_vertex history state_vertex"][0]
        history_index = x[
            start : start + 2, : self.sizes_dict["game_vertex history state_vertex"][1]
        ].to(torch.int64)
        start += 2
        history_attr = x[
            start : start + 1,
            : self.sizes_dict["attr game_vertex history state_vertex"][1],
        ].transpose(0, 1)

        state_x = self.conv3(
            (game_x, state_x),
            history_index,
            history_attr,
        ).relu()

        start = self.sizes_dict["game_vertex in state_vertex"][0]
        in_index = x[
            start : start + 2, : self.sizes_dict["game_vertex in state_vertex"][1]
        ].to(torch.int64)

        state_x = self.conv4(
            (game_x, state_x),
            in_index,
        ).relu()

        state_edges_n = self.sizes_dict["state_vertex parent_of state_vertex"][1]
        # if state_edges_n != 0: is it needed here?
        start = self.sizes_dict["state_vertex parent_of state_vertex"][0]
        parent_index = x[start : start + 2, :state_edges_n].to(torch.int64)

        state_x = self.conv2(
            state_x,
            parent_index,
        ).relu()

        return self.lin(state_x)
