import torch
from torch.nn import Linear
from torch_geometric.nn import GraphConv, SAGEConv, TAGConv


class StateGNNEncoderConvEdgeAttrCompact(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = TAGConv(5, hidden_channels, 2)
        self.conv2 = TAGConv(hidden_channels, hidden_channels, 3)  # TAGConv
        self.conv3 = GraphConv((-1, -1), hidden_channels)  # SAGEConv
        self.conv4 = SAGEConv((-1, -1), hidden_channels)
        self.lin = Linear(hidden_channels, out_channels)

    def forward(
        self,
        game_x,
        state_x,
        edge_index_v_v,
        edge_index_history_v_s,
        edge_attr_history_v_s,
        edge_index_in_v_s,
        edge_index_s_s,
    ):
        game_x = self.conv1(
            game_x,
            edge_index_v_v,
        ).relu()

        state_x = self.conv3(
            (game_x, state_x),
            edge_index_history_v_s,
            edge_attr_history_v_s,
        ).relu()

        state_x = self.conv4(
            (game_x, state_x),
            edge_index_in_v_s,
        ).relu()

        state_x = self.conv2(
            state_x,
            edge_index_s_s,
        ).relu()

        return self.lin(state_x)


class StateModelEncoder(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.state_encoder = StateGNNEncoderConvEdgeAttrCompact(
            hidden_channels, out_channels
        )

    def forward(
        self,
        game_x,
        state_x,
        edge_index_v_v,
        edge_index_history_v_s,
        edge_attr_history_v_s,
        edge_index_in_v_s,
        edge_index_s_s,
    ):
        return self.state_encoder(
            game_x,
            state_x,
            edge_index_v_v,
            edge_index_history_v_s,
            edge_attr_history_v_s,
            edge_index_in_v_s,
            edge_index_s_s,
        )
