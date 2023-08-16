import json
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import onnx
import onnxruntime
import torch
from dataclasses_json import dataclass_json
from torch.nn import Linear
from torch_geometric.data import HeteroData
from torch_geometric.nn import GraphConv, Linear, SAGEConv, TAGConv


@dataclass_json
@dataclass(slots=True)
class StateHistoryElem:
    GraphVertexId: int
    NumOfVisits: int


@dataclass_json
@dataclass(slots=True)
class State:
    Id: int
    Position: int
    PredictedUsefulness: float
    PathConditionSize: int
    VisitedAgainVertices: int
    VisitedNotCoveredVerticesInZone: int
    VisitedNotCoveredVerticesOutOfZone: int
    History: list[StateHistoryElem]
    Children: list[int]

    def __hash__(self) -> int:
        return self.Id.__hash__()


@dataclass_json
@dataclass(slots=True)
class GameMapVertex:
    Uid: int
    Id: int
    InCoverageZone: bool
    BasicBlockSize: int
    CoveredByTest: bool
    VisitedByState: bool
    TouchedByState: bool
    States: list[int]


@dataclass_json
@dataclass(slots=True)
class GameEdgeLabel:
    Token: int


@dataclass_json
@dataclass(slots=True)
class GameMapEdge:
    VertexFrom: int
    VertexTo: int
    Label: GameEdgeLabel


@dataclass_json
@dataclass(slots=True)
class GameState:
    GraphVertices: list[GameMapVertex]
    States: list[State]
    Map: list[GameMapEdge]


def convert_input_to_tensor(input: GameState) -> Tuple[HeteroData, Dict[int, int]]:
    """
    Converts game env to tensors
    """
    graphVertices = input.GraphVertices
    game_states = input.States
    game_edges = input.Map
    data = HeteroData()
    nodes_vertex_set = set()
    nodes_state_set = set()
    nodes_vertex = []
    nodes_state = []
    edges_index_v_v = []
    edges_index_s_s = []
    edges_index_s_v_in = []
    edges_index_v_s_in = []
    edges_index_s_v_history = []
    edges_index_v_s_history = []
    edges_attr_v_v = []

    edges_attr_s_v = []
    edges_attr_v_s = []

    state_map: Dict[int, int] = {}  # Maps real state id to its index
    vertex_map: Dict[int, int] = {}  # Maps real vertex id to its index
    vertex_index = 0
    state_index = 0

    # vertex nodes
    for v in graphVertices:
        vertex_id = v.Id
        if vertex_id not in vertex_map:
            vertex_map[vertex_id] = vertex_index  # maintain order in tensors
            vertex_index = vertex_index + 1
            nodes_vertex.append(
                np.array(
                    [
                        int(v.InCoverageZone),
                        v.BasicBlockSize,
                        int(v.CoveredByTest),
                        int(v.VisitedByState),
                        int(v.TouchedByState),
                    ]
                )
            )
    # vertex -> vertex edges
    for e in game_edges:
        edges_index_v_v.append(
            np.array([vertex_map[e.VertexFrom], vertex_map[e.VertexTo]])
        )
        edges_attr_v_v.append(
            np.array([e.Label.Token])
        )  # TODO: consider token in a model

    state_doubles = 0

    # state nodes
    for s in game_states:
        state_id = s.Id
        if state_id not in state_map:
            state_map[state_id] = state_index
            nodes_state.append(
                np.array(
                    [
                        s.Position,
                        s.PredictedUsefulness,
                        s.PathConditionSize,
                        s.VisitedAgainVertices,
                        s.VisitedNotCoveredVerticesInZone,
                        s.VisitedNotCoveredVerticesOutOfZone,
                    ]
                )
            )
            # history edges: state -> vertex and back
            for h in s.History:  # TODO: process NumOfVisits as edge label
                v_to = vertex_map[h.GraphVertexId]
                edges_index_s_v_history.append(np.array([state_index, v_to]))
                edges_index_v_s_history.append(np.array([v_to, state_index]))
                edges_attr_s_v.append(np.array([h.NumOfVisits]))
                edges_attr_v_s.append(np.array([h.NumOfVisits]))
            state_index = state_index + 1
        else:
            state_doubles += 1

    # state and its childen edges: state -> state
    for s in game_states:
        for ch in s.Children:
            edges_index_s_s.append(np.array([state_map[s.Id], state_map[ch]]))

    # state position edges: vertex -> state and back
    for v in graphVertices:
        for s in v.States:
            edges_index_s_v_in.append(np.array([state_map[s], vertex_map[v.Id]]))
            edges_index_v_s_in.append(np.array([vertex_map[v.Id], state_map[s]]))

    data["game_vertex"].x = torch.tensor(np.array(nodes_vertex), dtype=torch.float)
    data["state_vertex"].x = torch.tensor(np.array(nodes_state), dtype=torch.float)
    data["game_vertex to game_vertex"].edge_index = (
        torch.tensor(np.array(edges_index_v_v), dtype=torch.long).t().contiguous()
    )
    data["state_vertex in game_vertex"].edge_index = (
        torch.tensor(np.array(edges_index_s_v_in), dtype=torch.long).t().contiguous()
    )
    data["game_vertex in state_vertex"].edge_index = (
        torch.tensor(np.array(edges_index_v_s_in), dtype=torch.long).t().contiguous()
    )

    def tensor_not_empty(tensor):
        return tensor.numel() != 0

    # dumb fix
    def null_if_empty(tensor):
        return (
            tensor
            if tensor_not_empty(tensor)
            else torch.empty((2, 0), dtype=torch.int64)
        )

    data["state_vertex history game_vertex"].edge_index = null_if_empty(
        torch.tensor(np.array(edges_index_s_v_history), dtype=torch.long)
        .t()
        .contiguous()
    )
    data["game_vertex history state_vertex"].edge_index = null_if_empty(
        torch.tensor(np.array(edges_index_v_s_history), dtype=torch.long)
        .t()
        .contiguous()
    )
    data["state_vertex history game_vertex"].edge_attr = torch.tensor(
        np.array(edges_attr_s_v), dtype=torch.long
    )
    data["game_vertex history state_vertex"].edge_attr = torch.tensor(
        np.array(edges_attr_v_s), dtype=torch.long
    )
    # if (edges_index_s_s): #TODO: empty?
    data["state_vertex parent_of state_vertex"].edge_index = null_if_empty(
        torch.tensor(np.array(edges_index_s_s), dtype=torch.long).t().contiguous()
    )
    # print(data['state', 'parent_of', 'state'].edge_index)
    # data['game_vertex', 'to', 'game_vertex'].edge_attr = torch.tensor(np.array(edges_attr_v_v), dtype=torch.long)
    # data['state_vertex', 'to', 'game_vertex'].edge_attr = torch.tensor(np.array(edges_attr_s_v), dtype=torch.long)
    # data.state_map = state_map
    # print("Doubles", state_doubles, len(state_map))
    return data, state_map


class StateGNNEncoderConvEdgeAttrMod(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = TAGConv(5, hidden_channels, 2).jittable()
        self.conv2 = TAGConv(6, hidden_channels, 3).jittable()  # TAGConv
        self.conv3 = GraphConv((-1, -1), hidden_channels).jittable()  # SAGEConv
        self.conv32 = GraphConv((-1, -1), hidden_channels).jittable()
        self.conv4 = SAGEConv((-1, -1), hidden_channels).jittable()
        self.conv42 = SAGEConv((-1, -1), hidden_channels).jittable()
        self.lin = Linear(hidden_channels, out_channels)
        self.lin_last = Linear(out_channels, 1)

    def forward(self, x_dict, edge_index_dict, edge_attr=None):
        game_x = self.conv1(
            x_dict["game_vertex"],
            edge_index_dict["game_vertex to game_vertex"],
        ).relu()

        state_x = self.conv2(
            x_dict["state_vertex"],
            edge_index_dict["state_vertex parent_of state_vertex"],
        ).relu()

        state_x = self.conv3(
            (game_x, state_x),
            edge_index_dict["game_vertex history state_vertex"],
            edge_attr["game_vertex history state_vertex"],
        ).relu()

        state_x = self.conv32(
            (game_x, state_x),
            edge_index_dict["game_vertex history state_vertex"],
            edge_attr["game_vertex history state_vertex"],
        ).relu()

        state_x = self.conv4(
            (game_x, state_x),
            edge_index_dict["game_vertex in state_vertex"],
        ).relu()

        state_x = self.conv42(
            (game_x, state_x),
            edge_index_dict["game_vertex in state_vertex"],
        ).relu()

        state_x = self.lin(state_x)

        return self.lin_last(state_x)


class StateModelEncoderExport(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.state_encoder = StateGNNEncoderConvEdgeAttrMod(
            hidden_channels, out_channels
        )

    def forward(self, x_dict, edge_index_dict, edge_attr=None):
        z_dict = {}
        z_dict["state_vertex"] = self.state_encoder(x_dict, edge_index_dict, edge_attr)
        z_dict["game_vertex"] = x_dict["game_vertex"]
        return z_dict


def create_dummy_hetero_data(path):
    with open(path, "r") as dummy_file:
        dummy_input = json.load(dummy_file)
    dummy_input = GameState.from_json(dummy_input)
    hetero_data, _ = convert_input_to_tensor(dummy_input)

    return hetero_data


def create_torch_dummy_input(path):
    hetero_data = create_dummy_hetero_data(path)
    return hetero_data.x_dict, hetero_data.edge_index_dict, hetero_data.edge_attr_dict


def create_onnx_dummy_input(path):
    hetero_data = create_dummy_hetero_data(path)

    return {
        "x_dict": hetero_data.x_dict,
        "edge_index_dict": hetero_data.edge_index_dict,
        "edge_attr_dict": hetero_data.edge_attr_dict,
    }


DUMMY_PATH = "ml/onnx/dummy_input.json"


def main():
    model = StateModelEncoderExport(hidden_channels=32, out_channels=8)
    model.forward(*create_torch_dummy_input(DUMMY_PATH))
    model = torch.jit.script(model)
    model.eval()

    save_path = "test_model.onnx"

    torch.onnx.export(
        model=model,
        args=(*create_torch_dummy_input(DUMMY_PATH), {}),
        f=save_path,
        verbose=False,
        do_constant_folding=True,
        input_names=["x_dict", "edge_index_dict", "edge_attr_dict"],
        output_names=["out", "other_out"],
    )

    model = onnx.load(save_path)
    onnx.checker.check_model(model)
    print(onnx.helper.printable_graph(model.graph))

    ort_session = onnxruntime.InferenceSession(save_path)
    ort_inputs = create_onnx_dummy_input(DUMMY_PATH)
    ort_outs = ort_session.run(None, ort_inputs)  # ValueError


if __name__ == "__main__":
    main()
