from typing import Dict, Tuple

import numpy as np
import numpy.typing as npt
import torch
from torch_geometric.data import HeteroData

from common.game import GameMapEdge, GameMapVertex, GameState, State


def tensor_not_empty(tensor):
    return tensor.numel() != 0


# dumb fix
def null_if_empty(tensor):
    return (
        tensor if tensor_not_empty(tensor) else torch.empty((2, 0), dtype=torch.int64)
    )


GAME_VERTEX = "game_vertex"  # game_x
STATE_VERTEX = "state_vertex"  # state_x
GV_TO_GV = "game_vertex to game_vertex"  # edge_index_v_v
GV_HIS_SV = "game_vertex history state_vertex"
GV_HIS_SV_INDEX = GV_HIS_SV + " index"  # edge_index_history_v_s
GV_HIS_SV_ATTRS = GV_HIS_SV + " attrs"  # edge_attr_history_v_s
GV_IN_SV = "game_vertex in state_vertex"  # edge_index_in_v_s
SV_PARENTOF_SV = "state_vertex parent_of state_vertex"  # edge_index_s_s


def get_vertex_features(vertex: GameMapVertex) -> npt.NDArray:
    return np.array(
        (
            int(vertex.InCoverageZone),
            vertex.BasicBlockSize,
            int(vertex.CoveredByTest),
            int(vertex.VisitedByState),
            int(vertex.TouchedByState),
        )
    )


def get_state_features(state: State) -> npt.NDArray:
    return np.array(
        [
            state.Position,
            state.PredictedUsefulness,
            state.PathConditionSize,
            state.VisitedAgainVertices,
            state.VisitedNotCoveredVerticesInZone,
            state.VisitedNotCoveredVerticesOutOfZone,
        ]
    )


class ServerDataloaderHeteroVector:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.graph_types_and_data = {}
        self.dataset = []
        self.process_directory(data_dir)
        self.__process_files()

    @staticmethod
    def convert_input_to_tensor(
        game_state: GameState,
    ) -> Tuple[HeteroData, Dict[int, int]]:
        """
        Converts game env to tensors
        """
        graph_vertices = game_state.GraphVertices
        states = game_state.States
        game_edges = game_state.Map
        data = HeteroData()
        nodes_vertex = []
        nodes_state = []
        edges_index_v_v = []
        edges_index_s_s = []
        edges_index_v_s_in = []
        edges_index_v_s_history = []

        edges_attr_v_s = []

        state_map: Dict[int, int] = {}  # Maps real state id to its index
        vertex_map: Dict[int, int] = {}  # Maps real vertex id to its index

        # vertex nodes
        for vertex_index, vertex in enumerate(graph_vertices):
            vertex_map[vertex_index] = vertex.Id  # maintain order in tensors
            nodes_vertex.append(get_vertex_features(vertex))

        def create_numpy_edge(game_edge: GameMapEdge):
            return np.array(
                [vertex_map[game_edge.VertexFrom], vertex_map[game_edge.VertexTo]]
            )

        # vertex -> vertex edges
        edges_index_v_v = list(map(create_numpy_edge, game_edges))

        # state nodes
        for state_index, state in enumerate(states):
            if state.Id in state_map:
                assert False
            state_map[state.Id] = state_index
            nodes_state.append(get_state_features(state))

            # history edges: state -> vertex and back
            for his_edge in state.History:  # TODO: process NumOfVisits as edge label
                v_to = vertex_map[his_edge.GraphVertexId]
                edges_index_v_s_history.append(np.array([v_to, state_index]))
                edges_attr_v_s.append(np.array([his_edge.NumOfVisits]))

        # state and its childen edges: state -> state
        for state in states:
            for ch in state.Children:
                edges_index_s_s.append(np.array([state_map[state.Id], state_map[ch]]))

        # state position edges: vertex -> state and back
        for vertex in graph_vertices:
            for state in vertex.States:
                edges_index_v_s_in.append(
                    np.array([vertex_map[vertex.Id], state_map[state]])
                )

        data[GAME_VERTEX].x = torch.tensor(np.array(nodes_vertex), dtype=torch.float)
        data[STATE_VERTEX].x = torch.tensor(np.array(nodes_state), dtype=torch.float)
        data[GV_TO_GV].edge_index = (
            torch.tensor(np.array(edges_index_v_v), dtype=torch.long).t().contiguous()
        )
        data[GV_IN_SV].edge_index = (
            torch.tensor(np.array(edges_index_v_s_in), dtype=torch.long)
            .t()
            .contiguous()
        )

        data[GV_HIS_SV].edge_index = null_if_empty(
            torch.tensor(np.array(edges_index_v_s_history), dtype=torch.long)
            .t()
            .contiguous()
        )
        data[GV_HIS_SV].edge_attr = torch.tensor(
            np.array(edges_attr_v_s), dtype=torch.long
        )
        # if (edges_index_s_s): #TODO: empty?
        data[SV_PARENTOF_SV].edge_index = null_if_empty(
            torch.tensor(np.array(edges_index_s_s), dtype=torch.long).t().contiguous()
        )
        # return {
        #     GAME_VERTEX: data[GAME_VERTEX].x,
        #     STATE_VERTEX: data[STATE_VERTEX].x,
        #     GV_TO_GV: data[GV_TO_GV].edge_index,
        #     GV_HIS_SV_INDEX: data[GV_HIS_SV].edge_index,
        #     GV_HIS_SV_ATTRS: data[GV_HIS_SV].edge_attr,
        #     GV_IN_SV: data[GV_IN_SV].edge_index,
        #     SV_PARENTOF_SV: data[SV_PARENTOF_SV].edge_index,
        # }, state_map

        return data, state_map
