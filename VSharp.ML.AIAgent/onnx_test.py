import json
import os
import pathlib
from typing import Any, Callable

import numpy as np
import onnx
import onnxruntime
import torch

from common.game import GameState
from config import GeneralConfig
from ml.data_loader_compact import ServerDataloaderHeteroVector
from ml.onnx.onnx_import import (
    create_dummy_hetero_data,
    create_onnx_dummy_input,
    create_torch_dummy_input,
    export_onnx_model,
)
from new_encoder import StateModelEncoder

DUMMY_PATH = "ml/onnx/dummy_input.json"


def load_dummy_hetero_data(path):
    with open(path, "r") as dummy_file:
        dummy_input = json.load(dummy_file)
    dummy_input = GameState.from_dict(dummy_input)
    hetero_data, _ = ServerDataloaderHeteroVector.convert_input_to_tensor(dummy_input)

    return hetero_data


def split_data(data, with_modifier: Callable[[Any], Any] = lambda x: x):
    return (
        with_modifier(data["game_vertex"].x),
        with_modifier(data["state_vertex"].x),
        with_modifier(data["game_vertex to game_vertex"].edge_index),
        with_modifier(data["game_vertex history state_vertex"].edge_index),
        with_modifier(data["state_vertex history game_vertex"].edge_attr),
        with_modifier(data["game_vertex in state_vertex"].edge_index),
        with_modifier(data["state_vertex parent_of state_vertex"].edge_index),
    )


def create_input_for_onnx(
    data, with_modifier: Callable[[Any], Any] = lambda x: x.numpy()
):
    return {
        "game_vertex": with_modifier(data["game_vertex"].x),
        "state_vertex": with_modifier(data["state_vertex"].x),
        "game_vertex to game_vertex": with_modifier(
            data["game_vertex to game_vertex"].edge_index
        ),
        "game_vertex history state_vertex index": with_modifier(
            data["game_vertex history state_vertex"].edge_index
        ),
        "game_vertex history state_vertex attrs": with_modifier(
            data["game_vertex history state_vertex"].edge_attr
        ),
        "game_vertex in state_vertex": with_modifier(
            data["game_vertex in state_vertex"].edge_index
        ),
        "state_vertex parent_of state_vertex": with_modifier(
            data["state_vertex parent_of state_vertex"].edge_index
        ),
    }


def main():
    model = StateModelEncoder(32, 8)

    # x = model.convert_to_single_tensor(*create_dummy_hetero_data(), False)
    model_input = split_data(create_dummy_hetero_data())

    torch_out = model.forward(*model_input)
    # traced_cell = torch.jit.trace(model, x, strict=False)
    # print(traced_cell)
    # print(traced_cell.graph)
    # print(traced_cell.code)
    # traced_cell(x)

    save_path = "test_model.onnx"

    torch.onnx.export(
        model=model,
        args=model_input,
        f=save_path,
        verbose=False,
        dynamic_axes={
            "game_vertex": [0, 1],
            "state_vertex": [0, 1],
            "game_vertex to game_vertex": [0, 1],
            "game_vertex history state_vertex index": [0, 1],
            "game_vertex history state_vertex attrs": [0, 1],
            "game_vertex in state_vertex": [0, 1],
            "state_vertex parent_of state_vertex": [0, 1],
            # "out": [0, 1, 2, 3, 4, 5, 6, 7],
            "out": [0],
        },
        do_constant_folding=False,
        input_names=[
            "game_vertex",
            "state_vertex",
            "game_vertex to game_vertex",
            "game_vertex history state_vertex index",
            "game_vertex history state_vertex attrs",
            "game_vertex in state_vertex",
            "state_vertex parent_of state_vertex",
        ],
        output_names=["out"],
    )

    model_onnx = onnx.load(save_path)
    onnx.checker.check_model(model_onnx)
    print(onnx.helper.printable_graph(model_onnx.graph))

    ort_in = create_input_for_onnx(create_dummy_hetero_data())
    ort_session = onnxruntime.InferenceSession(save_path)
    ort_outs = ort_session.run(None, ort_in)

    print(f"{torch_out.tolist()=}")
    print(f"{ort_outs=}")

    test_folder = pathlib.Path("./test_game_states")

    for idx, test_gs in enumerate(os.listdir(test_folder)[:4]):
        x = load_dummy_hetero_data(test_folder / test_gs)
        torch_out = model(*split_data(x))

        ort_in = create_input_for_onnx(x)
        # ort_in = ort_in[:]
        # ort_session = onnxruntime.InferenceSession(save_path)
        ort_outs = ort_session.run(None, ort_in)

        print(f"{torch_out.shape}, {torch_out.flatten().tolist()=}")
        print(f"{ort_outs=}")
        print(f"{idx}/{len(os.listdir(test_folder))}")


if __name__ == "__main__":
    main()
