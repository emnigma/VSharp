import json
import os
import pathlib

import numpy as np
import onnx
import onnxruntime
import torch

from common.game import GameState
from config import GeneralConfig
from ml.data_loader_compact import ServerDataloaderHeteroVector
from ml.onnx.onnx_import import (
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

    return hetero_data.x_dict, hetero_data.edge_index_dict, hetero_data.edge_attr_dict


def main():
    model = StateModelEncoder(32, 8)

    x = model.convert_to_single_tensor(*create_torch_dummy_input(), False)

    torch_out = model.forward(x)
    # traced_cell = torch.jit.trace(model, x, strict=False)
    # print(traced_cell)
    # print(traced_cell.graph)
    # print(traced_cell.code)
    # traced_cell(x)

    save_path = "test_model.onnx"

    torch.onnx.export(
        model=model,
        args=x,
        f=save_path,
        verbose=False,
        dynamic_axes={
            "x": {1: "variable_input_size"},
            # "out": [0, 1, 2, 3, 4, 5, 6, 7],
        },
        do_constant_folding=False,
        input_names=["x"],
        output_names=["out"],
    )

    model_onnx = onnx.load(save_path)
    onnx.checker.check_model(model_onnx)
    print(onnx.helper.printable_graph(model_onnx.graph))

    ort_in = x.numpy()
    ort_session = onnxruntime.InferenceSession(save_path)
    ort_outs = ort_session.run(None, {"x": ort_in})

    test_folder = pathlib.Path("./test_game_states")

    for test_gs in os.listdir(test_folder)[:4]:
        x = model.convert_to_single_tensor(
            *load_dummy_hetero_data(test_folder / test_gs), False
        )

        ort_in = x.numpy()
        # ort_in = ort_in[:]
        ort_session = onnxruntime.InferenceSession(save_path)
        ort_outs = ort_session.run(None, {"x": ort_in})

        print(torch_out)
        print(ort_outs)
        print("")


if __name__ == "__main__":
    main()
