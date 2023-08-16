import json

import onnx
import onnxruntime
import torch

from common.constants import DUMMY_INPUT_PATH
from common.game import GameState
from ml.data_loader_compact import ServerDataloaderHeteroVector


def create_dummy_hetero_data():
    with open(DUMMY_INPUT_PATH, "r") as dummy_file:
        dummy_input = json.load(dummy_file)
    dummy_input = GameState.from_json(dummy_input)
    hetero_data, _ = ServerDataloaderHeteroVector.convert_input_to_tensor(dummy_input)

    return hetero_data


def create_onnx_dummy_input():
    hetero_data = create_dummy_hetero_data()

    return {
        "x_dict": hetero_data.x_dict,
        "edge_index_dict": hetero_data.edge_index_dict,
        "edge_attr_dict": hetero_data.edge_attr_dict,
        # "edge_index": np.array([1, 2]),
        # "edge_index.5": np.array([1, 2]),
        # "edge_index.3": np.array([1, 2]),
        # "onnx::Reshape_9": np.array([1, 2]),
    }


def create_torch_dummy_input():
    hetero_data = create_dummy_hetero_data()
    return hetero_data.x_dict, hetero_data.edge_index_dict, hetero_data.edge_attr_dict


def export_onnx_model(model: torch.nn.Module, save_path: str, opset_ver: int = None):
    torch.onnx.export(
        model=model,
        args=(*create_torch_dummy_input(), {}),
        f=save_path,
        verbose=False,
        do_constant_folding=True,
        dynamic_axes={
            "x_dict": [0, 1],
            "edge_index_dict": [0, 1],
            "edge_attr_dict": [0, 1],
        },
        input_names=["x_dict", "edge_index_dict", "edge_attr_dict"],
        output_names=["out", "other_out"],
        opset_version=opset_ver,
    )

    torch_model_out = model(*create_torch_dummy_input())
    check_onnx_model(save_path, check_against=torch_model_out)


def check_onnx_model(path: str, check_against):
    model = onnx.load(path)
    print(model)
    onnx.checker.check_model(model)
    print(onnx.helper.printable_graph(model.graph))

    ort_session = onnxruntime.InferenceSession(path)
    ort_inputs = create_onnx_dummy_input()
    ort_outs = ort_session.run(None, ort_inputs)

    print(ort_outs == check_against)

    print("ONNX model loaded succesfully")
