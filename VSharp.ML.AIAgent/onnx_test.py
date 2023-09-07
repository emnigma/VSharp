import json
import os
import pathlib
from typing import Any, Callable, OrderedDict

import onnx
import onnxruntime
import torch

from common.game import GameState
from ml.data_loader_compact import ServerDataloaderHeteroVector
from ml.onnx.onnx_import import create_dummy_hetero_data
from new_encoder import StateModelEncoder

DUMMY_PATH = "ml/onnx/dummy_input.json"

GAME_VERTEX = "game_vertex"  # game_x
STATE_VERTEX = "state_vertex"  # state_x
GV_TO_SV = "game_vertex to game_vertex"  # edge_index_v_v
GV_HIS_SV = "game_vertex history state_vertex"
GV_HIS_SV_INDEX = GV_HIS_SV + " index"  # edge_index_history_v_s
GV_HIS_SV_ATTRS = GV_HIS_SV + " attrs"  # edge_attr_history_v_s
GV_IN_SV = "game_vertex in state_vertex"  # edge_index_in_v_s
SV_PARENTOF_SV = "state_vertex parent_of state_vertex"  # edge_index_s_s


def load_hetero_data(path):
    with open(path, "r") as file:
        file_json = json.load(file)
    file_json = GameState.from_dict(file_json)
    hetero_data, _ = ServerDataloaderHeteroVector.convert_input_to_tensor(file_json)

    return hetero_data


def split_data(data, with_modifier: Callable[[Any], Any] = lambda x: x):
    return (
        with_modifier(data[GAME_VERTEX].x),
        with_modifier(data[STATE_VERTEX].x),
        with_modifier(data[GV_TO_SV].edge_index),
        with_modifier(data[GV_HIS_SV].edge_index),
        with_modifier(data[GV_HIS_SV].edge_attr),
        with_modifier(data[GV_IN_SV].edge_index),
        with_modifier(data[SV_PARENTOF_SV].edge_index),
    )


def hetero_data_to_onnx_in(
    data, with_modifier: Callable[[Any], Any] = lambda x: x.numpy()
):
    return {
        GAME_VERTEX: with_modifier(data[GAME_VERTEX].x),
        STATE_VERTEX: with_modifier(data[STATE_VERTEX].x),
        GV_TO_SV: with_modifier(data[GV_TO_SV].edge_index),
        GV_HIS_SV_INDEX: with_modifier(data[GV_HIS_SV].edge_index),
        GV_HIS_SV_ATTRS: with_modifier(data[GV_HIS_SV].edge_attr),
        GV_IN_SV: with_modifier(data[GV_IN_SV].edge_index),
        SV_PARENTOF_SV: with_modifier(data[SV_PARENTOF_SV].edge_index),
    }


def main():
    test_folder = pathlib.Path("./test_game_states")
    onnx_model_save_path = "test_model.onnx"

    model = StateModelEncoder(32, 8)

    strange_sd: OrderedDict = torch.load("GNNEncoderCompact.pth", map_location="cpu")
    for key in [
        "state_encoder.conv32.lin_rel.weight",
        "state_encoder.conv32.lin_rel.bias",
        "state_encoder.conv32.lin_root.weight",
        "state_encoder.conv42.lin_l.weight",
        "state_encoder.conv42.lin_l.bias",
        "state_encoder.conv42.lin_r.weight",
    ]:
        strange_sd.pop(key)

    model.load_state_dict(strange_sd)

    # x = model.convert_to_single_tensor(*create_dummy_hetero_data(), False)
    model_input = split_data(create_dummy_hetero_data())

    torch_out = model.forward(*model_input)
    # traced_cell = torch.jit.trace(model, x, strict=False)
    # print(traced_cell)
    # print(traced_cell.graph)
    # print(traced_cell.code)
    # traced_cell(x)

    torch.onnx.export(
        model=model,
        args=model_input,
        f=onnx_model_save_path,
        verbose=False,
        dynamic_axes={
            GAME_VERTEX: {0: "gv_count"},
            STATE_VERTEX: {0: "sv_count"},
            GV_TO_SV: {1: "gv2gv_rel_count"},
            GV_HIS_SV_INDEX: {1: "gv_his_sv_index_rel_count"},
            GV_HIS_SV_ATTRS: {0: "gv_his_sv_attrs_count"},
            GV_IN_SV: {1: "gv_in_sv_rel_count"},
            SV_PARENTOF_SV: {1: "sv_parentof_sv_rel_count"},
            # "out": [0, 1, 2, 3, 4, 5, 6, 7],
            # "out": [0],
        },
        do_constant_folding=False,
        input_names=[
            GAME_VERTEX,
            STATE_VERTEX,
            GV_TO_SV,
            GV_HIS_SV_INDEX,
            GV_HIS_SV_ATTRS,
            GV_IN_SV,
            SV_PARENTOF_SV,
        ],
        output_names=["out"],
        export_params=True,
    )

    model_onnx = onnx.load(onnx_model_save_path)
    onnx.checker.check_model(model_onnx)
    print(onnx.helper.printable_graph(model_onnx.graph))

    ort_in = hetero_data_to_onnx_in(create_dummy_hetero_data())
    ort_session = onnxruntime.InferenceSession(onnx_model_save_path)
    ort_outs = ort_session.run(None, ort_in)

    print(f"{shorten_output(torch_out)=}")
    print(f"{shorten_output(ort_outs[0])=}")

    for idx, test_gs in enumerate(os.listdir(test_folder)[:4]):
        x = load_hetero_data(test_folder / test_gs)
        torch_out = model(*split_data(x))

        ort_in = hetero_data_to_onnx_in(x)
        ort_outs = ort_session.run(None, ort_in)

        print(f"{shorten_output(torch_out)=}")
        print(f"{shorten_output(ort_outs[0])=}")
        print(f"{idx}/{len(os.listdir(test_folder))}")


def shorten_output(torch_out):
    shortened = list(map(lambda x: round(x, 2), torch_out.flatten().tolist()))
    if len(shortened) > 10:
        shortened = shortened[:10] + ["..."]
    return shortened


if __name__ == "__main__":
    main()
