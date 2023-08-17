import onnx
import onnxruntime
import torch
import numpy as np
from config import GeneralConfig
from ml.onnx.onnx_import import (
    create_onnx_dummy_input,
    create_torch_dummy_input,
    export_onnx_model,
)
from new_encoder import StateModelEncoder

DUMMY_PATH = "ml/onnx/dummy_input.json"


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
        do_constant_folding=True,
        input_names=["x"],
    )

    model = onnx.load(save_path)
    onnx.checker.check_model(model)
    # print(onnx.helper.printable_graph(model.graph))

    ort_in = x.numpy()
    ort_session = onnxruntime.InferenceSession(save_path)
    ort_outs = ort_session.run(None, {"x": ort_in})

    print(torch_out)
    print(ort_outs)


if __name__ == "__main__":
    main()
