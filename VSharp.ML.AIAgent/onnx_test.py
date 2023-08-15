import torch

from config import GeneralConfig
from ml.onnx.onnx_import import export_onnx_model
from ml.onnx.onnx_import import create_torch_dummy_input, create_onnx_dummy_input


def main():
    model = GeneralConfig.EXPORT_MODEL_INIT()
    model.load_state_dict(torch.load("test_model.pth"))
    model.eval()

    model = GeneralConfig.EXPORT_MODEL_INIT()
    model.load_state_dict(torch.load("test_model.pth"))
    a, b, c = create_torch_dummy_input()
    traced_cell = torch.jit.trace(model, (a, b, c), strict=False)
    print(traced_cell)
    print(traced_cell.graph)
    print(traced_cell.code)
    traced_cell(a, b, c)

    export_onnx_model(model, save_path="test_model.onnx")


if __name__ == "__main__":
    main()
