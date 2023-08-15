import torch

from config import GeneralConfig
from ml.onnx.onnx_import import export_onnx_model


def main():
    model = GeneralConfig.EXPORT_MODEL_INIT()
    model.load_state_dict(torch.load("test_model.pth"))
    model.eval()

    export_onnx_model(model, save_path="test_model.onnx")


if __name__ == "__main__":
    main()
