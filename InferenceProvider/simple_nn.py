import onnxruntime as rt
import torch
import numpy as np


class NeuralNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(1, 1)

    def forward(self, x):
        return self.lin(x)


def main():
    model = NeuralNetwork()
    model_path = "/Users/emax/Data/VSharp/InferenceProvider/test_nn.onnx"
    test_in = torch.Tensor([1])

    print(model)
    print(model(test_in))

    torch.onnx.export(
        args={"x": test_in},
        input_names=("input",),
        output_names=("output",),
        model=model,
        opset_version=17,
        # verbose=True,
        f=model_path,
    )

    sess = rt.InferenceSession(model_path)
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name

    result = sess.run(["output"], {"input": np.array(test_in)})

    print(input_name, output_name, result)


if __name__ == "__main__":
    main()
