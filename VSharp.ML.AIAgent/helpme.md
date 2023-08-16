# How to export GNN with dict inputs correctly?

## Problem description

I am having an issue when exporting of PyTorch GNN model to ONNX. Here is my export code:

```
torch.onnx.export(
    model=model,
    args=(x_dict, edge_index_dict, edge_attr_dict, {}),
    f=save_path,
    verbose=False,
    input_names=["x_dict", "edge_index_dict", "edge_attr_dict"],
    output_names=["out"],
)
```

`x_dict, edge_index_dict, edge_attr_dict` are of type `Dict[str, torch.Tensor]` (hetero_data is formed [like this](https://github.com/emnigma/VSharp/blob/408ba9800362285f420b3d9b51116f4b2cbb3391/VSharp.ML.AIAgent/ml/data_loader_compact.py#L30))

In addition to 3 inputs in my [model](https://github.com/emnigma/VSharp/blob/408ba9800362285f420b3d9b51116f4b2cbb3391/VSharp.ML.AIAgent/ml/models.py#L654)'s [forward](https://github.com/emnigma/VSharp/blob/408ba9800362285f420b3d9b51116f4b2cbb3391/VSharp.ML.AIAgent/ml/models.py#L659) , torch.onnx.export generates 4 additional inputs and when I try to use exported model with onnxruntime I get ValueError:

`ValueError: Required inputs (['edge_index', 'edge_index.5', 'edge_index.3', 'onnx::Reshape_9']) are missing from input feed (['x_dict', 'edge_index_dict', 'edge_attr_dict']).`

## Reproduction

here is a related packages from my conda env:

```
# This file may be used to create an environment using:
# $ conda create --name <env> --file <this file>
# platform: osx-arm64
# ...
dataclasses-json=0.5.7=pyhd8ed1ab_0
numpy=1.25.0=py311he598dae_0
onnx=1.13.1=py311h8472c4a_0
onnxruntime=1.15.1=py311h683bcd2_0
openblas=0.3.23=openmp_hf78f355_0
pandas=1.5.3=py311h6956b77_0
pytorch=2.0.1=py3.11_0
scikit-learn=1.3.0=pypi_0
scipy=1.11.1=pypi_0
sympy=1.11.1=py311hca03da5_0
torch-geometric=2.3.1=pypi_0
torch-scatter=2.1.1=pypi_0
torch-sparse=0.6.17=pypi_0
torchaudio=2.0.2=py311_cpu
torchvision=0.15.2=cpu_py311he74fb5d_0
# ...
```

here is a minimal reproduction script and dummy_data for it:

script: https://gist.github.com/emnigma/0b98cfbf3fff47be417c64489d83a2a2

data: https://gist.github.com/emnigma/e3ea559fe4db0adde886708f402473bb

## JIT trace output

here is the jit trace with strict=False .code output:

```
def forward(self,
    argument_1: Dict[str, Tensor],
    argument_2: Dict[str, Tensor],
    argument_3: Dict[str, Tensor]) -> Dict[str, Tensor]:
  state_encoder = self.state_encoder
  x = argument_1["game_vertex"]
  x0 = argument_1["state_vertex"]
  edge_index = argument_2["game_vertex to game_vertex"]
  edge_index0 = argument_2["game_vertex in state_vertex"]
  edge_index1 = argument_2["game_vertex history state_vertex"]
  edge_index2 = argument_2["state_vertex parent_of state_vertex"]
  edge_weight = argument_3["game_vertex history state_vertex"]
  _0 = (state_encoder).forward(x, edge_index, x0, edge_index2, edge_index1, edge_weight, edge_index0, )
  _1 = {"state_vertex": _0, "game_vertex": x}
  return _1
```

I am getting a feeling I am doing something wrong, how can i export my model correctly?
