import torch

from common.game import GameState
from ml.data_loader_compact import ServerDataloaderHeteroVector
from ml.model_wrappers.protocols import Predictor
from ml.predict_state_vector_hetero import PredictStateVectorHetGNN
from utils import back_prop


class CommonModelWrapper(Predictor):
    def __init__(self, model: torch.nn.Module, best_models: dict, optimizer, criterion) -> None:
        self.model = model
        self.best_models = best_models
        self._model = model
        self.optimizer = optimizer
        self.criterion = criterion

    def name(self) -> str:
        return "MY AWESOME MODEL"

    def model(self):
        return self._model

    def predict(self, input: GameState, map_name):
        hetero_input, state_map = ServerDataloaderHeteroVector.convert_input_to_tensor(
            input
        )
        assert self.model is not None

        next_step_id = PredictStateVectorHetGNN.predict_state_single_out(
            self.model, hetero_input, state_map
        )

        back_prop(self.best_models[map_name], self.model, hetero_input, self.optimizer, self.criterion)

        del hetero_input
        return next_step_id
