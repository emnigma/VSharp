import torch
from common.constants import DEVICE
from learning.play_game import play_game
from config import GeneralConfig
from connection.game_server_conn.utils import MapsType
from ml.common_model.models import CommonModel
from ml.common_model.wrapper import CommonModelWrapper
from ml.common_model.utils import csv2best_models, euclidean_dist
import ml.onnx.onnx_import
import logging
from common.constants import APP_LOG_FILE


open("./app.log", "w").close()

logging.basicConfig(
    level=GeneralConfig.LOGGER_LEVEL,
    filename=APP_LOG_FILE,
    filemode="a",
    format="%(asctime)s - p%(process)d: %(name)s - [%(levelname)s]: %(message)s",
)


def main():
    lr = 0.00001
    epochs = 10
    hidden_channels = 32
    num_gv_layers = 2
    num_sv_layers = 2
    print(DEVICE)
    model = CommonModel(hidden_channels, num_gv_layers, num_sv_layers)
    model.forward(*ml.onnx.onnx_import.create_torch_dummy_input())

    model.to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = euclidean_dist
    cmwrapper = CommonModelWrapper(model, csv2best_models(), optimizer, criterion)

    for epoch in range(epochs):
        result = play_game(
            with_predictor=cmwrapper,
            max_steps=GeneralConfig.MAX_STEPS,
            maps_type=MapsType.TRAIN,
        )


if __name__ == "__main__":
    main()
