import logging
from pathlib import Path

import torch

import ml.onnx.onnx_import
from common.constants import DEVICE
from config import GeneralConfig
from connection.game_server_conn.utils import MapsType
from epochs_statistics.tables import create_pivot_table, table_to_string
from learning.play_game import play_game
from ml.common_model.models import CommonModel
from ml.common_model.utils import csv2best_models, euclidean_dist
from ml.common_model.wrapper import CommonModelWrapper

LOG_PATH = Path("./ml_app.log")
TABLES_PATH = Path("./ml_tables.log")

logging.basicConfig(
    level=GeneralConfig.LOGGER_LEVEL,
    filename=LOG_PATH,
    filemode="a",
    format="%(asctime)s - p%(process)d: %(name)s - [%(levelname)s]: %(message)s",
)


def create_file(file: Path):
    open(file, "w").close()


def append_to_file(file: Path, s: str):
    with open(file, "a") as file:
        file.write(s)


def main():
    lr = 0.00001
    epochs = 10
    hidden_channels = 32
    num_gv_layers = 2
    num_sv_layers = 2
    print(DEVICE)
    model = CommonModel(hidden_channels, num_gv_layers, num_sv_layers)
    model.forward(*ml.onnx.onnx_import.create_torch_dummy_input())

    create_file(TABLES_PATH)
    create_file(LOG_PATH)

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
        table, _, _ = create_pivot_table({cmwrapper: result})
        table = table_to_string(table)
        append_to_file(TABLES_PATH, f"Epoch#{epoch}" + "\n")
        append_to_file(TABLES_PATH, table + "\n")


if __name__ == "__main__":
    main()
