import torch
from common.constants import DEVICE
from learning.play_game import play_game
from config import GeneralConfig
from connection.game_server_conn.utils import MapsType
from models import CommonModel
from wrapper import CommonModelWrapper
from utils import csv2best_models, euclidean_dist


def main():
    lr = 0.0001
    epochs = 10
    hidden_channels = 32
    num_gv_layers = 2
    num_sv_layers = 2

    model = CommonModel(hidden_channels, num_gv_layers, num_sv_layers)
    model.to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = euclidean_dist
    cmwrapper = CommonModelWrapper(model, csv2best_models(), optimizer, criterion)

    for epoch in range(epochs):
        play_game(
            with_predictor=cmwrapper,
            max_steps=GeneralConfig.MAX_STEPS,
            maps_type=MapsType.TRAIN,
        )


if __name__ == "__main__":
    main()
