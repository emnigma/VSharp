import ast
import csv
import os
import re

import torch

from common.constants import DEVICE
from ml.common_model.paths import csv_path, models_path
from ml.models import SAGEConvModel


def euclidean_dist(y_pred, y_true):
    y_pred_norm = y_pred
    y_true_norm = y_true
    if len(y_pred) > 1 and torch.min(y_pred) != torch.max(y_pred):
        y_pred_min, ind1 = torch.min(y_pred, dim=0)
        y_pred_norm = y_pred - y_pred_min

        y_true_min, ind1 = torch.min(y_true, dim=0)
        y_true_norm = y_true - y_true_min

    return torch.sqrt(torch.sum((y_pred_norm - y_true_norm) ** 2))


def get_last_epoch_num(path):
    epochs = list(map(lambda x: re.findall("[0-9]+", x), os.listdir(path)))
    return str(sorted(epochs)[-1][0])


def csv2best_models():
    best_models = {}
    path_to_csv = os.path.join(csv_path, get_last_epoch_num(csv_path) + ".csv")
    with open(path_to_csv, "r") as csv_file:
        csv_reader = csv.reader(csv_file)
        map_names = next(csv_reader)[1:]
        models = []
        for row in csv_reader:
            models_stat = dict()
            # int_row = list(map(lambda x: tuple(map(lambda y: int(y), x[1:-1].split(", "))), row[1:]))
            int_row = list(map(lambda x: ast.literal_eval(x), row[1:]))
            for i in range(len(int_row)):
                models_stat[map_names[i]] = int_row[i]
            models.append((row[0], models_stat))

        for map_name in map_names:
            best_model = max(models, key=(lambda m: m[1][map_name]))
            best_model_name = best_model[0]
            best_model_score = best_model[1]
            ref_model = SAGEConvModel(16)
            path_to_model = os.path.join(
                models_path,
                "epoch_" + get_last_epoch_num(models_path),
                best_model_name + ".pth",
            )
            ref_model.load_state_dict(torch.load(path_to_model))
            ref_model.to(DEVICE)
            best_models[map_name] = (ref_model, best_model_score[map_name])
        return best_models


def back_prop(best_model, model, data, optimizer, criterion):
    model.train()
    data.to(DEVICE)
    optimizer.zero_grad()
    out = model(data.x_dict, data.edge_index_dict, data.edge_attr_dict)
    y_true = best_model(data.x_dict, data.edge_index_dict, data.edge_attr_dict)
    if abs(torch.min(y_true)) > 1:
        y_true = 1 / y_true
    # print(out, "\n", y_true)
    loss = criterion(out, y_true)
    # print('loss:', loss)
    loss.backward()
    optimizer.step()
    return loss
