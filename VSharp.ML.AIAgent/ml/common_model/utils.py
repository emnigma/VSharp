import torch
from common.constants import DEVICE
import os
import re
import csv

from ml.models import SAGEConvModel
from ml.common_model.paths import csv_path, models_path


def euclidean_dist(y_pred, y_true):
    y_pred_min, ind1 = torch.min(y_pred, dim=0)
    y_pred_norm = y_pred - y_pred_min

    y_true_min, ind1 = torch.min(y_true, dim=0)
    y_true_norm = y_true - y_true_min

    return torch.sqrt(torch.sum((y_pred_norm - y_true_norm) ** 2))


def get_last_epoch_num(path):
    epochs = list(map(lambda x: re.findall('[0-9]+', x), os.listdir(path)))
    return str(sorted(epochs)[-1][0])


def get_sum(w):
    s = 0
    for i in range(len(w)):
        s += w[i] * 10**(len(w) - i - 1)
    return s


def csv2best_models():
    best_models = {}
    values = []
    models_names = []

    with open(csv_path + get_last_epoch_num(csv_path) + '.csv', 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        map_names = next(csv_reader)[1:]
        for row in csv_reader:
            models_names.append(row[0])
            int_row = list(map(lambda w: get_sum(w), list(map(lambda x: tuple(map(lambda y: int(y), x[1:-1].split(", "))), row[1:]))))
            values.append(int_row)
        val, ind = torch.max(torch.tensor(values), dim=0)
        for i in range(len(map_names)):
            best_models[map_names[i]] = models_names[ind[i]]
        return best_models


def back_prop(best_model, model, data, optimizer, criterion):
    model.train()
    data.to(DEVICE)
    optimizer.zero_grad()
    ref_model = SAGEConvModel(16)
    ref_model.load_state_dict(torch.load(models_path + "epoch_" + get_last_epoch_num(models_path) + "/" + best_model
                                         # + ".pth"
                                         ))
    ref_model.to(DEVICE)
    out = model(data.x_dict, data.edge_index_dict, data.edge_attr_dict)
    y_true = ref_model(data.x_dict, data.edge_index_dict, data.edge_attr_dict)

    loss = criterion(out, y_true)
    loss.backward()
    optimizer.step()
