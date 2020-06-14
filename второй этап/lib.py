import torch
import torch.nn as nn
from torch.utils import data
import numpy as np
import pandas as pd

class Net(nn.Module):
    def __init__(self, num_featers, hidden_sizes):
        super(Net, self).__init__()
        self.input_size = num_featers
        self.output_size = 3
        self.layers = nn.Sequential(

            nn.Linear(self.input_size, hidden_sizes[0]),
            nn.ReLU(),

            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),

            nn.Linear(hidden_sizes[1], hidden_sizes[2]),
            nn.ReLU(),

            nn.Linear(hidden_sizes[2], hidden_sizes[3]),
            nn.ReLU(),

            nn.Linear(hidden_sizes[3], self.output_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)


def getitem(X, y, ids, index):
    left, right = ids[index]
    if y is not None:
        return X[left:right, :], y[index]
    else:
        return X[left:right, :], None


class Dataset(data.Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X).float()
        self.y = torch.tensor(y).float()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        return self.X[index], self.y[index]


def get_smoothed_log_mape_column_value(responses_column, answers_column, epsilon):
    return torch.abs(torch.log((responses_column + epsilon) / (answers_column + epsilon))).mean()


def get_smoothed_mean_log_accuracy_ratio(answers, responses, epsilon=0.005):
    log_accuracy_ratio_mean = torch.stack(
        [
            get_smoothed_log_mape_column_value(responses[:, 0], answers[:, 0], epsilon),
            get_smoothed_log_mape_column_value(responses[:, 1], answers[:, 1], epsilon),
            get_smoothed_log_mape_column_value(responses[:, 2], answers[:, 2], epsilon),
        ]
    ).mean()
    percentage_error = 100 * (torch.exp(log_accuracy_ratio_mean) - 1)

    return percentage_error