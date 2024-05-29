import os
from typing import Tuple
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import yaml
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from sklearn.metrics import r2_score
from torch import tensor

torch.cuda.empty_cache()


class MLP(pl.LightningModule):
    def __init__(
        self,
        input_size_clim=20,
        input_size_met=12,
        input_size_state=7,
        output_size=7,
        diag_output_size=3,
        hidden_size=64,
        mu_norm=0,
        std_norm=1,
        dataset=None,
    ):
        super(MLP, self).__init__()
        # Normalization vector for delta_x's
        self.mu_norm = tensor(mu_norm)
        self.std_norm = tensor(std_norm)
        self.ds = dataset

        # Define layers
        self.diag_output_size = diag_output_size
        input_dim = input_size_clim + input_size_met + input_size_state

        #     self.fc1 = nn.Linear(input_dim, hidden_size)
        #     self.relu1 = nn.ReLU()
        #     self.fc2 = nn.Linear(hidden_size, hidden_size)
        #     self.relu2 = nn.LeakyReLU()
        #     self.fc3 = nn.Linear(hidden_size, hidden_size)
        #     self.relu3 = nn.LeakyReLU()
        #     self.fc4 = nn.Linear(hidden_size, hidden_size)
        #     self.relu4 = nn.LeakyReLU()
        #     self.dropout = nn.Dropout(0.2)
        #     self.fc5 = nn.Linear(hidden_size, hidden_size)
        #     self.relu5 = nn.LeakyReLU()
        #     self.fc6 = nn.Linear(hidden_size, hidden_size)
        #     self.relu6 = nn.LeakyReLU()
        #     self.fc7 = nn.Linear(hidden_size, output_size)
        #     self.fc8 = nn.Linear(hidden_size, diag_output_size)

        # def forward(self, clim_feats, met_feats, state_feats):
        #     combined = torch.cat((clim_feats, met_feats, state_feats), dim=-1)
        #     x = self.relu1(self.fc1(combined))
        #     x = self.relu2(self.fc2(x))
        #     x = self.relu3(self.fc3(x))
        #     x = self.dropout(self.relu4(self.fc4(x)))
        #     x = self.relu5(self.fc5(x))
        #     x = self.relu6(self.fc6(x))
        #     x_prog = self.fc7(x)
        #     x_diag = self.fc8(x)
        #     return x_prog, x_diag

        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.LeakyReLU()
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(0.15)
        self.relu3 = nn.LeakyReLU()
        self.fc4 = nn.Linear(hidden_size, output_size)
        self.fc5 = nn.Linear(hidden_size, diag_output_size)

    def forward(self, clim_feats, met_feats, state_feats):

        # remember to expand static for the time it should bein the dim 1
        combined = torch.cat((clim_feats, met_feats, state_feats), dim=-1)
        x = self.relu1(self.fc1(combined))
        x = self.dropout(self.relu2(self.fc2(x)))
        x = self.relu3(self.fc3(x))
        x_prog = self.fc4(x)
        x_diag = self.fc5(x)
        return x_prog, x_diag

    def transform(self, x, mean, std):
        x_norm = (x - mean) / (std + 1e-5)
        # x_norm = (x - mean) / (std)
        return x_norm








if __name__ == '__main__':

    # Define the config for the experiment
    PATH_NAME = os.path.dirname(os.path.abspath(__file__))
    with open(f"{PATH_NAME}/config.yaml") as stream:
        try:
            CONFIG = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)