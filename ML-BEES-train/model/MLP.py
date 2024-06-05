# ------------------------------------------------------------------
# Script to build the model Multi-Layer Perceptron (MLP)
# ------------------------------------------------------------------

from typing import Tuple
#import matplotlib.pyplot as plt
#import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import yaml
import os
#from pytorch_lightning.utilities.rank_zero import rank_zero_only
#from sklearn.metrics import r2_score
from torch import tensor
torch.cuda.empty_cache()
# ------------------------------------------------------------------

class MLP(pl.LightningModule):
    """
        MLP model for EC-Land dataset
    """
    def __init__(
        self,
        in_static: int = 22,
        in_dynamic: int = 12,
        in_prog:int = 7,
        out_prog: int = 7,
        out_diag: int = 3,
        hidden_size: int = 172,
        rollout: int = 6,
    ):
        super(MLP, self).__init__()

        # Initialize and define layers
        self.in_static = in_static
        self.in_dynamic = in_dynamic
        self.in_prog = in_prog
        self.out_prog = out_prog
        self.out_diag = out_diag
        self.hidden_size = hidden_size
        self.rollout = rollout

        input_dim = in_static + in_dynamic + in_prog

        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.LeakyReLU()
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(0.15)
        self.relu3 = nn.LeakyReLU()
        self.fc4 = nn.Linear(hidden_size, out_prog)
        self.fc5 = nn.Linear(hidden_size, out_diag)

    def forward(self, x_static: torch.tensor, x_dynamic: torch.tensor, x_prog: torch.tensor) -> Tuple[tensor, tensor]:

        combined = torch.cat((x_static, x_dynamic, x_prog), dim=-1)
        x = self.relu1(self.fc1(combined))
        x = self.dropout(self.relu2(self.fc2(x)))
        x = self.relu3(self.fc3(x))
        x_prog = self.fc4(x)
        x_diag = self.fc5(x)
        return x_prog, x_diag


if __name__ == '__main__':

    with open(r'../config.yaml') as stream:
        try:
            CONFIG = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    if CONFIG['devices'] != "-1":
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(CONFIG['devices'])
        device = 'cuda'
    else:
        device = 'cpu'

    x_dynamic = torch.randn((2, 6, 10051, 12), device=device)
    x_static = torch.randn((2, 6, 10051, 22), device=device)
    x_prog = torch.randn((2, 6, 10051, 7), device=device)

    model = MLP(in_static=22,
                in_dynamic=12,
                in_prog=7,
                out_prog=7,
                out_diag=3,
                hidden_size=172,
                rollout=6).to(device)

    print(model)
    #model.eval()
    n_parameters = sum(p.numel() for p in model.parameters())  # if p.requires_grad)
    print(f"number of parameters: {n_parameters}")

    x_prog, x_diag = model(x_static, x_dynamic, x_prog)

    print(x_prog.shape)
    print(x_diag.shape)

