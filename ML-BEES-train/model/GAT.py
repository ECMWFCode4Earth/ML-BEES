# ------------------------------------------------------------------
# Graph Attention Networks (https://arxiv.org/abs/1710.10903)
# ------------------------------------------------------------------

from typing import Tuple
# import pytorch_lightning as pl
import torch
import torch.nn as nn
import yaml
import os
# from metrics import r2_score_multi
from torch import tensor
from torch_geometric.nn import GATConv
torch.cuda.empty_cache()

# ------------------------------------------------------------------

class GAT(nn.Module):
    """
        Graph Attention Networks model for EC-Land dataset
    """

    def __init__(
            self,
            in_static: int = 22,
            in_dynamic: int = 12,
            in_prog: int = 7,
            out_prog: int = 7,
            out_diag: int = 3,
            hidden_dim: int = 172,
            dropout: float = 0.15,
            #head: int = 4,
            mu_norm: float = 0.,
            std_norm: float = 1.,
            pretrained: str = None,
    ):
        super(GAT, self).__init__()

        # Initialize
        self.in_static = in_static
        self.in_dynamic = in_dynamic
        self.in_prog = in_prog
        self.out_prog = out_prog
        self.out_diag = out_diag
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        #self.head = head
        self.pretrained = None if pretrained == "None" else pretrained
        self.register_buffer('mu_norm', tensor(mu_norm))
        self.register_buffer('std_norm', tensor(std_norm))
        self.register_buffer('zero', torch.tensor(0.), persistent=False)

        # TODO add temporal encoding as an option
        input_dim = in_static + in_dynamic + in_prog + 4

        # Define layers

        self.conv1 = GATConv(input_dim, hidden_dim, heads=1)
        self.conv2 = GATConv(hidden_dim, hidden_dim, heads=1)
        self.conv3 = GATConv(hidden_dim, hidden_dim, heads=1)
        self.conv4 = GATConv(hidden_dim, hidden_dim, heads=1)
        self.conv5 = GATConv(hidden_dim, hidden_dim, heads=1)

        self.fc1 = nn.Linear(hidden_dim, out_prog)
        self.fc2 = nn.Linear(hidden_dim, out_diag)

        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU()

        if self.pretrained is not None:
            print('initialize weights from pretrained model {} ...'.format(self.pretrained))
            checkpoint = torch.load(self.pretrained, map_location='cpu')
            state_dict = checkpoint['model_state_dict']
            self.load_state_dict(state_dict, strict=True)
            del state_dict, checkpoint
            torch.cuda.empty_cache()

    def predict(self, x_static: torch.tensor, x_dynamic: torch.tensor, x_prog: torch.tensor,
                x_time: torch.tensor, edge_index: torch.tensor) -> Tuple[tensor, tensor]:

        combined = torch.cat((x_static, x_dynamic, x_prog, x_time.float()), dim=-1)

        x = self.act(self.conv1(combined, edge_index))
        x = self.act(self.conv2(x, edge_index))
        x = self.dropout(self.act(self.conv3(x, edge_index)))
        x_prog_inc = self.act(self.conv4(x, edge_index))
        x_diag = self.act(self.conv5(x, edge_index))

        x_prog_inc = self.fc1(x_prog_inc)
        x_diag = self.fc2(x_diag)

        return x_prog_inc, x_diag

    def forward(self, x_static: torch.tensor, x_dynamic: torch.tensor, x_prog: torch.tensor, x_time: torch.tensor,
                edge_index: torch.tensor, x_prog_inc=None, x_diag=None):

        logits_prog_inc, logits_diag = self.predict(x_static, x_dynamic, x_prog, x_time, edge_index)
        # logits_prog_inc = self._transform(logits_prog_inc, self.mu_norm, self.std_norm)

        if x_prog_inc is not None:
            loss_prog = self.MSE_loss(logits_prog_inc, x_prog_inc)
        else:
            loss_prog = self.zero

        if x_diag is not None:
            loss_diag = self.MSE_loss(logits_diag, x_diag)
        else:
            loss_diag = self.zero

        #logits_prog_inc = self._inv_transform(logits_prog_inc, self.mu_norm, self.std_norm)

        return logits_prog_inc, logits_diag, loss_prog, loss_diag

    @staticmethod
    def _transform(x: torch.tensor, mean: torch.tensor, std: torch.tensor) -> torch.tensor:
        """
        Normalize data with mean and standard deviation. The normalization is done as x_norm = (x - mean) / std

        Args:
            x (torch.tensor): Tensor to be normalized
            mean (torch.tensor): Mean to be used for the normalization
            std (torch.tensor): Standard deviation to be used for the normalization
        Returns:
            x_norms (torch.tensor): Tensor with normalized values
        """
        x_norm = (x - mean) / (std + 1e-5)
        return x_norm

    @staticmethod
    def _inv_transform(x_norm: torch.tensor, mean: torch.tensor, std: torch.tensor) -> torch.tensor:
        """
        Denormalize data with mean and standard deviation. The de-normalization is done as x = (x_norm * std) + mean

        Args:
            x_norm (torch.tensor): Tensor with normalized values
            mean (torch.tensor): Mean to be used for the de-normalization
            std (torch.tensor): Standard deviation to be used for the de-normalization
        Returns:
            x (torch.tensor): Tensor with denormalized values
        """
        x = (x_norm * (std + 1e-5)) + mean
        return x

    def MSE_loss(self, logits, labels):
        criterion = nn.MSELoss()
        return criterion(logits, labels)


if __name__ == '__main__':

    with open(r'../configs/config.yaml') as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    if config['devices'] != "-1":
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(config['devices'])
        device = 'cuda'
    else:
        device = 'cpu'

    x_dynamic = torch.randn((10051, 12), device=device)
    x_static = torch.randn((10051, 24), device=device)
    x_prog = torch.randn((10051, 12), device=device)
    x_time = torch.randn((10051, 4), device=device)
    edge_index = torch.randint(0, 10051, (2, 10051), device=device)

    model = GAT(in_static=24,
                in_dynamic=12,
                in_prog=12,
                out_prog=12,
                out_diag=5,
                hidden_dim=172,
                dropout=0.15,
                #head=4,
                mu_norm=0.,
                std_norm=1.,
                pretrained=config['pretrained']
                ).to(device)

    print(model)
    # model.eval()
    n_parameters = sum(p.numel() for p in model.parameters())  # if p.requires_grad)
    print(f"number of parameters: {n_parameters}")

    x_prog, x_diag, loss_prog, loss_diag = model(x_static, x_dynamic, x_prog, x_time, edge_index)

    print(x_prog.shape)
    print(x_diag.shape)

