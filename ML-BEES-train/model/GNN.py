# ------------------------------------------------------------------
# Graph Neural Network
# ------------------------------------------------------------------

from typing import Tuple
import torch
import torch.nn as nn
import yaml
import os
from torch import tensor
from torch_geometric.nn import TransformerConv, GATConv, GATv2Conv, GCNConv, SuperGATConv, ARMAConv
torch.cuda.empty_cache()

# ------------------------------------------------------------------

class GNN_block(nn.Module):
    def __init__(self, layer: str = 'UniMP', in_channels: int = 64, out_channels: int = 64, heads: int = 4):
        super(GNN_block, self).__init__()

        self.layer = layer
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads

        self.norm = nn.LayerNorm(in_channels)

        if layer == 'UniMP':
            conv_layer = TransformerConv  # Unified Message Passing Model (https://arxiv.org/abs/2009.03509)
        elif layer == 'GCN':
            conv_layer = GCNConv  # Graph Convolutional Networks (https://arxiv.org/abs/1609.02907)
        elif layer == 'GAT':
            conv_layer = GATConv  # Graph Attention Networks (https://arxiv.org/abs/1710.10903)
        elif layer == 'GATv2':
            conv_layer = GATv2Conv  # Graph Attention Networks V2 (https://arxiv.org/abs/2105.14491)
        elif layer == 'ARMA':
            conv_layer = ARMAConv  # Graph Neural Networks with auto-regressive moving average (ARMA) filters (https://arxiv.org/abs/1901.01343)
        elif layer == 'SuperGAT':
            conv_layer = SuperGATConv  # Self-supervised Graph Attention Networks (https://openreview.net/forum?id=Wi5KUNlqWty)
        else:
            raise ValueError('Unexpected model {}'.format(layer))

        if layer in ['UniMP', 'GAT', 'GATv2']:
            self.conv = conv_layer(in_channels, out_channels, heads=heads)
        else:
            self.conv = conv_layer(in_channels, out_channels)
        self.act = nn.ReLU()

    def forward(self, x: torch.tensor, edge_index: torch.tensor) -> torch.tensor:
        x = x + self.act(self.conv(self.norm(x), edge_index))
        return x



class GNN(nn.Module):
    """ Graph Neural Network model for EC-Land dataset """
    def __init__(
            self,
            model: str = 'UniMP',
            in_static: int = 22,
            in_dynamic: int = 12,
            in_prog: int = 7,
            out_prog: int = 7,
            out_diag: int = 3,
            hidden_dim: int = 32,
            dropout: float = 0.1,
            heads: int = 4,
            rollout: int = 6,
            mu_norm: float = 0.,
            std_norm: float = 1.,
            pretrained: str = None,
    ):
        super(GNN, self).__init__()
        """
        Args:
            model (str): name of the graph neural network
            in_static (int): number of the input static features
            in_dynamic (int): number of the input dynamic features
            in_prog (int): number of the input prognostic features
            out_prog (int): number of the output prognostic features
            out_diag (int): number of the output prognostic features
            hidden_dim (int): hidden dimension of the model
            head (int): multi-attention heads of the model
            rollout (int): number of rollouts
            dropout (float): dropout ratio
            mu_norm (float): mean of the output prognostic variables to be used for normalization
            std_norm (float): standard deviations of the output prognostic variables to be used for normalization
            pretrained (str): model checkpoint
        """
        # Initialize
        self.in_static = in_static
        self.in_dynamic = in_dynamic
        self.in_prog = in_prog
        self.out_prog = out_prog
        self.out_diag = out_diag
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.heads = heads if model in ['UniMP', 'GAT', 'GATv2'] else 1
        self.rollout = rollout
        self.pretrained = None if pretrained == "None" else pretrained
        self.register_buffer('mu_norm', tensor(mu_norm))
        self.register_buffer('std_norm', tensor(std_norm))
        self.register_buffer('zero', torch.tensor(0.), persistent=False)

        input_dim = in_static + in_dynamic + in_prog + 4

        # Define layers
        assert model in ['UniMP', 'GCN', 'GAT', 'GATv2', 'ARMA', 'SuperGAT'], ('{} is not supported. '
        'Supported models are \'UniMP\', \'GCN\', \'GAT\', \'GATv2\', \'ARMA\', or \'SuperGAT\'').format(model)

        self.layer = model
        self.proj = nn.Linear(input_dim, hidden_dim*self.heads, bias=False)

        self.layers = nn.ModuleList()
        # TODO make number of layers dynamic
        for l in range(4):
            self.layers.append(GNN_block(self.layer, hidden_dim*self.heads, hidden_dim, self.heads))

        self.fc1 = nn.Linear(hidden_dim*self.heads, hidden_dim*self.heads)
        self.fc2 = nn.Linear(hidden_dim*self.heads, hidden_dim*self.heads)
        self.fc3 = nn.Linear(hidden_dim*self.heads, out_prog)
        self.fc4 = nn.Linear(hidden_dim*self.heads, out_diag)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.LeakyReLU(0.2)

        if self.pretrained is not None:
            print('initialize weights from pretrained model {} ...'.format(self.pretrained))
            checkpoint = torch.load(self.pretrained, map_location='cpu')
            state_dict = checkpoint['model_state_dict']
            self.load_state_dict(state_dict, strict=True)
            del state_dict, checkpoint
            torch.cuda.empty_cache()

    def predict(self, x_static: torch.tensor, x_dynamic: torch.tensor, x_prog: torch.tensor,
                x_time: torch.tensor, edge_index: torch.tensor) -> Tuple[tensor, tensor]:
        """
        Args:
            x_static (torch.tensor) : static features [batch size * points, in_static]
            x_dynamic (torch.tensor): dynamic features [batch size * points, in_dynamic]
            x_prog (torch.tensor): initial state of prognostic variables [batch size * points, in_prog]
            x_time (torch.tensor): temporal encoding [batch size * points, 4]
            edge_index (torch.tensor): edge connections [2, number of edges]
        Returns:
            logits_prog_inc (torch.tensor): predicted increments for prognostic variables [batch size * points, out_prog]
            logits_diag (torch.tensor): predicted diagnostic variables [batch size * points, out_diag]
        """

        combined = torch.cat((x_static, x_dynamic, x_prog, x_time.float()), dim=-1)

        x = self.proj(combined)
        x = self.layers[0](x, edge_index)
        x = self.layers[1](x, edge_index)

        x_prog_inc = self.dropout(self.layers[2](x, edge_index))
        x_diag = self.dropout(self.layers[3](x, edge_index))

        x_prog_inc = self.act(self.fc1(x_prog_inc))
        x_diag = self.act(self.fc2(x_diag))
        x_prog_inc = self.fc3(x_prog_inc)
        x_diag = self.fc4(x_diag)

        return x_prog_inc, x_diag

    def forward(self, x_static: torch.tensor, x_dynamic: torch.tensor, x_prog: torch.tensor, x_time: torch.tensor,
                edge_index: torch.tensor, x_prog_inc: torch.tensor = None, x_diag: torch.tensor = None):
        """
        Args:
            x_static (torch.tensor) : static features [batch size * points, rollout, in_static]
            x_dynamic (torch.tensor): dynamic features [batch size * points, rollout, in_dynamic]
            x_prog (torch.tensor): initial state of prognostic variables [batch size * points, rollout, in_prog]
            x_time (torch.tensor): temporal encoding [batch size * points, rollout, 4]
            edge_index (torch.tensor): edge connections [2, number of edges]
            x_prog_inc (torch.tensor) [optional]: target increments for prognostic variables [batch size * points, rollout, out_prog]
            x_diag (torch.tensor) [optional]: target diagnostic variables [batch size * points, rollout, out_diag]
        Returns:
            logits_prog_inc (torch.tensor): predicted increments for prognostic variables [batch size * points, out_prog]
            logits_diag (torch.tensor): predicted diagnostic variables [batch size * points, out_diag]
            loss_prog (torch.tensor): loss for the predicted increments for prognostic variables
            loss_diag (torch.tensor): loss for the predicted diagnostic variables

            prediction is computed only for the first timestep
        """

        if x_static.ndim == 3:
            logits_prog_inc, logits_diag = self.predict(x_static[:, 0, ...], x_dynamic[:, 0, ...],
                                                        x_prog[:, 0, ...], x_time[:, 0, ...], edge_index)
        else:
            logits_prog_inc, logits_diag = self.predict(x_static, x_dynamic, x_prog, x_time, edge_index)

        #logits_prog_inc = self._transform(logits_prog_inc, self.mu_norm, self.std_norm)

        if x_prog_inc is not None:
            loss_prog = self.MSE_loss(logits_prog_inc, x_prog_inc[:, 0, :])
        else:
            loss_prog = self.zero

        if x_diag is not None:
            loss_diag = self.MSE_loss(logits_diag, x_diag[:, 0, :])
        else:
            loss_diag = self.zero

        if x_prog_inc is not None and x_diag is not None and self.rollout > 1 and self.training:
            x_state_rollout = x_prog.clone()
            y_rollout = x_prog_inc.clone()
            y_rollout_diag = x_diag.clone()
            for step in range(self.rollout):
                # select input with lookback
                x0 = x_state_rollout[:, step, :].clone()
                # prediction at rollout step
                y_hat, y_hat_diag = self.predict(x_static[:, step, :], x_dynamic[:, step, :], x0, x_time[:, step, :], edge_index)
                y_rollout_diag[:, step, :] = y_hat_diag.clone()

                if step < self.rollout - 1:
                    # overwrite x with prediction
                    x_state_rollout[:, step + 1, :] = (x_state_rollout[:, step, :].clone() +
                                                       self._inv_transform(y_hat, self.mu_norm, self.std_norm))

                # overwrite y with prediction
                y_rollout[:, step, :] = y_hat.clone()

            step_loss_prog = self.MSE_loss(y_rollout, x_prog_inc)
            step_loss_diag = self.MSE_loss(y_rollout_diag, x_diag)

            loss_prog += step_loss_prog
            loss_diag += step_loss_diag

        #logits_prog_inc = self._transform(logits_prog_inc, self.mu_norm, self.std_norm)

        return logits_prog_inc, logits_diag, loss_prog, loss_diag

    @staticmethod
    def _transform(x: torch.tensor, mean: torch.tensor, std: torch.tensor) -> torch.tensor:
        """
        Normalize data with mean and standard deviation. The normalization is done as x_norm = (x - mean) / std

        Args:
            x (torch.tensor): tensor to be normalized
            mean (torch.tensor): mean to be used for the normalization
            std (torch.tensor): standard deviation to be used for the normalization
        Returns:
            x_norms (torch.tensor): tensor with normalized values
        """
        x_norm = (x - mean) / (std + 1e-5)
        return x_norm

    @staticmethod
    def _inv_transform(x_norm: torch.tensor, mean: torch.tensor, std: torch.tensor) -> torch.tensor:
        """
        Denormalize data with mean and standard deviation. The de-normalization is done as x = (x_norm * std) + mean

        Args:
            x_norm (torch.tensor): tensor with normalized values
            mean (torch.tensor): mean to be used for the de-normalization
            std (torch.tensor): standard deviation to be used for the de-normalization
        Returns:
            x (torch.tensor): tensor with denormalized values
        """
        x = (x_norm * (std + 1e-5)) + mean
        return x

    def MSE_loss(self, logits, labels):
        """
        compute the mean squared error (squared L2 norm) between the input logits and target labels

        Args:
            logits (torch.tensor): prediction
            labels (torch.tensor): ground truth

        Returns:
            MSE_loss (torch.tensor): mean squared error
        """
        criterion = nn.MSELoss(reduction='mean')
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

    x_dynamic = torch.randn((10051, 2, 12), device=device)
    x_static = torch.randn((10051, 2, 24), device=device)
    x_prog = torch.randn((10051, 2, 12), device=device)
    x_time = torch.randn((10051, 2, 4), device=device)
    edge_index = torch.randint(0, 10051, (2, 10051), device=device)

    for layer in ['UniMP', 'GCN', 'GAT', 'GATv2', 'ARMA', 'SuperGAT']:
        model = GNN(model=layer,
                    in_static=24,
                    in_dynamic=12,
                    in_prog=12,
                    out_prog=12,
                    out_diag=5,
                    hidden_dim=32,
                    rollout=2,
                    dropout=0.15,
                    heads=3,
                    mu_norm=0.,
                    std_norm=1.,
                    #pretrained=config['pretrained']
                    ).to(device)

        print(model)
        # model.eval()
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"number of parameters: {n_parameters}")

        x_prog_out, x_diag_out, loss_prog, loss_diag = model(x_static, x_dynamic, x_prog, x_time, edge_index)

        print(x_prog_out.shape)
        print(x_diag_out.shape)

