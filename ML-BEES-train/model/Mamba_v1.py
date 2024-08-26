# ------------------------------------------------------------------
# Script to build the Mamba v1 model
# ------------------------------------------------------------------

from typing import Tuple
import torch
import torch.nn as nn
import yaml
import os
from torch import tensor
from mamba_ssm import Mamba

torch.cuda.empty_cache()

# ------------------------------------------------------------------

class Mamba_v1(nn.Module):
    """ Mamba model for EC-Land dataset """

    def __init__(
            self,
            in_static: int = 22,
            in_dynamic: int = 12,
            in_prog: int = 7,
            out_prog: int = 7,
            out_diag: int = 3,
            hidden_dim: int = 32,
            rollout: int = 6,
            d_state: int = 64,
            d_conv: int = 4,
            expand: int = 2,
            dt_min: float = 0.01,
            dt_max: float = 0.1,
            mu_norm: float = 0.,
            std_norm: float = 1.,
            pretrained: str = None,
    ):

        """
        Args:
            in_static (int): number of the input static features
            in_dynamic (int): number of the input dynamic features
            in_prog (int): number of the input prognostic features
            out_prog (int): number of the output prognostic features
            out_diag (int): number of the output prognostic features
            hidden_dim (int): hidden dimension of the model
            rollout (int): number of rollouts
            d_state (int): SSM state expansion factor
            d_conv (int): local convolution width
            expand (int): block expansion factor
            dt_min (float): minimum delta t
            dt_max (float): maximum delta t
            mu_norm (float): mean of the output prognostic variables to be used for normalization
            std_norm (float): standard deviations of the output prognostic variables to be used for normalization
            pretrained (str): model checkpoint
        """

        super(Mamba_v1, self).__init__()

        # Initialize
        self.in_static = in_static
        self.in_dynamic = in_dynamic
        self.in_prog = in_prog
        self.out_prog = out_prog
        self.out_diag = out_diag
        self.hidden_dim = hidden_dim
        self.rollout = rollout

        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.dt_min = dt_min
        self.dt_max = dt_max

        self.pretrained = None if pretrained == "None" else pretrained
        self.register_buffer('mu_norm', tensor(mu_norm))
        self.register_buffer('std_norm', tensor(std_norm))
        self.register_buffer('zero', torch.tensor(0.), persistent=False)

        input_dim = in_static + in_dynamic + in_prog + 4  # 4 is because of the temporal encoding

        # Define layers

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.mamba1 = Mamba(
            d_model=hidden_dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dt_min=dt_min,
            dt_max=dt_max)

        self.norm2 = nn.LayerNorm(hidden_dim)
        self.mamba2 = Mamba(
            d_model=hidden_dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dt_min=dt_min,
            dt_max=dt_max)

        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu2 = nn.LeakyReLU(0.2)
        self.fc3 = nn.Linear(hidden_dim, out_prog)
        self.fc4 = nn.Linear(hidden_dim, out_diag)

        if self.pretrained is not None:
            print('initialize weights from pretrained model {} ...'.format(self.pretrained))
            checkpoint = torch.load(self.pretrained, map_location='cpu')
            state_dict = checkpoint['model_state_dict']
            self.load_state_dict(state_dict, strict=True)
            del state_dict, checkpoint
            torch.cuda.empty_cache()

    def predict(self, x_static: torch.tensor, x_dynamic: torch.tensor, x_prog: torch.tensor,
                x_time: torch.tensor) -> Tuple[tensor, tensor]:
        """
        Args:
            x_static (torch.tensor) : static features [batch size, rollout or 1, points, in_static]
            x_dynamic (torch.tensor): dynamic features [batch size, rollout, points, in_dynamic]
            x_prog (torch.tensor): initial state of prognostic variables [batch size, rollout, points, in_prog]
            x_time (torch.tensor): temporal encoding [batch size, rollout, 4]
        Returns:
            logits_prog_inc (torch.tensor): predicted increments for prognostic variables [batch size, 1, points, out_prog]
            logits_diag (torch.tensor): predicted diagnostic variables [batch size, 1, points, out_diag]
        """

        combined = torch.cat((x_static, x_dynamic, x_prog, x_time.float()), dim=-1)
        B, L, P, C = combined.shape
        combined = combined.permute(0, 2, 1, 3).reshape(B*P, L, C)

        x = self.relu1(self.fc1(combined))
        skip = x
        x = self.norm1(x)
        x = self.mamba1(x)
        x = x + skip
        skip = x
        x = self.norm2(x)
        x = self.mamba2(x)
        x = x + skip
        x = x.reshape(B, P, L, self.hidden_dim).permute(0, 2, 1, 3)[:, -1:, :, :]
        x = self.relu2(self.fc2(x))
        x_prog_inc = self.fc3(x)
        x_diag = self.fc4(x)

        return x_prog_inc, x_diag


    def forward(self, x_static: torch.tensor, x_dynamic: torch.tensor, x_prog: torch.tensor, x_time: torch.tensor,
                x_prog_inc: torch.tensor = None, x_diag: torch.tensor = None):
        """
        Args:
            x_static (torch.tensor) : static features [batch size, rollout or 1, points, in_static]
            x_dynamic (torch.tensor): dynamic features [batch size, rollout, points, in_dynamic]
            x_prog (torch.tensor): initial state of prognostic variables [batch size, rollout, points, in_prog]
            x_time (torch.tensor): temporal encoding [batch size, rollout, 4]
            x_prog_inc (torch.tensor) [optional]: target increments for prognostic variables [batch size, rollout, points, out_prog]
            x_diag (torch.tensor) [optional]: target diagnostic variables [batch size, rollout, points, out_diag]
        Returns:
            logits_prog_inc (torch.tensor): predicted increments for prognostic variables [batch size, 1, points, out_prog]
            logits_diag (torch.tensor): predicted diagnostic variables [batch size, 1, points, out_diag]
            loss_prog (torch.tensor): loss for the predicted increments for prognostic variables
            loss_diag (torch.tensor): loss for the predicted diagnostic variables
        """

        if x_static.shape[1] != self.rollout:
            x_static = x_static.repeat(1, self.rollout, 1, 1)
        x_time = x_time.unsqueeze(2).repeat(1, 1, x_static.shape[2], 1)

        logits_prog_inc, logits_diag = self.predict(x_static, x_dynamic, x_prog, x_time)
        #logits_prog_inc = self._transform(logits_prog_inc, self.mu_norm, self.std_norm)

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

    def MSE_loss(self, logits: torch.tensor, labels: torch.tensor):
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

    x_dynamic = torch.randn((2, 32, 10051, 12), device=device)
    x_static = torch.randn((2, 32, 10051, 24), device=device)
    x_prog = torch.randn((2, 32, 10051, 7), device=device)
    x_time = torch.randn((2, 32, 4), device=device)

    model = Mamba_v1(in_static=24,
                     in_dynamic=12,
                     in_prog=7,
                     out_prog=7,
                     out_diag=3,
                     hidden_dim=32,
                     rollout=32,
                     mu_norm=0.,
                     std_norm=1.,
                     # pretrained=config['pretrained']
                     ).to(device)

    print(model)
    # model.eval()
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"number of parameters: {n_parameters}")

    x_diag = torch.randn((2, 1, 10051, 3), device=device)
    x_prog_inc = torch.randn((2, 1, 10051, 7), device=device)
    x_prog, x_diag, loss_prog, loss_diag = model(x_static, x_dynamic, x_prog, x_time, x_prog_inc, x_diag)

    print(x_prog.shape)
    print(x_diag.shape)
