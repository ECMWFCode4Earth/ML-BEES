# ------------------------------------------------------------------
# Script to build the model Multi-Layer Perceptron (MLP)
# ------------------------------------------------------------------

from typing import Tuple
# import pytorch_lightning as pl
import torch
import torch.nn as nn
import yaml
import os
# from metrics import r2_score_multi
from torch import tensor

torch.cuda.empty_cache()


# ------------------------------------------------------------------

class MLP_Obs(nn.Module):
    """
        MLP model for EC-Land dataset
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
            rollout: int = 6,
            mu_norm: float = 0.,
            std_norm: float = 1.,
            swvl1_idx: int = None,
            skt_idx: int = None,
            pretrained: str = None,
    ):
        super(MLP_Obs, self).__init__()

        # Initialize
        self.in_static = in_static
        self.in_dynamic = in_dynamic
        self.in_prog = in_prog
        self.out_prog = out_prog
        self.out_diag = out_diag
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.rollout = rollout
        self.pretrained = None if pretrained == "None" else pretrained
        self.register_buffer('mu_norm', tensor(mu_norm))
        self.register_buffer('std_norm', tensor(std_norm))
        self.register_buffer('zero', torch.tensor(0.), persistent=False)

        self.swl1_idx = swvl1_idx
        self.skt_idx = skt_idx
        #self._swvl1_idx_list = [False if i != self.swl1_idx else True for i in range(self.out_prog)]
        #self._skt_idx_list = [False if i != self.skt_idx else True for i in range(self.out_diag)]

        # TODO add temporal encoding as an option
        input_dim = in_static + in_dynamic + in_prog + 4

        # Define layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu2 = nn.LeakyReLU(0.2)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.norm3 = nn.BatchNorm1d(hidden_dim)
        self.relu3 = nn.LeakyReLU(0.2)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.relu4 = nn.LeakyReLU(0.2)
        self.fc5 = nn.Linear(hidden_dim, out_prog)
        # self.fc6 = nn.Linear(hidden_dim, hidden_dim)
        # self.relu6 = nn.LeakyReLU()
        self.fc7 = nn.Linear(hidden_dim, out_diag)

        if self.pretrained is not None:
            print('initialize weights from pretrained model {} ...'.format(self.pretrained))
            checkpoint = torch.load(self.pretrained, map_location='cpu')
            state_dict = checkpoint['model_state_dict']
            self.load_state_dict(state_dict, strict=True)
            del state_dict, checkpoint
            torch.cuda.empty_cache()

    def predict(self, x_static: torch.tensor, x_dynamic: torch.tensor, x_prog: torch.tensor,
                x_time: torch.tensor) -> Tuple[tensor, tensor]:

        combined = torch.cat((x_static, x_dynamic, x_prog, x_time.float()), dim=-1)
        # combined = torch.cat((x_static, x_dynamic, x_prog), dim=-1)
        x = self.relu1(self.fc1(combined))
        skip = x
        x = self.relu2(self.fc2(x))

        if x.ndim != 4:
            B, P, C = x.shape
            x = self.relu3(self.norm3(self.fc3(x).view(B * P, C)).view(B, P, C))
        else:
            B, T, P, C = x.shape
            x = self.relu3(self.norm3(self.fc3(x).view(B * T * P, C)).view(B, T, P, C))

        x = x + skip
        x = self.relu4(self.fc4(x))
        x_prog_inc = self.fc5(x)
        x_diag = self.fc7(x)
        return x_prog_inc, x_diag

    def forward(self, x_static: torch.tensor, x_dynamic: torch.tensor, x_prog: torch.tensor, x_time: torch.tensor,
                x_prog_inc=None, x_diag=None, x_prog_obs_inc=None, x_diag_obs=None):

        if self.training:
            if x_static.shape[1] != self.rollout:
                x_static = x_static.repeat(1, self.rollout, 1, 1)

        x_time = x_time.unsqueeze(2).repeat(1, 1, x_static.shape[2], 1)

        logits_prog_inc, logits_diag = self.predict(x_static, x_dynamic, x_prog, x_time)
        # logits_prog_inc = self._transform(logits_prog_inc, self.mu_norm, self.std_norm)

        # remove invalid observation points
        if x_prog_obs_inc is not None:
            x_prog_obs_inc = x_prog_obs_inc[:, :, :, 0]
            valid_x_prog_obs_inc = ~torch.isnan(x_prog_obs_inc)
            x_prog_obs_inc = x_prog_obs_inc[valid_x_prog_obs_inc]
        if x_diag_obs is not None:
            x_diag_obs = x_diag_obs[:, :, :, 0]
            valid_x_diag_obs = ~torch.isnan(x_diag_obs)
            x_diag_obs = x_diag_obs[valid_x_diag_obs]

        # compute the losses

        if x_prog_inc is not None:
            loss_prog = self.MSE_loss(logits_prog_inc, x_prog_inc)
        else:
            loss_prog = self.zero

        if x_diag is not None:
            loss_diag = self.MSE_loss(logits_diag, x_diag)
        else:
            loss_diag = self.zero

        if x_prog_obs_inc is not None and len(x_prog_obs_inc) > 1:
            loss_prog_obs = self.MSE_loss(logits_prog_inc[:, :, :, self.swl1_idx][valid_x_prog_obs_inc], x_prog_obs_inc)
        else:
            loss_prog_obs = self.zero

        if x_diag_obs is not None and len(x_diag_obs) > 1:
            loss_diag_obs = self.MSE_loss(logits_diag[:, :, :, self.skt_idx][valid_x_diag_obs], x_diag_obs)

        else:
            loss_diag_obs = self.zero

        # TODO check when rollout > 1
        if x_prog_inc is not None and x_diag is not None and self.rollout > 1 and self.training:
            x_state_rollout = x_prog.clone()
            y_rollout = x_prog_inc.clone()
            y_rollout_diag = x_diag.clone()
            # y_rollout_diag = []
            for step in range(self.rollout):
                # select input with lookback
                x0 = x_state_rollout[:, step, :, :].clone()
                # prediction at rollout step
                y_hat, y_hat_diag = self.predict(x_static[:, step, :, :], x_dynamic[:, step, :, :], x0,
                                                 x_time[:, step, :, :])
                y_rollout_diag[:, step, :, :] = y_hat_diag.clone()
                # y_rollout_diag.append(y_hat_diag[:, None, :, :])

                # y_hat = self._inv_transform(y_hat, self.mu_norm, self.std_norm)

                if step < self.rollout - 1:
                    # overwrite x with prediction
                    x_state_rollout[:, step + 1, :, :] = (x_state_rollout[:, step, :, :].clone() +
                                                          self._inv_transform(y_hat, self.mu_norm, self.std_norm))

                # overwrite y with prediction
                y_rollout[:, step, :, :] = y_hat.clone()

            # y_rollout_diag = torch.cat(y_rollout_diag, dim=1)

            step_loss_prog = self.MSE_loss(y_rollout, x_prog_inc)
            step_loss_diag = self.MSE_loss(y_rollout_diag, x_diag)

            if x_prog_obs_inc is not None and len(x_prog_obs_inc) > 1:
                step_loss_prog_obs = self.MSE_loss(y_rollout[:, :, :, self.swl1_idx][valid_x_prog_obs_inc], x_prog_obs_inc)
            else:
                step_loss_prog_obs = self.zero

            if x_diag_obs is not None and len(x_diag_obs) > 1:
                step_loss_diag_obs = self.MSE_loss(y_rollout_diag[:, :, :, self.skt_idx][valid_x_diag_obs], x_diag_obs)
            else:
                step_loss_diag_obs = self.zero

            loss_prog = loss_prog + step_loss_prog + step_loss_prog_obs
            loss_diag = loss_diag + step_loss_diag + step_loss_diag_obs

        # logits_prog_inc = self._inv_transform(logits_prog_inc, self.mu_norm, self.std_norm)

        return logits_prog_inc, logits_diag, loss_prog + loss_prog_obs, loss_diag + loss_diag_obs

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

    x_dynamic = torch.randn((2, 6, 10051, 12), device=device)
    x_static = torch.randn((2, 6, 10051, 24), device=device)
    x_prog = torch.randn((2, 6, 10051, 7), device=device)
    x_time = torch.randn((2, 6, 4), device=device)

    model = MLP_Obs(in_static=24,
                in_dynamic=12,
                in_prog=7,
                out_prog=7,
                out_diag=3,
                hidden_dim=256,
                rollout=1,
                dropout=0.15,
                mu_norm=0.,
                std_norm=1.,
                # pretrained=config['pretrained']
                ).to(device)

    print(model)
    # model.eval()
    n_parameters = sum(p.numel() for p in model.parameters())  # if p.requires_grad)
    print(f"number of parameters: {n_parameters}")

    x_diag = torch.randn((2, 6, 10051, 3), device=device)
    x_prog_inc = torch.randn((2, 6, 10051, 7), device=device)
    x_prog, x_diag, loss_prog, loss_diag = model(x_static, x_dynamic, x_prog, x_time, x_prog_inc, x_diag)

    print(x_prog.shape)
    print(x_diag.shape)

