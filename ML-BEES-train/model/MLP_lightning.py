# ------------------------------------------------------------------
# Script to build the model Multi-Layer Perceptron (MLP)
# ------------------------------------------------------------------

from typing import Tuple
import pytorch_lightning as pl
import torch
import torch.nn as nn
import yaml
import os
from metrics import r2_score_multi
#from pytorch_lightning.utilities.rank_zero import rank_zero_only
from torch import tensor
torch.cuda.empty_cache()
# ------------------------------------------------------------------

class LitMLP(pl.LightningModule):
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
        hidden_size: int = 172,
        dropout: float = 0.15,
        rollout: int = 6,
        mu_norm: float = 0.,
        std_norm: float = 1.,
    ):
        super(LitMLP, self).__init__()

        # Initialize
        self.in_static = in_static
        self.in_dynamic = in_dynamic
        self.in_prog = in_prog
        self.out_prog = out_prog
        self.out_diag = out_diag
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.rollout = rollout

        self.mu_norm = tensor(mu_norm).to(self.device)
        self.std_norm = tensor(std_norm).to(self.device)

        input_dim = in_static + in_dynamic + in_prog

        # Define layers
        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.LeakyReLU()
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.relu3 = nn.LeakyReLU()
        self.fc4 = nn.Linear(hidden_size, out_prog)
        self.fc5 = nn.Linear(hidden_size, out_diag)

    def forward(self, x_static: torch.tensor, x_dynamic: torch.tensor, x_prog: torch.tensor) -> Tuple[tensor, tensor]:

        if self.rollout > 1:
            x_static = x_static.repeate(1, self.rollout, 1, 1)

        combined = torch.cat((x_static, x_dynamic, x_prog), dim=-1)
        x = self.relu1(self.fc1(combined))
        x = self.dropout(self.relu2(self.fc2(x)))
        x = self.relu3(self.fc3(x))
        x_prog = self.fc4(x)
        x_diag = self.fc5(x)
        return x_prog, x_diag


    def predict_step(self, x_static, x_dynamic, x_prog, x_diag) -> Tuple[tensor, tensor]:
        """
        Given arrays of features produces predictions for all timesteps
        :return: (prognost_targets, diagnostic_targets)
        """
        preds = x_prog.clone().to(self.device)
        preds_diag = x_diag.clone().to(self.device)
        len_run = preds.shape[0]

        for t in range(len_run):
            preds_dx, preds_diag_x = self.forward(x_static, x_dynamic[[t]], preds[[t]])
            if t < (len_run - 1):
                preds[t + 1] = preds[t] + self._inv_transform(preds_dx, self.mu_norm, self.std_norm)
            preds_diag[t] = preds_diag_x
        return preds, preds_diag

    def training_step(self, train_batch, batch_idx):

        x_dynamic, x_prog, x_prog_inc, x_diag, x_static, _ = train_batch
        logits, logits_diag = self.forward(x_static, x_dynamic, x_prog)
        #loss = self.MSE_loss(self._transform(logits, self.mu_norm, self.std_norm),
        #                     self._transform(x_prog_inc, self.mu_norm, self.std_norm))
        loss = self.MSE_loss(self._inv_transform(logits, self.mu_norm, self.std_norm), x_prog_inc)
        loss_diag = self.MSE_loss(logits_diag, x_diag)

        self.log("train_loss", loss)
        self.log("train_diag_loss", loss_diag)

        if self.rollout > 1:
            x_state_rollout = x_prog.clone()
            y_rollout = x_prog_inc.clone()
            y_rollout_diag = x_diag.clone()
            for step in range(self.rollout):
                # select input with lookback
                x0 = x_state_rollout[:, step, :, :].clone()
                # prediction at rollout step
                y_hat, y_hat_diag = self.forward(x_static[:, step, :, :], x_dynamic[:, step, :, :], x0)
                y_rollout_diag[:, step, :, :] = y_hat_diag

                y_hat = self._inv_transform(y_hat, self.mu_norm, self.std_norm)

                if step < self.rollout - 1:
                    # overwrite x with prediction
                    x_state_rollout[:, step + 1, :, :] = x_state_rollout[:, step, :, :].clone() + y_hat

                # overwrite y with prediction
                y_rollout[:, step, :, :] = y_hat

                step_loss = self.MSE_loss(y_rollout, x_prog_inc)
                step_loss_diag = self.MSE_loss(y_rollout_diag, x_diag)
                # step_loss = step_loss / ROLLOUT
      #          self.log("step_loss", step_loss)
       #         self.log("step_loss_diag", step_loss_diag)
                loss += step_loss
                loss_diag += step_loss_diag

        return loss + loss_diag

    def validation_step(self, val_batch, batch_idx):

        x_dynamic, x_prog, x_prog_inc, x_diag, x_static, _ = val_batch
        logits, logits_diag = self.forward(x_static, x_dynamic, x_prog)

        logits = self._inv_transform(logits, self.mu_norm, self.std_norm)

        loss = self.MSE_loss(logits, x_prog_inc)
        loss_diag = self.MSE_loss(logits_diag, x_diag)

        r2 = r2_score_multi(logits.cpu(), x_prog_inc.cpu())
        r2_diag = r2_score_multi(logits_diag.cpu(), x_diag.cpu())

        self.log("val_loss", loss, on_step=False, on_epoch=True, rank_zero_only=True)
        self.log("val_R2", r2, on_step=False, on_epoch=True, rank_zero_only=True)
        self.log("val_diag_loss", loss_diag, on_step=False, on_epoch=True, rank_zero_only=True)
        self.log("val_diag_R2", r2_diag, on_step=False, on_epoch=True, rank_zero_only=True)

        if self.rollout > 1:
            x_state_rollout = x_prog.clone()
            y_rollout = x_prog_inc.clone()
            y_rollout_diag = x_diag.clone()
            for step in range(self.rollout):
                # select input with lookback
                x0 = x_state_rollout[:, step, :, :].clone()
                # prediction at rollout step
                y_hat, y_hat_diag = self.forward(x_static[:, step, :, :], x_dynamic[:, step, :, :], x0)
                y_rollout_diag[:, step, :, :] = y_hat_diag

                y_hat = self._inv_transform(y_hat, self.mu_norm, self.std_norm)

                if step < self.rollout - 1:
                    # overwrite x with prediction
                    x_state_rollout[:, step + 1, :, :] = (x_state_rollout[:, step, :, :].clone() + y_hat)
                # overwrite y with prediction
                y_rollout[:, step, :, :] = y_hat

                step_loss = self.MSE_loss(y_rollout, x_prog_inc)
                step_loss_diag = self.MSE_loss(y_rollout_diag, x_diag)
                # step_loss = step_loss / ROLLOUT
                self.log("val_step_loss", step_loss, on_step=False, on_epoch=True, rank_zero_only=True)
                self.log("val_step_loss_diag", step_loss_diag, on_step=False, on_epoch=True, rank_zero_only=True)
                loss += step_loss
                loss_diag += step_loss_diag

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

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


if __name__ == '__main__':

    with open(r'../configs/config.yaml') as stream:
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

    model = LitMLP(in_static=22,
                   in_dynamic=12,
                   in_prog=7,
                   out_prog=7,
                   out_diag=3,
                   hidden_size=172,
                   rollout=6,
                   dropout=0.15,
                   mu_norm=0.,
                   std_norm=1.).to(device)

    print(model)
    #model.eval()
    n_parameters = sum(p.numel() for p in model.parameters())  # if p.requires_grad)
    print(f"number of parameters: {n_parameters}")

    x_prog, x_diag = model(x_static, x_dynamic, x_prog)

    print(x_prog.shape)
    print(x_diag.shape)

