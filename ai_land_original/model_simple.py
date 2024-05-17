# Class for scaling features/targets
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

# Define the config for the experiment
PATH_NAME = os.path.dirname(os.path.abspath(__file__))
with open(f"{PATH_NAME}/config.yaml") as stream:
    try:
        CONFIG = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

torch.cuda.empty_cache()


def make_map_val_plot(pred_arr, targ_arr, lat_arr, lon_arr, name_lst):
    fig, axes = plt.subplots(
        nrows=3,
        ncols=int(np.ceil(len(name_lst) / 3)),
        figsize=(18, 9),
    )

    map_errs = 100 * np.abs((pred_arr - targ_arr) / (targ_arr + 1e-5))
    # map_errs = 100 * np.abs((pred_arr - targ_arr) / np.mean(targ_arr))
    mape = np.mean(map_errs, axis=(0, 1))

    for i, axis in enumerate(axes.flatten()):
        if i < len(name_lst):
            var = name_lst[i]
            c = axis.scatter(
                lon_arr[::1],
                lat_arr[::1],
                c=mape[::1, name_lst.index(var)],
                vmin=0,
                vmax=100,
                s=1,
            )
            plt.colorbar(c)
            axis.set_title(f"MAPE {var}")
        else:
            axis.set_axis_off()

    fig.tight_layout()
    return fig


def r2_score_multi(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Calculated the r-squared score between 2 arrays of values.

    :param y_pred: predicted array :param y_true: "truth" array :return: r-squared
    metric
    """
    return r2_score(y_pred.flatten(), y_true.flatten())


# Define a neural network model with hidden layers and activation functions
class NonLinearRegression(pl.LightningModule):
    def __init__(
        self,
        input_size_clim,
        input_size_met,
        input_size_state,
        hidden_size,
        output_size,
        diag_output_size,
        mu_norm=0,
        std_norm=1,
        dataset=None,
    ):
        super().__init__()
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

    def predict_step(
        self, clim_feats, met_feats, states, diagnostics
    ) -> Tuple[tensor, tensor]:
        """Given arrays of features produces a prediction for all timesteps.

        :return: (prognost_targets, diagnostic_targets)
        """
        preds = states.clone().to(self.device)
        preds_diag = diagnostics.clone().to(self.device)
        # preds = torch.zeros_like(states).to(self.device)
        # preds_diag = torch.zeros_like(diagnostics).to(self.device)
        # preds[0] = states[0]
        len_run = preds.shape[0]

        for x in range(len_run):
            preds_dx, preds_diag_x = self.forward(
                clim_feats, met_feats[[x]], preds[[x]]
            )
            if x < (len_run - 1):
                preds[x + 1] = preds[x] + preds_dx
            preds_diag[x] = preds_diag_x
        return preds, preds_diag

    def MSE_loss(self, logits, labels):
        criterion = nn.MSELoss()
        # criterion = nn.SmoothL1Loss()
        return criterion(logits, labels)

    def training_step(self, train_batch, batch_idx):
        x_clim, x_met, x_state, y, y_diag = train_batch
        logits, logits_diag = self.forward(x_clim, x_met, x_state)
        mean = self.mu_norm.to(self.device)
        std = self.std_norm.to(self.device)
        loss = self.MSE_loss(
            self.transform(logits, mean, std), self.transform(y, mean, std)
        )
        loss_diag = self.MSE_loss(logits_diag, y_diag)
        self.log(
            "train_loss",
            loss,
        )
        self.log(
            "train_diag_loss",
            loss_diag,
        )

        if CONFIG["roll_out"] > 1:
            x_state_rollout = x_state.clone()
            y_rollout = y.clone()
            y_rollout_diag = y_diag.clone()
            for step in range(CONFIG["roll_out"]):
                # x = [batch_size=8, lookback (7) + rollout (3) = 10, n_feature = 37]
                x0 = x_state_rollout[
                    :, step, :, :
                ].clone()  # select input with lookback.
                y_hat, y_hat_diag = self.forward(
                    x_clim[:, step, :, :], x_met[:, step, :, :], x0
                )  # prediction at rollout step
                y_rollout_diag[:, step, :, :] = y_hat_diag
                if step < CONFIG["roll_out"] - 1:
                    x_state_rollout[:, step + 1, :, :] = (
                        x_state_rollout[:, step, :, :].clone() + y_hat
                    )  # overwrite x with prediction.
                y_rollout[:, step, :, :] = y_hat  # overwrite y with prediction.
            step_loss = self.MSE_loss(
                self.transform(y_rollout, mean, std), self.transform(y, mean, std)
            )
            step_loss_diag = self.MSE_loss(y_rollout_diag, y_diag)
            # step_loss = step_loss / ROLLOUT
            self.log(
                "step_loss",
                step_loss,
            )
            self.log(
                "step_loss_diag",
                step_loss_diag,
            )
            loss += step_loss
            loss_diag += step_loss_diag

        return loss + loss_diag  # + loss_abs

    def validation_step(self, val_batch, batch_idx):
        x_clim, x_met, x_state, y, y_diag = val_batch
        mean = self.mu_norm.to(self.device)
        std = self.std_norm.to(self.device)
        logits, logits_diag = self.forward(x_clim, x_met, x_state)
        loss = self.MSE_loss(
            self.transform(logits, mean, std), self.transform(y, mean, std)
        )
        loss_diag = self.MSE_loss(logits_diag, y_diag)
        r2 = r2_score_multi(
            self.transform(logits, mean, std).cpu(),
            self.transform(y, mean, std).cpu(),
        )
        r2_diag = r2_score_multi(
            logits_diag.cpu(),
            y_diag.cpu(),
        )
        self.log("val_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val_R2", r2, on_step=False, on_epoch=True, sync_dist=True)
        self.log(
            "val_diag_loss", loss_diag, on_step=False, on_epoch=True, sync_dist=True
        )
        self.log("val_diag_R2", r2_diag, on_step=False, on_epoch=True, sync_dist=True)

        if CONFIG["roll_out"] > 1:
            x_state_rollout = x_state.clone()
            y_rollout = y.clone()
            y_rollout_diag = y_diag.clone()
            for step in range(CONFIG["roll_out"]):
                # x = [batch_size=8, lookback (7) + rollout (3) = 10, n_feature = 37]
                x0 = x_state_rollout[
                    :, step, :, :
                ].clone()  # select input with lookback.
                y_hat, y_hat_diag = self.forward(
                    x_clim[:, step, :, :], x_met[:, step, :, :], x0
                )  # prediction at rollout step
                y_rollout_diag[:, step, :, :] = y_hat_diag
                if step < CONFIG["roll_out"] - 1:
                    x_state_rollout[:, step + 1, :, :] = (
                        x_state_rollout[:, step, :, :].clone() + y_hat
                    )  # overwrite x with prediction.
                y_rollout[:, step, :, :] = y_hat  # overwrite y with prediction.
            step_loss = self.MSE_loss(
                self.transform(y_rollout, mean, std), self.transform(y, mean, std)
            )
            step_loss_diag = self.MSE_loss(y_rollout_diag, y_diag)
            # step_loss = step_loss / ROLLOUT
            self.log(
                "val_step_loss",
                step_loss,
            )
            self.log(
                "val_step_loss_diag",
                step_loss_diag,
            )
            loss += step_loss
            loss_diag += step_loss_diag

        if ((self.current_epoch + 1) % CONFIG["logging"]["plot_freq"] == 0) & (
            batch_idx == 0
        ):
            self.log_fig_mlflow(x_state_rollout, x_state, mean, std)

    @rank_zero_only
    def log_fig_mlflow(self, logits, y, mean, std):
        fig = make_map_val_plot(
            # (x_state + logits).cpu().numpy(),
            # (x_state + y).cpu().numpy(),
            # self.transform(logits, mean, std).cpu().numpy(),
            # self.transform(y, mean, std).cpu().numpy(),
            logits.cpu().numpy(),
            y.cpu().numpy(),
            self.ds.lats,
            self.ds.lons,
            self.ds.targ_lst,
        )
        if CONFIG["logging"]["logger"] == "mlflow":
            self.logger.experiment.log_figure(
                self.logger.run_id,
                fig,
                f"map_val_epoch{self.current_epoch + 1}.png",
            )
        else:
            fig.savefig(
                f"{CONFIG['logging']['location']}/plots/map_val_epoch{self.current_epoch + 1}.png"
            )
        plt.close(fig)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
