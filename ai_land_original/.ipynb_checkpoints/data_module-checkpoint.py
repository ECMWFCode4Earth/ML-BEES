import os
from typing import Tuple

import cftime
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import yaml
import zarr
from torch import tensor
from torch.utils.data import DataLoader, Dataset

# Open up experiment config
PATH_NAME = os.path.dirname(os.path.abspath(__file__))
with open(f"{PATH_NAME}/config.yaml") as stream:
    try:
        CONFIG = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

torch.cuda.empty_cache()


class EcDataset(Dataset):
    # load the dataset
    def __init__(
        self,
        start_yr=CONFIG["start_year"],
        end_yr=CONFIG["end_year"],
        x_idxs=CONFIG["x_slice_indices"],
        path=CONFIG["file_path"],
        roll_out=CONFIG["roll_out"],
    ):
        self.ds_ecland = zarr.open(path)
        # Create time index to select appropriate data range
        date_times = pd.to_datetime(
            cftime.num2pydate(
                self.ds_ecland["time"], self.ds_ecland["time"].attrs["units"]
            )
        )
        self.start_index = min(np.argwhere(date_times.year == int(start_yr)))[0]
        self.end_index = max(np.argwhere(date_times.year == int(end_yr)))[0]
        self.times = np.array(date_times[self.start_index : self.end_index])
        self.len_dataset = self.end_index - self.start_index

        # Select points in space
        self.x_idxs = (0, None) if "None" in x_idxs else x_idxs
        self.x_size = len(self.ds_ecland["x"][slice(*self.x_idxs)])
        self.lats = self.ds_ecland["lat"][slice(*self.x_idxs)]
        self.lons = self.ds_ecland["lon"][slice(*self.x_idxs)]

        # List of climatological time-invariant features
        self.static_feat_lst = CONFIG["clim_feats"]
        self.clim_index = [
            list(self.ds_ecland["clim_variable"]).index(x) for x in CONFIG["clim_feats"]
        ]
        # List of features that change in time
        self.dynamic_feat_lst = CONFIG["dynamic_feats"]
        self.dynamic_index = [
            list(self.ds_ecland["variable"]).index(x) for x in CONFIG["dynamic_feats"]
        ]
        # Prognostic target list
        self.targ_lst = CONFIG["targets_prog"]
        self.targ_index = [
            list(self.ds_ecland["variable"]).index(x) for x in CONFIG["targets_prog"]
        ]
        # Diagnostic target list
        self.targ_diag_lst = CONFIG["targets_diag"]
        self.targ_diag_index = [
            list(self.ds_ecland["variable"]).index(x) for x in CONFIG["targets_diag"]
        ]

        # Define the statistics used for normalising the data
        self.x_dynamic_means = tensor(self.ds_ecland.data_means[self.dynamic_index])
        self.x_dynamic_stdevs = tensor(self.ds_ecland.data_stdevs[self.dynamic_index])

        # Create time-invariant static climatological features
        x_static = tensor(
            self.ds_ecland.clim_data[slice(*self.x_idxs), self.clim_index]
        )
        clim_means = tensor(self.ds_ecland.clim_means[self.clim_index])
        clim_stdevs = tensor(self.ds_ecland.clim_stdevs[self.clim_index])
        self.x_static_scaled = self.transform(
            x_static, clim_means, clim_stdevs
        ).reshape(1, self.x_size, -1)

        # Define statistics for normalising the targets
        self.y_prog_means = tensor(self.ds_ecland.data_means[self.targ_index])
        self.y_prog_stdevs = tensor(self.ds_ecland.data_stdevs[self.targ_index])

        self.y_diag_means = tensor(self.ds_ecland.data_means[self.targ_diag_index])
        self.y_diag_stdevs = tensor(self.ds_ecland.data_stdevs[self.targ_diag_index])

        self.rollout = roll_out

    def transform(self, x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
        """Transform data with mean and stdev.

        :param x: data :param mean: mean :param std: standard deviation :return:
        normalised data
        """
        x_norm = (x - mean) / (std + 1e-5)
        # x_norm = (x - mean) / std
        return x_norm

    def inv_transform(
        self, x_norm: np.ndarray, mean: np.ndarray, std: np.ndarray
    ) -> np.ndarray:
        """Inverse transform on data with mean and stdev.

        :param x_norm: normlised data :param mean: mean :param std: standard deviation
        :return: unnormalised data
        """
        x = (x_norm * (std + 1e-5)) + mean
        # x = (x_norm * (std)) + mean
        return x

    def load_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load data into memory. **CAUTION ONLY USE WHEN WORKING WITH DATASET THAT FITS
        IN MEM**

        :return: static_features, dynamic_features, prognostic_targets,
        diagnostic_targets
        """
        ds_slice = tensor(
            self.ds_ecland.data[
                self.start_index : self.end_index, slice(*self.x_idxs), :
            ]
        )

        X = ds_slice[:, :, self.dynamic_index]
        X = self.transform(X, self.x_dynamic_means, self.x_dynamic_stdevs)

        X_static = self.x_static_scaled

        Y_prog = ds_slice[:, :, self.targ_index]
        Y_prog = self.transform(Y_prog, self.y_prog_means, self.y_prog_stdevs)

        Y_diag = ds_slice[:, :, self.targ_diag_index]
        Y_diag = self.transform(Y_diag, self.y_diag_means, self.y_diag_stdevs)
        return X_static, X, Y_prog, Y_diag

    # number of rows in the dataset
    def __len__(self):
        return self.len_dataset - 1 - self.rollout

    # get a row at an index
    def __getitem__(self, idx):
        idx = idx + self.start_index
        ds_slice = tensor(
            self.ds_ecland.data[
                slice(idx, idx + self.rollout + 1), slice(*self.x_idxs), :
            ]
        )

        X = ds_slice[:, :, self.dynamic_index]
        X = self.transform(X, self.x_dynamic_means, self.x_dynamic_stdevs)

        X_static = self.x_static_scaled.expand(self.rollout, -1, -1)

        Y_prog = ds_slice[:, :, self.targ_index]
        Y_prog = self.transform(Y_prog, self.y_prog_means, self.y_prog_stdevs)

        Y_diag = ds_slice[:, :, self.targ_diag_index]
        Y_diag = self.transform(Y_diag, self.y_diag_means, self.y_diag_stdevs)

        # Is it faster to slice data by time idx first and then select columns or is the below more optimal?
        # X = tensor(self.ds_ecland.data[slice(idx, idx + self.rollout + 1), slice(*self.x_idxs), self.dynamic_index])
        # X = self.transform(X, self.x_dynamic_means, self.x_dynamic_stdevs)

        # X_static = self.x_static_scaled.expand(self.rollout, -1, -1)

        # Y_prog = tensor(self.ds_ecland.data[slice(idx, idx + self.rollout + 1), slice(*self.x_idxs), self.targ_index])
        # Y_prog = self.transform(Y_prog, self.y_prog_means, self.y_prog_stdevs)

        # Y_diag = tensor(self.ds_ecland.data[slice(idx, idx + self.rollout + 1), slice(*self.x_idxs), self.targ_diag_index])
        # Y_diag = self.transform(Y_diag, self.y_diag_means, self.y_diag_stdevs)

        # Calculate delta_x update for corresponding x state
        Y_inc = Y_prog[1:, :, :] - Y_prog[:-1, :, :]
        return X_static, X[:-1], Y_prog[:-1], Y_inc, Y_diag[:-1]


class NonLinRegDataModule(pl.LightningDataModule):
    """Pytorch lightning specific data class."""

    def setup(self, stage):
        # generator = torch.Generator().manual_seed(42)
        self.train = EcDataset(start_yr=CONFIG["start_year"], end_yr=CONFIG["end_year"])
        self.test = EcDataset(
            start_yr=CONFIG["validation_start"], end_yr=CONFIG["validation_end"]
        )

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=CONFIG["batch_size"],
            shuffle=True,
            num_workers=CONFIG["num_workers"],
            persistent_workers=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.test,
            batch_size=CONFIG["batch_size"],
            shuffle=False,
            num_workers=CONFIG["num_workers"],
            persistent_workers=True,
            pin_memory=True,
        )
