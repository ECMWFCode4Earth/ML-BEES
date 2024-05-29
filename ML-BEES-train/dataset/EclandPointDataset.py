import os
from typing import Tuple

import cftime
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import yaml
import zarr
import xarray as xr
from torch import tensor
from torch.utils.data import DataLoader, Dataset

# ------------------------------------------------------------------

class EcDataset(Dataset):
    # load the dataset
    def __init__(self, start_year=2015, end_year=2020, x_slice_indices=(0, None), root=None, roll_out=6,
                 clim_features=None, dynamic_features=None, target_features=None, target_diag_features=None):
        super().__init__()

        # initialize
        self.root = root
        self.start_year = start_year
        self.end_year = end_year
        self.x_slice_indices = x_slice_indices
        self.roll_out = roll_out

        self.clim_features = clim_features
        self.dynamic_features = dynamic_features
        self.target_prog_features = target_features
        self.target_diag_features = target_diag_features

        # open the dataset
        self.ds_ecland = zarr.open(root)

        print(list(self.ds_ecland.keys()))
        print(xr.open_dataset(root, engine='zarr'))

        # create time index to select appropriate data range
        date_times = pd.to_datetime(cftime.num2pydate(self.ds_ecland["time"], self.ds_ecland["time"].attrs["units"]))

        # create time index to select appropriate data range
        self.start_index = min(np.argwhere(date_times.year == int(start_year)))[0]
        self.end_index = max(np.argwhere(date_times.year == int(end_year)))[0]

        # get the number of samples
        self.len_dataset = self.end_index - self.start_index

        # select points in space
        self.x_slice_indices = (0, None) if "None" in x_slice_indices else x_slice_indices

        # slice time and space
        self.times = np.array(date_times[self.start_index: self.end_index])
        self.x_size = len(self.ds_ecland["x"][slice(*self.x_slice_indices)])
        self.lat = self.ds_ecland["lat"][slice(*self.x_slice_indices)]
        self.lon = self.ds_ecland["lon"][slice(*self.x_slice_indices)]

        # get lists of indices for the features
        # list of climatological time-invariant features
        self.clim_index = [list(self.ds_ecland["clim_variable"]).index(x) for x in self.clim_features]
        # list of features that change in time
        self.dynamic_index = [list(self.ds_ecland["variable"]).index(x) for x in self.dynamic_features]
        # list of prognostic target features
        self.targ_index = [list(self.ds_ecland["variable"]).index(x) for x in self.target_prog_features]
        # list of diagnostic target features
        self.targ_diag_index = [list(self.ds_ecland["variable"]).index(x) for x in self.target_diag_features]

        # define the statistics used for normalising the data
        self.x_dynamic_means = self.ds_ecland.data_means[self.dynamic_index]
        self.x_dynamic_stdevs = self.ds_ecland.data_stdevs[self.dynamic_index]

        # create time-invariant static climatological features
        x_static = self.ds_ecland.clim_data[slice(*self.x_slice_indices), self.clim_index]  # [10051, 20]
        clim_means = self.ds_ecland.clim_means[self.clim_index]
        clim_stdevs = self.ds_ecland.clim_stdevs[self.clim_index]
        self.x_static_scaled = self.transform(x_static, clim_means, clim_stdevs).reshape(1, self.x_size, -1)  # [1, 10051, 20]

        # define statistics for normalising the targets
        self.y_prog_means = self.ds_ecland.data_means[self.targ_index]
        self.y_prog_stdevs = self.ds_ecland.data_stdevs[self.targ_index]
        self.y_diag_means = self.ds_ecland.data_means[self.targ_diag_index]
        self.y_diag_stdevs = self.ds_ecland.data_stdevs[self.targ_diag_index]

        print(self.len_dataset - self.roll_out - 1)

        print(self.times[:8])

        quit()

    def transform(self, x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
        """Transform data with mean and stdev.

        :param x: data :param mean: mean :param std: standard deviation :return:
        normalised data
        """
        x_norm = (x - mean) / (std + 1e-5)
        return x_norm

    def inv_transform(self, x_norm: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
        """Inverse transform on data with mean and stdev.

        :param x_norm: normlised data :param mean: mean :param std: standard deviation
        :return: unnormalised data
        """
        x = (x_norm * (std + 1e-5)) + mean
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


    def __getitem__(self, idx):

        # add random dropout
        # compute time features

        idx = idx + self.start_index

        ds_slice = self.ds_ecland.data[slice(idx, idx + self.roll_out + 1), slice(*self.x_slice_indices), :]  # [7, 10051, :]
        # get dynamic features
        data_dynamic = ds_slice[:, :, self.dynamic_index]  # [7, 10051, 12]
        data_dynamic = self.transform(data_dynamic, self.x_dynamic_means, self.x_dynamic_stdevs)
        # get static features
        data_static = self.x_static_scaled.copy()
        # get time features
       # data_time =
        # get prognostic target features
        data_prognostic = ds_slice[:, :, self.targ_index]
        data_prognostic = self.transform(data_prognostic, self.y_prog_means, self.y_prog_stdevs)
        # get diagnostic target features
        data_diagnostic = ds_slice[:, :, self.targ_diag_index]
        data_diagnostic = self.transform(data_diagnostic, self.y_diag_means, self.y_diag_stdevs)
        # get delta_x update for corresponding x state
        data_prognostic_inc = data_prognostic[1:, :, :] - data_prognostic[:-1, :, :]

        return data_dynamic[:-1], data_prognostic[:-1], data_prognostic_inc, data_diagnostic[:-1], data_static, data_time

    def __len__(self):
        return self.len_dataset - self.roll_out - 1

if __name__ == "__main__":

    with open(r'../config.yaml') as stream:
        try:
            CONFIG = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    dataset = EcDataset(start_year=2011,#CONFIG["start_year"],
                        end_year=2011,#CONFIG["end_year"],
                        x_slice_indices=CONFIG["x_slice_indices"],
                        root=CONFIG["file_path"],
                        roll_out=1,#CONFIG["roll_out"],
                        clim_features=CONFIG["clim_feats"],
                        dynamic_features=CONFIG["dynamic_feats"],
                        target_features=CONFIG["targets_prog"],
                        target_diag_features=CONFIG["targets_diag"],
                        )

    print('number of sampled data:', dataset.__len__())

    print('data static shape:', dataset.__getitem__(0)[0].shape)
    print('data dynamic shape:', dataset.__getitem__(0)[1].shape)
    print('targets prognosis shape:', dataset.__getitem__(0)[2].shape)
    print('targets prognosis inc shape:', dataset.__getitem__(0)[3].shape)
    print('targets diagnosis shape:', dataset.__getitem__(0)[4].shape)

    is_test_run = True

    if is_test_run:

        import time
        import random

        manual_seed = 0
        random.seed(manual_seed)
        torch.manual_seed(manual_seed)
        torch.cuda.manual_seed_all(manual_seed)

        train_loader = torch.utils.data.DataLoader(dataset,
                                                   batch_size=1,
                                                   shuffle=False,
                                                   pin_memory=False,
                                                   num_workers=8,
                                                   prefetch_factor=1)

        end = time.time()

        for i, (data_dynamic, data_prognostic, data_prognostic_inc, data_diagnostic, data_static, data_time) in enumerate(train_loader):
            print('time: {}/{}--{}'.format(i + 1, len(train_loader), time.time() - end))
            end = time.time()
