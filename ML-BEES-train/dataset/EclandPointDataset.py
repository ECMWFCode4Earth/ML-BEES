# ------------------------------------------------------------------
# Script to load point-based data from EC-Land dataset
# ------------------------------------------------------------------

#from typing import Tuple
import cftime
import numpy as np
import pandas as pd
#import os
#import pytorch_lightning as pl
import torch
import yaml
import zarr
#import xarray as xr
#from torch import tensor
from torch.utils.data import DataLoader, Dataset
# ------------------------------------------------------------------

class EcDataset(Dataset):
    """
    Dataset class for EC-Land dataset

    Methods:
        _encode_time(): Private method to convert date time to Month, Day, Hour
        transform(): Method to normalize data with mean and standard deviation
        inv_transform(): Method to denormalize data with mean and standard deviation
        __getitem__(): Method to load datacube by the time step index
        __len__(): Method to get the number of time steps in the dataset
    """
    def __init__(self, start_year: int = 2015, end_year: int = 2020, x_slice_indices: tuple = (0, None),
                 root: str = None, roll_out: int = 6, clim_features: list = None, dynamic_features: list = None,
                 target_prog_features: list = None, target_diag_features: list = None,
                 is_add_lat_lon: bool = True, is_norm: bool = True):
        """
        Args:
            start_year (int): Start year
            end_year (int): End year
            x_slice_indices (tuple): Indices to slice the point by x coordinates
            root (str): Path to dataset in zarr format
            roll_out (int): Rollout in the future
            clim_features (list): A list of static features
            dynamic_features (list): A list of dynamic features
            target_prog_features (list): A list of target prognostic features
            target_diag_features (list): A list of target diagnostic features
            is_add_lat_lon (bool): Whether to add lat and lon coordinates as static features
            is_norm (bool): Whether to normalize the data
        """
        super().__init__()

        # initialize
        self.root = root
        self.start_year = start_year
        self.end_year = end_year
        self.x_slice_indices = x_slice_indices
        self.roll_out = roll_out

        self.clim_features = clim_features
        self.dynamic_features = dynamic_features
        self.target_prog_features = target_prog_features
        self.target_diag_features = target_diag_features

        self.is_add_lat_lon = is_add_lat_lon
        self.is_norm = is_norm

        # open the dataset
        self.ds_ecland = zarr.open(root)

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
        self.times = np.array(date_times[self.start_index: self.end_index + 1], dtype=np.datetime64)
        self.x_size = len(self.ds_ecland["x"][slice(*self.x_slice_indices)])
        self.lat = self.ds_ecland["lat"][slice(*self.x_slice_indices)]
        self.lon = self.ds_ecland["lon"][slice(*self.x_slice_indices)]

        # get lists of indices for the features
        # list of climatological time-invariant features
        self.clim_index = [list(self.ds_ecland["clim_variable"]).index(x) for x in self.clim_features]
        # list of features that change in time
        self.dynamic_index = [list(self.ds_ecland["variable"]).index(x) for x in self.dynamic_features]
        # list of prognostic target features
        self.targ_prog_index = [list(self.ds_ecland["variable"]).index(x) for x in self.target_prog_features]
        # list of diagnostic target features
        self.targ_diag_index = [list(self.ds_ecland["variable"]).index(x) for x in self.target_diag_features]

        # get time-invariant static climatological features
        self.data_static = self.ds_ecland.clim_data[slice(*self.x_slice_indices), self.clim_index].reshape(1, self.x_size,
                                                                                                        -1).data  # [10051, 20]
        # get time features
        self.data_time = np.zeros((len(self.times), 3))
        for t in range(len(self.times)):
            self.data_time[t] = self._encode_time(self.times[t])

        # normalize data
        if is_norm:
            self.x_dynamic_means = self.ds_ecland.data_means[self.dynamic_index]
            self.x_dynamic_stdevs = self.ds_ecland.data_stdevs[self.dynamic_index]
            self.y_prog_means = self.ds_ecland.data_means[self.targ_prog_index]
            self.y_prog_stdevs = self.ds_ecland.data_stdevs[self.targ_prog_index]
            self.y_diag_means = self.ds_ecland.data_means[self.targ_diag_index]
            self.y_diag_stdevs = self.ds_ecland.data_stdevs[self.targ_diag_index]
            clim_means = self.ds_ecland.clim_means[self.clim_index]
            clim_stdevs = self.ds_ecland.clim_stdevs[self.clim_index]
            self.data_static = self.transform(self.data_static, clim_means,
                                              clim_stdevs)  #.reshape(1, self.x_size, -1).data  # [1, 10051, 20]

            self.data_time = self.data_time / np.array([12, 31, 24])

        # add latitude and longitude features to data_static
        if is_add_lat_lon:
            if is_norm:
                self.lat_scaled = self.transform(self.lat, np.mean(self.lat), np.std(self.lat)).reshape(1, self.x_size,
                                                                                                        1)
                self.lon_scaled = self.transform(self.lon, np.mean(self.lon), np.std(self.lon)).reshape(1, self.x_size,
                                                                                                        1)
                self.data_static = np.concatenate((self.data_static,
                                                   self.lat_scaled,
                                                   self.lon_scaled), axis=-1)
            else:
                self.data_static = np.concatenate((self.data_static,
                                                   self.lat.reshape(1, self.x_size, 1),
                                                   self.lon.reshape(1, self.x_size, 1)), axis=-1)

    @staticmethod
    def _encode_time(x_time: np.datetime64) -> np.array:
        """
        Convert datetime64 to Month, Day, Hour

        Args:
            x_time (np.datetime64): Date time YYYY-MM-DDTXX:XX:XX
        Returns:
            numpy array [3,] representing the [Month, Day, Hour]
        """
        x_time = str(x_time)
        #year = int(data_time[:4])
        month = int(x_time[5:7])
        day = int(x_time[8:10])
        hour = int(x_time[11:13])

        return np.array([month, day, hour])

    def transform(self, x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
        """
        Normalize data with mean and standard deviation. The normalization is done as x_norm = (x - mean) / std

        Args:
            x (np.ndarray): Numpy array to be normalized
            mean (np.ndarray): Mean to be used for the normalization
            std (np.ndarray): Standard deviation to be used for the normalization
        Returns:
            x_norms (np.ndarray): Numpy array with normalized values
        """
        x_norm = (x - mean) / (std + 1e-5)
        return x_norm

    def inv_transform(self, x_norm: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
        """
        Denormalize data with mean and standard deviation. The de-normalization is done as x = (x_norm * std) + mean

        Args:
            x_norm (np.ndarray): Numpy array with normalized values
            mean (np.ndarray): Mean to be used for the de-normalization
            std (np.ndarray): Standard deviation to be used for the de-normalization
        Returns:
            x (np.ndarray): Numpy array with denormalized values
        """
        x = (x_norm * (std + 1e-5)) + mean
        return x

    def __getitem__(self, idx):
        """
        Method to load datacube by the time step index

        Args:
            idx (int): index of the time step
        Returns:
            data_dynamic (np.array): numpy array [rollout, x_size, dynamic features] representing the dynamic features
            data_prognostic (np.array): numpy array [rollout, x_size, prognostic features] representing the target prognostic features
            data_prognostic_inc (np.array): numpy array [rollout, x_size, prognostic features] representing the target update of the prognostic features
            data_diagnostic (np.array): numpy array [rollout, x_size, diagnostic features] representing the target diagnostic features
            data_static (np.array): numpy array [1, x_size, static features] representing the static features
            data_time (np.array): numpy array [rollout, 3] representing the time at the current state [month, day, hour]
       """

        # get time features
        data_time = self.data_time[idx:idx + self.roll_out]
        # get index
        idx = idx + self.start_index
        ds_slice = self.ds_ecland.data[slice(idx, idx + self.roll_out + 1), slice(*self.x_slice_indices), :]  # [7, 10051, :]
        # get dynamic features
        data_dynamic = ds_slice[:, :, self.dynamic_index]  # [7, 10051, 12]
        # get static features
        data_static = self.data_static.copy()
        # get prognostic target features
        data_prognostic = ds_slice[:, :, self.targ_prog_index]
        # get diagnostic target features
        data_diagnostic = ds_slice[:, :, self.targ_diag_index]

        # normalize data
        if self.is_norm:
            data_dynamic = self.transform(data_dynamic, self.x_dynamic_means, self.x_dynamic_stdevs)
            data_prognostic = self.transform(data_prognostic, self.y_prog_means, self.y_prog_stdevs)
            data_diagnostic = self.transform(data_diagnostic, self.y_diag_means, self.y_diag_stdevs)

        # get delta_x update for corresponding x state
        data_prognostic_inc = data_prognostic[1:, :, :] - data_prognostic[:-1, :, :]

        # TODO add dropout ratio to data

        return (data_dynamic[:-1], data_prognostic[:-1], data_prognostic_inc, data_diagnostic[:-1],
                data_static, data_time)

    def __len__(self):
        """
        Method to get the number of time steps in the dataset

        Returns:
            the length of the dataset
        """
        return self.len_dataset - self.roll_out + 1


if __name__ == "__main__":

    with open(r'../config.yaml') as stream:
        try:
            CONFIG = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    dataset = EcDataset(start_year=2020,  #CONFIG["start_year"],
                        end_year=2020,  #CONFIG["end_year"],
                        x_slice_indices=CONFIG["x_slice_indices"],
                        root=CONFIG["file_path"],
                        roll_out=2,  #CONFIG["roll_out"],
                        clim_features=CONFIG["clim_feats"],
                        dynamic_features=CONFIG["dynamic_feats"],
                        target_prog_features=CONFIG["targets_prog"],
                        target_diag_features=CONFIG["targets_diag"],
                        is_add_lat_lon=True,
                        is_norm=True,
                        )

    print('number of sampled data:', dataset.__len__())
    print('data dynamic shape:', dataset.__getitem__(0)[0].shape)
    print('targets prognosis shape:', dataset.__getitem__(0)[1].shape)
    print('targets prognosis inc shape:', dataset.__getitem__(0)[2].shape)
    print('targets diagnosis shape:', dataset.__getitem__(0)[3].shape)
    print('data static shape:', dataset.__getitem__(0)[4].shape)
    print('data time shape:', dataset.__getitem__(0)[5].shape)

    is_test_run = False
    is_plot = True

    if is_test_run:

        import time
        import random

        manual_seed = 0
        random.seed(manual_seed)
        torch.manual_seed(manual_seed)
        torch.cuda.manual_seed_all(manual_seed)

        time.sleep(2)

        train_loader = torch.utils.data.DataLoader(dataset,
                                                   batch_size=1,
                                                   shuffle=False,
                                                   pin_memory=False,
                                                   num_workers=8,
                                                   prefetch_factor=1)

        end = time.time()

        for i, (
        data_dynamic, data_prognostic, data_prognostic_inc, data_diagnostic, data_static, data_time) in enumerate(
                train_loader):
            print('time: {}/{}--{}'.format(i + 1, len(train_loader), time.time() - end))
            end = time.time()

    if is_plot:
        import matplotlib as mpl
        import matplotlib.pyplot as plt

        mpl.use('TkAgg')

        s = 8

        for i in range(len(dataset)):

            print('time step ', i)

            data_dynamic, data_prognostic, data_prognostic_inc, data_diagnostic, data_static, data_time = dataset[i]

            # plot dynamic features
            for v in range(data_dynamic.shape[-1]):
                plt.scatter(dataset.lon, dataset.lat, c=data_dynamic[0, :, v], linewidths=0, s=s)
                plt.title(dataset.dynamic_features[v] + ', rollout=1')
                plt.colorbar()
                plt.show()

            # plot target prognostic features
            for v in range(data_prognostic.shape[-1]):
                plt.scatter(dataset.lon, dataset.lat, c=data_prognostic[0, :, v], linewidths=0, s=s)
                plt.title(dataset.target_prog_features[v] + ', rollout=1')
                plt.colorbar()
                plt.show()

            # plot target prognostic features
            for v in range(data_diagnostic.shape[-1]):
                plt.scatter(dataset.lon, dataset.lat, c=data_diagnostic[0, :, v], linewidths=0, s=s)
                plt.title(dataset.target_diag_features[v] + ', rollout=1')
                plt.colorbar()
                plt.show()

            # plot static features
            for v in range(len(dataset.clim_features)):
                plt.scatter(dataset.lon, dataset.lat, c=data_static[0, :, v], linewidths=0, s=s)
                plt.title(dataset.clim_features[v])
                plt.colorbar()
                plt.show()

            # plot lat and lon features if they exist
            if dataset.is_add_lat_lon:
                fig, axs = plt.subplots(1, 2)
                lat = axs[0].scatter(dataset.lon, dataset.lat, c=data_static[0, :, -2], linewidths=0, s=s)
                axs[0].set_title('latitude')
                plt.colorbar(lat, ax=axs[0])
                lon = axs[1].scatter(dataset.lon, dataset.lat, c=data_static[0, :, -1], linewidths=0, s=s)
                axs[1].set_title('longitude')
                plt.colorbar(lon, ax=axs[1])

                plt.show()

