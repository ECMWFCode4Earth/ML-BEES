# ------------------------------------------------------------------
# Modified from ML-BEES-train for running sensitivity tests
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
from datetime import datetime
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
                 is_add_lat_lon: bool = True, is_norm: bool = True, point_dropout: float = 0.0):
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
            is_add_lat_lon (bool): Whether to add lat and lon positional encoding as static features
                                   the encoding adds 4 additional features to the static features based on a sinusoidal encoding
            is_norm (bool): Whether to normalize the data
            point_dropout (float): Ratio of data points to be dropped randomly

        """
        super().__init__()

        # initialize
        self.root = root
        self.start_year = start_year
        self.end_year = end_year
        self.x_slice_indices = x_slice_indices
        self.roll_out = roll_out

        # None is made a keyword for the config used at the time of writing this code
        if clim_features == None:
            self.clim_features = ['clim_clake', 'clim_ldepth', 'clim_ctype', 'clim_cell_area',
                                  'clim_landsea', 'clim_glm', 'clim_cu', 'clim_cvh', 'clim_cvl', 
                                  'clim_geopot', 'clim_sdfor', 'clim_sdor', 'clim_theta_cap', 
                                  'clim_theta_pwp', 'clim_veg_covh', 'clim_veg_covl', 
                                  'clim_veg_z0mh', 'clim_veg_z0ml', 'clim_veg_rsminh', 'clim_veg_rsminl']
        else:
            self.clim_features = clim_features
        
        if dynamic_features == None:
            self.dynamic_features = ['met_ctpf',
                                    'met_lwdown',
                                    'met_psurf',
                                    'met_qair',
                                    'met_rainf',
                                    'met_swdown',
                                    'met_snowf',
                                    'met_tair',
                                    'met_wind_e',
                                    'met_wind_n',
                                    'lai_hv',
                                    'lai_lv']
        else:
            self.dynamic_features = dynamic_features

        if target_prog_features == None:
            self.target_prog_features = ['swvl1',
                                        'swvl2',
                                        'swvl3',
                                        'stl1',
                                        'stl2',
                                        'stl3',
                                        'snowc',
                                        'aco2gpp',
                                        'dis',
                                        'e',
                                        'sro',
                                        'ssro']
        else:
            self.target_prog_features = target_prog_features
        
        if target_diag_features == None:
            self.target_diag_features = ['d2m', 't2m', 'skt', 'sshf', 'slhf']
        else:
            self.target_diag_features = target_diag_features

        self.is_add_lat_lon = is_add_lat_lon
        self.is_norm = is_norm
        self.point_dropout = point_dropout

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
        self.x = self.ds_ecland["x"][slice(*self.x_slice_indices)]
        self.x_size = len(self.x)
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
        self.data_time = np.zeros((len(self.times), 4))
        for t in range(len(self.times)):
            self.data_time[t] = self._encode_time(self.times[t])

        self.data_time = self.data_time.astype(float)

        # normalize data
        if is_norm:
            self.x_dynamic_means = self.ds_ecland.data_means[self.dynamic_index]
            self.x_dynamic_stdevs = self.ds_ecland.data_stdevs[self.dynamic_index]
            self.y_prog_means = self.ds_ecland.data_means[self.targ_prog_index]
            self.y_prog_stdevs = self.ds_ecland.data_stdevs[self.targ_prog_index]
            self.y_diag_means = self.ds_ecland.data_means[self.targ_diag_index]
            self.y_diag_stdevs = self.ds_ecland.data_stdevs[self.targ_diag_index]
            self.clim_means = self.ds_ecland.clim_means[self.clim_index]
            self.clim_stdevs = self.ds_ecland.clim_stdevs[self.clim_index]
            self.data_static = EcDataset.transform(self.data_static, self.clim_means, self.clim_stdevs)

            # get statistic to normalize the output data_prognostic_inc
            self.y_prog_inc_mean = self.ds_ecland.data_1stdiff_means[self.targ_prog_index] / (self.y_prog_stdevs + 1e-5)
            self.y_prog_inc_std = self.ds_ecland.data_1stdiff_stdevs[self.targ_prog_index] / (self.y_prog_stdevs + 1e-5)

        # add latitude and longitude features to data_static
        if is_add_lat_lon:
            lat = self.lat.reshape(1, self.x_size, 1)
            lon = self.lon.reshape(1, self.x_size, 1)
            encoded_lat = np.concatenate((np.sin(lat * np.pi / 180), np.cos(lat * np.pi / 180)), axis=-1)
            encoded_lon = np.concatenate((np.sin(lon * np.pi / 180), np.cos(lon * np.pi / 180)), axis=-1)

            self.data_static = np.concatenate((self.data_static, encoded_lat, encoded_lon), axis=-1)

        # get number of points to be dropped
        if self.point_dropout > 0:
            self._is_dropout = True
            self._dropout_samples = int(self.x_size * (1 - self.point_dropout))
        else:
            self._is_dropout = False

    @staticmethod
    def _encode_time(x_time: np.datetime64) -> np.array:
        """
        Convert datetime64 to (day of the year, hour)

        Args:
            x_time (np.datetime64): Date time YYYY-MM-DDTXX:XX:XX
        Returns:
            numpy array [4,] representing the [sin(day of the year), cos(day of the year), sin(hour), cos(hour)]
        """
        x_time = str(x_time)
        year, month, day, hour = int(x_time[:4]), int(x_time[5:7]), int(x_time[8:10]), int(x_time[11:13])
        day_of_year = datetime(year, month, day).timetuple().tm_yday

        return np.array([np.sin(day_of_year * np.pi/183),
                         np.cos(day_of_year * np.pi/183),
                         np.sin(hour * np.pi/12),
                         np.cos(hour * np.pi/12)])

    @staticmethod
    def transform(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
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

    @staticmethod
    def inv_transform(x_norm: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
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
            data_time (np.array): numpy array [rollout, 4] representing the time at the current state [sin(day-of-year), cos(day-of-year), sin(hour), cos(hour)]
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
        data_diagnostic[np.isnan(data_diagnostic)] = 0 # some random nans in sshf...

        # normalize data
        if self.is_norm:
            data_dynamic = EcDataset.transform(data_dynamic, self.x_dynamic_means, self.x_dynamic_stdevs)
            data_prognostic = EcDataset.transform(data_prognostic, self.y_prog_means, self.y_prog_stdevs)
            data_diagnostic = EcDataset.transform(data_diagnostic, self.y_diag_means, self.y_diag_stdevs)

        # drop points from the current time step
        if self._is_dropout:
            random_indices = np.random.choice(self.x_size, size=self._dropout_samples, replace=False)

            data_dynamic = data_dynamic[:, random_indices, :]
            data_prognostic = data_prognostic[:, random_indices, :]
            data_diagnostic = data_diagnostic[:, random_indices, :]
            data_static = data_static[:, random_indices, :]

        # get delta_x update for corresponding x state
        data_prognostic_inc = data_prognostic[1:, :, :] - data_prognostic[:-1, :, :]
        if self.is_norm:
            data_prognostic_inc = EcDataset.transform(data_prognostic_inc, self.y_prog_inc_mean, self.y_prog_inc_std)

        return (data_dynamic[:-1], data_prognostic[:-1], data_prognostic_inc, data_diagnostic[:-1],
                data_static, data_time)

    def __len__(self) -> int:
        """
        Method to get the number of time steps in the dataset

        Returns:
            the length of the dataset
        """
        return self.len_dataset - self.roll_out + 1

    @property
    def n_static(self):
        """
        getter method to get the number of static features in the dataset

        Returns:
            number of static features
        """
        return len(self.clim_features) + 4 if self.is_add_lat_lon else len(self.clim_features)

    @property
    def n_dynamic(self):
        """
        getter method to get the number of dynamic features in the dataset

        Returns:
            number of dynamic features
        """
        return len(self.dynamic_features)


    @property
    def n_prog(self):
        """
        getter method to get the number of target prognostic features in the dataset

        Returns:
            number of target prognostic features
        """
        return len(self.targ_prog_index)

    @property
    def n_diag(self):
        """
        getter method to get the number of target diagnostic features in the dataset

        Returns:
            number of target diagnostic features
        """
        return len(self.target_diag_features)