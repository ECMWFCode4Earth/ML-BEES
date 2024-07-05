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
from datetime import datetime
#from torch.utils.data import Dataset
from torch_geometric.data import Dataset, Data

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
                 root: str = None, clim_features: list = None, dynamic_features: list = None,
                 target_prog_features: list = None, target_diag_features: list = None,
                 is_add_lat_lon: bool = True, is_norm: bool = True,
                 graph_type: str = 'distance-graph', k_graph: int = 8, d_graph: float = 10, max_num_points: int = 8):
        """
        Args:
            start_year (int): Start year
            end_year (int): End year
            x_slice_indices (tuple): Indices to slice the point by x coordinates
            root (str): Path to dataset in zarr format
            clim_features (list): A list of static features
            dynamic_features (list): A list of dynamic features
            target_prog_features (list): A list of target prognostic features
            target_diag_features (list): A list of target diagnostic features
            is_add_lat_lon (bool): Whether to add lat and lon positional encoding as static features
                                   the encoding adds 4 additional features to the static features based on a sinusoidal encoding
            is_norm (bool): Whether to normalize the data
            graph_type (str): Graph type either knn-graph or distance-graph.
                              If graph type is 'none', the graph will not be computed
            k_graph (int): The number of neighbors to compute the graph edges
            d_graph (float): The distance in km to compute the graph edges within it
            max_num_points (int): The maximum number of neighbored points to return for each node.
                                  Returned neighbors are chosen randomly
        """
        super().__init__()

        # initialize
        self.root = root
        self.start_year = start_year
        self.end_year = end_year
        self.x_slice_indices = x_slice_indices
        self.roll_out = 1

        self.clim_features = clim_features
        self.dynamic_features = dynamic_features
        self.target_prog_features = target_prog_features
        self.target_diag_features = target_diag_features

        self.is_add_lat_lon = is_add_lat_lon
        self.is_norm = is_norm

        self.graph_type = graph_type.lower()
        self.k_graph = k_graph
        self.d_graph = d_graph
        self.max_num_points = max_num_points

        assert graph_type.lower() in ['knn-graph', 'distance-graph']
        assert k_graph >= 1
        assert d_graph > 0
        assert max_num_points >= 1

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

        # compute graph edges
        if graph_type == 'knn-graph':
            self.edge_index = self._knn_graph(self.lat, self.lon, k=self.k_graph)
        elif graph_type == 'distance-graph':
            self.edge_index = self._distance_graph(self.lat, self.lon,
                                                   d=self.d_graph, max_num_points=self.max_num_points)

    @staticmethod
    def _knn_graph(lat: np.array, lon: np.array, k: int) -> np.array:
        """
        Computes graph edges based on the nearest k points assuming a spherical Earth.
        Args:
            lat (np.array): latitude in degree [N]
            lon (np.array): longitude in degree [N]
            k (int): The number of neighbors
        Returns:
            edge_index [2, k*N] representing the edge indices of the k-nearest neighbors on a spherical Earth
        """
        assert len(lat) == len(lon)

        R = 6371  # Earth radius in km

        lat = lat * np.pi / 180.
        lon = lon * np.pi / 180.

        # distance is based on https://en.wikipedia.org/wiki/Haversine_formula

        edge_index = np.zeros((2, k*len(lat)), dtype=int)
        edge_index[0, :] = np.repeat(np.arange(0, len(lat)), k)

        for i in range(len(lat)):
            d_lat = lat[i] - lat
            d_lon = lon[i] - lon
            a = np.sin(d_lat / 2) ** 2 + np.cos(lat[i]) * np.cos(lat) * np.sin(d_lon / 2) ** 2
            c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
            distance = R * c
            ind_knn = np.argpartition(distance, k+1)[:k+1]
            ind_knn = ind_knn[ind_knn != i]
            edge_index[1, (i*k):(i*k)+k] = ind_knn

        return edge_index


    @staticmethod
    def _distance_graph(lat: np.array, lon: np.array, d: float, max_num_points: int) -> np.array:
        """
        Computes graph edges within a given distance on spherical Earth
        Args:
            lat (np.array): latitude in degree [N]
            lon (np.array): longitude in degree [N]
            d (float): The distance in km to compute the graph edges within it
            max_num_points (int): The maximum number of neighbored points to return for each node.
                                  Returned neighbors are chosen randomly

        Returns:
            edge_index [2, ...] representing the edge indices of the nearest neighbors within a given distance on a spherical Earth
        """
        assert len(lat) == len(lon)

        R = 6371  # Earth radius in km

        lat = lat * np.pi / 180.
        lon = lon * np.pi / 180.

        # distance is based on https://en.wikipedia.org/wiki/Haversine_formula

        edge_index = None

        for i in range(len(lat)):
            d_lat = lat[i] - lat
            d_lon = lon[i] - lon
            a = np.sin(d_lat / 2) ** 2 + np.cos(lat[i]) * np.cos(lat) * np.sin(d_lon / 2) ** 2
            c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
            distance = R * c
            ind_distance = np.argwhere(distance <= d)
            ind_distance = ind_distance[ind_distance != i]

            if len(ind_distance) > max_num_points:
                random_indices = np.random.choice(len(ind_distance), size=max_num_points, replace=False)
                ind_distance = ind_distance[random_indices]

            if len(ind_distance) != 0:
                edge_index_i = np.repeat([i], len(ind_distance))
                edge_index_i = np.vstack((ind_distance, edge_index_i))

                if edge_index is None:
                    edge_index = edge_index_i
                else:
                    edge_index = np.hstack((edge_index, edge_index_i))

        return edge_index


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

        return np.array([np.sin(day_of_year * np.pi/183.),
                         np.cos(day_of_year * np.pi/183.),
                         np.sin(hour * np.pi/12.),
                         np.cos(hour * np.pi/12.)])

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

        # get delta_x update for corresponding x state
        data_prognostic_inc = data_prognostic[1:, :, :] - data_prognostic[:-1, :, :]
        if self.is_norm:
            data_prognostic_inc = EcDataset.transform(data_prognostic_inc, self.y_prog_inc_mean, self.y_prog_inc_std)#

        return Data(data_dynamic=torch.from_numpy(data_dynamic[0]),
                    data_prognostic=torch.from_numpy(data_prognostic[0]),
                    data_prognostic_inc=torch.from_numpy(data_prognostic_inc[0]),
                    data_diagnostic=torch.from_numpy(data_diagnostic[0]),
                    data_static=torch.from_numpy(data_static[0]),
                    data_time=torch.from_numpy(data_time).repeat(self.x_size, 1),
                    edge_index=torch.from_numpy(self.edge_index),
                    num_nodes=torch.from_numpy(np.array(self.x_size))
                    )

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



if __name__ == "__main__":

    with open(r'../configs/config.yaml') as stream:
        try:
            CONFIG = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    dataset = EcDataset(start_year=2021,  #CONFIG["start_year"],
                        end_year=2021,  #CONFIG["end_year"],
                        x_slice_indices=CONFIG["x_slice_indices"],
                        root=CONFIG["file_path"],
                        #roll_out=1,  #CONFIG["roll_out"],
                        clim_features=CONFIG["clim_feats"],
                        dynamic_features=CONFIG["dynamic_feats"],
                        target_prog_features=CONFIG["targets_prog"],
                        target_diag_features=CONFIG["targets_diag"],
                        is_add_lat_lon=True,
                        is_norm=True,
                        graph_type='distance-graph',
                        d_graph=100,
                        max_num_points=8,
                        #graph_type='knn-graph',
                        #k_graph=8,
                        )

    # check
    #"""
    print('number of sampled data:', dataset.__len__())
    print('data dynamic shape:', dataset.__getitem__(0).data_dynamic.shape)
    print('targets prognosis shape:', dataset.__getitem__(0).data_prognostic.shape)
    print('targets prognosis inc shape:', dataset.__getitem__(0).data_prognostic_inc.shape)
    print('targets diagnosis shape:', dataset.__getitem__(0).data_diagnostic.shape)
    print('data static shape:', dataset.__getitem__(0).data_static.shape)
    print('data time shape:', dataset.__getitem__(0).data_time.shape)
    print()
    print('number of static features:', dataset.n_static)
    print('number of dynamic features:', dataset.n_dynamic)
    print('number of target prognostic features:', dataset.n_prog)
    print('number of target diagnostic features:', dataset.n_diag)
    print()
    #"""
    is_test_run = True
    is_plot = False
    is_plot_graph = True

    if is_test_run:

        from torch_geometric.loader import DataLoader
        import time
        import random

        manual_seed = 0
        random.seed(manual_seed)
        torch.manual_seed(manual_seed)
        torch.cuda.manual_seed_all(manual_seed)

        time.sleep(2)

        train_loader = DataLoader(dataset,
                                  batch_size=1,
                                  shuffle=False,
                                  pin_memory=False,
                                  num_workers=8,
                                  prefetch_factor=1
                                  )

        end = time.time()

        for i, data in enumerate(train_loader):

            data_dynamic = data.data_dynamic
            data_prognostic = data.data_prognostic
            data_prognostic_inc = data.data_prognostic_inc
            data_diagnostic = data.data_diagnostic
            data_static = data.data_static
            data_time = data.data_time

            print('time: {}/{}--{}'.format(i + 1, len(train_loader), time.time() - end))
            end = time.time()

    if is_plot:

        import matplotlib as mpl
        import matplotlib.pyplot as plt

        mpl.use('TkAgg')

        s = 8

        for i in range(len(dataset)):

            print('time step -', i)

            data_i = dataset[i]

            data_dynamic = data_i.data_dynamic
            data_prognostic = data_i.data_prognostic
            data_prognostic_inc = data_i.data_prognostic_inc
            data_diagnostic = data_i.data_diagnostic
            data_static = data_i.data_static
            data_time = data_i.data_time

            # plot dynamic features
            for v in range(data_dynamic.shape[-1]):
                break
                plt.scatter(dataset.lon, dataset.lat, c=data_dynamic[0, :, v], linewidths=0, s=s)
                plt.title(dataset.dynamic_features[v] + ', rollout=1')
                plt.colorbar()
                plt.show()

            # plot target prognostic features
            for v in range(data_prognostic.shape[-1]):
                break
                plt.scatter(dataset.lon, dataset.lat, c=data_prognostic[0, :, v], linewidths=0, s=s)
                plt.title(dataset.target_prog_features[v] + ', rollout=1')
                plt.colorbar()
                plt.show()

            # plot target prognostic features
            for v in range(data_diagnostic.shape[-1]):
                break
                plt.scatter(dataset.lon, dataset.lat, c=data_diagnostic[0, :, v], linewidths=0, s=s)
                plt.title(dataset.target_diag_features[v] + ', rollout=1')
                plt.colorbar()
                plt.show()

            # plot static features
            for v in range(len(dataset.clim_features)):
                break
                plt.scatter(dataset.lon, dataset.lat, c=data_static[0, :, v], linewidths=0, s=s)
                plt.title(dataset.clim_features[v])
                plt.colorbar()
                plt.show()

            # plot lat and lon features if they exist
            if dataset.is_add_lat_lon:
                fig, axs = plt.subplots(2, 2)
                lat_sin = axs[0, 0].scatter(dataset.lon, dataset.lat, c=data_static[0, :, -4], linewidths=0, s=s, cmap="Dark2")
                axs[0, 0].set_title('sin(latitude)')
                plt.colorbar(lat_sin, ax=axs[0, 0])
                lat_cos = axs[1, 0].scatter(dataset.lon, dataset.lat, c=data_static[0, :, -3], linewidths=0, s=s, cmap="Dark2")
                axs[1, 0].set_title('cos(latitude)')
                plt.colorbar(lat_cos, ax=axs[1, 0])

                lon_sin = axs[0, 1].scatter(dataset.lon, dataset.lat, c=data_static[0, :, -2], linewidths=0, s=s, cmap="Dark2")
                axs[0, 1].set_title('sin(longitude)')
                plt.colorbar(lon_sin, ax=axs[0, 1])
                lon_cos = axs[1, 1].scatter(dataset.lon, dataset.lat, c=data_static[0, :, -1], linewidths=0, s=s, cmap="Dark2")
                axs[1, 1].set_title('cos(longitude)')
                plt.colorbar(lon_cos, ax=axs[1, 1])
                plt.show()

    if is_plot_graph:

        from torch_geometric.data import Data
        from torch_geometric.utils import to_networkx
        import networkx as nx
        import matplotlib.pyplot as plt
        import matplotlib as mpl
        mpl.use('TkAgg')

        """
        # 3D Graph
        x = 6371 * np.cos(dataset.lat * np.pi/180) * np.cos(dataset.lon * np.pi/180)
        y = 6371 * np.cos(dataset.lat * np.pi/180) * np.sin(dataset.lon * np.pi/180)
        z = 6371 * np.sin(dataset.lat * np.pi/180)
        # Create the 3D figure
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        node_colors = mpl.cm.viridis(dataset.data_static[0, :, 6])
        ax.scatter(x, y, z, s=20, ec="w")
        for edge in dataset.edge_index.T:
            ax.plot(x[edge], y[edge], z[edge], color="tab:gray")
        def _format_axes(ax):
            ax.grid(False)
            for dim in (ax.xaxis, ax.yaxis, ax.zaxis):
                dim.set_ticks([])
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")
        _format_axes(ax)
        fig.tight_layout()
        plt.show()
        """

        # 2D Graph
        x = np.concatenate((dataset.lon[:, None], dataset.lat[:, None]), axis=1)
        data = Data(x=dataset.data_static[0, :, 6], edge_index=torch.from_numpy(dataset.edge_index))
        G = to_networkx(data, to_undirected=True)

        plt.figure(figsize=(14, 14))
        plt.xticks([])
        plt.yticks([])
        pos = {i: x[i] for i in range(len(x))}
        node_colors = mpl.cm.viridis(dataset.data_static[0, :, 6])
        nx.draw_networkx(G,
                         pos=pos,
                         with_labels=False,
                         node_color=node_colors,
                         edge_color='gray',
                         node_size=20,
                         linewidths=0.01)
        plt.show()



