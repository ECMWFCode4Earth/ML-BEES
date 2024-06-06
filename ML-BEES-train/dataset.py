import math

import xarray as xr
import numpy as np
import torch
from torch.utils.data import Dataset

# Variable lists
clim_feat_lst = [
    'clim_clake',
    'clim_ctype',
    'clim_landsea',
    'clim_cu',
    'clim_cvh',
    'clim_cvl',
    'clim_geopot',
    'clim_sdfor',
    'clim_sdor',
    # 'clim_sotype',
    # 'clim_tvh',
    # 'clim_tvl',
    'clim_theta_cap',
    'clim_theta_pwp',
    'clim_veg_covh',
    'clim_veg_covl',
    'clim_veg_z0mh',
    'clim_veg_z0ml',
    'clim_veg_rsminh',
    'clim_veg_rsminl'
]

feat_lst = [
    'lai_hv',
    'lai_lv',
    'met_ctpf',
    'met_lwdown',
    'met_psurf',
    'met_qair',
    'met_rainf',
    'met_swdown',
    'met_snowf',
    'met_tair',
    'met_wind_e',
    'met_wind_n',
    'swvl1',
    'swvl2',
    'swvl3',
    'stl1',
    'stl2',
    'stl3',
    'snowc'
]

targ_lst = [
    'swvl1',
    'swvl2',
    'swvl3',
    'stl1',
    'stl2',
    'stl3',
    'snowc',
]


# Dataset class
class EcDataset(Dataset):

    def __init__(self, path, start_yr, end_yr, spatial_encoding=False):
        # Open data file
        ds = xr.open_zarr(path).sel(time=slice(start_yr, end_yr))

        # Input features (static, dynamic)
        all_feats = []
        clim_feats_ds = (ds.sel(clim_variable=clim_feat_lst).clim_data
                         .expand_dims(time=ds.time)
                         .isel(time=slice(0, -1))
                         .stack(z=("x", "time",))
                         .transpose()
                         .rename({"clim_variable": "variable"})
                         )
        all_feats.append(clim_feats_ds)
        feats_ds = (ds.sel(variable=feat_lst)
                    .isel(time=slice(0, -1))
                    .data.stack(z=("x", "time",))
                    .transpose()
                    )
        all_feats.append(feats_ds)
        self.feats = torch.tensor(xr.concat(all_feats, dim="variable").chunk({"variable": -1}).values)

        # Output features
        target_ds = ds.sel(variable=targ_lst).data
        self.targets = torch.tensor(
            target_ds.isel(time=slice(1, None)).stack(z=("x", "time",)).values.T - target_ds.isel(
                time=slice(0, -1)).stack(z=("x", "time",)).values.T)

        # Statistics used for normalization
        self.feat_means = torch.concat((torch.tensor(ds.clim_means.sel(clim_variable=clim_feat_lst).values),
                                        torch.tensor(ds.data_means.sel(variable=feat_lst).values)))
        self.target_means = torch.tensor(ds.data_means.sel(variable=targ_lst).values)
        self.feat_stdevs = torch.concat((torch.tensor(ds.clim_stdevs.sel(clim_variable=clim_feat_lst).values),
                                         torch.tensor(ds.data_stdevs.sel(variable=feat_lst).values)))
        self.target_stdevs = torch.tensor(ds.data_stdevs.sel(variable=targ_lst).values)

        # Spatial encodings
        if spatial_encoding:
            ds = ds.assign(cos_lat=np.cos((math.pi / 180) * ds.lat))
            lats = torch.tensor((ds.lat
                                 .expand_dims(time=ds.time)
                                 .isel(time=slice(0, -1))
                                 .stack(z=("x", "time",))
                                 .transpose()
                                 ).values)[:, None] * (math.pi / 180)
            lons = torch.tensor((ds.lon
                                 .expand_dims(time=ds.time)
                                 .isel(time=slice(0, -1))
                                 .stack(z=("x", "time",))
                                 .transpose()
                                 ).values)[:, None] * (math.pi / 180)
            self.feats = torch.concat((self.feats, torch.cos(lats), torch.sin(lats), torch.cos(lons), torch.sin(lats)),
                                      dim=1)
            self.feat_means = torch.concat((self.feat_means, torch.zeros((4,))))
            self.feat_stdevs = torch.concat((self.feat_stdevs, torch.ones((4,))))

    # number of rows in the dataset
    def __len__(self):
        return self.feats.shape[0]

    # get a row at an index
    def __getitem__(self, idx):
        X = self.transform(self.feats[idx], self.feat_means, self.feat_stdevs)
        y = self.transform(self.targets[idx], self.target_means, self.target_stdevs)

        return X, y

    # Static methods
    def transform(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
        """Transform data with mean and stdev.

        :param x: data :param mean: mean :param std: standard deviation :return:
        normalised data
        """
        x_norm = (x - mean) / (std + 1e-5)

        return x_norm

    def inv_transform(x_norm: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
        """Inverse transform on data with mean and stdev.

        :param x_norm: normlised data :param mean: mean :param std: standard deviation
        :return: unnormalised data
        """
        x = (x_norm * (std + 1e-5)) + mean

        return x