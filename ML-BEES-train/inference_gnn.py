"""
THIS SCRIPT should use EclandGraphDataset_rollout insead of EclandGraphDataset
"""
# ------------------------------------------------------------------
# Script for running inference of the GNN emulator
# ------------------------------------------------------------------

import yaml
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

from model.GNN import GNN
from dataset.EclandGraphDataset import EcDataset

# ------------------------------------------------------------------

# Setup data path
data_path = "/home/ssd4tb/shams/ecland/ecland_i6aj_o400_2010_2022_6h_euro.zarr"

# For ensembles
files = ['UniMP_' + str(i+1) for i in range(28)]
files = files[1:]

# Define functions to apply to each emulator output...
# ... to constrain prognostic variables
def apply_constraints_prog(x):
    """
    Apply constraints to prognostic output of the emulator

    Args:
        x (np.ndarray): Numpy array with prognostic variables
    Returns:
        x (np.ndarray): Numpy array with prognostic variables, updated according to constraints
    """
    # All prognostic variables are positive
    x = np.clip(x, 0, None) 
    # Snow cover cannot be higher than 100
    x[:, np.array(CONFIG["targets_prog"]) == "snowc"] = np.clip(x[:, np.array(CONFIG["targets_prog"]) == "snowc"], None, 100)
    return x

# ... to constrain diagnostic variables
def apply_constraints_diag(x):
    """
    Apply constraints to diagnostic output of the emulator

    Args:
        x (np.ndarray): Numpy array with diagnostic variables
    Returns:
        x (np.ndarray): Numpy array with diagnostic variables, updated according to constraints
    """
    # All but three variables are positive
    for i in range(x.shape[1]):
        if CONFIG["targets_diag"][i] not in ["e", "slhf", "sshf"]:
            x[:,i] = np.clip(x[:,i], 0, None)
    return x

for file in files:

    # Setup model and results paths
    model_path = "/home/hdd16tb/shams/log_ecmwf/log_4/{}/model_checkpoints/best_loss_model.pth".format(file)
    result_path = r'/home/ssd4tb/shams/unimp_ens/euro_unimp_{}_train_2010_2019_val_2020_2020.zarr'.format(file[6:])

    # Settings
    spatial_encoding = True # wether to use the additional spatial encodings
    temporal_encoding = True # whether to use the additional temporal encodings

    # get remaining settings from config file
    with open('configs/config.yaml') as stream:
        try:
            CONFIG = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    # Initialize dataset to run the inference for
    ds_inf = EcDataset(
        start_year=2020,
        end_year = 2022,
        x_slice_indices=CONFIG["x_slice_indices"],
        root = data_path,
        clim_features=CONFIG["clim_feats"],
        dynamic_features=CONFIG["dynamic_feats"],
        target_prog_features=CONFIG["targets_prog"],
        target_diag_features=CONFIG["targets_diag"],
        is_add_lat_lon=spatial_encoding,
        is_norm=CONFIG["is_norm"],
        graph_type = CONFIG["graph_type"],
        d_graph = CONFIG["d_graph"],
        max_num_points = CONFIG["max_num_points"],
        k_graph = CONFIG["k_graph"]
    )
    edge_index = torch.from_numpy(ds_inf.edge_index) # edge_index is static

    # Initialize the GNN Model
    model = GNN(model=CONFIG['model'],
        in_static=ds_inf.n_static,
        in_dynamic=ds_inf.n_dynamic,
        in_prog=ds_inf.n_prog,
        out_prog=ds_inf.n_prog,
        out_diag=ds_inf.n_diag,
        hidden_dim=CONFIG["hidden_dim"],
        rollout=CONFIG["roll_out"],
        heads=CONFIG["heads"],
        dropout=CONFIG["dropout"],
        mu_norm=ds_inf.y_prog_inc_mean,
        std_norm=ds_inf.y_prog_inc_std,
        pretrained=model_path
    )
    model.eval()

    with torch.no_grad():

        # setup storage for prognostic outputs and get initial state
        prognostic_preds = []
        data = ds_inf[0]
        x_state, x_clim = data.data_prognostic, data.data_static
        prognostic_preds.append(EcDataset.inv_transform(x_state, ds_inf.y_prog_means, ds_inf.y_prog_stdevs))
        
        # setup storage for diagnostic outputs
        diagnostic_preds = []

        # Inference
        for i in tqdm(range(len(ds_inf)), desc="Running ECLand emulator..."):

            # Get dynamic input data from dataset
            data = ds_inf[i]
            x_met, x_time = data.data_dynamic, data.data_time

            # Run model
            y_state_inc_pred, y_diag_pred, _, _ = model(x_clim, x_met, x_state, x_time, edge_index)
            y_state_inc_pred, y_diag_pred, x_state = y_state_inc_pred.detach().numpy().squeeze(), y_diag_pred.detach().numpy().squeeze(), x_state.numpy().squeeze()
            
            # Prognostic variables
            y_state_inc_pred = EcDataset.inv_transform(y_state_inc_pred, ds_inf.y_prog_inc_mean, ds_inf.y_prog_inc_std) # Unnormalize so that it can be added to the normalized state vector
            x_state += y_state_inc_pred
            x_state = apply_constraints_prog(EcDataset.inv_transform(x_state, ds_inf.y_prog_means, ds_inf.y_prog_stdevs)) # Unnormalize updated state vector and apply consistency constraints
            prognostic_preds.append(x_state)
            x_state = EcDataset.transform(x_state, ds_inf.y_prog_means, ds_inf.y_prog_stdevs) # Re-normalize state vector for next iteration
            x_state = torch.tensor(x_state)
            
            # Diagnostic variables
            y_diag_pred = apply_constraints_diag(EcDataset.inv_transform(y_diag_pred, ds_inf.y_diag_means, ds_inf.y_diag_stdevs))
            diagnostic_preds.append(y_diag_pred)

        # Diagnostic variables for the last timestep are not part of the dataset, so we add a "dummy"
        diagnostic_preds.append(y_diag_pred)

        # Save emulator output to zarr file
        all_preds = np.concatenate((np.stack(prognostic_preds), np.stack(diagnostic_preds)), axis=2)
        preds_xr = xr.DataArray(
            data = all_preds,
            coords = {"x":ds_inf.ds_ecland["x"], "time":ds_inf.times, "variable":CONFIG["targets_prog"] + CONFIG["targets_diag"]},
            dims = ["time", "x", "variable"],
            name = "data"
        )
        preds_xr = preds_xr.assign_coords(lon=("x", ds_inf.lon))
        preds_xr = preds_xr.assign_coords(lat=("x", ds_inf.lat))
        preds_xr = preds_xr.to_dataset()
        preds_xr.to_zarr(result_path)

        # continue
#    #     true = xr.open_zarr(data_path).sel(time=slice("2020", "2022")).data
#    #     pred = xr.open_zarr(result_path).sel(time=slice("2020", "2022")).data

#         def find_nearest_idx(
#                 arr1: np.ndarray,
#                 arr2: np.ndarray,
#                 val1: float,
#                 val2: float,
#         ) -> int:
#             """Find first nearest index for a given tolerance for two arrays and 2 values

#             :param arr1: first array
#             :param arr2: second arrat
#             :param val1: value to find in first array
#             :param val2: value to find in second array
#             :return: index as int
#             """
#             return (np.abs(arr1 - val1) + np.abs(arr2 - val2)).argmin()


#       #  lat, lon = 50.72, 7.11
#       #  x_idx = find_nearest_idx(true.lat, true.lon, lat, lon).values

#         def ailand_plot(var_name, label=None, test_date="2021-01-01"):
#             """Plotting function for the ec-land database and ai-land model output

#             :param var_name: parameter variable name
#             :param ax: the axes to plot on
#             :param ylabel: y-label for plot
#             :param ax_title: title for plot
#             :param test_date: date to plot vertical line (train/test split), defaults to "2021-01-01"
#             :return: plot axes
#             """

#             fig = plt.figure(figsize=(9, 4))
#             true.isel(x=x_idx).sel(variable=var_name).plot(label="ec-land", ax=plt.gca())
#             pred.isel(x=x_idx).sel(variable=var_name).plot(label="ai-land", ax=plt.gca())

#             plt.gca().axvline(pred.sel(time=test_date).time.values[0], color="k", linestyle="--")
#             plt.gca().set_xlim(pred.time.values[[0, -1]])
#             plt.gca().set_ylabel(label)
#             plt.show()

#             return


#      #   for var in pred["variable"].values:
#      #       ailand_plot(var)
