# ------------------------------------------------------------------
# Script for running inference of the MLP emulator with time variable LAI
# ------------------------------------------------------------------

import yaml
import xarray as xr
import numpy as np
import torch
from tqdm import tqdm

from model.MLP_Obs import MLP_Obs
from dataset.EclandObsPointDataset import EcObsDataset

# ------------------------------------------------------------------

# Setup paths
data_path = "/home/ssd4tb/shams/ecland/ecland_i6aj_o400_2010_2022_6h_euro.zarr"
model_path = "/home/hdd16tb/shams/log_ecmwf/log_5/MLP_Obs_8/model_checkpoints/best_loss_model.pth"
result_path = "/home/hdd16tb/shams/log_ecmwf/euro_mlp_v2_obs_without_timevarying_lai_train_2010_2019_val_2020_2020.zarr"

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
ds_inf = EcObsDataset(
    start_year = 2020,
    end_year = 2022,
    root = data_path,
    root_sm = CONFIG["smap_file"],
    root_temp = CONFIG["modis_temp_file"],
    use_time_var_lai = CONFIG["use_time_var_lai"],
    root_lail = CONFIG["lail_file"],
    root_laih = CONFIG["laih_file"],
    roll_out = 1,
    clim_features=CONFIG["clim_feats"],
    dynamic_features=CONFIG["dynamic_feats"],
    target_prog_features=CONFIG["targets_prog"],
    target_diag_features=CONFIG["targets_diag"],
    is_add_lat_lon = spatial_encoding,
    is_norm = True,
    point_dropout = 0.0
)

# Initialize the MLP Model
model = MLP_Obs(in_static=ds_inf.n_static,
    in_dynamic=ds_inf.n_dynamic,
    in_prog=ds_inf.n_prog,
    out_prog=ds_inf.n_prog,
    out_diag=ds_inf.n_diag,
    hidden_dim=CONFIG["hidden_dim"],
    rollout=CONFIG["roll_out"],
    dropout=CONFIG["dropout"],
    mu_norm=ds_inf.y_prog_inc_mean,
    std_norm=ds_inf.y_prog_inc_std,
    pretrained=model_path
)
model.eval()

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

with torch.no_grad():

    # setup storage for prognostic outputs and get initial state
    prognostic_preds = []
    _, x_state, _, _, x_clim, _, _, _ = ds_inf[0] 
    x_state, x_clim = x_state.squeeze(), torch.from_numpy(x_clim)[:, None, :, :]
    prognostic_preds.append(EcObsDataset.inv_transform(x_state, ds_inf.y_prog_means, ds_inf.y_prog_stdevs))

    # setup storage for diagnostic outputs
    diagnostic_preds = []

    # Inference
    for i in tqdm(range(len(ds_inf)), desc="Running ECLand emulator..."):

        # Get dynamic input data from dataset
        x_met, _, _, _, _, _, _, x_time = ds_inf[i]
        x_met, x_state, x_time = torch.from_numpy(x_met)[:, None, :, :], torch.from_numpy(x_state)[None, None, :, :], torch.from_numpy(x_time)[:, None, :]

        # Run model
        y_state_inc_pred, y_diag_pred, _, _ = model(x_clim, x_met, x_state, x_time)
        y_state_inc_pred, y_diag_pred, x_state = y_state_inc_pred.detach().numpy().squeeze(), y_diag_pred.detach().numpy().squeeze(), x_state.numpy().squeeze()
        
        # Prognostic variables
        y_state_inc_pred = EcObsDataset.inv_transform(y_state_inc_pred, ds_inf.y_prog_inc_mean, ds_inf.y_prog_inc_std) # Unnormalize so that it can be added to the normalized state vector
        x_state += y_state_inc_pred
        x_state = apply_constraints_prog(EcObsDataset.inv_transform(x_state, ds_inf.y_prog_means, ds_inf.y_prog_stdevs)) # Unnormalize updated state vector and apply consistency constraints
        prognostic_preds.append(x_state)
        x_state = EcObsDataset.transform(x_state, ds_inf.y_prog_means, ds_inf.y_prog_stdevs) # Re-normalize state vector for next iteration
        
        # Diagnostic variables
        y_diag_pred = apply_constraints_diag(EcObsDataset.inv_transform(y_diag_pred, ds_inf.y_diag_means, ds_inf.y_diag_stdevs)) # Unnormalize diagnostic variables and apply constraints
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