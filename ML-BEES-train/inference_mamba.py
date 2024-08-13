import yaml
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

from model.Mamba_v1 import Mamba_v1
from dataset.EclandPointDataset import EcDataset

data_path = "/home/ssd4tb/shams/ecland/ecland_i6aj_o400_2010_2022_6h_euro.zarr"
model_path = "/home/hdd16tb/shams/log_ecmwf/log_2/Mamba_prog_con/model_checkpoints/best_loss_model.pth"
result_path = "/home/hdd16tb/shams/log_ecmwf/Mamba_prog_con.zarr"
spatial_encoding = True
temporal_encoding = True

with open('configs/config.yaml') as stream:
    try:
        CONFIG = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

rollout = CONFIG["roll_out"]

# Dataset
ds_inf_tmp = EcDataset(
    start_year=2019,
    end_year=2019,
    root = data_path,
    roll_out = 1,
    clim_features=CONFIG["clim_feats"],
    dynamic_features=CONFIG["dynamic_feats"],
    target_prog_features=CONFIG["targets_prog"],
    target_diag_features=CONFIG["targets_diag"],
    is_add_lat_lon = spatial_encoding,
    is_norm = True,
    point_dropout = 0.0
)

ds_inf = EcDataset(
    start_year=2020,
    end_year=2022,
    root = data_path,
    roll_out = 1,
    clim_features=CONFIG["clim_feats"],
    dynamic_features=CONFIG["dynamic_feats"],
    target_prog_features=CONFIG["targets_prog"],
    target_diag_features=CONFIG["targets_diag"],
    is_add_lat_lon = spatial_encoding,
    is_norm = True,
    point_dropout = 0.0
)

# MLP Model
model = Mamba_v1(in_static=ds_inf.n_static,
    in_dynamic=ds_inf.n_dynamic,
    in_prog=ds_inf.n_prog,
    out_prog=ds_inf.n_prog,
    out_diag=ds_inf.n_diag,
    hidden_dim=CONFIG["hidden_dim"],
    rollout=CONFIG["roll_out"],
    d_state=CONFIG["d_state"],
    d_conv=CONFIG["d_conv"],
    expand=CONFIG["expand"],
    dt_min=CONFIG["dt_min"],
    dt_max=CONFIG["dt_max"],
    dropout=CONFIG["dropout"],
    mu_norm=ds_inf.y_prog_inc_mean,
    std_norm=ds_inf.y_prog_inc_std,
    pretrained=model_path
).cuda()

# Define function to apply to each model step
def apply_constraints_prog(x):
    x = np.clip(x, 0, None) # All prog. variables are positive
    x[:,np.array(CONFIG["targets_prog"]) == "snowc"] = np.clip(x[:,np.array(CONFIG["targets_prog"]) == "snowc"], None, 100) # Snow cover cannot be higher than 100
    return x

def apply_constraints_diag(x):
    for i in range(x.shape[1]):
        if CONFIG["targets_diag"][i] not in ["e", "slhf", "sshf"]:
            x[:,i] = np.clip(x[:,i], 0, None)
    # x[:,np.array(CONFIG["targets_diag"]) not in ["slhf", "sshf", "e"]] = np.clip(x[:,np.array(CONFIG["targets_diag"]) not in ["slhf", "sshf", "e"]], 0, None) # All variables except e are positive
    return x

with torch.no_grad():

    model.eval()

    # Initial state
    prognostic_preds = []
    x_met, x_state, _, _, x_clim, x_time = ds_inf[0]
    x_clim = torch.from_numpy(x_clim)[:, None, :, :].repeat(1, rollout, 1, 1)

    x_state_all = []
    x_time_all = []
    x_met_all = []

    for t in range(rollout-1):
        x_met_t, x_state_t, _, _, _, x_time_t = ds_inf_tmp[len(ds_inf_tmp)-t]

        x_met_all.append(torch.from_numpy(x_met_t))
        x_state_all.append(torch.from_numpy(x_state_t))
        x_time_all.append(torch.from_numpy(x_time_t))

    x_met_all = torch.cat(x_met_all, dim=0)
    x_state_all = torch.cat(x_state_all, dim=0)
    x_time_all = torch.cat(x_time_all, dim=0)

    prognostic_preds.append(EcDataset.inv_transform(x_state.squeeze(), ds_inf.y_prog_means, ds_inf.y_prog_stdevs))

    x_met_all = torch.cat((x_met_all, torch.from_numpy(x_met)), dim=0)[None, :, :, :]
    x_state_all = torch.cat((x_state_all, torch.from_numpy(x_state)), dim=0)[None, :, :, :]
    x_time_all = torch.cat((x_time_all, torch.from_numpy(x_time)), dim=0)
    x_time_all = x_time_all[None, :, :]

    diagnostic_preds = []

    # Inference
    for i in tqdm(range(len(ds_inf)), desc="Running ECLand emulator..."):
        x_met, _, _, _, _, x_time = ds_inf[i]
        # x_met, x_time = torch.from_numpy(x_met).unsqueeze(1).cuda(), torch.from_numpy(x_time).unsqueeze(1).cuda()
        x_met, x_time = torch.from_numpy(x_met)[:, None, :, :], torch.from_numpy(x_time)[None, :, :]

        x_met_all = torch.cat((x_met_all, x_met), dim=1)[:, 1:, :, :]
        x_time_all = torch.cat((x_time_all, x_time), dim=1)[:, 1:, :]

        y_state_inc_pred, y_diag_pred, _, _ = model(x_clim.cuda(), x_met_all.cuda(), x_state_all.cuda(), x_time_all.cuda())
        y_state_inc_pred, y_diag_pred = y_state_inc_pred.cpu().numpy().squeeze(), y_diag_pred.cpu().numpy().squeeze()

        x_state = x_state_all[0, -1, :, :].clone().cpu().numpy()

        # Prognostic variables
        y_state_inc_pred = EcDataset.inv_transform(y_state_inc_pred, ds_inf.y_prog_inc_mean, ds_inf.y_prog_inc_std) # Unnormalize so that it can be added to the normalized state vector
        x_state += y_state_inc_pred
        x_state = EcDataset.inv_transform(x_state, ds_inf.y_prog_means, ds_inf.y_prog_stdevs)
        x_state = apply_constraints_prog(x_state) # Unnormalize updated state vector and apply consistency constraints
        prognostic_preds.append(x_state)
        x_state = EcDataset.transform(x_state, ds_inf.y_prog_means, ds_inf.y_prog_stdevs) # Re-normalize state vector for next iteration
        x_state_all = torch.cat((x_state_all, torch.from_numpy(x_state)[None, None, :, :]), dim=1)[:, 1:, :, :]

        # Diagnostic variables
        y_diag_pred = apply_constraints_diag(EcDataset.inv_transform(y_diag_pred, ds_inf.y_diag_means, ds_inf.y_diag_stdevs))
        diagnostic_preds.append(y_diag_pred)

# Diagnostic variables for the last timestep are not part of the dataset, so we add a "dummy"
diagnostic_preds.append(y_diag_pred)

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

true = xr.open_zarr(data_path).sel(time=slice("2020", "2022")).data
pred = xr.open_zarr(result_path).sel(time=slice("2020", "2022")).data

print(np.unique(pred.values))

def find_nearest_idx(
        arr1: np.ndarray,
        arr2: np.ndarray,
        val1: float,
        val2: float,
) -> int:
    """Find first nearest index for a given tolerance for two arrays and 2 values

    :param arr1: first array
    :param arr2: second arrat
    :param val1: value to find in first array
    :param val2: value to find in second array
    :return: index as int
    """
    return (np.abs(arr1 - val1) + np.abs(arr2 - val2)).argmin()


lat, lon = 50.72, 7.11
# lat, lon = 70.94, 24.31
x_idx = find_nearest_idx(true.lat, true.lon, lat, lon).values


def ailand_plot(var_name, label=None, test_date="2021-01-01"):
    """Plotting function for the ec-land database and ai-land model output

    :param var_name: parameter variable name
    :param ax: the axes to plot on
    :param ylabel: y-label for plot
    :param ax_title: title for plot
    :param test_date: date to plot vertical line (train/test split), defaults to "2021-01-01"
    :return: plot axes
    """

    fig = plt.figure(figsize=(9, 4))
    true.isel(x=x_idx).sel(variable=var_name).plot(label="ec-land", ax=plt.gca())
    pred.isel(x=x_idx).sel(variable=var_name).plot(label="ai-land", ax=plt.gca())

    plt.gca().axvline(pred.sel(time=test_date).time.values[0], color="k", linestyle="--")
    plt.gca().set_xlim(pred.time.values[[0, -1]])
    plt.gca().set_ylabel(label)
    plt.show()

    return


for var in pred["variable"].values:
    ailand_plot(var)