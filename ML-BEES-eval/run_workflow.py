######
# This script executes the whole pipe line of evaluation.
# 
######


### IMPORTS ###
import os
import numpy as np
import xarray as xr

from eval_utilities import spatial_temporal_metrics as stm
from eval_utilities import visualization as vis

import cartopy.crs as ccrs
import cartopy.feature as cfeature

### SETUP ###
domain = "euro" # euro | glob
eval_timespan = slice("2021-01-01T00", "2022-11-30T00")

variables = None # set None for the prog. and diag. variables in the config, list otherwise

metric_fnames = {"Bias": "nor_bias.zarr",
                 "RMSE": "nor_rmse.zarr",
                 "ACC": "acc.zarr",
                 }#"Phase Shift": "phase_shift.zarr"}

compute_metrics = False
create_visualizations = False
compute_scoreboard = True


### LOAD CONFIG ###
import yaml
with open(f"config.yaml") as stream:
    try:
        CONFIG = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

ds_ref = xr.open_zarr(CONFIG[f"path_ec_{domain}"]).sel(time=eval_timespan)
cell_areas = ds_ref.clim_data.sel(clim_variable="clim_cell_area")

inf_paths = CONFIG["inf_paths"]
eval_paths = CONFIG["eval_paths"]

if variables == None:
    variables = CONFIG["targets_prog"] + CONFIG["targets_diag"]

weights = {"swvl1": ds_ref.clim_data.sel(clim_variable="clim_theta_cap"), #use field capacity to emphasize potentially moist grid points
           "swvl2": ds_ref.clim_data.sel(clim_variable="clim_theta_cap"),
           "swvl3": ds_ref.clim_data.sel(clim_variable="clim_theta_cap"),
           "stl1": None,
           "stl2": None,
           "stl3": None,
           "snowc": ds_ref.data.sel(variable="snowc").mean(dim="time"), #weigh by how much snow there is at all
           "d2m": None,
           "t2m": None,
           "skt": None,
           "sshf": None,
           "slhf": None,
           "aco2gpp": ds_ref.clim_data.sel(clim_variable=["clim_veg_covl", "clim_veg_covh"]).sum(dim="clim_variable"), #"amount" of plant cover
           "dis": None,
           "e": None,
           "sro": None,
           "ssro": None}


### COMPUTE METRICS AND CREATE VISUALIZATIONS ###
desired_chunks = (4, 10051, 17)  # Adjust based on your desired chunk size

for model in inf_paths.keys():

    if compute_metrics:
        # Compute and store metrics:
        ds_mod = xr.open_zarr(inf_paths[model]).sel(time=eval_timespan)
        ds_mod = ds_mod.chunk({'time': 4, 'x': 10051, 'variable': 17})
        
        path = eval_paths[model] + '/'
        if not os.path.exists(path):
            os.makedirs(path)
        mc = stm.Metrics(ds_mod, ds_ref, path)
        mc.evaluate()

    if create_visualizations:
        # Visualization:
        path = eval_paths[model] + '/visualization/'
        if not os.path.exists(path):
            os.makedirs(path)

        # Bias
        bias = mc.bias(relative=False)
        for var in bias.variable.values:
            vis.vis_zarr_map(bias, var, path + 'bias', 1, 99)

        # Normalized bias
        nor_bias = mc.bias(relative=True)
        for var in nor_bias.variable.values:
            vis.vis_zarr_map(nor_bias, var, path + 'nor_bias', 1, 99)  
        
        # RMSE
        rmse = mc.rmse(relative=False)
        for var in rmse.variable.values:
            vis.vis_zarr_map(rmse, var, path + 'rmse', 1, 99)

        # Normalized RMSE
        nor_rmse = mc.rmse(relative=True)
        for var in nor_rmse.variable.values:
            vis.vis_zarr_map(nor_rmse, var, path + 'nor_rmse', 1, 99)  

        # ACC
        acc = mc.acc()
        for var in acc.variable.values:
            vis.vis_zarr_map(acc, var, path + 'acc', 1, 99)

        # Power Spectra
        for var in variables:
            vis.power_spectrum(ds_mod, ds_ref, var, path + 'spectrum')

        # Amplitude Maps
        for var in variables:
            time_axis = np.where(np.array(ds_ref.data.sel(variable=var).shape) == len(ds_ref.time))[0][0]
            fft_ref = np.fft.rfft(ds_ref.data.sel(variable=var), axis=time_axis)
            fft_mod = np.fft.rfft(ds_mod.data.sel(variable=var), axis=time_axis)
            freq = np.fft.rfftfreq(ds_ref.sizes["time"], d=(ds_ref.time[1] - ds_ref.time[0]).item() / 1e9)

            i_day = np.argmin(np.abs(freq - 1/(24*60*60)))
            vis.plot_amplitude_map(abs(fft_ref[i_day]), abs(fft_mod[i_day]), path + 'harmonic_analysis', "Diurnal")
            
            i_month =  np.argmin(np.abs(freq - 1/(30*24*60*60)))
            vis.plot_amplitude_map(abs(fft_ref[i_month]), abs(fft_mod[i_month]), path + 'harmonic_analysis', "Monthly")

            i_season = np.argmin(np.abs(freq - 4/(365*24*60*60)))
            vis.plot_amplitude_map(abs(fft_ref[i_season]), abs(fft_mod[i_season]), path + 'harmonic_analysis', "Seasonal")

            i_year = np.argmin(np.abs(freq - 1/(365*24*60*60))) 
            vis.plot_amplitude_map(abs(fft_ref[i_year]), abs(fft_mod[i_year]), path + 'harmonic_analysis', "Annual")
        

### SCOREBOARD ###
if compute_scoreboard:
    def gen_table_header(f, metric, vars):
        """
        Script to generate a simple markdown table header and write it to file stream `f`.
        """
        first_line = f"|{metric}|" #first line contains the metric and the variable names
        second_line = "|-|" #second line is just filled with dashes

        for var in vars: #automatically match number of variables
            first_line += f"{var}|"
            second_line += ":-:|"

        # Write:
        f.write(first_line + "\n")
        f.write(second_line + "\n")

    def smean_filtered(ds_metric, vars, cell_areas, weights):
        """
        Modify the spatial mean in the metrics module to remove infinite values 
        and cut outliers at the 99th percentile to not skew scores too much.
        """
        relative_error = ds_metric.data.sel(variable=vars).values
        relative_error[np.isinf(relative_error)] = np.nan
        mask = relative_error < np.nanpercentile(relative_error, 99)

        if weights is None:
            weights = np.ones(ds_metric.sizes["x"]) #uniform weights

        total_weighted_area = (weights[mask] * cell_areas[mask]).sum(dim="x") 
        spatial_integral = (weights[mask] * relative_error[mask] * cell_areas[mask]).sum(dim="x")

        return( spatial_integral/total_weighted_area )

    def score(relative_error, alpha=1):
        return( np.exp(-alpha * relative_error) )

    with open("scoreboard.md", "w") as f:
        # Write title:
        f.write("# AILand Score Board\n")
        f.write("\n")
        f.write("A score of one corresponds to a perfect relative error across all grid points. "\
                "The score asymptotically approaches zero for large relative errors.")
        f.write("\n")
        
        # Generate a table for every metric seperately:
        for metric in metric_fnames.keys():
            # Write metric sub titles:
            f.write(f"## {metric}\n")
            f.write("\n")
            gen_table_header(f, metric, variables)

            # Add a line for every model:
            for model in eval_paths.keys():
                ds_metric = xr.open_zarr(f"{eval_paths[model]}/spatial/{metric_fnames[metric]}")
                current_line = f"|{model}|"

                for var in variables:
                    if var not in ds_metric["variable"]:
                        var_score = np.nan
                    elif metric == "ACC":
                        var_score = smean_filtered(ds_metric, vars=var, cell_areas=cell_areas, weights=weights[var]).values.item()
                    elif metric == "Phase Shift":
                        normalized = 0.5 * (1. + np.cos(2. * np.pi * ds_metric / 365.))
                        var_score = smean_filtered(normalized, vars=var, cell_areas=cell_areas, weights=weights[var]).values.item()
                    else:
                        var_score = score(smean_filtered(ds_metric, vars=var, cell_areas=cell_areas, weights=weights[var])).values.item()

                    current_line += f"{var_score:.2f}|"
                f.write(current_line + "\n")
            f.write("\n")