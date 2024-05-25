import xarray as xr
import numpy as np
import matplotlib.pyplot as plt


def fig_plot():
    pass


def vis_zarr_map(zarr_eval, var, min_perc, max_perc, time_point=False):

    """
    Visualize the original zarr file -- ecland or ai-land output;
    select a single time point of one variable; Or 

    --- Parameters ---
    zarr_eval:   the zarr file; zarr should be xarray.Dataset
    vars:       str or iterable of str
    min_prec:   percentile for lower limit, by default 1%
    max_prec:   percentile for upper limit, by default 99%
    time_point:   bool-by daulft False or int

    --- Returns ---
    map
    """
    if time_point==False:
        zarr_eval_selected = zarr_eval.sel(variable=var)
    else:
        zarr_eval_selected = zarr_eval.isel(time=time_point).sel(variable=var)

    # Create the scatter plot
    fig, ax = plt.subplots(figsize=(12, 6))

    # pre-define a min and max for a quick visualization; vmin/vmax based on the 1 and 99 percentile 
    vmin=np.percentile(zarr_eval_selected.data.values, min_perc, axis=0)
    vmax=np.percentile(zarr_eval_selected.data.values, max_perc, axis=0)

    zarr_eval_selected.plot.scatter(
        x="lon", y="lat", hue="data", s=10, edgecolors="none", ax=ax, vmin=vmin,vmax=vmax)
    
    plt.show()