import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

def fig_plot():
    pass


def vis_zarr_map(zarr_eval, var, path_png, min_perc, max_perc, time_point=False):

    """
    Visualize the original zarr file -- ecland or ai-land output;
    select a single time point of one variable; Or plot the metrics for one variable;
    save the figure to the path

    --- Parameters ---
    zarr_eval:   the zarr file; zarr should be xarray.Dataset
    vars:       str or iterable of str
    path_png:   path to save the figure; should include the metrics name if plot the metric
    min_prec:   percentile for lower limit, by default 1%
    max_prec:   percentile for upper limit, by default 99%
    time_point:   bool-by daulft False or int

    --- Returns ---
    show the map and save in the path
    """
    if time_point==False:
        zarr_eval_selected = zarr_eval.sel(variable=var)
    else:
        zarr_eval_selected = zarr_eval.isel(time=time_point).sel(variable=var)

    # Create the scatter plot
    fig, ax = plt.subplots(figsize=(12, 6))

    # filter the nan and inf value to calculate min/max percentile

    valid_mask = ~np.isnan(zarr_eval_selected.data.values) & ~np.isinf(zarr_eval_selected.data.values)

    # Filter the array to keep only the valid values
    compressed_array = zarr_eval_selected.data.values[valid_mask]

    # pre-define a min and max for a quick visualization; vmin/vmax based on the 1 and 99 percentile 
    vmin=np.percentile(compressed_array, min_perc, axis=0)
    vmax=np.percentile(compressed_array, max_perc, axis=0)

    zarr_eval_selected.plot.scatter(
        x="lon", y="lat", hue="data", s=10, edgecolors="none", ax=ax, vmin=vmin,vmax=vmax)
    
    fig.savefig(path_png+'_%s.png' % var, bbox_inches="tight") # path_png should include the metrics name

    plt.show()


def power_spectrum(mod, ref, var, path_png):
    """
    Computes and displays the power spectrum of variable `var` in dataset `mod` against the spectrum
    created from the reference dataset `ref. The image is saved under `path_png`. The plot displays 
    the spatial mean by a solid line and the standard deviation by a shaded area. 

    --- Parameters ---
    ds (comp):  xarray.DataSet/DataArray
    var:        str
    path_png:   str
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.tight_layout()

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set(xlabel="Frequency / Hz", ylabel="Magnitude", title=var)

    # Day:
    ax.axvline(1/(24*60*60), color="tab:grey", ls="dashed")
    ax.text(1/(24*60*60), 0.99, 'day', color='tab:grey', ha='right', va='top', rotation=90, transform=ax.get_xaxis_transform())

    # Month:
    ax.axvline(1/(30*24*60*60), color="tab:grey", ls="dashed")
    ax.text(1/(30*24*60*60), 0.99, 'month', color='tab:grey', ha='right', va='top', rotation=90, transform=ax.get_xaxis_transform())

    # Season:
    ax.axvline(4/(365*24*60*60), color="tab:grey", ls="dashed")
    ax.text(4/(365*24*60*60), 0.99, 'season', color='tab:grey', ha='right', va='top', rotation=90, transform=ax.get_xaxis_transform())

    # Year:
    ax.axvline(1/(365*24*60*60), color="tab:grey", ls="dashed")
    ax.text(1/(365*24*60*60), 0.99, 'year', color='tab:grey', ha='right', va='top', rotation=90, transform=ax.get_xaxis_transform())

    
    # Spectrum reference:
    #areas = ref.clim_data.sel(clim_variable="clim_cell_area")
    space_axis = np.where(np.array(ref.data.sel(variable=var).shape) == len(ref.lat))[0][0]
    time_axis = np.where(np.array(ref.data.sel(variable=var).shape) == len(ref.time))[0][0]

    fft = np.fft.fft(ref.data.sel(variable=var), axis=time_axis)
    freq = np.fft.fftfreq(ref.sizes["time"], d=(ref.time[1] - ref.time[0]).item() / 1e9)
    mask = freq > 0
    
    fft_mean = np.mean(abs(fft), axis=space_axis)
    fft_std = np.std(abs(fft), axis=space_axis)
    ax.plot(freq[mask], fft_mean[mask], color="tab:blue", label="ECLand")
    #ax.fill_between(freq[mask], (fft_mean - fft_std)[mask], (fft_mean + fft_std)[mask], alpha=0.5, color="tab:blue")


    # Spectrum modeled:
    fft = np.fft.fft(mod.data.sel(variable=var), axis=time_axis)
    freq = np.fft.fftfreq(mod.sizes["time"], d=(mod.time[1] - mod.time[0]).item() / 1e9)
    mask = freq > 0

    fft_mean = np.mean(abs(fft), axis=space_axis)
    fft_std = np.std(abs(fft), axis=space_axis)
    ax.plot(freq[mask], fft_mean[mask], color="tab:orange", label="Emulator")
    #ax.fill_between(freq[mask], (fft_mean - fft_std)[mask], (fft_mean + fft_std)[mask], alpha=0.5, color="tab:orange")

    plt.legend()
    fig.savefig(f"{path_png}_{var}.png", bbox_inches="tight")
    plt.show()