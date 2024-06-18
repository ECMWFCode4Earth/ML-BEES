import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import mpl_scatter_density # adds projection='scatter_density'
from matplotlib.colors import LinearSegmentedColormap
from shapely.geometry import Point
import geopandas as gpd


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


def boxplot_percentile(df, var1, var2, ymin, ymax):
    """
    Make the boxplot of evaluation metrics against the 10% bin percentile of the climate variable;
    Note: this only applies for the climate variables with continueous values;
    `clim_cvl`, `clim_cvh` - low/high vegetation cover fraction of gridcell
    `clim_cu` - urban cover fraction
    `clim_theta_pwp` - partial wilting point of soil
    `clim_theta_cap` - field capacity of soil
    `lat`, `lon` 

    --- Parameters ---
    df:   the pd.DataFrame combining evaluation metrics and climatic variables
    var1: name of the climate variable -->str
    var2: name of the evaluation metrics -->str
    ymin: min value of the metrics -->int; need a first guess by setting ymin and ymax to be None
    ymax: min value of the metrics -->int 
    """
    
    num_of_bins=10
    df['%s_percentile' % var1] = pd.qcut(df[var1], q=num_of_bins, labels=[f'{i*10}%-{(i+1)*10}%' for i in range(10)], 
                                         duplicates='drop')

    # Create a boxplot for swvl1_bias at each 10% CVL percentile
    plt.figure(figsize=(12, 6))
    num_colors = num_of_bins
    colors = sns.color_palette("Set3", num_colors)
    np.random.shuffle(colors)  # Randomize colors
    boxplot=sns.boxplot(x='%s_percentile' % var1, y=var2, data=df,palette=colors)
    plt.title('%s by %s percentile' % (var2, var1), fontsize=18)
    plt.suptitle('')
    plt.xlabel('%s Percentile' % var1, fontsize=16)
    plt.ylabel('%s' % var2, fontsize=16)
    plt.xticks(rotation=45)
    boxplot.tick_params(axis='x', labelsize=14)
    boxplot.tick_params(axis='y', labelsize=14)
    if ymin!=None and ymax!=None:
        boxplot.set_ylim(ymin, ymax)
    plt.show()

def boxplot_value_range(df, var1, var2, ymin, ymax):
    """
    Make the boxplot of evaluation metrics against the 10% bin value range of the climate variable;
    Note: this only applies for the climate variables with continueous values;
    `clim_cvl`, `clim_cvh` - low/high vegetation cover fraction of gridcell
    `clim_cu` - urban cover fraction
    `clim_theta_pwp` - partial wilting point of soil
    `clim_theta_cap` - field capacity of soil
    `lat`, `lon` 

    --- Parameters ---
    df:  the pd.DataFrame combining evaluation metrics and climatic variables
    var1: name of the climate variable -->str
    var2: name of the evaluation metrics -->str
    ymin: min value of the metrics -->int; need a first guess by setting ymin and ymax to be None
    ymax: min value of the metrics -->int 
    """

    # Define bins for the clim_cvl value range
    #bins = [i/10 for i in range(11)]
    #labels = [f'{i/10}-{(i+1)/10}' for i in range(10)]
    bins = np.linspace(df[var1].min(), df[var1].max(), 11)
    labels = [f'{bins[i]:.2f}-{bins[i+1]:.2f}' for i in range(len(bins)-1)]
    
    # Create a new column for the CVL value range
    df['%s_value_range' % var1] = pd.cut(df[var1], bins=bins, labels=labels, include_lowest=True)
    # Create a boxplot for swvl1_bias at each 10% CVL percentile
    plt.figure(figsize=(12, 6))
    num_colors = len(bins)-1
    colors = sns.color_palette("Set3", num_colors)
    np.random.shuffle(colors)  # Randomize colors
    boxplot=sns.boxplot(x='%s_value_range' % var1, y=var2, data=df,palette=colors)
    plt.title('%s by %s range' % (var2, var1), fontsize=18)
    plt.suptitle('')
    plt.xlabel('%s ratio' % var1, fontsize=16)
    plt.ylabel(var2, fontsize=16)
    plt.xticks(rotation=45)
    boxplot.tick_params(axis='x', labelsize=14)
    boxplot.tick_params(axis='y', labelsize=14)
    if ymin!=None and ymax!=None:
        boxplot.set_ylim(ymin, ymax)
    plt.show()


def boxplot_type(df, var1, var2, ymin, ymax):
    """
    Make the boxplot of evaluation metrics against the types of the climate variable;
    Note: this only applies for the climate variables with categoristic values;
    `clim_tvl`, `clim_tvh` - low/high vegetation type at gridcell
    `clim_sotype` - soil type
    `clim_glm` - glacier land mask
    `clim_veg_covl`, `clim_veg_covh` - average veg cover in categories

    --- Parameters ---
    df:   the pd.DataFrame combining evaluation metrics and climatic variables
    var1: name of the climate variable -->str
    var2: name of the evaluation metrics -->str
    ymin: min value of the metrics -->int; need a first guess by setting ymin and ymax to be None
    ymax: min value of the metrics -->int 
    """
    plt.figure(figsize=(12, 6))
    num_colors = len(df[var1].unique())
    colors = sns.color_palette("Set3", num_colors)
    np.random.shuffle(colors)  # Randomize colors
    boxplot=sns.boxplot(x=var1, y=var2, data=df,palette=colors)
    if ymin!=None and ymax!=None:
        boxplot.set_ylim(ymin, ymax)
    plt.xlabel('%s type' % var1, fontsize=16)
    plt.ylabel(var2, fontsize=16)
    plt.title('%s by %s range' % (var2, var1), fontsize=18)
    boxplot.tick_params(axis='x', labelsize=14)
    boxplot.tick_params(axis='y', labelsize=14)
    plt.show()

def boxplot_ar5(error_zarr, gdf, region_list, var1, var2, ymin, ymax):
    """
    Make the boxplot of evaluation metrics at different AR regions;
    source: https://www.ipcc-data.org/guidelines/pages/ar5_regions.html

    --- Parameters ---
    error_zarr:   the original error metrics .zarr file
    gdf: the AR5 region shape .shp file
    region_list: the index of selected AR5 regions based on gdf --> list [int]
    var1: name of the output variable -->str
    var2: name of the evaluation metrics -->str
    ymin: min value of the metrics -->int; need a first guess by setting ymin and ymax to be None
    ymax: min value of the metrics -->int 
    """
    def create_ar5_mask(error_zarr, region_num, gdf):
        lat=error_zarr.lat.values
        lon=error_zarr.lon.values

        mask_region=np.zeros((10051,))*np.nan # create a mask with only 1 dimension; then use this mask to mask out the .values file
        for i in range(lat.shape[0]):  # go through the lat and lon from original zarr file
            
            lo=lon[i]
            la=lat[i]
            p = Point(lo,la)
            if p.within(gdf.geometry[region_num]): # check if the point is in the selected AR5 region
                mask_region[i] =1 # let selected region to be 1 and other regions to be NAN
        return mask_region

    plt.figure(figsize=(12, 6))

    num_colors = len(region_list)
    colors = sns.color_palette("Set3", num_colors)
    np.random.shuffle(colors)  # Randomize colors plates

    masked_data=[] 

    for num in region_list:
        # create the AR5 mask at each selected region
        region_mask=create_ar5_mask(error_zarr, num, gdf)
        # apply the mask on the metric dataset
        masked_data.append(region_mask*error_zarr.data.sel(variable=var1).values)

    # Filter out NaN values from each array
    cleaned_data=[]
    label_list=[]
    for i in range(len(masked_data)):
        cleaned_data.append(masked_data[i][~np.isnan(masked_data[i])]) # remove the NAN for boxplot
        label_list.append(gdf.LAB[region_list[i]])
    # Create a boxplot
    box = plt.boxplot(cleaned_data, patch_artist=True, labels=label_list)

    # Apply colors to each box
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)

    if ymin!=None and ymax!=None:
        plt.ylim(ymin, ymax)

    plt.xlabel('Region', fontsize=20)
    plt.ylabel(var2, fontsize=20)
    plt.title('%s for %s' % (var2,var1), fontsize=22)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.show()


def density_scatter_plot(df, var1, var2, ymin, ymax):
    """
    Make the density scatter plot of evaluation metrics against the types of the climate variable;
    Note: this only applies for the climate variables with continueous values;
    `clim_cvl`, `clim_cvh` - low/high vegetation cover fraction of gridcell
    `clim_cu` - urban cover fraction
    `clim_theta_pwp` - partial wilting point of soil
    `clim_theta_cap` - field capacity of soil

    --- Parameters ---
    df:   the pd.DataFrame combining evaluation metrics and climatic variables
    var1: name of the climate variable -->str
    var2: name of the evaluation metrics -->str
    ymin: min value of the metrics -->int; need a first guess by setting ymin and ymax to be None
    ymax: min value of the metrics -->int 
    """

    white_viridis = LinearSegmentedColormap.from_list('white_viridis', [
        (0, '#ffffff'),
        (1e-20, '#440053'),
        (0.2, '#404388'),
        (0.4, '#2a788e'),
        (0.6, '#21a784'),
        (0.8, '#78d151'),
        (1, '#fde624'),], N=256)
    
    x=df[var1]
    y=df[var2]

    def using_mpl_scatter_density(fig, x, y):
        ax = fig.add_subplot(1, 1, 1, projection='scatter_density')
        density = ax.scatter_density(x, y, cmap=white_viridis)
        density.set_clim(0, 2)
        fig.colorbar(density, label='Number of points per pixel')

    fig = plt.figure(12,6)
    using_mpl_scatter_density(fig, x, y)
    plt.xlabel(var1, fontsize=16)
    plt.ylabel(var2, fontsize=16)
    plt.title('Density scatter plot between %s and %s ' % (var1, var2), fontsize=18)
    if ymin!=None and ymax!=None:
        plt.ylim(ymin, ymax)
    plt.show()
