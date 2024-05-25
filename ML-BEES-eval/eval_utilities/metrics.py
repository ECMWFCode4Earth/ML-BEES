#######
# Module for evaluations metrics
#######

import numpy as np
import xarray as xr

def bias(mod, ref, vars, relative=False):
    """
    Computes the bias of model dataset `mod` compared to reference
    dataset `ref`. If `relative` is "True", output the relative bias.

    See doi.org/10.1029/2018MS001354 for details.

    --- Parameters ---
    mod, ref:   xarray.Dataset
    vars:       str or iterable of str
    relative:   bool

    --- Returns ---
    xarray.Dataset of biases
    """
    bias = mod.data.sel(variable=vars).mean(dim="time") - ref.global_data_means.sel(variable=vars)
    
    if relative:
        # Normalize bias using the central residual mean square of the reference data:
        crms = np.sqrt( (( ref.data.sel(variable=vars) - ref.global_data_means.sel(variable=vars) )**2).mean(dim="time") )
        
        return( np.abs(bias)/crms )
    else:
        return( bias )
    

def spatial_mean(ds, var, weights=None):
    """
    Computes the (weighted) spatial mean for selected variables `vars``
    on dataset `ds`.

    See doi.org/10.1029/2018MS001354 for details.

    --- Parameters ---
    ds:         xarray.Dataset
    vars:       str or iterable of str
    weights:    np.array

    --- Returns ---
    xarray.Dataset
    """
    if weights is None:
        weights = np.ones(ds.dims["x"]) #uniform weights

    total_weighted_area = (weights * ds.clim_data.sel(clim_variable="clim_cell_area")).sum(dim="x") 
    spatial_integral = (weights * ds.data.sel(variable=var) * ds.clim_data.sel(clim_variable="clim_cell_area")).sum(dim="x")

    return( spatial_integral/total_weighted_area )


def rmse(mod, ref, vars, relative=False):
    """
    Computes the RMSE of model dataset `mod` compared to reference
    dataset `ref`. If `relative` is "True", output the relative bias.

    See doi.org/10.1029/2018MS001354 for details.

    --- Parameters ---
    mod, ref:   xarray.Dataset
    vars:       str or iterable of str
    relative:   bool

    --- Returns ---
    xarray.Dataset of biases
    """
    
    if relative:
        # Normalize centralized RMSE using the central residual mean square of the reference data:
        anomalies_mod = mod.data.sel(variable=vars) - mod.data.sel(variable=vars).mean(dim="time")
        anomalies_ref = ref.data.sel(variable=vars) - ref.global_data_means.sel(variable=vars)
        crmse = np.sqrt( (( anomalies_mod - anomalies_ref )**2).mean(dim="time") )
        
        crms = np.sqrt( (( ref.data.sel(variable=vars) - ref.global_data_means.sel(variable=vars) )**2).mean(dim="time") )

        return( crmse/crms )
    else:
        rmse = np.sqrt( (( mod.data.sel(variable=vars) - ref.data.sel(variable=vars) )**2).mean(dim="time") )
        return( rmse )
    

def phase_shift(mod, ref, vars, agg_span="1D", cycle_res="dayofyear"):
    """
    Computes the RMSE of model dataset `mod` compared to reference dataset `ref`.
    `agg_span` controls the aggregation for computing the mean cycle using pandas'
    frequency aliases[1].
    `cycle_res` sets the resolution for computing the mean cycle and can be 
    chosen from the options available in the pandas datetime accessor construct[2].

    Example:
    The pre-set `agg_span`="1D" will compute daily mean values. Those daily values
    are averaged over the all available years keeping the daily resolution, because
    `cycle_res` is set to "dayofyear". The pre-set results in a diurnal phase-shift
    given in days.

    See doi.org/10.1029/2018MS001354 for details.

    --- Parameters ---
    mod, ref:   xarray.Dataset
    vars:       str or iterable of str
    agg_span:   str
    cycle_res:  str

    --- Returns ---
    xarray.Dataset of biases

    [1] https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timeseries-offset-aliases
    [2] https://pandas.pydata.org/docs/user_guide/timeseries.html#time-date-components
    """
    
    # Aggregation:
    aggregate_ref = ref.data.sel(variable=vars).resample(time=agg_span).mean(dim="time")
    aggregate_mod = mod.data.sel(variable=vars).resample(time=agg_span).mean(dim="time")

    # Compute mean cycle:
    cycle_ref = aggregate_ref.groupby(f"time.{cycle_res}").mean(dim="time")
    cycle_mod = aggregate_mod.groupby(f"time.{cycle_res}").mean(dim="time")


    shift = cycle_mod.argmax(dim=cycle_res) - cycle_ref.argmax(dim=cycle_res)
    
    # Respect periodicity (e.g. phase shift might happen across start/end of year):
    pl = cycle_ref.sizes[cycle_res] #periodicity depends on the selected cycle resolution
    stack = np.stack((shift, shift - pl), axis=-1) #stack normal and shifted version
    shift = np.where(np.argmin(abs(stack), axis=-1), shift - pl, shift) #element-wise absolute min

    return(shift)


def interannual_var():
    pass

def acc(mod, ref, vars):
    '''
    Calculate the pixel-wise ACC scores;

    The anomaly correlation coefficient (ACC) is the correlation between anomalies of forecasts and anomalies of verifying values.

    Equation according to ECMWF:

    https://confluence.ecmwf.int/display/FUG/
    Section+12.A+Statistical+Concepts+-+Deterministic+Data#Section12.AStatisticalConceptsDeterministicData-
    MeasureofSkill-theAnomalyCorrelationCoefficient(ACC)

    --- Parameters ---
    ref: ml-emulator output; 
    mod: ec-land output; 
    vars: desired variable to evaluate; name according to namelist 

    --- Return ---
    acc_score: return acc at pixel-scale; xarray.DataArray
    
    '''
    # anomalies of ML-emulator data
    anomalies_mod = mod.data.sel(variable=vars) - mod.data.sel(variable=vars).mean(dim="time")
    # anomalies of ecland-emulator data
    anomalies_ref = ref.data.sel(variable=vars) - ref.global_data_means.sel(variable=vars)

    # Calculate the covariance between the anomalies
    covariance = (anomalies_mod * anomalies_ref).mean(dim='time')
    
    # Calculate the standard deviations of the anomalies
    std1 = anomalies_mod.std(dim='time')
    std2 = anomalies_ref.std(dim='time')
    
    # Calculate the ACC
    acc_score = covariance / (std1 * std2)

    return acc_score

def reg_spat_dist_score(mod, ref):
    '''
    Evaluate the spatial distribution pattern at regional scale:
    score the spatial distribution of the time averaged variable

    Details from https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2018MS001354; https://agupubs.onlinelibrary.wiley.com/doi/epdf/10.1029/2000JD900719

    --- Parameters ---
    ref: ml-emulator output; 
    mod: ec-land output; 
    vars: desired variable to evaluate; name according to namelist 

    --- Return ---
    Sdist: return a single value for a certain region during a certain time period
    
    '''

    # Calculate period mean for both datasets
    mod_mean = mod.data.sel(variable=vars).mean(dim="time")
    ref_mean = ref.global_data_means.sel(variable=vars)

    # Calculate standard deviations spatially
    mod_std = mod_mean.std(dim=("lat", "lon"))
    ref_std = ref_mean.std(dim=("lat", "lon"))

    # Calculate the normalized standard deviation (σ) of the period mean
    sigma = mod_std / ref_std

    # Calculate the spatial correlation (R) of period mean values （?）
    # the anomaly of period mean; 
    # all the mean should be calculated over the domain
    mod_anomaly = mod_mean - mod_mean.mean(dim=("lat", "lon"))
    ref_anomaly = ref_mean - ref_mean.mean(dim=("lat", "lon"))
    
    covariance = (mod_anomaly * ref_anomaly).mean(dim=("lat", "lon"))

    # calculate the std of the spatial anomaly 
    forecast_anomaly_std = mod_anomaly.std(dim=("lat", "lon"))
    observed_anomaly_std = ref_anomaly.std(dim=("lat", "lon"))
    
    # calulation of spatial correlation (?)
    spatial_r = covariance / (forecast_anomaly_std * observed_anomaly_std)

    # Calculate Sdist using the provided relationship
    Sdist = 2 * (1 + spatial_r) / ((sigma + 1/sigma)**2)

    return Sdist


def plot():
    pass