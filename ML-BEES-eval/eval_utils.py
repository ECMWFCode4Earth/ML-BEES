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
    bias = mod.global_data_means.sel(variable=vars) - ref.global_data_means.sel(variable=vars)
    
    if relative:
        # Normalize bias using the central residual mean square of the reference data:
        crms = np.sqrt( (( ref.data.sel(variable=vars) - ref.global_data_means.sel(variable=vars) )**2).mean(dim="time") )
        
        return( np.abs(bias)/crms )
    else:
        return( bias )
    

def spatial_mean(ds, vars, weights=None):
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
    spatial_integral = (weights * ds.data.sel(variable="e") * ds.clim_data.sel(clim_variable="clim_cell_area")).sum(dim="x")

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
        anomalies_mod = mod.data.sel(variable=vars) - mod.global_data_means.sel(variable=vars)
        anomalies_ref = ref.data.sel(variable=vars) - ref.global_data_means.sel(variable=vars)
        crmse = np.sqrt( (( anomalies_mod - anomalies_ref )**2).mean(dim="time") )
        
        crms = np.sqrt( (( ref.data.sel(variable=vars) - ref.global_data_means.sel(variable=vars) )**2).mean(dim="time") )

        return( crmse/crms )
    else:
        rmse = np.sqrt( (( mod.data.sel(variable=vars) - ref.data.sel(variable=vars) )**2).mean(dim="time") )
        return( rmse )