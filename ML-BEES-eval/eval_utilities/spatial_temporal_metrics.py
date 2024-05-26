#######
# Module for evaluations metrics
#######

import numpy as np
import xarray as xr

class Metrics:

    def __init__(self, mod, ref, path):
        self.mod=mod
        self.ref=ref
        self.path=path

    def bias(self, relative=False):
        """
        Computes the bias of model dataset `mod` compared to reference
        dataset `ref`. If `relative` is "True", output the relative bias.

        See doi.org/10.1029/2018MS001354 for details.

        --- Parameters ---
        mod, ref:   xarray.Dataset; ml-emulator, ecland
        vars:       str or iterable of str
        relative:   bool

        --- Returns ---
        xarray.Dataset of biases --> xarray.DataArray
        """
        bias = self.mod.data.mean(dim="time") - self.ref.global_data_means # xarray.DataArray
        
        if relative:
            # Normalize bias using the central residual mean square of the reference data:
            crms = np.sqrt( (( self.ref.data - self.ref.global_data_means )**2).mean(dim="time") )

            nor_bias=(np.abs(bias)/crms).to_dataset(name='data')
            nor_bias.to_zarr(self.path + 'nor_bias.zarr', mode='w') # overwrite the existed .zarr
            
            return( nor_bias ) # xarray.DataArray
        else:
            bias_xr=bias.to_dataset(name='data')
            bias_xr.to_zarr(self.path + 'bias.zarr', mode='w')

            return( bias_xr ) # convert to xarray.Dataset -- easier for plotting
        
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

    def rmse(self, relative=False):
        """
        Computes the RMSE of model dataset `mod` (ailand) compared to reference
        dataset `ref` (ecland). If `relative` is "True", output the relative bias.

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
            anomalies_mod = self.mod.data - self.mod.data.mean(dim="time")
            anomalies_ref = self.ref.data - self.ref.global_data_means
            crmse = np.sqrt( (( anomalies_mod - anomalies_ref )**2).mean(dim="time") )
            
            crms = np.sqrt( (( self.ref.data - self.ref.global_data_means )**2).mean(dim="time") )

            nor_rmse=(crmse/crms).to_dataset(name='data')
            nor_rmse.to_zarr(self.path + 'nor_rmse.zarr', mode='w')

            return( nor_rmse )
        else:
            rmse = np.sqrt( (( self.mod.data - self.ref.data )**2).mean(dim="time") )

            rmse_xr=rmse.to_dataset(name='data')
            rmse_xr.to_zarr(self.path + 'rmse.zarr', mode='w')
            return( rmse_xr ) # convert to xarray.Dataset -- easier for plotting
        

    def phase_shift(self, agg_span="1D", cycle_res="dayofyear"):
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
        xarray.Dataset -- pixel-wise shift value?

        [1] https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timeseries-offset-aliases
        [2] https://pandas.pydata.org/docs/user_guide/timeseries.html#time-date-components
        """
        
        # Aggregation:
        aggregate_ref = self.ref.data.sel(variable=vars).resample(time=agg_span).mean(dim="time")
        aggregate_mod = self.mod.data.sel(variable=vars).resample(time=agg_span).mean(dim="time")

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

    def acc(self):
        '''
        Calculate the pixel-wise ACC scores for all the target variables;

        The anomaly correlation coefficient (ACC) is the correlation between anomalies of forecasts and anomalies of verifying values.

        If the variation pattern of the anomalies of forecast is perfectly coincident with that of the anomalies
        of verifying value, ACC will take the maximum value of 1. In turn, if the variation pattern is completely
        reversed, ACC takes the minimum value of -1. -- ACC [-1, 1]

        Equation according to ECMWF:

        https://confluence.ecmwf.int/display/FUG/
        Section+12.A+Statistical+Concepts+-+Deterministic+Data#Section12.AStatisticalConceptsDeterministicData-
        MeasureofSkill-theAnomalyCorrelationCoefficient(ACC)

        --- Parameters ---
        mod: ml-emulator output; 
        ref: ec-land output; 
        # vars: desired variable to evaluate; name according to namelist 

        --- Return ---
        acc_score: return acc at pixel-scale; xarray.DataArray
        
        '''
        # anomalies of ML-emulator data
        # anomalies_mod = mod.data.sel(variable=vars) - mod.data.sel(variable=vars).mean(dim="time")
        # the climatology should be the same -- using reference climatology
        anomalies_mod = self.mod.data - self.ref.global_data_means
        # anomalies of ecland-emulator data
        anomalies_ref = self.ref.data - self.ref.global_data_means

        acc_score = xr.corr(anomalies_mod, anomalies_ref, dim="time")

        acc_xr=acc_score.to_dataset(name='data')

        acc_xr.to_zarr(self.path + 'acc.zarr', mode='w')

        return( acc_xr )

    def reg_spat_dist_score(self, vars):
        '''
        Evaluate the spatial distribution pattern at regional scale:
        score the spatial distribution of the time averaged variable

        Details from https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2018MS001354; https://agupubs.onlinelibrary.wiley.com/doi/epdf/10.1029/2000JD900719

        --- Parameters ---
        ref: ml-emulator output; 
        mod: ec-land output; 
        vars: desired variable to evaluate; name according to namelist 

        --- Return ---
        Sdist: return a SINGLE value for a certain region during a certain time period
        
        '''

        # Calculate period mean for both datasets
        mod_mean = self.mod.data.sel(variable=vars).data.mean(dim="time")
        ref_mean = self.ref.data.sel(variable=vars).global_data_means

        # Calculate standard deviations spatially
        mod_std = mod_mean.std()
        ref_std = ref_mean.std()

        # Calculate the normalized standard deviation (σ) of the period mean
        sigma = mod_std / ref_std

        # Calculate the spatial correlation (R) of period mean values （?）
        # the anomaly of period mean; 
        # all the mean should be calculated over the domain
        mod_anomaly = mod_mean - mod_mean.mean()
        ref_anomaly = ref_mean - ref_mean.mean()
        
        covariance = (mod_anomaly * ref_anomaly).mean()

        # calculate the std of the spatial anomaly 
        forecast_anomaly_std = mod_anomaly.std()
        observed_anomaly_std = ref_anomaly.std()
        
        # calulation of spatial correlation (?)
        spatial_r = covariance / (forecast_anomaly_std * observed_anomaly_std)

        # Calculate Sdist using the provided relationship
        Sdist = 2 * (1 + spatial_r) / ((sigma + 1/sigma)**2)

        return Sdist

