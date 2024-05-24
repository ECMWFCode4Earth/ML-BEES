#######
# Script for unit testing eval_utilities.metrics functionality
# Run using "python -m unittest tests.test_metrics" form parent directory.
#######

import unittest

import numpy as np
import xarray as xr

from eval_utilities import metrics


class TestBias(unittest.TestCase):
    def test_expected_bias(self):
        """
        Test that function returns expected bias on mock datasets
        """

        expected_bias = 1.

        # Define dimensions:
        lats = np.arange(60.0, 61.0, step=0.1)
        lons = np.arange(20.0, 21.0, step=0.1)
        time = np.arange(np.datetime64("2010-02-01T00", "ns"),
                        np.datetime64("2010-02-02T00", "ns"), 
                        np.timedelta64(6,"h"))
        variables = ["Bees", "Supercalifragilisticexpialidocious"]

        # Generate data:
        np.random.seed(0)
        data_ref = np.random.normal(size=(len(time), len(lats), len(variables)))
        data_mod = expected_bias + data_ref

        # Create datasets:
        ds_mod = xr.Dataset(
            data_vars=dict(data=(["time", "x", "variable"], data_mod),
                        global_data_means=(["x", "variable"], np.mean(data_mod, axis=0))),
            coords=dict(lat=("x", lats), lon=("x", lons), time=("time", time), 
                        variable=("variable", variables), x=("x", np.arange(len(lats))))
        )

        ds_ref = xr.Dataset(
            data_vars=dict(data=(["time", "x", "variable"], data_ref),
                        global_data_means=(["x", "variable"], np.mean(data_ref, axis=0))),
            coords=dict(lat=("x", lats), lon=("x", lons), time=("time", time), 
                        variable=("variable", variables), x=("x", np.arange(len(lats))))
        )

        # Test:
        np.testing.assert_allclose(expected_bias, metrics.bias(ds_mod, ds_ref, vars=variables))




if __name__ == '__main__':
    unittest.main()