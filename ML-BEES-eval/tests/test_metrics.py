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
        Test that `metrics.bias` returns expected bias on mock datasets
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
            data_vars=dict(data=(["time", "x", "variable"], data_mod)),
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


    def test_relative_bias(self):
        """
        Test that `metrics.bias`'s relative error functionality works on mock datasets
        """

        expected_bias = 1.
        ref_scale = 3.1415 #see assertion below

        # Define dimensions:
        lats = np.arange(60.0, 60.1, step=0.1)
        lons = np.arange(20.0, 20.1, step=0.1)
        time = np.arange(np.datetime64("2010-02-01T00", "ns"),
                        np.datetime64("2012-02-01T00", "ns"), 
                        np.timedelta64(6,"h"))
        variables = ["Bees", "Supercalifragilisticexpialidocious"]

        # Generate data:
        np.random.seed(0)
        data_ref = np.random.normal(scale=ref_scale, size=(len(time), len(lats), len(variables)))
        data_mod = expected_bias + data_ref

        # Create datasets:
        ds_mod = xr.Dataset(
            data_vars=dict(data=(["time", "x", "variable"], data_mod)),
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
        # Since `data_ref` is normally distributed with scale `ref_scale`, the normalization is expected to
        # be 1/`ref_scale`. The expected relative error is then `expected_bias`/`ref_scale`.
        abs_diff = np.abs(metrics.bias(ds_mod, ds_ref, variables, relative=True) - expected_bias/ref_scale)
        self.assertTrue( (abs_diff < 0.1).all().item() )



class TestPhaseShift(unittest.TestCase):
    def test_expected_shift(self):
        """
        Test that `metrics.phase_shift` returns expected shift on mock datasets
        """

        periodicity = 365 * 24/6 #diurnal cycle for a 6h resolution dataset
        expected_shift = 10 * 24/6

        # Define dimensions:
        lats = np.arange(60.0, 60.1, step=0.1)
        lons = np.arange(20.0, 20.1, step=0.1)
        time = np.arange(np.datetime64("2010-02-01T00", "ns"),
                        np.datetime64("2012-02-01T00", "ns"), 
                        np.timedelta64(6,"h"))
        variables = ["t2m", "e"]

        # Generate data:
        data_ref = np.sin(np.arange(len(time)) * 2*np.pi/periodicity)
        data_ref = np.repeat(data_ref[:, np.newaxis], len(lats), axis=-1)
        data_ref = np.repeat(data_ref[:, :, np.newaxis], len(variables), axis=-1)

        data_mod = np.sin((np.arange(len(time)) - expected_shift) * 2*np.pi/periodicity)
        data_mod = np.repeat(data_mod[:, np.newaxis], len(lats), axis=-1)
        data_mod = np.repeat(data_mod[:, :, np.newaxis], len(variables), axis=-1)

        # Create datasets:
        ds_ref = xr.Dataset(
            data_vars=dict(data=(["time", "x", "variable"], data_ref),
                        global_data_means=(["x", "variable"], np.mean(data_ref, axis=0))),
            coords=dict(lat=("x", lats), lon=("x", lons), time=("time", time), 
                        variable=("variable", variables), x=("x", np.arange(len(lats))))
        )

        ds_mod = xr.Dataset(
            data_vars=dict(data=(["time", "x", "variable"], data_mod)),
            coords=dict(lat=("x", lats), lon=("x", lons), time=("time", time), 
                        variable=("variable", variables), x=("x", np.arange(len(lats))))
        )

        # Test:
        np.testing.assert_allclose(expected_shift, metrics.phase_shift(ds_mod, ds_ref, vars=variables) * 24/6)



if __name__ == '__main__':
    unittest.main()