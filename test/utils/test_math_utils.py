from unittest import TestCase
import numpy as np

from utils.math_utils import rolling_mean


def nan_equal(a, b):
    try:
        np.testing.assert_equal(a, b)
    except AssertionError:
        return False
    return True


class TestRollingMean(TestCase):
    def test_integer_rolling_mean(self):
        x = [5, 4, 3, 2]
        n = 3
        res = list(rolling_mean(x, n))
        # Use below if allowing mean calculation with number of samples < n
        # self.assertListEqual(res, [5., 4.5, 4., 3.])
        assert nan_equal(res, [np.nan, np.nan, 4., 3.])
        assert isinstance(res[3], float)

    def test_float_rolling_mean(self):
        x = [22, 42, 45, 29, 2]
        n = 2
        res = list(rolling_mean(x, n))
        assert nan_equal(res, [np.nan, 32., 43.5, 37., 15.5])