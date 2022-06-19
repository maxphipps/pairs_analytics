from unittest import TestCase
import numpy as np
import copy

from model_utils import calculate_dynamic_data


class test_calculate_dynamic_data(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = dict()
        cls.data['y0'] = np.array([2., 5., 3.,])
        cls.data['scale_factor'] = np.array([3., 3., 3.,])
        cls.data['y1_unscaled'] = np.array([3., 6., 4.])

    def test_scale_factor(self):
        """
        Test price scaling
        :return:
        """
        _data = copy.copy(self.data)
        calculate_dynamic_data(_data, ma_window_days=1)
        self.assertListEqual(list(_data['y1_times_f']), [9., 18., 12.])

    def test_price_residue(self):
        """
        Test price residue
        :return:
        """
        _data = copy.copy(self.data)
        calculate_dynamic_data(_data, ma_window_days=1)
        self.assertListEqual(list(_data['y_residue']), [9.-2., 18.-5., 12.-3.])
