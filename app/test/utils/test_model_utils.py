from unittest import TestCase
import numpy as np
import copy

from app.utils.model_utils import calculate_dynamic_data


class test_calculate_dynamic_data(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = dict()
        cls.data['x_index'] = np.array([0, 1, 2])
        cls.data['y0'] = np.array([2., 5., 3.,])
        cls.data['y1_unscaled'] = np.array([3., 6., 4.])

        # 2x lines: y=3.0 and y=2.0
        cls.mdl_params = [dict(l=3.0, m=3.0, k=1.0, x0=0., active=1),
                          dict(l=2.0, m=2.0, k=1.0, x0=0., active=1)]

    def test_hedge_ratio(self):
        """
        Test price scaling
        :return:
        """
        _data = copy.copy(self.data)
        _mdl_params = copy.copy(self.mdl_params)
        calculate_dynamic_data(_data, _mdl_params)
        self.assertListEqual(list(_data['y1_times_f']), [3.*5., 6.*5., 4.*5.])

    def test_price_spread(self):
        """
        Test price spread
        :return:
        """
        _data = copy.copy(self.data)
        _mdl_params = copy.copy(self.mdl_params)
        calculate_dynamic_data(_data, _mdl_params)
        self.assertListEqual(list(_data['y_spread']), [15.-2., 30.-5., 20.-3.])
