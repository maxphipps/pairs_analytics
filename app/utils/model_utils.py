from scipy.optimize import minimize
import numpy as np
import copy
from app.utils.math_utils import rolling_mean

from app.utils.constants import PRICE_DELTA_MA_WINDOW_DAYS
from app.utils.math_utils import generalised_logistic


def calculate_dynamic_data(data: dict, mdl_params: list, calculate_indicators=True) -> None:
    """
    Calculates quantities for the dynamic pairs model
    :return:
    """
    # TODO: handle case where no functions are active
    active_mdl_params = [_params for _params in copy.deepcopy(mdl_params) if _params.get('active', True)]
    for _mdl_params in active_mdl_params:
        _mdl_params.pop('active', None)
    data['hedge_ratio'] = sum([generalised_logistic(**_params).calculate(data['x_index'])
                               for _params in active_mdl_params])
    data['y1_times_f'] = np.multiply(data['y1_unscaled'], data['hedge_ratio'])
    data['y_spread'] = data['y1_times_f'] - data['y0']
    if calculate_indicators:
        data['y_spread_ma'] = rolling_mean(data['y_spread'], PRICE_DELTA_MA_WINDOW_DAYS)


def optimise_hedge_ratio(data: dict,
                         mdl_params: dict,
                         num_functions: int) -> float:
    """
    Optimises the hedge ratio
    :param data: Price data
    :param mdl_params: Model parameters object
    :param num_functions: Number of logistic functions in linear combination
    :return:
    """
    def cost_fcn(params):
        # Unpack active functions' parameters
        params_dict_list = []
        for ifunc in range(num_active_functions):
            skip = ifunc * 4
            params_dict_list.append(dict(l=params[0 + skip],
                                         m=params[1 + skip],
                                         k=params[2 + skip],
                                         x0=params[3 + skip]))
        calculate_dynamic_data(data, params_dict_list, calculate_indicators=False)
        cost = sum(abs(data['y_spread']))
        return cost

    # TODO: messy, needs cleanup
    num_active_functions = 0
    x0 = []
    for ifunc in range(num_functions):
        if mdl_params[ifunc].get('active', True):
            num_active_functions += 1
            for key in ('l', 'm', 'k', 'x0'):
                x0.append(mdl_params[ifunc][key])
    res = minimize(cost_fcn, x0, method='nelder-mead')
    opt_params = copy.deepcopy(mdl_params)
    for ifunc in range(num_functions):
        if mdl_params[ifunc].get('active'):
            skip = ifunc * 4
            opt_params[ifunc] = dict(l=res.x[0+skip],
                                     m=res.x[1+skip],
                                     k=res.x[2+skip],
                                     x0=res.x[3+skip])
    return opt_params
