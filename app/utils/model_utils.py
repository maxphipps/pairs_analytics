from scipy.optimize import minimize
import numpy as np
from app.utils.math_utils import rolling_mean

from app.utils.constants import PRICE_DELTA_MA_WINDOW_DAYS
from app.utils.math_utils import generalised_logistic


def calculate_dynamic_data(data: dict, mdl_params: dict, calculate_indicators=True) -> None:
    """
    Calculates quantities for the dynamic pairs model
    :return:
    """
    data['hedge_ratio'] = generalised_logistic(**mdl_params).calculate(data['x_index'])
    data['y1_times_f'] = np.multiply(data['y1_unscaled'], data['hedge_ratio'])
    data['y_spread'] = data['y1_times_f'] - data['y0']
    if calculate_indicators:
        data['y_spread_ma'] = rolling_mean(data['y_spread'], PRICE_DELTA_MA_WINDOW_DAYS)


def optimise_hedge_ratio(data: dict,
                         mdl_params: dict) -> float:
    def cost_fcn(params):
        mdl_params = {'l': params[0], 'm': params[1], 'k': params[2], 'x0': params[3]}
        calculate_dynamic_data(data, mdl_params, calculate_indicators=False)
        cost = sum(abs(data['y_spread']))
        return cost

    x0 = [mdl_params[k] for k in ('l', 'm', 'k', 'x0')]
    res = minimize(cost_fcn, x0, method='nelder-mead')
    opt_params = dict(l=res.x[0], m=res.x[1], k=res.x[2], x0=res.x[3])
    return opt_params
