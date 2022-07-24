from scipy.optimize import minimize
import numpy as np
from app.utils.math_utils import rolling_mean

from app.utils.constants import PRICE_DELTA_MA_WINDOW_DAYS
from app.utils.math_utils import generalised_logistic


def calculate_dynamic_data(data: dict, mdl_params: list, calculate_indicators=True) -> None:
    """
    Calculates quantities for the dynamic pairs model
    :return:
    """
    data['hedge_ratio'] = sum([generalised_logistic(**_params).calculate(data['x_index']) for _params in mdl_params])
    data['y1_times_f'] = np.multiply(data['y1_unscaled'], data['hedge_ratio'])
    data['y_spread'] = data['y1_times_f'] - data['y0']
    if calculate_indicators:
        data['y_spread_ma'] = rolling_mean(data['y_spread'], PRICE_DELTA_MA_WINDOW_DAYS)


def optimise_hedge_ratio(data: dict,
                         mdl_params: dict,
                         n_functions: int) -> float:
    """
    Optimises the hedge ratio
    :param data: Price data
    :param mdl_params: Model parameters object
    :param n_functions: Number of logistic functions in linear combination
    :return:
    """
    def cost_fcn(params):
        mdl_params = [{'l': params[0], 'm': params[1], 'k': params[2], 'x0': params[3]}]
        calculate_dynamic_data(data, mdl_params, calculate_indicators=False)
        cost = sum(abs(data['y_spread']))
        return cost

    x0 = [mdl_params[n][k] for k in ('l', 'm', 'k', 'x0') for n in range(n_functions)]
    res = minimize(cost_fcn, x0, method='nelder-mead')
    opt_params = []
    for ifunc in range(n_functions):
        skip = ifunc * 4
        opt_params.append(dict(l=res.x[0+skip], m=res.x[1+skip], k=res.x[2+skip], x0=res.x[3+skip]))
    return opt_params
