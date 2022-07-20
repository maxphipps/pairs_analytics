import pandas as pd
from scipy.optimize import minimize
import numpy as np
from app.utils.math_utils import rolling_mean
import math


def calculate_dynamic_data(data: dict,
                           ma_window_days: int) -> None:
    """
    Calculates quantities for the dynamic pairs model
    :return:
    """
    data['y1_times_f'] = np.multiply(data['y1_unscaled'], data['scale_factor'])
    data['y_residue'] = data['y1_times_f'] - data['y0']
    data['y_residue_ma'] = rolling_mean(data['y_residue'], ma_window_days)


def scan_discontinuities(data: dict,
                         initial_scale_factor: float,
                         discontinuity_idx_ser: list):
    """
    Performs grid search for possible discontinuities
    Algorithm:
    1) Let tDiscontinuity = t0 + tDelta * n
    2) Minimise 'left' cost by varying scale factor for 'left' data < tDiscontinuity
    3) Minimise 'right' cost by varying scale factor for 'right' data >= tDiscontinuity
    4) Total cost = 'left' cost + 'right' cost
    5) Iterate n
    """
    # TODO: Downsample data from daily resolution to accelerate calculation
    df_data = pd.DataFrame(data)
    num_samples = 50
    data_len = len(data['x_data'])
    skip = int(data_len / num_samples)
    idx_min = skip
    idx_max = data_len - skip
    results = pd.DataFrame(columns=['l_scale_factor', 'r_scale_factor',
                                    'l_cost', 'r_cost',
                                    'net_cost'])
    l_scale_factor = initial_scale_factor
    r_scale_factor = initial_scale_factor
    for discontinuity_idx in range(idx_min, idx_max, skip):
        # TODO: After grid search, perform local search
        # Find closest discontinuities previously found
        # Next is supremum (if exists)
        mask = discontinuity_idx_ser > discontinuity_idx
        discontinuity_idx_next = discontinuity_idx_ser[mask].min()
        if math.isnan(discontinuity_idx_next): discontinuity_idx_next = data_len
        discontinuity_idx_next = int(np.round(discontinuity_idx_next))  # cast to int
        # Previous is infimum (if exists)
        mask = discontinuity_idx_ser < discontinuity_idx
        discontinuity_idx_prev = discontinuity_idx_ser[mask].max()
        if math.isnan(discontinuity_idx_prev): discontinuity_idx_prev = 0
        discontinuity_idx_prev = int(np.round(discontinuity_idx_prev))  # cast to int

        left_data = df_data.iloc[discontinuity_idx_prev:discontinuity_idx].copy()
        right_data = df_data.iloc[discontinuity_idx:discontinuity_idx_next].copy()
        l_scale_factor, l_cost = optimise_scale_factor(left_data, l_scale_factor)
        r_scale_factor, r_cost = optimise_scale_factor(right_data, r_scale_factor)
        results = results.append({'discontinuity_idx_prev': discontinuity_idx_prev,
                                  'discontinuity_idx_next': discontinuity_idx_next,
                                  'discontinuity_idx': discontinuity_idx,
                                  'x_timestamp': df_data['x_data'].iloc[discontinuity_idx],
                                  'l_scale_factor': l_scale_factor,
                                  'r_scale_factor': r_scale_factor,
                                  'l_cost': l_cost,
                                  'r_cost': r_cost,
                                  'net_cost': l_cost + r_cost},
                                 ignore_index=True)
    # Normalise cost values for plotting
    results['norm_net_cost'] = results['net_cost'] / results['net_cost'].max() * 100.
    return results

def optimise_scale_factor(data: dict,
                          initial_scale_factor: float) -> float:
    def cost_fcn(scale_factor):
        data['y1_times_f'] = data['y1_unscaled'] * scale_factor
        data['y_residue'] = data['y1_times_f'] - data['y0']
        cost = sum(abs(data['y_residue']))
        return cost

    res = minimize(cost_fcn, x0=initial_scale_factor, method='nelder-mead')
    return res.x[0], abs(sum(data['y_residue']))
