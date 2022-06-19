import numpy as np
from app.utils.math_utils import rolling_mean


def calculate_dynamic_data(data: dict,
                           ma_window_days: int) -> None:
    """
    Calculates quantities for the dynamic pairs model
    :return:
    """
    data['y1_times_f'] = np.multiply(data['y1_unscaled'], data['scale_factor'])
    data['y_residue'] = data['y1_times_f'] - data['y0']
    data['y_residue_ma'] = rolling_mean(data['y_residue'], ma_window_days)