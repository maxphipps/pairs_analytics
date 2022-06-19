import numpy as np


def rolling_mean(x, n):
    ret = np.cumsum(x, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    ret = ret[n - 1:] / n
    # Prepend Nones so length matches length of x
    return np.insert(ret, 0, [np.nan]*(n-1))
