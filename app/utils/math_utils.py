from dataclasses import dataclass
import numpy as np


def rolling_mean(x, n):
    ret = np.cumsum(x, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    ret = ret[n - 1:] / n
    # Prepend Nones so length matches length of x
    return np.insert(ret, 0, [np.nan]*(n-1))

@dataclass
class generalised_logistic:
    """
    Modified logistic function
    :param l: left asymptote
    :param m: right asymptote
    :param k: sigmoid steepness
    :param x0: sigmoid midpoint x-shift
    :return:
    """
    l: float
    m: float
    k: float
    x0: float

    def _func(self, x):
        """
        The modified logistic function
        :param x:
        :return:
        """
        return self.l + (self.m - self.l) / (1 + np.exp(-self.k * (x - self.x0)))

    def calculate(self, x_list) -> float:
        """
        Returns modified logistic for given list of x values
        :param x: List of x value
        :return:
        """
        return np.array([self._func(x) for x in x_list])
