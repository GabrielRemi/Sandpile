import typing as tp

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from sandpile import *
from utils import *


def draw_slope(system: SandpileND, **kwargs) -> None:
    plt_kwargs = {}
    plt_kwargs.update(kwargs)

    plt.plot(range(len(system.average_slopes)), system.average_slopes, **plt_kwargs)


def load_system(dim, grid, bound, perturb, crit) -> tuple[SandpileND, pd.DataFrame]:
    system = SandpileND.load_from_file(f"data/data_{dim}_{grid}_{bound}_{perturb}_{crit}")
    data = system.get_avalanche_data()

    return system, data


def draw_distribution(x: tp.Sequence, y: tp.Sequence, log_scale: bool = True, **kwargs) -> None:
    if log_scale:
        plt.xscale("log")
        plt.yscale("log")

    plt_kwargs = {
        "s": 3
    }
    plt_kwargs.update(kwargs)
    plt.scatter(x, y, **plt_kwargs)


def _fit_func(x, m, b):
    return x * m + b


def exponent_fit(x: tp.Sequence, y: tp.Sequence, limit: int | None = None) -> list:
    x, y = np.asarray(x), np.asarray(y)

    ind = (y > 0) & (x < (limit or np.inf))
    x = np.log(x[ind])
    y = np.log(y[ind])

    output = curve_fit(_fit_func, x, y, full_output=True)

    return [output[0]]
