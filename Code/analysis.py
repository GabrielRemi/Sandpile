import typing
import typing as tp

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from sandpile import *
from utils import *


def draw_slope(slope: typing.Sequence, step: int = 1, **kwargs) -> None:
    plt_kwargs = {}
    plt_kwargs.update(kwargs)

    plt.plot(np.array(range(len(slope))) * step, slope, **plt_kwargs)
    plt.xlabel("time steps $t$")
    plt.ylabel(r"average slope $\langle s \rangle$")


def load_system(dim, grid, bound, perturb, crit) -> tuple[SandpileND, pd.DataFrame]:
    system = SandpileND.load_from_file(f"data/data_{dim}_{grid}_{bound}_{perturb}_{crit}")
    data = system.get_avalanche_data()

    return system, data


def draw_distribution(x: tp.Sequence, y: tp.Sequence, log_scale: bool = True, axis=None, **kwargs) -> None:
    if log_scale:
        if axis is None:
            plt.xscale("log")
            plt.yscale("log")
        else:
            axis.set_xscale("log")
            axis.set_yscale("log")

    plt_kwargs = {
        "s": 3
    }
    plt_kwargs.update(kwargs)
    if axis is None:
        axis = plt
    axis.scatter(x, y, **plt_kwargs)


def _fit_func(x, m, b):
    return x * m + b


def exponent_fit(x: tp.Sequence, y: tp.Sequence, limit: int | None = None) -> list:
    x, y = np.asarray(x), np.asarray(y)

    ind = (y > 0) & (x < (limit or np.inf))
    x = np.log(x[ind])
    y = np.log(y[ind])

    output = curve_fit(_fit_func, x, y, full_output=True)

    return [output[0]]
