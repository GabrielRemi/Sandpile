import typing
import typing as tp

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from sandpile import *
from utils import *


def draw_slope(slope: typing.Sequence, step: int = 1, ax=None, **kwargs) -> None:
    plt_kwargs = {}
    plt_kwargs.update(kwargs)

    if ax is None:
        ax = plt
    ax.plot(np.array(range(len(slope))) * step, slope, **plt_kwargs)
    # plt.xlabel("time steps $t$")
    # plt.ylabel(r"average slope $\langle s \rangle$")


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

    ind = y != 0
    axis.scatter(x[ind], y[ind], **plt_kwargs)


def _fit_func(x, m, b):
    return x * m + b


def calculate_scaling_exponent(x: tp.Sequence, y: tp.Sequence, lower_limit: int | None = None,
                               upper_limit: int | None = None
                               ) -> float:
    x, y = np.asarray(x), np.asarray(y)

    ind = (y > 0) & (x < (upper_limit or np.inf)) & (x > (lower_limit or -np.inf))
    x = np.log(x[ind])
    y = np.log(y[ind])

    output = curve_fit(_fit_func, x, y)

    return output[0][0]


def get_conditional_expectation(x_sample: typing.Sequence, y_sample: typing.Sequence, **kwargs):
    x, y, joint_dist = get_2d_hist(x_sample, y_sample, **kwargs)

    _, y_dist = get_hist(y_sample)

    ind = y_dist > 0
    y = y[ind]

    e = (joint_dist.transpose() * x).sum(axis=1)[ind] / y_dist[ind]

    return y, e


def calculate_all_scaling_exponents(data_dir: str | pathlib.Path,
                                    limits: list[tuple[float | int | None, float | int | None]]
                                    ) -> pd.DataFrame:
    if isinstance(data_dir, str):
        data_dir = pathlib.Path(data_dir)
    elif not isinstance(data_dir, pathlib.Path):
        raise TypeError("path_dir variable has to be of type str or pathlib.Path")

    df = load_combine_avalanche_data_samples(data_dir.__str__())

    exponents = []

    # scaling of distributions
    for i, obs in enumerate("size time reach".split()):
        x, y = get_hist(df[obs])
        lower, upper = limits[i]
        exponent = calculate_scaling_exponent(x, y, lower_limit=lower, upper_limit=upper)
        exponents.append(exponent)

    # scaling of expectation values
    comb = [
        "size time".split(),
        "time size".split(),
        "size reach".split(),
        "reach size".split(),
        "time reach".split(),
        "reach time".split()
    ]
    for i, (x_obs, y_obs) in enumerate(comb):
        x, y = get_conditional_expectation(df[x_obs], df[y_obs])
        lower, upper = limits[i + 3]
        exponent = calculate_scaling_exponent(x, y, lower_limit=lower, upper_limit=upper)
        exponents.append(exponent)


    return pd.DataFrame({
        "tau": 1 - exponents[0],
        "alpha": 1 - exponents[1],
        "lambda": 1 - exponents[2],
        "gamma1": exponents[3],
        "1/gamma1": exponents[4],
        "gamma2": exponents[5],
        "1/gamma2": exponents[6],
        "gamma3": exponents[7],
        "1/gamma3": exponents[8],
    }, index=[0])
