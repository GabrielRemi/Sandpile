import typing as tp

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from scipy.optimize import curve_fit

from sandpile import *
from utils import *

import uncertainties as unc


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

    ind = (y != 0) & (x != 0)
    axis.scatter(x[ind], y[ind], **plt_kwargs)


def _fit_func(x, m, b):
    return x * m + b


def calculate_scaling_exponent(x: tp.Sequence, y: tp.Sequence, lower_limit: int | None = None,
                               upper_limit: int | None = None
                               ) -> tuple[unc.core.Variable, unc.core.Variable]:
    x, y = np.asarray(x), np.asarray(y)

    ind = (y > 0) & (x != 0) & (x < (upper_limit or np.inf)) & (x > (lower_limit or -np.inf))
    x = np.log(x[ind])
    y = np.log(y[ind])

    output = curve_fit(_fit_func, x, y)

    m, m_err = output[0][0], np.sqrt(output[1][0, 0])
    n, n_err = output[0][1], np.sqrt(output[1][1, 1])
    return unc.ufloat(m, m_err), unc.ufloat(n, n_err)


def get_conditional_expectation(x_sample: typing.Sequence, y_sample: typing.Sequence, **kwargs):
    x, y, joint_dist = get_2d_hist(x_sample, y_sample, **kwargs)

    _, y_dist = get_hist(y_sample)

    ind = y_dist > 0
    y = y[ind]

    e = (joint_dist.transpose() * x).sum(axis=1)[ind] / y_dist[ind]

    return y, e


def calculate_all_scaling_exponents(data: pd.DataFrame,
                                    limits: list[tuple[float | int | None, float | int | None]]
                                    ) -> tuple[pd.DataFrame, list[unc.core.AffineScalarFunc]]:
    exponents: list[unc.core.Variable] = []
    amplitudes: list[unc.core.AffineScalarFunc] = []

    df = data
    # scaling of distributions
    for i, obs in enumerate("size time reach".split()):
        x, y = get_hist(df[obs])
        lower, upper = limits[i]
        exponent, amp = calculate_scaling_exponent(x, y, lower_limit=lower, upper_limit=upper)
        exponents.append(exponent)
        amplitudes.append(np.e ** amp)

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
        exponent, amp = calculate_scaling_exponent(x, y, lower_limit=lower, upper_limit=upper)
        exponents.append(exponent)
        amplitudes.append(np.e ** amp)

    return pd.DataFrame({
        "tau"     : unc.ufloat(1, 0) - exponents[0],
        "alpha"   : unc.ufloat(1, 0) - exponents[1],
        "lambda"  : unc.ufloat(1, 0) - exponents[2],
        "gamma1"  : exponents[3],
        "1/gamma1": exponents[4],
        "gamma2"  : exponents[5],
        "1/gamma2": exponents[6],
        "gamma3"  : exponents[7],
        "1/gamma3": exponents[8],
    }, index=[0]), amplitudes


def plot_scaling_exponents(data: pd.DataFrame, exponents: pd.Series | None = None,
                           amplitudes: typing.Sequence[unc.core.AffineScalarFunc] | None = None,
                           fig_ax: tuple[Figure, Axes] | None = None
                           ) -> None:
    if exponents is None and amplitudes is None:
        with_fits = False
    elif (exponents is not None and amplitudes is None) or (exponents is None and amplitudes is not None):
        raise ValueError("both exponents and amplitudes have to be non None values")
    else:
        with_fits = True

    # Draw distributions
    fig: Figure
    ax: Axes
    if fig_ax is None:
        fig, ax = plt.subplots(3, 3, figsize=(9, 7))
    else:
        fig, ax = fig_ax
    for i, obs in enumerate("size time reach".split()):
        if obs == "reach":
            bins = 50
        else:
            bins = None
        bins = None # TODO what with this
        x, y = get_hist(data[obs], bins=bins)
        ind = (x != 0) & (y != 0)
        draw_distribution(x[ind], y[ind], axis=ax[0, i], s=3)
        if with_fits:
            ax[0, i].plot(x[ind], unc.nominal_value(amplitudes[i]) * x[ind] ** unc.nominal_value(1 - exponents.iloc[i]))
    comb = [
        "size time".split(),
        "time size".split(),
        "size reach".split(),
        "reach size".split(),
        "time reach".split(),
        "reach time".split()
    ]

    indices = [
        (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)
    ]

    for i, (index, (xn, yn)) in enumerate(zip(indices, comb)):
        x, y = get_conditional_expectation(data[xn], data[yn])
        ind = (x != 0) & (y != 0)
        x, y = x[ind], y[ind]
        draw_distribution(x, y, axis=ax[*index])
        if with_fits:
            ax[*index].plot(x, unc.nominal_value(amplitudes[i + 3]) * x ** unc.nominal_value(exponents.iloc[i + 3]))
