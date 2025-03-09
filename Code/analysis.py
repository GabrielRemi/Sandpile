import typing as tp
from numpy.typing import ArrayLike

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from scipy.optimize import curve_fit

from sandpile import *
from utils import *

import uncertainties as unc


def draw_slope(slope: ArrayLike, step: int = 1, ax=None, **kwargs) -> None:
    plt_kwargs = {}
    plt_kwargs.update(kwargs)

    if ax is None:
        ax = plt
    ax.plot(np.asarray(range(len(slope))) * step, slope, **plt_kwargs)  # type: ignore
    # plt.xlabel("time steps $t$")
    # plt.ylabel(r"average slope $\langle s \rangle$")


def draw_distribution(x: ArrayLike, y: ArrayLike, log_scale: bool = True, axis=None, **kwargs) -> None:
    if log_scale:
        if axis is None:
            plt.xscale("log")
            plt.yscale("log")
        else:
            axis.set_xscale("log")
            axis.set_yscale("log")

    plt_kwargs = {"s": 3}
    plt_kwargs.update(kwargs)
    if axis is None:
        axis = plt

    ind = (y != 0) & (x != 0)
    axis.scatter(x[ind], y[ind], **plt_kwargs)  # type: ignore


def _fit_func(x, m, b):
    return x * m + b


def calculate_scaling_exponent(
    x: ArrayLike, y: ArrayLike, lower_limit: int | float | None = None, upper_limit: int | float | None = None
) -> tuple[unc.core.Variable, unc.core.Variable]:
    x, y = np.asarray(x), np.asarray(y)

    ind = (y > 0) & (x != 0) & (x < (upper_limit or np.inf)) & (x > (lower_limit or -np.inf))
    x = np.log(x[ind])
    y = np.log(y[ind])

    output = curve_fit(_fit_func, x, y)

    m, m_err = output[0][0], np.sqrt(output[1][0, 0])
    n, n_err = output[0][1], np.sqrt(output[1][1, 1])
    return unc.ufloat(m, m_err), unc.ufloat(n, n_err)


# def get_conditional_expectation(x_sample: typing.Sequence, y_sample: typing.Sequence, **kwargs):
#     x, y, joint_dist = get_2d_hist(x_sample, y_sample, **kwargs)

#     _, y_dist = get_hist(y_sample)

#     ind = y_dist > 0
#     y = y[ind]

#     e = (joint_dist.transpose() * x).sum(axis=1)[ind] / y_dist[ind]

#     return y, e


# TODO Change this to new
# def calculate_all_scaling_exponents(
#     data: pd.DataFrame, limits: list[tuple[float | int | None, float | int | None]]
# ) -> tuple[pd.DataFrame, list[unc.core.AffineScalarFunc]]:
#     exponents: list[unc.core.Variable] = []
#     amplitudes: list[unc.core.AffineScalarFunc] = []

#     df = data
#     # scaling of distributions
#     for i, obs in enumerate("size time reach".split()):
#         x, y = get_hist(df[obs])  # type: ignore
#         lower, upper = limits[i]
#         exponent, amp = calculate_scaling_exponent(x, y, lower_limit=lower, upper_limit=upper)
#         exponents.append(exponent)
#         amplitudes.append(np.e**amp)

#     # scaling of expectation values
#     comb = [
#         "size time".split(),
#         "time size".split(),
#         "size reach".split(),
#         "reach size".split(),
#         "time reach".split(),
#         "reach time".split(),
#     ]
#     for i, (x_obs, y_obs) in enumerate(comb):
#         x, y = get_conditional_expectation(df[x_obs], df[y_obs])
#         lower, upper = limits[i + 3]
#         exponent, amp = calculate_scaling_exponent(x, y, lower_limit=lower, upper_limit=upper)
#         exponents.append(exponent)
#         amplitudes.append(np.e**amp)

#     return (
#         pd.DataFrame(
#             {
#                 "tau": unc.ufloat(1, 0) - exponents[0],
#                 "alpha": unc.ufloat(1, 0) - exponents[1],
#                 "lambda": unc.ufloat(1, 0) - exponents[2],
#                 "gamma1": exponents[3],
#                 "1/gamma1": exponents[4],
#                 "gamma2": exponents[5],
#                 "1/gamma2": exponents[6],
#                 "gamma3": exponents[7],
#                 "1/gamma3": exponents[8],
#             },
#             index=[0],
#         ),
#         amplitudes,
#     )


def plot_scaling_exponents(s: NDArray, t: NDArray, r: NDArray, bins, axs: tp.Any):
    s_dist = bins.sum(axis=(1, 2))
    t_dist = bins.sum(axis=(0, 2))
    r_dist = bins.sum(axis=(0, 1))

    # Plot s vs bins.sum(axis=(1, 2))
    axs[0, 0].set_ylabel("$P(S=s)$")
    axs[0, 0].set_xlabel("$s$")
    axs[0, 1].set_ylabel("$P(T=t)$")
    axs[0, 1].set_xlabel("$t$")
    axs[0, 2].set_ylabel("$P(R=r)$")
    axs[0, 2].set_xlabel("$r$")
    draw_distribution(s, s_dist, axis=axs[0, 0])
    draw_distribution(t, t_dist, axis=axs[0, 1])
    draw_distribution(r, r_dist, axis=axs[0, 2])

    axs[1, 0].set_ylabel("$E[S | T=t]$")
    axs[1, 0].set_xlabel("$t$")
    axs[1, 1].set_ylabel("$E[T | S=s]$")
    axs[1, 1].set_xlabel("$s$")

    axs[1, 2].set_ylabel("$E[S | R=r]$")
    axs[1, 2].set_xlabel("$r$")
    axs[2, 0].set_ylabel("$E[R | S=s]$")
    axs[2, 0].set_xlabel("$s$")

    axs[2, 1].set_ylabel("$E[T | R=r]$")
    axs[2, 1].set_xlabel("$r$")
    axs[2, 2].set_ylabel("$E[R | T=t]$")
    axs[2, 2].set_xlabel("$t$")

    # E(S | T=t)
    ind = t_dist > 0
    est = (bins.sum(axis=2) * s.reshape(len(s), 1)).sum(axis=0)[ind] / t_dist[ind]
    draw_distribution(t[ind], est, axis=axs[1, 0])

    # E[T | S=s]
    ind = s_dist > 0
    ets = (bins.sum(axis=2) * t).sum(axis=1)[ind] / s_dist[ind]
    draw_distribution(s[ind], ets, axis=axs[1, 1])

    # E[S | R=r]
    ind = r_dist > 0
    esr = (bins.sum(axis=1) * s.reshape(len(s), 1)).sum(axis=0)[ind] / r_dist[ind]
    draw_distribution(r[ind], esr, axis=axs[1, 2])

    # E[R | S=s]
    ind = s_dist > 0
    ers = (bins.sum(axis=1) * r).sum(axis=1)[ind] / s_dist[ind]
    draw_distribution(s[ind], ers, axis=axs[2, 0])

    # E[T | R=r]
    ind = r_dist > 0
    etr = (bins.sum(axis=0) * t.reshape(len(t), 1)).sum(axis=0)[ind] / r_dist[ind]
    draw_distribution(r[ind], etr, axis=axs[2, 1])

    # E[R | T=t]
    ind = t_dist > 0
    ert = (bins.sum(axis=0) * r).sum(axis=1)[ind] / t_dist[ind]
    draw_distribution(t[ind], ert, axis=axs[2, 2])
