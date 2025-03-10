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


def expo(x, m: unc.core.Variable, n: unc.core.Variable):
    return np.exp(n.nominal_value) * x**m.nominal_value


def plot_scaling_exponents(
    s: NDArray,
    t: NDArray,
    r: NDArray,
    bins,
    axs: tp.Any,
    limits: list[tuple[float | None, float | None]] | None = None,
    do_plot: bool = True,
) -> pd.DataFrame | None:
    """Plot the scaling exponents for a given distribution and fit the exponents, only if limits are given."""

    if limits is not None and len(limits) != 9:
        raise ValueError("Limits must be a list of 9 tuples")

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

    ind_t = t_dist > 0
    ind_s = s_dist > 0
    ind_r = r_dist > 0

    # E(S | T=t)
    est = (bins.sum(axis=2) * s.reshape(len(s), 1)).sum(axis=0)[ind_t] / t_dist[ind_t]
    draw_distribution(t[ind_t], est, axis=axs[1, 0])

    # E[T | S=s]
    ets = (bins.sum(axis=2) * t).sum(axis=1)[ind_s] / s_dist[ind_s]
    draw_distribution(s[ind_s], ets, axis=axs[1, 1])

    # E[S | R=r]
    esr = (bins.sum(axis=1) * s.reshape(len(s), 1)).sum(axis=0)[ind_r] / r_dist[ind_r]
    draw_distribution(r[ind_r], esr, axis=axs[1, 2])

    # E[R | S=s]
    ers = (bins.sum(axis=1) * r).sum(axis=1)[ind_s] / s_dist[ind_s]
    draw_distribution(s[ind_s], ers, axis=axs[2, 0])

    # E[T | R=r]
    etr = (bins.sum(axis=0) * t.reshape(len(t), 1)).sum(axis=0)[ind_r] / r_dist[ind_r]
    draw_distribution(r[ind_r], etr, axis=axs[2, 1])

    # E[R | T=t]
    ert = (bins.sum(axis=0) * r).sum(axis=1)[ind_t] / t_dist[ind_t]
    draw_distribution(t[ind_t], ert, axis=axs[2, 2])

    if limits is None:
        return
    else:
        s_m, s_n = calculate_scaling_exponent(s, s_dist, *limits[0])
        t_m, t_n = calculate_scaling_exponent(t, t_dist, *limits[1])
        r_m, r_n = calculate_scaling_exponent(r, r_dist, *limits[2])
        st_m, st_n = calculate_scaling_exponent(t[ind_t], est, *limits[3])
        ts_m, ts_n = calculate_scaling_exponent(s[ind_s], ets, *limits[4])
        sr_m, sr_n = calculate_scaling_exponent(r[ind_r], esr, *limits[5])
        rs_m, rs_n = calculate_scaling_exponent(s[ind_s], ers, *limits[6])
        tr_m, tr_n = calculate_scaling_exponent(r[ind_r], etr, *limits[7])
        rt_m, rt_n = calculate_scaling_exponent(t[ind_t], ert, *limits[8])

        if do_plot:
            axs[0, 0].plot(s, expo(s, s_m, s_n))
            axs[0, 1].plot(t, expo(t, t_m, t_n))
            axs[0, 2].plot(r, expo(r, r_m, r_n))
            axs[1, 0].plot(t[ind_t], expo(t[ind_t], st_m, st_n))
            axs[1, 1].plot(s[ind_s], expo(s[ind_s], ts_m, ts_n))
            axs[1, 2].plot(r[ind_r], expo(r[ind_r], sr_m, sr_n))
            axs[2, 0].plot(s[ind_s], expo(s[ind_s], rs_m, rs_n))
            axs[2, 1].plot(r[ind_r], expo(r[ind_r], tr_m, tr_n))
            axs[2, 2].plot(t[ind_t], expo(t[ind_t], rt_m, rt_n))

        return pd.DataFrame(
            {
                "tau": unc.ufloat(1, 0) - s_m,  # type: ignore
                "alpha": unc.ufloat(1, 0) - t_m,  # type: ignore
                "lambda": unc.ufloat(1, 0) - r_m,  # type: ignore
                "gamma1": st_m,
                "1/gamma1": ts_m,
                "gamma2": sr_m,
                "1/gamma2": rs_m,
                "gamma3": tr_m,
                "1/gamma3": rt_m,
            },
            index=[0],
        )
