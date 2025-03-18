import os
import sys
from copy import deepcopy
from multiprocessing import cpu_count, Pool
from pathlib import Path
from typing import *

import numpy as np
import psutil  # type: ignore
from IPython.core.getipython import get_ipython
from numpy.typing import NDArray

from .sandpile import Sandpile16Bit, Sandpile8Bit


def get_memory() -> float:
    """Return memory usage in MB"""
    process = psutil.Process(os.getpid())
    memory_usage = process.memory_info().rss  # in bytes
    return memory_usage / (1024 ** 2)


def is_notebook() -> bool:
    try:
        # Check if the environment is an IPython shell (which includes Jupyter)
        if 'ipykernel' in sys.modules or 'IPython' in sys.modules:
            if get_ipython().__class__.__name__ == 'ZMQInteractiveShell':
                # We're in a Jupyter notebook
                return True
        return False
    except NameError:
        return False  # If 'IPython' is not available, we're not in a notebook


if is_notebook():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

__all__ = ["Sandpile", "get_avalanche_hist_3d", "save_avalanche_distribution", "save_avalanche_data", "load_3d_dist",
           "generate_3d_distribution_from_data_samples", "calculate_power_spectrum", "calculate_power_frequencies",
           "run_multiple_samples"]

Sandpile: TypeAlias = Sandpile8Bit | Sandpile16Bit


@overload
def get_avalanche_hist_3d(*, system: Sandpile) -> tuple[list[NDArray[np.float64]], NDArray[np.float64]]:
    ...


@overload
def get_avalanche_hist_3d(size: NDArray, time: NDArray, reach: NDArray) -> tuple[
    list[NDArray[np.float64]], NDArray[np.float64]]:
    ...


def get_avalanche_hist_3d(size: Optional[NDArray] = None, time: Optional[NDArray] = None,
                          reach: Optional[NDArray] = None, *, system: Sandpile | None = None
                          ) -> tuple[
    list[NDArray[np.float64]], NDArray[np.float64]]:
    edges = []
    if system is not None:
        data = [system.get_size(), system.get_time(), system.get_reach()]
    else:
        data = [size, time, reach]
    for p in data:
        edges.append(np.array(range(np.floor(p).min().astype(int), np.ceil(p).max().astype(int) + 1)) - 0.5)

    bins, _ = np.histogramdd((data[0], data[1], data[2]), bins=[*edges])  # type: ignore
    for i in range(3):
        edges[i] = 0.5 * (edges[i][1:] + edges[i][:-1])

    return edges, bins


def save_avalanche_distribution(system: Sandpile, path: str):
    edges, bins = get_avalanche_hist_3d(system=system)
    np.savez_compressed(path, size=edges[0], time=edges[1], reach=edges[2], bins=bins)


def save_avalanche_data(system: Sandpile, path: str):
    s, r, t = system.get_size(), system.get_reach(), system.get_time()

    np.savez_compressed(path, size=s, time=t, reach=r)


def load_3d_dist(data_dir: str | Path):
    if isinstance(data_dir, str):
        data_dir = Path(data_dir)
    data = np.load(data_dir)

    return (data["size"], data["time"], data["reach"]), data["bins"]


def generate_3d_distribution_from_data_samples(data_files: list[str]) -> tuple[
    list[NDArray[np.float64]], NDArray[np.float64]]:
    # df = load_combine_avalanche_data_samples(data_dir, with_dissipation=False, sample_count=sample_count)

    s, t, r = np.array([]), np.array([]), np.array([])

    for file in data_files:
        data = np.load(file)
        s = np.append(s, data["size"])
        t = np.append(t, data["time"])
        r = np.append(r, data["reach"])

    return get_avalanche_hist_3d(s, t, r)


from dataclasses import dataclass


def calculate_power_spectrum(dissipation_rate: NDArray):
    return (np.fft.rfft(dissipation_rate).__abs__() ** 2)[1:]


def calculate_power_frequencies(dissipation_rate: NDArray):
    return np.fft.rfftfreq(len(dissipation_rate))[1:]


@dataclass
class ProcessMeta:
    bit: int
    dim: int
    grid: int
    crit_slope: int
    is_open: bool
    is_conservative: bool
    time_cut_off: int
    time_steps: int
    total_dissipation_time: int
    index: int
    data_dir: Path
    desc: str
    start_cfg: Optional[NDArray] = None


def _process(meta: ProcessMeta):
    np.random.seed(int.from_bytes(os.urandom(4), "big"))
    if meta.bit == 8:
        system = Sandpile8Bit(meta.dim, meta.grid, meta.crit_slope, meta.is_open, meta.is_conservative)
    elif meta.bit == 16:
        system = Sandpile16Bit(meta.dim, meta.grid, meta.crit_slope, meta.is_open, meta.is_conservative)
    else:
        raise ValueError(f"unknown bit value: {meta.bit}")
    system.time_cuf_off = meta.time_cut_off

    system.simulate(meta.time_steps, meta.start_cfg)
    # system.save_separate(
    #     (data_dir / f"data_{index}").absolute().__str__(), step, time_cut_off=time_cut_off, max_time=max_time
    # )
    save_avalanche_data(system, (meta.data_dir / f"avalanche_data_{meta.index}.npz").absolute().__str__())

    d = system.generate_total_dissipation_rate(meta.total_dissipation_time)
    # power_spectrum = np.fft.rfft(d).__abs__() ** 2
    power_spectrum = calculate_power_spectrum(d)
    np.save(meta.data_dir / f"power_spectrum_{meta.index}.npy", power_spectrum)

    # df = system.get_avalanche_data()
    # bins, _ = np.histogramdd((df["size"], df["time"], df["reach"]), bins=[*edges])
    del system
    # gc.collect()


def run_multiple_samples(
    system: Sandpile,
    folder_path: str,
    time_steps: int,
    total_dissipation_time: int,
    sample_count: int,
    start_cfg: NDArray | None = None,
    desc: str = "sample",
    **kwargs,
) -> None:
    """
    Run multiple samples for that system in parallel so that a statistical average of the data
    can be calculated

    :param system: Sandpile system to simulate
    :param folder_path: Folder where to save generated data
    :param time_steps: number of perturbation updates
    :param total_dissipation_time: number of time steps to calculate the total dissipation rate
    :param sample_count: number of samples to simulate
    :param desc: Description in the progress bar
    :param start_cfg: Starting configuration of the system

    :param kwargs:
    """

    system_desc = f"d{system.dim}_g{system.grid}_c{system.crit_slope}_"
    bound = "op" if system.get_has_open_boundary() else "cl"
    perturb = "co" if system.get_has_conservative_perturbation() else "nco"
    system_desc += f"{bound}_{perturb}"

    data_dir = Path(folder_path)
    if not data_dir.exists():
        os.mkdir(data_dir)
    data_dir = (data_dir / system_desc).resolve().absolute()

    pre_run_num = 0  # if data was simulated before, the number of samples before this
    if not data_dir.exists():
        os.mkdir(data_dir)
    elif not data_dir.is_dir():
        print("Name for folder already exists for a file", file=sys.stderr)
    else:
        for file in data_dir.iterdir():
            file.unlink()

    if isinstance(system, Sandpile8Bit):
        bit = 8
    elif isinstance(system, Sandpile16Bit):
        bit = 16
    else:
        raise ValueError("System argument has to be a Sandpile8Bit or Sandpile16Bit object")

    tasks = [ProcessMeta(bit, system.dim, system.grid, system.crit_slope, system.get_has_open_boundary(),
                         system.get_has_conservative_perturbation(), system.time_cuf_off, time_steps,
                         total_dissipation_time, i, data_dir, desc, start_cfg) for i in
             range(pre_run_num, sample_count + pre_run_num)]

    # ------RUN----------
    with Pool(cpu_count() - 2) as pool:
        list(tqdm(pool.imap_unordered(_process, tasks), total=sample_count, desc=system_desc))

    # Generate Data to analyze
    edges, bins = generate_3d_distribution_from_data_samples(list(data_dir.glob("avalanche_data_*.npz")))
    for file in data_dir.glob("avalanche_data_*.npz"):
        file.unlink()

    np.savez_compressed(data_dir / "avalanche_distribution.npz", size=edges[0], time=edges[1], reach=edges[2],
                        bins=bins)

    mean_power_spectrum = np.zeros(shape=(total_dissipation_time // 2 + 1,), dtype=np.float64)

    for file in data_dir.glob("power_spectrum_*.npy"):
        mean_power_spectrum += np.load(file)
        file.unlink()
    mean_power_spectrum /= sample_count

    np.save(data_dir / "mean_power_spectrum.npy", mean_power_spectrum)
