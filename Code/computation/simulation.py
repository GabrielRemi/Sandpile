import gc
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path
from typing import *

import numpy as np
import psutil  # type: ignore
from IPython.core.getipython import get_ipython
from numpy.typing import NDArray

from .sandpile import *


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
           "run_multiple_samples", "generate_3d_distribution_from_directory", "generate_power_spectrum_from_directory",
           "clean_up_directory"]

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

    for i in range(3):
        if len(edges[i]) > 1000:
            edges[i] = edges[i][::len(edges[i]) // 1000]

    bins, _ = np.histogramdd((data[0], data[1], data[2]), bins=[*edges], density=True)
    for i in range(3):
        edges[i] = 0.5 * (edges[i][1:] + edges[i][:-1])

    return edges, bins / bins.sum()


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
    return (np.fft.rfft(dissipation_rate).__abs__() ** 2)


def calculate_power_frequencies(dissipation_rate: NDArray):
    return np.fft.rfftfreq(len(dissipation_rate))


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
    tqdm_update_steps: Optional[int] = 1000


def _process(meta: ProcessMeta, shared_tqdm_value: mp.Value):
    np.random.seed(int.from_bytes(os.urandom(4), "big"))
    if meta.bit == 8:
        sandpile = Sandpile8Bit
        # sandpile_simulate_worker = sandpile_simulate_worker_8Bit
    elif meta.bit == 16:
        sandpile = Sandpile16Bit
        # sandpile_simulate_worker = sandpile_simulate_worker_16Bit
    else:
        raise ValueError(f"unknown bit value: {meta.bit}")

    system = sandpile(meta.dim, meta.grid, meta.crit_slope, meta.is_open, meta.is_conservative)
    system.time_cut_off = meta.time_cut_off

    # system.simulate(meta.time_steps, None, None)
    sandpile_simulate_worker(system, shared_tqdm_value, meta.time_steps, meta.tqdm_update_steps)

    save_avalanche_data(system, (meta.data_dir / f"avalanche_data_{meta.index}.npz").absolute().__str__())

    d = system.generate_total_dissipation_rate(meta.total_dissipation_time)
    # power_spectrum = np.fft.rfft(d).__abs__() ** 2
    power_spectrum = calculate_power_spectrum(d)
    np.save(meta.data_dir / f"power_spectrum_{meta.index}.npy", power_spectrum)

    # df = system.get_avalanche_data()
    # bins, _ = np.histogramdd((df["size"], df["time"], df["reach"]), bins=[*edges])
    del system
    gc.collect()


def _worker(input_queue, shared_tqdm_value: mp.Value):
    while True:
        process_meta = input_queue.get()
        if process_meta is None:
            break
        _process(process_meta, shared_tqdm_value)


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
        worker_count: number of workers to use for the simulation, defaults to cpu_count() - 2
        tqdm_update_interval: interval in seconds for updating the progress bar, defaults to 0.01 seconds
        tqdm_update_steps: number of updates if the progress bar for one sample, defaults to 1000 steps
    """
    worker_count = kwargs.get("worker_count", mp.cpu_count() - 2)

    system_desc = f"d{system.dim}_g{system.grid}_c{system.crit_slope}_"
    bound = "op" if system.get_has_open_boundary() else "cl"
    perturb = "co" if system.get_has_conservative_perturbation() else "nco"
    system_desc += f"{bound}_{perturb}"

    data_dir = Path(folder_path)
    if not data_dir.exists():
        os.mkdir(data_dir)
    data_dir = (data_dir / system_desc).resolve().absolute()

    # momentarily deprecated
    if not data_dir.exists():
        os.mkdir(data_dir)
    elif not data_dir.is_dir():
        print("Name for folder already exists for a file", file=sys.stderr)
    else:
        if kwargs.get("clear", True):
            for file in data_dir.iterdir():
                file.unlink()

    if isinstance(system, Sandpile8Bit):
        bit = 8
    elif isinstance(system, Sandpile16Bit):
        bit = 16
    else:
        raise ValueError("System argument has to be a Sandpile8Bit or Sandpile16Bit object")

    shared_tqdm_value = mp.Value("i", 0)
    input_queue = mp.Queue()
    for i in range(sample_count):
        args = ProcessMeta(bit, system.dim, system.grid, system.crit_slope, system.get_has_open_boundary(),
                           system.get_has_conservative_perturbation(), system.time_cut_off, time_steps,
                           total_dissipation_time, i, data_dir, desc, start_cfg, kwargs.get("tqdm_update_steps", 1000))
        input_queue.put(args)
    for _ in range(worker_count):
        input_queue.put(None)

    processes = []
    for _ in range(worker_count):
        p = mp.Process(target=_worker, args=(input_queue, shared_tqdm_value))
        p.start()
        processes.append(p)

    pbar_amount = sample_count * time_steps
    pbar = tqdm(total=pbar_amount, desc=system_desc)

    while shared_tqdm_value.value < pbar_amount:
        time.sleep(kwargs.get("tqdm_update_interval", 0.01))
        pbar.n = shared_tqdm_value.value
        pbar.refresh()

    pbar.n = shared_tqdm_value.value
    pbar.refresh()
    pbar.close()

    for p in processes:
        p.join()

    generate_3d_distribution_from_directory(data_dir)
    generate_power_spectrum_from_directory(data_dir)
    # mean_power_spectrum = np.zeros(shape=(total_dissipation_time // 2 + 1,), dtype=np.float64)
    #
    # for file in data_dir.glob("power_spectrum_*.npy"):
    #     mean_power_spectrum += np.load(file)
    #     file.unlink()
    # mean_power_spectrum /= sample_count
    #
    # np.save(data_dir / "mean_power_spectrum.npy", mean_power_spectrum)


def generate_3d_distribution_from_directory(dir: Path):
    files = dir.glob("avalanche_data_*.npz")
    if dir.joinpath("avalanche_distribution.npz").exists():
        centers, bins = load_3d_dist(dir.joinpath("avalanche_distribution.npz"))
    else:
        f = next(files)
        centers, bins = generate_3d_distribution_from_data_samples([f.__str__()])
        f.unlink()

    edges = []
    for c in centers:
        if len(c) < 2:
            print(f"No Avalanches generated for {dir}")
            for file in files:
                file.unlink()
            return
        w = c[1] - c[0]
        e = [x - w / 2 for x in c]
        e.append(c[-1] + w / 2)
        edges.append(np.array(e))

    i = 1
    for file in files:
        i += 1
        data = np.load(file)
        b, _ = np.histogramdd((data["size"], data["time"], data["reach"]), bins=[*edges], density=True)
        bins += b / b.sum()
        file.unlink()
    bins /= bins.sum()
    np.savez_compressed(dir / "avalanche_distribution.npz", size=centers[0], time=centers[1], reach=centers[2],
                        bins=bins)


def generate_power_spectrum_from_directory(dir: Path):
    files = dir.glob("power_spectrum_*.npy")
    try:
        f = next(files)
    except StopIteration:
        return
    power_spectrum = np.load(f)
    n = len(power_spectrum)

    if dir.joinpath("mean_power_spectrum.npy").exists():
        print("Exists")
        mean_power_spectrum = np.load(dir.joinpath("mean_power_spectrum.npy"))
        if len(mean_power_spectrum) != n:
            raise ValueError("Mean power spectrum has the wrong length")
    else:
        mean_power_spectrum = np.zeros(shape=(n,), dtype=np.float64)
    mean_power_spectrum += power_spectrum / power_spectrum.max()
    f.unlink()

    for file in files:
        mean_power_spectrum += np.load(file)
        file.unlink()
    mean_power_spectrum /= mean_power_spectrum.max()
    np.save(dir / "mean_power_spectrum.npy", mean_power_spectrum)


def clean_up_directory(dir: Path):
    generate_3d_distribution_from_directory(dir)
    generate_power_spectrum_from_directory(dir)
