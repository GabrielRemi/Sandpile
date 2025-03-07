from copy import deepcopy
from dataclasses import dataclass, field
from multiprocessing import Pool
from os import cpu_count, urandom

from computation import relax_avalanche, get_total_dissipation_rate
from utils import *
import json


if is_notebook():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

import numpy as np
from numpy.typing import NDArray
import pandas as pd
import gc


# Array: type = NDArray[np.int8]
Array: type = NDArray[np.uint8] | NDArray[np.uint32]


@dataclass
class SandpileND:
    dimension: int
    linear_grid_size: int
    critical_slope: int
    boundary_condition: typing.Literal["open", "closed"] = "open"
    perturbation: typing.Literal["conservative", "non conservative"] = "conservative"
    start_cfg: Array | None = None
    _shape: tuple = field(init=False, repr=False)

    _curr_slope: NDArray[np.uint8] = field(init=False, repr=False)
    average_slopes_list: list[float] = field(init=False, repr=False)
    average_slopes: Array = field(init=False, repr=False)
    _avalanches: dict[str, int | float | Array] = field(init=False, repr=False, default_factory=dict)
    _perturb_func: typing.Callable[[Array, typing.Sequence[int]], None] = field(init=False, repr=False)

    @property
    def shape(self):
        return self._shape

    def get_average_slopes(self) -> Array:
        return np.asarray(self.average_slopes_list)

    def __post_init__(self):
        if self.linear_grid_size > 255:
            raise ValueError("grid size limit = 255")
        self._shape = tuple([self.linear_grid_size] * self.dimension)
        self._initialize_system(self.start_cfg)

    def get_avalanche_data(self) -> pd.DataFrame:
        return pd.DataFrame(self._avalanches)

    def get_distribution(self, cut_off_time: int = 0) -> tuple[list[NDArray], NDArray]:
        return get_avalanche_hist_3d(pd.DataFrame(self._avalanches).query(f"time_step > {cut_off_time}"))

    def _append_avalanche_data_to_dict(self, av_data: tuple[int, int, int, float],
                                       dissipation_rate: NDArray[np.uint8]
                                       ) -> None:
        self._avalanches["time_step"].append(av_data[0])
        self._avalanches["size"].append(av_data[1])
        self._avalanches["time"].append(av_data[2])
        self._avalanches["reach"].append(av_data[3])
        self._avalanches["dissipation_rate"].append(dissipation_rate)

    def _initialize_system(self, start_cfg: Array | None = None) -> None:
        """Initialize the system for the simulation"""
        if self.perturbation == "conservative":
            self._perturb_func = self._conservative_perturbation
        elif self.perturbation == "non conservative":
            self._perturb_func = self._non_conservative_perturbation
        else:
            raise ValueError(f"unknown perturbation type: {self.perturbation}")

        if start_cfg is None:
            start_cfg = np.zeros(shape=self._shape, dtype=np.int8)
            if self.perturbation == "conservative" and self.boundary_condition == "closed":
                start_cfg = start_cfg.astype(np.int16)

        if start_cfg.shape != self._shape:
            raise Exception("Shape mismatch")

        self._curr_slope = deepcopy(start_cfg)
        self.average_slopes_list = [self._curr_slope.mean()]
        self._avalanches = {
            "time_step"       : [],
            "size"            : [],
            "time"            : [],
            "reach"           : [],
            "dissipation_rate": []
        }

    def _conservative_perturbation(self, cfg: Array, position_index: typing.Sequence[int]):
        if len(position_index) != self.dimension:
            Exception("position index dimension mismatch")

        cfg[*position_index] += self.dimension
        for dimension, index in enumerate(position_index):
            if index == 0:
                continue

            shifted_position_index = deepcopy(position_index)
            shifted_position_index[dimension] -= 1

            cfg[*shifted_position_index] -= 1

    def _non_conservative_perturbation(self, cfg: Array, position_index: typing.Sequence[int]):
        if len(position_index) != self.dimension:
            Exception("position index dimension mismatch")

        cfg[*position_index] += self.dimension

    def step(self, perturb_position: Array | None = None):
        if perturb_position is None:
            perturb_position = np.random.randint(low=0, high=self.linear_grid_size, size=self.dimension, dtype=np.uint8)
        perturb_position = np.asarray(perturb_position, dtype=np.uint8)

        self._perturb_func(self._curr_slope, perturb_position)

        if self._curr_slope[*perturb_position] > self.critical_slope:
            b = True if self.boundary_condition == "closed" else False
            s = (self.dimension, self.linear_grid_size, self.critical_slope, b)
            c = self._curr_slope.copy()
            try:
                av_data, dissipation_rate = relax_avalanche(len(self.average_slopes_list), self._curr_slope.reshape(-1),
                                                            perturb_position, s)
                self._append_avalanche_data_to_dict(av_data, dissipation_rate)
            except Exception as e:
                # with open("sandpile_error.log", "w") as f:
                #     print(len(self.average_slopes_list), self._curr_slope.tolist(),
                #           perturb_position, s, file=f)
                err = {
                    "time stamp": len(self.average_slopes_list),
                    "curr slope": c.tolist(),
                    "pos"       : perturb_position.tolist(),
                    "system"    : s,
                }
                with open("sandpile_error.log", "w") as f:
                    json.dump(err, f, indent=4)
                raise e

        self.average_slopes_list.append(self._curr_slope.mean())

    def __call__(self, time_steps: int, start_cfg: Array | None = None,
                 with_progress_bar: bool = True,
                 desc: str | None = None,
                 tqdm_position: int = 1
                 ) -> None:
        self._initialize_system(start_cfg)

        random_positions = np.random.randint(low=0, high=self.linear_grid_size,
                                             size=(time_steps - 1, self.dimension), dtype=np.uint8)

        if desc is None:
            desc = f"dim{self.dimension} grid{self.linear_grid_size} {self.boundary_condition} {self.perturbation}"
        min_iters = int(np.ceil(time_steps / 500))
        min_iters = min(min_iters, 100)

        if with_progress_bar:
            if is_notebook():
                tqdm.write("", end=" ")
            for position_index, _ in zip(
                    random_positions,
                    tqdm(range(1, time_steps), desc=desc, miniters=min_iters, leave=True, position=tqdm_position)):
                self.step()
        else:
            for position_index in random_positions:
                self.step()

        self.average_slopes = np.asarray(self.average_slopes_list)

    def save_separate(self, path: str, step: int = 1, time_cut_off: int = 0, max_time: int = 2000) -> None:
        """

        Save the data into separate files. One file for the average slopes,
        one for the computation data and one for the computation dissipation rates.

        :param str path: Path to save the files. Appends some text for each file.
        :param int step: The spacing of average slopes to save in file. higher
        numbers reduces disk space
        :param int time_cut_off: do not include any avalanches before that time step
        :param int max_time: maximum time in the total dissipation rate

        """
        np.save(path + ".slopes", [step, *self.average_slopes[::step]])

        df = self.get_avalanche_data().query(f"time_step > {time_cut_off}")
        df["size time reach".split()].to_csv(path + ".avalanche.csv", index=False)

        # np.savez_compressed(path + ".avalanche.npz", *df["dissipation_rate"])
        tdr = get_total_dissipation_rate(list(df["dissipation_rate"]), max_time)
        np.save(path + ".total_dissipation_rate.npy", tdr)

    def copy(self) -> "SandpileND":
        return deepcopy(self)

    def run_multiple_samples(self, folder_path: str, time_steps: int, sample_count: int,
                             start_cfg: Array | None = None, run: bool = True, desc: str = "sample", **kwargs
                             ) -> None | list[any]:
        """
        Run multiple samples for that system in parallel so that a statistical average of the data
        can be calculated

        :param folder_path: Folder where to save generated data
        :param time_steps: number of perturbation updates
        :param sample_count: number of samples to simulate
        :param desc: Description in the progress bar
        :param start_cfg: Starting configuration of the system
        :param run: If yes, run the simulations. If not, return a list of the processes
        """
        system = self

        system_desc = f"d{system.dimension}_g{system.linear_grid_size}_c{system.critical_slope}_"
        bound = "op" if system.boundary_condition == "open" else "cl"
        perturb = "co" if system.perturbation == "conservative" else "nco"
        system_desc += f"{bound}_{perturb}"

        data_dir = pathlib.Path(folder_path) / system_desc
        data_dir = data_dir.resolve().absolute()
        # edges = [
        # np.array(range(0, 2001)) - 0.5,
        # np.array(range(1, 251)) - 0.5,
        # np.array(range(0, 251)) - 0.5
        # ]

        pre_run_num = 0  # if data was simulated before, the number of samples before this
        if not data_dir.exists():
            os.mkdir(data_dir)
        elif not data_dir.is_dir():
            print("Name for folder already exists for a file", file=sys.stderr)
        else:
            clear = kwargs.get("clear", False)
            if clear:
                for file in data_dir.iterdir():
                    file.unlink()
            else:
                pre_run_num = len(list(data_dir.glob("*.csv.gz")))

        system._initialize_system()
        tasks = [
            (system, time_steps, start_cfg, i, data_dir, kwargs.get("step") or 1, desc,
             kwargs.get("time_cut_off", 0), kwargs.get("max_time", 2000))
            for i in range(pre_run_num, sample_count + pre_run_num)]

        # if run:
        with Pool(cpu_count() - 2) as pool:
            list(tqdm(pool.imap_unordered(_process_args_list, tasks), total=sample_count, desc=system_desc))

        # total_bin_values = np.sum(bins, axis=0)
        # bin_centers = [0.5 * (x[1:] + x[:-1]) for x in edges]
        # np.savez_compressed(data_dir / "distribution.npz", size=bin_centers[0], time=bin_centers[1],
        #                     reach=bin_centers[2],
        #                     bins=total_bin_values)
        # else:
        #     return [(_process, task) for task in tasks]
        if kwargs.get("to_distribution", False):
            generate_3d_distribution_from_data_sample(data_dir)
            for file in data_dir.glob("*.csv"):
                file.unlink()


def _process(system: SandpileND, time_steps: int, start_cfg: Array, index: int, data_dir, step: int,
             desc: str, time_cut_off: int = 0, max_time: int = 2000
             ):
    np.random.seed(int.from_bytes(urandom(4), "big"))
    system = deepcopy(system)
    system(time_steps, start_cfg, with_progress_bar=False)
    system.save_separate((data_dir / f"data_{index}").absolute().__str__(), step,
                         time_cut_off=time_cut_off, max_time=max_time)

    # df = system.get_avalanche_data()
    # bins, _ = np.histogramdd((df["size"], df["time"], df["reach"]), bins=[*edges])
    del system
    gc.collect()

    # return bins


def _process_args_list(args):
    return _process(*args)


def get_avalanche_hist_3d(df: pd.DataFrame):
    parameter = "size time reach".split()
    edges = []
    for p in parameter:
        edges.append(np.array(range(
            np.floor(df[p]).min().astype(int),
            np.ceil(df[p]).max().astype(int) + 1)) - 0.5)

    bins, _ = np.histogramdd((df["size"], df["time"], df["reach"]), bins=[*edges])
    for i in range(3):
        edges[i] = 0.5 * (edges[i][1:] + edges[i][:-1])

    return edges, bins


def generate_3d_distribution_from_data_sample(data_dir: str | pathlib.Path, sample_count: int | None = None):
    df = load_combine_avalanche_data_samples(data_dir, with_dissipation=False, sample_count=sample_count)
    if isinstance(data_dir, str):
        data_dir = pathlib.Path(data_dir)

    centers, bins = get_avalanche_hist_3d(df)
    np.savez_compressed(data_dir / "distribution.npz",
                        size=centers[0],
                        time=centers[1],
                        reach=centers[2],
                        bins=bins)


def load_3d_dist(data_dir: str | pathlib.Path):
    if isinstance(data_dir, str):
        data_dir = pathlib.Path(data_dir)
    data = np.load(data_dir / "distribution.npz")

    return (data["size"], data["time"], data["reach"]), data["bins"]
