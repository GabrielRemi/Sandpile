from copy import deepcopy
from dataclasses import dataclass, field
from multiprocessing import Pool
from os import cpu_count, urandom

from avalanche import relax_avalanche
from utils import *


if is_notebook():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

import numpy as np
from numpy.typing import NDArray
import pandas as pd
import gc


__all__ = "SandpileND".split()

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

        self._perturb_func(self._curr_slope, perturb_position)

        if self._curr_slope[*perturb_position] > self.critical_slope:
            b = True if self.boundary_condition == "closed" else False
            s = (self.dimension, self.linear_grid_size, self.critical_slope, b)
            av_data, dissipation_rate = relax_avalanche(len(self.average_slopes_list), self._curr_slope.reshape(-1),
                                                        perturb_position, s)
            self._append_avalanche_data_to_dict(av_data, dissipation_rate)

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

        if is_notebook():
            # print("\r", end=" ", flush=True)
            tqdm.write("", end=" ")

        if with_progress_bar:
            for position_index, _ in zip(
                    random_positions,
                    tqdm(range(1, time_steps), desc=desc, miniters=min_iters, leave=True, position=tqdm_position)):
                self.step()
        else:
            for position_index in random_positions:
                self.step()

        self.average_slopes = np.asarray(self.average_slopes_list)

    def save_separate(self, path: str, step: int = 1) -> None:
        """

        Save the data into separate files. One file for the average slopes,
        one for the avalanche data and one for the avalanche dissipation rates.

        :param str path: Path to save the files. Appends some text for each file.
        :param int step: The spacing of average slopes to save in file. higher
        numbers reduces disk space

        """
        np.save(path + ".slopes", [step, *self.average_slopes[::step]])

        df = self.get_avalanche_data()
        df["time_step size time reach".split()].to_csv(path + ".avalanche.csv.gz", index=False, compression="gzip")

        np.savez_compressed(path + ".avalanche.npz", *df["dissipation_rate"])

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
        tasks = [(system, time_steps, start_cfg, i, data_dir, kwargs.get("step") or 1, desc) for i in
                 range(pre_run_num, sample_count + pre_run_num)]

        if run:
            with Pool(cpu_count() - 2) as pool:
                pool.starmap(_process, tasks)
        else:
            return [(_process, task) for task in tasks]


def _process(system: SandpileND, time_steps: int, start_cfg: Array, index: int, data_dir, step: int,
             desc: str
             ):
    np.random.seed(int.from_bytes(urandom(4), "big"))
    system = deepcopy(system)
    system(time_steps, start_cfg, desc=f"{desc} {index}", tqdm_position=index)
    system.save_separate((data_dir / f"data_{index}").absolute().__str__(), step)
    del system
    gc.collect()
