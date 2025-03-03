import ast
from copy import deepcopy
from dataclasses import dataclass, field, InitVar
from multiprocessing import Pool
from os import cpu_count, urandom

from utils import *


if is_notebook():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

import numpy as np
from numpy.typing import NDArray
import pandas as pd
import gc


__all__ = ["Avalanche", "SandpileND", "get_critical_points", "check_create_avalanche", "run_multiple_samples"]

# Array: type = NDArray[np.int8]
Array: type = NDArray[np.uint8] | NDArray[np.uint32]


@dataclass(repr=True)
class Avalanche:
    # critical_slope: int
    system: "SandpileND"
    _starting_point: NDArray[np.uint8]
    """The starting position of the avalanche."""
    time_step: int
    """Time step at which the avalanche occurred."""
    start_cfg: InitVar[NDArray[np.uint8] | None] = None
    """Starting configuration of the system to relax. If not given, do nothing."""
    termination_time: int = 2500
    """Avalanche time, after which the avalanche is brought to a halt"""
    _size: int = 0
    """Number of critical points integrated over all time steps of the avalanche."""
    _time: int = 0
    """Lifetime of the avalanche."""
    _reach: float = 0
    """The distance between the starting point and the most distant critical point of the avalanche."""

    _dissipation_rate: list[int] = field(repr=False, init=False, default_factory=list)

    def __post_init__(self, start_cfg):
        if start_cfg is None:
            return

        # Relax the system
        while self._do_step(start_cfg):
            # print(start_cfg)
            if self._time == self.termination_time:
                raise Exception(f"Avalanche was found in a loop after {self.termination_time} relaxations")

    @property
    def starting_point(self) -> NDArray[np.uint8]:
        return deepcopy(self._starting_point)

    @property
    def size(self) -> int:
        return self._size

    @property
    def time(self) -> int:
        return self._time

    @property
    def reach(self) -> float:
        return self._reach

    @property
    def dissipation_rate(self) -> Array:
        arr = self._dissipation_rate

        if np.vectorize(lambda n: np.can_cast(n, np.uint8))(arr).all():
            return np.asarray(arr, dtype=np.uint8)
        else:
            return np.asarray(arr, dtype=np.uint32)

    def _do_step(self, cfg: Array) -> bool:
        """
        Do a relaxation update of the system configuration and update the avalanche properties.

        :param cfg: current avalanche update.
        :return: True if cfg was in critical state, False if already relaxed.
        """

        critical_points = get_critical_points(self.system.critical_slope, cfg)
        if len(critical_points) == 0:
            return False

        self._size += len(critical_points)
        self._time += 1
        max_distance = np.sqrt(((critical_points - self._starting_point) ** 2).sum(axis=1)).max()
        self._reach = max(max_distance, self._reach)

        self._dissipation_rate.append(len(critical_points))

        return False  # TODO
        for critical_point in np.random.permutation(critical_points):
            if self.system.boundary_condition == "open":
                self._obound_check_criticality(cfg, critical_point)
            elif self.system.boundary_condition == "closed":
                self._cbound_check_criticality(cfg, critical_point)
            else:
                raise ValueError("unknown boundary condition type")

        return True

    # noinspection SpellCheckingInspection
    def _obound_check_criticality(self, cfg: Array, position_index: Array) -> None:
        """
        Relax the system by using open boundary conditions. The 'left' borders are always closed.

        :param cfg: configuration to relax.
        :param position_index: position of the relaxation process.
        """
        if np.any(position_index == 0):
            cfg[*position_index] = 0
            return

        boundary_indices = np.asarray(position_index == (self.system.linear_grid_size - 1)).sum()
        cfg[*position_index] += -2 * self.system.dimension + boundary_indices

        for dimension, single_index in enumerate(position_index):
            single_index: NDArray[np.uint32]
            shifted_position_index = deepcopy(position_index)

            shifted_position_index[dimension] -= 1

            cfg[*shifted_position_index] += 1

            if single_index < (self.system.linear_grid_size - 1):
                shifted_position_index[dimension] += 2
                cfg[*shifted_position_index] += 1

    def _cbound_check_criticality(self, cfg: Array, position_index: Array) -> None:
        """
        Relax the system by using open boundary conditions. The 'left' borders are always closed.

        :param cfg: configuration to relax.
        :param position_index: position of the relaxation process.
        """
        # print(position_index)
        if np.any(position_index == 0) or np.any(position_index == (self.system.linear_grid_size - 1)):
            # print("border!")
            cfg[*position_index] = 0
            return

        cfg[*position_index] -= 2 * self.system.dimension

        for dimension, single_index in enumerate(position_index):
            shifted_position_index = deepcopy(position_index)

            shifted_position_index[dimension] -= 1
            cfg[*shifted_position_index] += 1

            shifted_position_index[dimension] += 2
            cfg[*shifted_position_index] += 1

    def to_str(self) -> str:
        """Turn Avalanche data into a string"""

        s = ""
        s += f"{self._starting_point.tolist()}\n{self._size}\n{self._time}\n{self._reach}"
        s += f"\n{self.time_step}"

        for r in self.dissipation_rate:
            s += f"\n{r}"

        return s


def get_critical_points(critical_slope: int, cfg: Array) -> NDArray[np.uint8]:
    """
    Find all critical points in the system.

    :param critical_slope: Critical slope of the system.
    :param cfg: current system configuration.
    :return: Array of indices of critical points.
    """

    return np.asarray((cfg > critical_slope).nonzero()).swapaxes(0, 1).astype(np.uint8)


def check_create_avalanche(system: "SandpileND", start_cfg: Array) -> Avalanche | None:
    """

    Check if the system configuration is in a critical system

    :return: None if non-critical, return Avalanche object if otherwise.
    """

    critical_points = get_critical_points(system.critical_slope, start_cfg)

    if len(critical_points) == 0:
        return None
    elif len(critical_points) == 1:
        return Avalanche(system=system, _starting_point=critical_points[0], start_cfg=start_cfg,
                         time_step=len(system.average_slopes_list))
    else:
        raise Exception("Configuration not the beginning of an avalanche")


@dataclass
class SandpileND:
    dimension: int
    linear_grid_size: int
    critical_slope: int
    boundary_condition: typing.Literal["open", "closed"] = "open"
    perturbation: typing.Literal["conservative", "non conservative"] = "conservative"
    start_cfg: Array | None = None
    _shape: tuple = field(init=False, repr=False)

    _curr_slope: Array = field(init=False, repr=False)
    average_slopes_list: list[float] = field(init=False, repr=False)
    average_slopes: Array = field(init=False, repr=False)
    _avalanches: dict[str, int | float | Array] = field(init=False, repr=False, default_factory=dict)

    @property
    def avalanches(self) -> NDArray[Avalanche]:
        return np.array(self._avalanches)

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

    def _append_avalanche_data_to_dict(self, av: Avalanche) -> None:
        self._avalanches["time_step"].append(av.time_step)
        self._avalanches["size"].append(av.size)
        self._avalanches["time"].append(av.time)
        self._avalanches["reach"].append(av.reach)
        self._avalanches["dissipation_rate"].append(av.dissipation_rate)

    def _initialize_system(self, start_cfg: Array | None = None) -> None:
        """Initialize the system for the simulation"""

        if start_cfg is None:
            start_cfg = np.zeros(shape=self._shape)

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
            perturb_position = np.random.randint(low=0, high=self.linear_grid_size, size=self.dimension)

        avalanche = check_create_avalanche(self, self._curr_slope)
        if avalanche is not None:
            # self._avalanches.append(avalanche)
            self._append_avalanche_data_to_dict(avalanche)

        if self.perturbation == "conservative":
            self._conservative_perturbation(self._curr_slope, perturb_position)
        elif self.perturbation == "non conservative":
            self._non_conservative_perturbation(self._curr_slope, perturb_position)
        else:
            raise ValueError(f"unknown perturbation type: {self.perturbation}")

        self.average_slopes_list.append(self._curr_slope.mean())

    def __call__(self, time_steps: int, start_cfg: Array | None = None, desc: str | None = None,
                 tqdm_position: int = 1
                 ) -> None:
        self._initialize_system(start_cfg)

        random_positions = np.random.randint(low=0, high=self.linear_grid_size, size=(time_steps - 1, self.dimension))

        if desc is None:
            desc = f"dim{self.dimension} grid{self.linear_grid_size} {self.boundary_condition} {self.perturbation}"
        min_iters = int(np.ceil(time_steps / 500))
        min_iters = min(min_iters, 100)

        if is_notebook():
            # print("\r", end=" ", flush=True)
            tqdm.write("", end=" ")
        for position_index, _ in zip(
                random_positions,
                tqdm(range(1, time_steps), desc=desc, miniters=min_iters, leave=True, position=tqdm_position)):
            self.step()

        self.average_slopes = np.asarray(self.average_slopes_list)

    def save_data(self, path: str) -> None:
        """Save the data into a file"""

        # System specifications
        s = f"dimension: {self.dimension} linear_grid_size: {self.linear_grid_size} "
        s += f"critical_slope: {self.critical_slope} "
        s += f"boundary: {self.boundary_condition} "
        s += f"perturbation: {self.perturbation}"
        s += f"\n{self.average_slopes.tolist()}\n"

        for a in self.avalanches:
            s += a.to_str()
            s += "\n---\n"

        with open(path, "w") as file:
            file.write(s)

    @classmethod
    def load_from_file(cls, path: str) -> "SandpileND":
        file = open(path, "r")

        lines = file.readlines()

        parameters = [int(x) for x in re.findall(r"\d+", lines[0])]
        system = SandpileND(
            dimension=parameters[0],
            linear_grid_size=parameters[1],
            critical_slope=parameters[2],
        )
        if "closed" in lines[0]:
            system.boundary_condition = "closed"
        if "non conservative" in lines[0]:
            system.perturbation = "non conservative"

        system.average_slopes = np.array(ast.literal_eval(lines[1]))
        system.average_slopes_list = system.average_slopes.tolist()

        aval_data = []
        curr_data = []

        for line in lines[2:]:
            if line == "---\n":
                aval_data.append(curr_data)
                curr_data = []
                continue

            curr_data.append(line.strip())

        avalanches: list[Avalanche] = []

        for d in aval_data:
            # starting_point = np.fromstring(d[0].strip("[]"), sep=" ", dtype=np.uint8)
            starting_point = np.array(ast.literal_eval(d[0]))
            size = int(d[1])
            time = int(d[2])
            reach = float(d[3])
            time_step = int(d[4])

            dissipation_rate = [int(x) for x in d[5:]]

            a = Avalanche(system=system, _starting_point=starting_point, time_step=time_step)
            a._size = size
            a._time = time
            a._reach = reach

            a._dissipation_rate = dissipation_rate
            avalanches.append(a)

        system._avalanches = avalanches

        file.close()
        return system

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


def run_multiple_samples(system: SandpileND, folder_path: str, time_steps: int, sample_count: int,
                         start_cfg: Array | None = None, run: bool = True, desc: str = "sample", **kwargs
                         ) -> None | list[any]:
    """
    Run multiple samples for that system in parallel so that a stastical average of the data
    can be calculated

    :param system: System to simulate
    :param folder_path: Folder where to save generated data
    :param time_steps: number of perturbation updates
    :param sample_count: number of samples to simulate
    :param desc: Description in the progress bar
    :param start_cfg: Starting configuration of the system
    :param run: If yes, run the simulations. If not, return a list of the processes
    """
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
