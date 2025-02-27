import ast
import re
import typing
from copy import deepcopy
from dataclasses import dataclass, field, InitVar

# test test test
# problem

from utils import *


if is_notebook():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

import numpy as np
from numpy.typing import NDArray


__all__ = ["Avalanche", "SandpileND", "get_critical_points", "check_create_avalanche"]

Array: type = NDArray[np.int8]
Index: type = typing.Sequence[int]


@dataclass(repr=True)
class Avalanche:
    # critical_slope: int
    system: "SandpileND"
    _starting_point: Index
    """The starting position of the avalanche"""
    start_cfg: InitVar[Array | None] = None
    """Starting configuration of the system to relax. If not given, do nothing."""
    termination_time: int = 500
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
    def starting_point(self) -> Index:
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
        return np.asarray(self._dissipation_rate)

    def _do_step(self, cfg: Array) -> bool:
        """
        Do a relaxation update of the system configuration and update the avalanche properties.

        :param cfg: current avalanche update.
        :return: True if cfg was in critical state, False if already relaxed.
        """

        # critical_points = np.asarray((cfg > self.critical_slope).nonzero()).swapaxes(0, 1)
        critical_points = get_critical_points(self.system.critical_slope, cfg)
        if len(critical_points) == 0:
            return False

        self._size += len(critical_points)
        self._time += 1
        max_distance = np.sqrt(((critical_points - self._starting_point) ** 2).sum(axis=1)).max()
        self._reach = max(max_distance, self._reach)

        self._dissipation_rate.append(len(critical_points))

        for critical_point in critical_points:
            self._obound_check_criticality(cfg, critical_point)

        return True

    def _obound_check_criticality(self, cfg: Array, position_index: Array) -> None:
        """
        Relax the system by using open boundary conditions. The 'left' borders are always closed.

        :param cfg: configuration to relax.
        :param position_index: position of the relaxation process.
        """
        if np.any(position_index == 0):
            cfg[*position_index] = 0

        boundary_indices = np.asarray(position_index == (self.system.linear_grid_size - 1)).sum()
        cfg[*position_index] += -2 * self.system.dimension + boundary_indices

        for dimension, single_index in enumerate(position_index):
            shifted_position_index = deepcopy(position_index)

            shifted_position_index[dimension] -= 1
            cfg[*shifted_position_index] += 1

            if single_index < (self.system.linear_grid_size - 1):
                shifted_position_index[dimension] += 2
                cfg[*shifted_position_index] += 1

    def _cbound_check_criticality(self, cfg: Array, position_index: Array) -> None:
        raise NotImplementedError()
        if np.any(position_index == 0) or np.any(position_index == (self.system.linear_grid_size - 1)):
            return

    def to_str(self) -> str:
        """Turn Avalanche data into a string"""

        s = ""
        s += f"{list(self._starting_point)}\n{self._size}\n{self._time}\n{self._reach}"

        for r in self.dissipation_rate:
            s += f"\n{r}"

        return s


def get_critical_points(critical_slope: int, cfg: Array) -> Array:
    """
    Find all critical points in the system.

    :param critical_slope: Critical slope of the system.
    :param cfg: current system configuration.
    :return: Array of indices of critical points.
    """

    return np.asarray((cfg > critical_slope).nonzero()).swapaxes(0, 1)


def check_create_avalanche(system: "SandpileND", start_cfg: Array) -> Avalanche | None:
    """

    Check if the system configuration is in a critical system

    :return: None if non-critical, return Avalanche object if otherwise.
    """

    critical_points = get_critical_points(system.critical_slope, start_cfg)

    if len(critical_points) == 0:
        return None
    elif len(critical_points) == 1:
        return Avalanche(system=system, _starting_point=critical_points[0], start_cfg=start_cfg)
    else:
        raise Exception("Configuration not the beginning of an avalanche")


@dataclass
class SandpileND:
    dimension: int
    linear_grid_size: int
    critical_slope: int
    start_cfg: Array | None = None
    _shape: tuple = field(init=False, repr=False)

    _curr_slope: Array = field(init=False, repr=False)
    average_slopes: Array = field(init=False, repr=False)
    _avalanches: list[Avalanche] = field(init=False, repr=False)

    @property
    def avalanches(self):
        return np.array(self._avalanches)

    @property
    def shape(self):
        return self._shape

    def __post_init__(self):
        self._shape = tuple([self.linear_grid_size] * self.dimension)
        # self._initialize_system(self.start_cfg)

    def _initialize_system(self, time_steps: int, start_cfg: Array | None = None) -> None:
        """Initialize the system for the simulation"""

        if start_cfg is None:
            start_cfg = np.zeros(shape=self._shape)

        if start_cfg.shape != self._shape:
            raise Exception("Shape mismatch")

        self._curr_slope = deepcopy(start_cfg)
        self.average_slopes = np.zeros(time_steps)
        self.average_slopes[0] = self._curr_slope.mean()
        self._avalanches = []

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

    def __call__(self, time_steps: int, start_cfg: Array | None = None) -> None:
        self._initialize_system(time_steps, start_cfg)

        random_positions = np.random.randint(low=0, high=self.linear_grid_size, size=(time_steps - 1, self.dimension))

        desc = f"dim {self.dimension} grid {self.linear_grid_size}"
        miniters = int(np.ceil(time_steps / 500))

        print("\r ", end="")
        for i, position_index, _ in zip(
                range(1, time_steps),
                random_positions,
                tqdm(range(1, time_steps), desc=desc, miniters=miniters, leave=True)):
            # self._slopes[i] = deepcopy(self._slopes[i - 1])

            avalanche = check_create_avalanche(self, self._curr_slope)
            if avalanche is not None:
                self._avalanches.append(avalanche)
            self._conservative_perturbation(self._curr_slope, position_index)

            self.average_slopes[i] = self._curr_slope.mean()

    def save_data(self, path: str) -> None:
        """Save the data into a file"""

        # System specifications
        s = f"dimension: {self.dimension}, linear_grid_size: {self.linear_grid_size}, "
        s += f"critical_slope: {self.critical_slope}\n{list(self.average_slopes)}\n"

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
        system = SandpileND(dimension=parameters[0], linear_grid_size=parameters[1], critical_slope=parameters[2])
        system.average_slopes = np.array(ast.literal_eval(lines[1]))

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

            dissipation_rate = [int(x) for x in d[4:]]

            a = Avalanche(system=system, _starting_point=starting_point)
            a._size = size
            a._time = time
            a._reach = reach

            a._dissipation_rate = dissipation_rate
            avalanches.append(a)

        system._avalanches = avalanches

        file.close()
        return system
