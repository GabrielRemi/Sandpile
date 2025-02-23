import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass, field, InitVar
import typing

from copy import deepcopy


__all__ = ["Avalanche", "SandpileND", "get_critical_points", "check_create_avalanche"]

Array: type = NDArray[np.int8]
Index: type = typing.Sequence[int]


@dataclass(repr=True)
class Avalanche:
    # critical_slope: int
    system: "SandpileND"
    start_cfg: InitVar[Array]
    _starting_point: Index
    """The starting position of the avalanche"""
    _size: int = 0
    """Number of critical points integrated over all time steps of the avalanche."""
    _time: int = 0
    """Lifetime of the avalanche."""
    _reach: float = 0
    """The distance between the starting point and the most distant critical point of the avalanche."""

    _dissipation_rate: list[int] = field(repr=False, init=False, default_factory=list)

    def __post_init__(self, start_cfg):
        # Relax the system
        while self._do_step(start_cfg):
            continue

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
        if np.any(position_index == 0):
            return

        boundary_indices = np.asarray(position_index == (self.system.linear_grid_size - 1)).sum()
        cfg[*position_index] += -2 * self.system.dimension - boundary_indices

        for dimension, single_index in enumerate(position_index):
            shifted_position_index = deepcopy(position_index)

            shifted_position_index[dimension] -= 1
            cfg[*shifted_position_index] += 1

            if single_index < (self.system.linear_grid_size - 1):
                shifted_position_index[dimension] += 2
                cfg[*shifted_position_index] += 1


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

    _slopes: NDArray[Array] = field(init=False, repr=False)
    _avalanches: list[tuple[int, int, int]] = field(init=False, repr=False)

    @property
    def slopes(self):
        return self._slopes

    @property
    def average_slopes(self) -> NDArray[np.float64]:
        return self.slopes.mean(axis=1)

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

        self._slopes = np.zeros(shape=(time_steps, *self._shape))
        self._slopes[0] = start_cfg
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

    def _crit_update(self, cfg: Array) -> None:
        """
        Check for criticality and relax the system if necessary. Save the avalanche
        in self._avalanches.
        """

        avalanche = check_create_avalanche(self.critical_slope, cfg)
        if avalanche is None:
            return

        # self._obound_avalanche_update(cfg)
        # while ()

    def __call__(self, time_steps: int, start_cfg: Array | None = None) -> None:
        self._initialize_system(time_steps, start_cfg)

        random_positions = np.random.randint(low=0, high=self.linear_grid_size, size=(time_steps - 1, self.dimension))

        for i, position_index in zip(range(1, time_steps), random_positions):
            self._slopes[i] = deepcopy(self._slopes[i - 1])
            self._conservative_perturbation(self._slopes[i], position_index)


if __name__ == "__main__":
    # np.random.seed(3)
    # system = SandpileND(1, 3)
    # system(4)
    # print(system.slopes)
    system = SandpileND(1, 5, 3)
    start_cfg = np.array([1, 1, 1, 6, 1])
    a = Avalanche(system=system, start_cfg=start_cfg, _starting_point=np.array([3]))
    print(a)
    print(start_cfg)
