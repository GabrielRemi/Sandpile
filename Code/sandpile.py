import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass, field
import typing

from copy import deepcopy


array: type = NDArray[np.int8]


@dataclass
class SandpileND:
    dimension: int
    linear_grid_size: int
    start_cfg: array | None = None
    _shape: tuple = field(init=False, repr=False)

    _slopes: NDArray[array] = field(init=False, repr=False)
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

    def _initialize_system(self, time_steps: int, start_cfg: array | None = None) -> None:
        """Initialize the system for the simulation"""

        if start_cfg is None:
            start_cfg = np.zeros(shape=self._shape)

        if start_cfg.shape != self._shape:
            raise Exception("Shape mismatch")

        self._slopes = np.zeros(shape=(time_steps, *self._shape))
        self._slopes[0] = start_cfg
        self._avalanches = []

    def conservative_perturbation(self, cfg: array, position_index: typing.Sequence[int]):
        if len(position_index) != self.dimension:
            Exception("position index dimension mismatch")

        cfg[*position_index] += self.dimension
        for dimension, index in enumerate(position_index):
            if index == 0:
                continue

            shifted_position_index = deepcopy(position_index)
            shifted_position_index[dimension] -= 1

            cfg[*shifted_position_index] -= 1

    def __call__(self, time_steps: int, start_cfg: array | None = None) -> None:
        self._initialize_system(time_steps, start_cfg)

        random_positions = np.random.randint(low=0, high=self.linear_grid_size, size=(time_steps - 1, self.dimension))

        for i, position_index in zip(range(1, time_steps), random_positions):
            self._slopes[i] = deepcopy(self._slopes[i - 1])
            self.conservative_perturbation(self._slopes[i], position_index)


if __name__ == "__main__":
    np.random.seed(3)
    system = SandpileND(1, 3)
    system(4)
    print(system.slopes)
