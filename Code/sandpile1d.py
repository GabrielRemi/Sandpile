from copy import deepcopy

import numpy as np
from numpy.typing import NDArray


class Sandpile1D:
    """

    For N+1 particles with heights h_i, we define the slopes between the particles
    s_i = h_i - h_(i+1) which results in N slopes.

    Adding one particle at position lattice point i (h_i -> h_i + 1) results in slope space
    s(i) -> s(i) + 1
    s(i-1) -> s(i-1) - 1

    If the slope s(i) > s_c, then a particles falls down, which yields
    s(i) -> s(i) - 2
    s(i +/- 1) -> s(i +/- 1) + 1

    """

    def __init__(self, size: int, critical_slope: int, starting_cfg: NDArray[np.int8] | None = None):
        self.size: int = size
        self.critical_slope: int = critical_slope
        self.slopes: list[NDArray[np.int8]] = [starting_cfg or np.zeros(self.size)]

    def step(self) -> None:
        """Add 1 grain of sand at a random position."""

        index = np.random.randint(low=0, high=self.size)
        new_cfg = deepcopy(self.slopes[-1])

        new_cfg[index] += 1

        if index != 0:
            new_cfg[index - 1] -= 1

        self.check_criticality(new_cfg, index)

        self.slopes.append(new_cfg)

    def check_criticality(self, slope: NDArray[np.int8], index):
        """Checks recursively for criticality."""
        # print("before", slope, index)
        if index < 0 or index >= self.size:
            return
        if slope[index] <= self.critical_slope:
            return

        if index == self.size - 1:
            slope[index] -= 1
            slope[index - 1] += 1
        elif index == 0:
            slope[index] -= 2
            slope[index + 1] += 1
        else:
            slope[index] -= 2
            slope[index + 1] += 1
            slope[index - 1] += 1

        self.check_criticality(slope, index + 1)
        self.check_criticality(slope, index - 1)
        # print("after", slope, index, end="\n\n")

    @classmethod
    def get_height_from_slope(cls, slope: NDArray[np.int8]) -> NDArray[np.int8]:
        n = len(slope)
        height = np.zeros(n + 1)
        for i in reversed(range(n)):
            height[i] = slope[i] + height[i + 1]

        return height


# np.random.seed(2)
# system = Sandpile1D(5, 2)
# for i in range(100):
#     system.step()
#
# for slope in system.slopes:
#     print(slope, system.get_height_from_slope(slope))
