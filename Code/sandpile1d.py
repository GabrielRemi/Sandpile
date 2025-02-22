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
        """The number of slopes in the grid."""

        self.critical_slope: int = critical_slope
        """The critical slope of the system. No slope on the grid can be bigger than this value."""

        self._slopes: list[NDArray[np.int8]] = [starting_cfg or np.zeros(self.size)]
        """The time evolution of the system of the last simulation as a list."""

        self._avalanches: list[NDArray[NDArray[np.int8]]] = []
        """The list of avalanches that occured during the last simulation"""

    # make them somehow immutable
    @property
    def slopes(self):
        return tuple(self._slopes)

    @property
    def avalanches(self):
        return tuple(self._avalanches)

    def step(self) -> None:
        """Check for criticality and add 1 grain of sand at a random position."""
        """ 
        TODO Check criticality and update the system. Save any occured avalanche in 
        the self.__avalanches variable.
        """

        index = np.random.randint(low=0, high=self.size)
        new_cfg = deepcopy(self._slopes[-1])

        new_cfg[index] += 1

        if index != 0:
            new_cfg[index - 1] -= 1

        self.check_criticality(new_cfg, index)
        self._slopes.append(new_cfg)

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

    def get_average_slopes(self) -> NDArray[np.float64]:
        """
        Calculate the average slope of the system of the last simulation for every time step
        """

        return np.mean(self._slopes, axis=1)

    def __call__(self, time_steps: int, starting_cfg: NDArray[np.int8] | None = None) -> None:
        """
        Do a simulation of the system for a number of time steps.
        :param time_steps: Number of time steps. In one step, criticality is checked and a grain added randomly.
        """

        self._slopes = [starting_cfg or np.zeros(self.size)]
        self._avalanches = []

        for _ in range(time_steps):
            self.step()

# np.random.seed(2)
# system = Sandpile1D(5, 2)
# for i in range(100):
#     system.step()
#
# for slope in system.slopes:
#     print(slope, system.get_height_from_slope(slope))
