from __future__ import annotations
import numpy
import typing
__all__ = ['Sandpile']
class Sandpile:
    crit_slope: int
    dim: int
    grid: int
    @typing.overload
    def __init__(self, dim: int, grid: int, crit_slope: int) -> None:
        """
        A Class for simulating Sandpiles. Can be used with any dimension and grid size, as long
        as the number of points inside the grid is small enough for to work.
        
        :param dim: Dimension of the system
        :param grid: Grid size per dimension
        :param crit_slope: Critical Slope. If the slope value on lattice point is above this value, the system relaxes.
        """
    @typing.overload
    def __init__(self, dim: int, grid: int, crit_slope: int, has_open_boundary: bool, has_conservative_perturbation: bool) -> None:
        """
        A Class for simulating Sandpiles. Can be used with any dimension and grid size, as long
        as the number of points inside the grid is small enough for to work.
        
        :param dim: Dimension of the system
        :param grid: Grid size per dimension
        :param crit_slope: Critical Slope. If the slope value on lattice point is above this value, the system relaxes.
        :param has_open_boundary: Specify if the system uses open boundaries for relaxation.
        :param has_conservative_perturbation: Specify if the system uses conservative perturbations for relaxations.
        """
    def get_average_slopes(self) -> numpy.ndarray[numpy.float64[m, 1]]:
        """
        Array of average slopes calculated during the last simulation.
        """
    def get_has_conservative_perturbation(self) -> bool:
        ...
    def get_has_open_boundary(self) -> bool:
        ...
    def initialize_system(self, time_steps: int, start_cfg: numpy.ndarray[numpy.int8[m, 1]] | None = None, seed: int | None = None) -> None:
        """
        Initialize the system to start a simulation. You need to run this function every time
        you start a manual simulation with the `step()` function.
        
        :param time_steps: The number of time steps to simulate.
        :param start_cfg: Initial configuration of the system.
        :param seed: Random seed for reproducibility.
        """
    def simulate(self, time_steps: int, start_cfg: numpy.ndarray[numpy.int8[m, 1]] | None = None, seed: int | None = None) -> None:
        """
        Simulates the system over a given number of time steps.
        
        :param time_steps: The number of time steps to simulate.
        :param start_cfg: Initial configuration of the system.
        :param seed: Random seed for reproducibility.
        
        Example:
        ```python
        from cpp_computation import Sandpile
        system = Sandpile(2, 20, 7)
        system.simulate(100)
        print(system.average_slopes)
        """
