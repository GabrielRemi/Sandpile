from __future__ import annotations
import numpy
import typing
__all__ = ['Sandpile16Bit', 'Sandpile8Bit', 'sandpile_simulate_single', 'sandpile_simulate_worker']
class Sandpile16Bit:
    crit_slope: int
    dim: int
    grid: int
    def __init__(self, dim: int, grid: int, crit_slope: int, has_open_boundary: bool = True, has_conservative_perturbation: bool = True) -> None:
        """
        A Class for simulating Sandpiles. Can be used with any dimension and grid size, as long
        as the number of points inside the grid is small enough for to work.
        
        :param dim: Dimension of the system
        :param grid: Grid size per dimension
        :param crit_slope: Critical Slope. If the slope value on lattice point is above this value, the system relaxes.
        :param has_open_boundary: Specify if the system uses open boundaries for relaxation.
        :param has_conservative_perturbation: Specify if the system uses conservative perturbations for relaxations.
        """
    def generate_total_dissipation_rate(self, time_steps: int, seed: int | None = None) -> numpy.ndarray[numpy.uint32[m, 1]]:
        """
        Generate the total dissipation rate by randomly placing the individual dissipation rates on
        a grid.
        
        :param time_steps: defines the time scale of the total dissipation rate.
        :param seed: Uses this seed for random number generation.
        
        :return: total dissipation rate
        """
    def get_average_slopes(self) -> numpy.ndarray[numpy.float64[m, 1]]:
        """
        Array of average slopes calculated during the last simulation.
        """
    def get_has_conservative_perturbation(self) -> bool:
        ...
    def get_has_open_boundary(self) -> bool:
        ...
    def get_reach(self) -> numpy.ndarray[numpy.float64[m, 1]]:
        ...
    def get_size(self) -> numpy.ndarray[numpy.uint32[m, 1]]:
        ...
    def get_time(self) -> numpy.ndarray[numpy.uint32[m, 1]]:
        ...
    def initialize_system(self, time_steps: int, start_cfg: numpy.ndarray[numpy.int16[m, 1]] | None = None, seed: int | None = None) -> None:
        """
        Initialize the system to start a simulation. You need to run this function every time
        you start a manual simulation with the `step()` function.
        
        :param time_steps: The number of time steps to simulate.
        :param start_cfg: Initial configuration of the system.
        :param seed: Random seed for reproducibility.
        """
    def simulate(self, time_steps: int, start_cfg: numpy.ndarray[numpy.int16[m, 1]] | None = None, seed: int | None = None) -> None:
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
    def step(self, perturb_position: numpy.ndarray[numpy.uint8[m, 1]] | None = None) -> None:
        """
        Make one perturbation step of the system.
        
        :param perturb_position: The position of the perturbation. If not specified, perturb the system at random.
        """
    @property
    def time_cut_off(self) -> int:
        """
        All avalanche data registered before this time step will be ignored
        """
    @time_cut_off.setter
    def time_cut_off(self, arg0: int) -> None:
        ...
class Sandpile8Bit:
    crit_slope: int
    dim: int
    grid: int
    def __init__(self, dim: int, grid: int, crit_slope: int, has_open_boundary: bool = True, has_conservative_perturbation: bool = True) -> None:
        """
        A Class for simulating Sandpiles. Can be used with any dimension and grid size, as long
        as the number of points inside the grid is small enough for to work.
        
        :param dim: Dimension of the system
        :param grid: Grid size per dimension
        :param crit_slope: Critical Slope. If the slope value on lattice point is above this value, the system relaxes.
        :param has_open_boundary: Specify if the system uses open boundaries for relaxation.
        :param has_conservative_perturbation: Specify if the system uses conservative perturbations for relaxations.
        """
    def generate_total_dissipation_rate(self, time_steps: int, seed: int | None = None) -> numpy.ndarray[numpy.uint32[m, 1]]:
        """
        Generate the total dissipation rate by randomly placing the individual dissipation rates on
        a grid.
        
        :param time_steps: defines the time scale of the total dissipation rate.
        :param seed: Uses this seed for random number generation.
        
        :return: total dissipation rate
        """
    def get_average_slopes(self) -> numpy.ndarray[numpy.float64[m, 1]]:
        """
        Array of average slopes calculated during the last simulation.
        """
    def get_has_conservative_perturbation(self) -> bool:
        ...
    def get_has_open_boundary(self) -> bool:
        ...
    def get_reach(self) -> numpy.ndarray[numpy.float64[m, 1]]:
        ...
    def get_size(self) -> numpy.ndarray[numpy.uint32[m, 1]]:
        ...
    def get_time(self) -> numpy.ndarray[numpy.uint32[m, 1]]:
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
    def step(self, perturb_position: numpy.ndarray[numpy.uint8[m, 1]] | None = None) -> None:
        """
        Make one perturbation step of the system.
        
        :param perturb_position: The position of the perturbation. If not specified, perturb the system at random.
        """
    @property
    def time_cut_off(self) -> int:
        """
        All avalanche data registered before this time step will be ignored
        """
    @time_cut_off.setter
    def time_cut_off(self, arg0: int) -> None:
        ...
@typing.overload
def sandpile_simulate_single(system: Sandpile8Bit, time_steps: int, tqdm_update_steps: int = 1000, position: int = 0) -> None:
    ...
@typing.overload
def sandpile_simulate_single(system: Sandpile16Bit, time_steps: int, tqdm_update_steps: int = 1000, position: int = 0) -> None:
    ...
@typing.overload
def sandpile_simulate_worker(system: Sandpile8Bit, shared_value: typing.Any, time_steps: int, tqdm_update_steps: int = 1000) -> None:
    ...
@typing.overload
def sandpile_simulate_worker(system: Sandpile16Bit, shared_value: typing.Any, time_steps: int, tqdm_update_steps: int = 1000) -> None:
    ...
