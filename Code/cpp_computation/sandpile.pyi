from __future__ import annotations
import numpy
import typing
__all__ = ['AvalancheData', 'Sandpile', 'cl_bound_system_relax', 'get_critical_points', 'op_bound_system_relax', 'ravel_index', 'relax_avalanche', 'unravel_index']
class AvalancheData:
    dissipation_rate: numpy.ndarray[numpy.uint16]
    reach: float
    size: int
    time: int
    time_step: int
    def __init__(self, arg0: int) -> None:
        ...
class Sandpile:
    average_slopes: list[float]
    closed_boundary: bool
    crit_slope: int
    dim: int
    grid: int
    @typing.overload
    def __init__(self, arg0: int, arg1: int, arg2: int) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: int, arg1: int, arg2: int, arg3: bool, arg4: bool) -> None:
        ...
    def simulate(self, arg0: int, arg1: numpy.ndarray[numpy.uint8] | None) -> None:
        ...
@typing.overload
def cl_bound_system_relax(arg0: numpy.ndarray[numpy.int8], arg1: numpy.ndarray[numpy.uint8], arg2: int) -> None:
    ...
@typing.overload
def cl_bound_system_relax(arg0: numpy.ndarray[numpy.int16], arg1: numpy.ndarray[numpy.uint8], arg2: int) -> None:
    ...
@typing.overload
def get_critical_points(arg0: numpy.ndarray[numpy.int8], arg1: ...) -> list[numpy.ndarray[numpy.uint8]]:
    ...
@typing.overload
def get_critical_points(arg0: numpy.ndarray[numpy.int16], arg1: ...) -> list[numpy.ndarray[numpy.uint8]]:
    ...
@typing.overload
def op_bound_system_relax(arg0: numpy.ndarray[numpy.int8], arg1: numpy.ndarray[numpy.uint8], arg2: int) -> None:
    ...
@typing.overload
def op_bound_system_relax(arg0: numpy.ndarray[numpy.int16], arg1: numpy.ndarray[numpy.uint8], arg2: int) -> None:
    ...
def ravel_index(arg0: numpy.ndarray[numpy.uint8], arg1: int) -> int:
    ...
@typing.overload
def relax_avalanche(arg0: int, arg1: numpy.ndarray[numpy.int8], arg2: numpy.ndarray[numpy.uint8], arg3: ...) -> AvalancheData:
    ...
@typing.overload
def relax_avalanche(arg0: int, arg1: numpy.ndarray[numpy.int16], arg2: numpy.ndarray[numpy.uint8], arg3: ...) -> AvalancheData:
    ...
def unravel_index(arg0: int, arg1: int, arg2: int) -> numpy.ndarray[numpy.uint8]:
    ...
