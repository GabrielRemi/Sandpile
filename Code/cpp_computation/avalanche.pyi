from __future__ import annotations
import numpy
import typing
__all__ = ['AvalancheData', 'SystemMeta', 'cl_bound_system_relax', 'get_critical_points', 'op_bound_system_relax', 'ravel_index', 'relax_avalanche', 'unravel_index']
class AvalancheData:
    dissipation_rate: numpy.ndarray[numpy.uint8]
    reach: float
    size: int
    time: int
    time_step: int
    def __init__(self, arg0: int) -> None:
        ...
class SystemMeta:
    closed_bounday: bool
    crit_slope: int
    dim: int
    grid: int
    def __init__(self, arg0: int, arg1: int, arg2: int, arg3: bool) -> None:
        ...
@typing.overload
def cl_bound_system_relax(arg0: numpy.ndarray[numpy.int8], arg1: numpy.ndarray[numpy.uint8], arg2: int) -> None:
    ...
@typing.overload
def cl_bound_system_relax(arg0: numpy.ndarray[numpy.int16], arg1: numpy.ndarray[numpy.uint8], arg2: int) -> None:
    ...
@typing.overload
def get_critical_points(arg0: numpy.ndarray[numpy.int8], arg1: SystemMeta) -> list[numpy.ndarray[numpy.uint8]]:
    ...
@typing.overload
def get_critical_points(arg0: numpy.ndarray[numpy.int16], arg1: SystemMeta) -> list[numpy.ndarray[numpy.uint8]]:
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
def relax_avalanche(arg0: int, arg1: numpy.ndarray[numpy.int8], arg2: numpy.ndarray[numpy.uint8], arg3: SystemMeta) -> AvalancheData:
    ...
@typing.overload
def relax_avalanche(arg0: int, arg1: numpy.ndarray[numpy.int16], arg2: numpy.ndarray[numpy.uint8], arg3: SystemMeta) -> AvalancheData:
    ...
def unravel_index(arg0: int, arg1: int, arg2: int) -> numpy.ndarray[numpy.uint8]:
    ...
