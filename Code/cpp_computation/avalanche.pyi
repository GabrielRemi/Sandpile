from __future__ import annotations
import numpy
import typing
__all__ = ['SystemMeta', 'get_critical_points', 'ravel_index']
class SystemMeta:
    closed_bounday: bool
    crit_slope: int
    dim: int
    grid: int
    def __init__(self, arg0: int, arg1: int, arg2: int, arg3: bool) -> None:
        ...
@typing.overload
def get_critical_points(arg0: numpy.ndarray[numpy.int8], arg1: SystemMeta) -> list[numpy.ndarray[numpy.uint8]]:
    ...
@typing.overload
def get_critical_points(arg0: numpy.ndarray[numpy.int16], arg1: SystemMeta) -> list[numpy.ndarray[numpy.uint8]]:
    ...
def ravel_index(arg0: numpy.ndarray[numpy.uint8], arg1: int) -> int:
    ...
