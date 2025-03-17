from __future__ import annotations
import numpy
__all__ = ['AvalancheData']
class AvalancheData:
    dissipation_rate: numpy.ndarray[numpy.uint16[m, 1]]
    reach: float
    size: int
    time: int
    time_step: int
    def __init__(self, arg0: int) -> None:
        ...
