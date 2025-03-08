import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass

# @dataclass
class SystemMeta:
    """Meta"""

    dim: np.uint8
    grid: np.uint8
    crit_slope: np.uint8
    closed_boundary: bool
    
    def __init__(self, dim: int, grid: int, crit_slope: int, 
                 closed_boundary: bool)
    """Some"""
    
def ravel_index(multi_index: NDArray[np.uint8], grid: int) -> np.uint64
    """Ravel Index."""

def get_critical_points(cfg: NDArray, meta) -> list[NDArray[np.uint8]]:
    """
    Find all critical points in the system.

    :param critical_slope: Critical slope of the system.
    :param cfg: current system configuration.
    :param dim: dimension of the system.
    :param grid_size: grid size of the system.
    :return: Array of indices of critical points.
    """
