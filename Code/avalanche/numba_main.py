import numpy as np
from numba import njit


@njit
def get_critical_points(cfg, critical_slope):
    """
    Find all critical points in the system.

    :param critical_slope: Critical slope of the system.
    :param cfg: current system configuration.
    :return: Array of indices of critical points.
    """

    # return np.asarray(np.nonzero(cfg > critical_slope)).swapaxes(0, 1).astype(np.uint8)

    return np.column_stack(np.nonzero(cfg > critical_slope)).astype(np.uint8)


@njit
def op_bound_system_relax(cfg, position_index) -> None:
    """
    Relax the system by using open boundary conditions. The 'left' borders are always closed.

    :param cfg: configuration to relax.
    :param position_index: position of the relaxation process.
    """
    # if np.any(position_index == 0):
    #     cfg[tuple(list(position_index))] = 0
    print(tuple([2, 3, 4]))
        # return
    return
    #
    # dimension = cfg.ndim
    # linear_grid_size = cfg.shape[0]
    #
    # boundary_indices = np.asarray(position_index == (linear_grid_size - 1)).sum()
    # cfg[tuple(position_index)] += -2 * dimension + boundary_indices
    #
    # for dimension, single_index in enumerate(position_index):
    #     shifted_position_index = position_index.copy()
    #
    #     shifted_position_index[dimension] -= 1
    #
    #     cfg[tuple(shifted_position_index)] += 1
    #
    #     if single_index < (linear_grid_size - 1):
    #         shifted_position_index[dimension] += 2
    #         cfg[tuple(shifted_position_index)] += 1
