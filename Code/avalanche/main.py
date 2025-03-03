import numpy as np


# pythran export get_critical_points(uint8[:], uint8)
# pythran export get_critical_points(uint8[:,:], uint8)
# pythran export get_critical_points(uint8[:,:,:], uint8)
# pythran export get_critical_points(uint8[:,:,:,:], uint8)
# pythran export get_critical_points(uint8[:,:,:,:,:], uint8)
# pythran export get_critical_points(uint8[:,:,:,:,:,:], uint8)
# pythran export get_critical_points(uint8[:,:,:,:,:,:,:], uint8)
def get_critical_points(cfg, critical_slope):
    """
    Find all critical points in the system.

    :param critical_slope: Critical slope of the system.
    :param cfg: current system configuration.
    :return: Array of indices of critical points.
    """

    return np.asarray(np.nonzero(cfg > critical_slope)).swapaxes(0, 1).astype(np.uint8)



# pythran export to_flattened_index(uint8[:], uint8) -> uint32
def to_flattened_index(multiindex, grid):
    result = 0
    curr_pow = 0

    for i in multiindex[::-1]:
        result += i * grid ** curr_pow
        curr_pow += 1

    return result


# pythran export op_bound_system_relax(int8[:], uint8[:], uint8)
# noinspection SpellCheckingInspection
def op_bound_system_relax(cfg, position_index, grid_size) -> None:
    """
    Relax the system by using open boundary conditions. The 'left' borders are always closed.

    :param cfg: configuration to relax.
    :param position_index: position of the relaxation process.
    :param grid_size: linear size of the grid
    """
    dim = len(position_index)

    if np.any(position_index == 0):
        cfg[to_flattened_index(position_index, grid_size)] = 0
        return

    boundary_indices = np.nonzero(position_index == (grid_size - 1))[0]
    # return cfg[to_flattened_index(position_index, grid_size)]
    cfg[to_flattened_index(position_index, grid_size)] += -2 * dim + len(boundary_indices)

    for dimension, single_index in enumerate(position_index):
        shifted_position_index = position_index.copy()

        shifted_position_index[dimension] -= 1

        cfg[to_flattened_index(shifted_position_index, grid_size)] += 1

        if single_index < (grid_size - 1):
            shifted_position_index[dimension] += 2
            cfg[to_flattened_index(shifted_position_index, grid_size)] += 1

# def _cbound_check_criticality(self, cfg: Array, position_index: Array) -> None:
#     """
#     Relax the system by using open boundary conditions. The 'left' borders are always closed.
#
#     :param cfg: configuration to relax.
#     :param position_index: position of the relaxation process.
#     """
#     # print(position_index)
#     if np.any(position_index == 0) or np.any(position_index == (self.system.linear_grid_size - 1)):
#         # print("border!")
#         cfg[*position_index] = 0
#         return
#
#     cfg[*position_index] -= 2 * self.system.dimension
#
#     for dimension, single_index in enumerate(position_index):
#         shifted_position_index = deepcopy(position_index)
#
#         shifted_position_index[dimension] -= 1
#         cfg[*shifted_position_index] += 1
#
#         shifted_position_index[dimension] += 2
#         cfg[*shifted_position_index] += 1
