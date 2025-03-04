import numpy as np


# pythran export get_critical_points(int8[:], uint8, uint8, uint8) -> uint8[:] list
def get_critical_points(cfg, critical_slope, dim, grid_size):
    """
    Find all critical points in the system.

    :param critical_slope: Critical slope of the system.
    :param cfg: current system configuration.
    :param dim: dimension of the system.
    :param grid_size: grid size of the system.
    :return: Array of indices of critical points.
    """

    points = np.asarray(np.nonzero(cfg > critical_slope)).swapaxes(0, 1).astype(np.uint8)
    return [unravel_index(p[0], dim, grid_size) for p in points]


# pythran export ravel_index(uint8[:], uint8) -> uint32
def ravel_index(multiindex, grid):
    result = 0
    curr_pow = 0

    for i in multiindex[::-1]:
        result += i * grid ** curr_pow
        curr_pow += 1

    return result


# pythran export unravel_index(uint32, uint8, uint8) -> uint8[:]
def unravel_index(index, dim, grid_size):
    indices = [0] * dim
    for i in reversed(range(dim)):
        indices[i] = index % grid_size
        index = np.floor(index / grid_size).astype(int)

    return np.array(indices)


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
        cfg[ravel_index(position_index, grid_size)] = 0
        return

    boundary_indices = np.nonzero(position_index == (grid_size - 1))[0]
    # return cfg[to_flattened_index(position_index, grid_size)]
    cfg[ravel_index(position_index, grid_size)] += -2 * dim + len(boundary_indices)

    for dimension, single_index in enumerate(position_index):
        shifted_position_index = position_index.copy()

        shifted_position_index[dimension] -= 1

        cfg[ravel_index(shifted_position_index, grid_size)] += 1

        if single_index < (grid_size - 1):
            shifted_position_index[dimension] += 2
            cfg[ravel_index(shifted_position_index, grid_size)] += 1


# pythran export cl_bound_system_relax(int8[:], uint8[:], uint8)
# noinspection SpellCheckingInspection
def cl_bound_system_relax(cfg, position_index, grid_size) -> None:
    """
    Relax the system by using open boundary conditions. The 'left' borders are always closed.

    :param cfg: configuration to relax.
    :param position_index: position of the relaxation process.
    :param grid_size: linear size of the grid
    """
    if np.any(position_index == 0) or np.any(position_index == (grid_size - 1)):
        cfg[ravel_index(position_index, grid_size)] = 0
        return

    dim = len(position_index)
    cfg[ravel_index(position_index, grid_size)] -= 2 * dim

    for dimension, single_index in enumerate(position_index):
        shifted_position_index = position_index.copy()

        shifted_position_index[dimension] -= 1
        cfg[ravel_index(shifted_position_index, grid_size)] += 1

        shifted_position_index[dimension] += 2
        cfg[ravel_index(shifted_position_index, grid_size)] += 1


# pythran export relax_avalanche(uint64, int8[:], uint8[:], (uint64, uint8, uint8, bool))
def relax_avalanche(time_step, start_cfg, start_point, system):
    """

    :param time_step:
    :param start_cfg:
    :param start_point:
    :param system: (dim, grid, critical_slope, closed)
    :return:
    """
    dim, grid, critical_slope, closed = system
    dissipation_rate = []
    size, time, reach = 0, 0, 0

    max_step = 5000
    i = 0
    if closed:
        relax = cl_bound_system_relax
    else:
        relax = op_bound_system_relax

    for i in range(max_step):
        critical_points = get_critical_points(start_cfg, critical_slope, dim, grid)
        if len(critical_points) == 0:
            break

        size += len(critical_points)
        time += 1
        dissipation_rate.append(len(critical_points))

        max_distance = np.max([np.sqrt(((_p - start_point) ** 2).sum()) for _p in critical_points])
        reach = max(max_distance, reach)

        # np.random.shuffle(critical_points)
        for critical_point in critical_points:
            relax(start_cfg, critical_point, grid)

    if i == (max_step - 1):
        raise Exception("Max number of step iterations reached.")

    return (time_step, size, time, reach), np.array(dissipation_rate, dtype=np.uint8)
