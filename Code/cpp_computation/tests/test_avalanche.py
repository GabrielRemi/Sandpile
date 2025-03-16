from ..sandpile import *
import numpy as np
import pytest
from typing import Any


def setup_ravel_index():
    np.random.seed(0)
    for dim in range(1, 8):
        for grid in range(1, 201, 10):
            multi_index = np.random.randint(0, grid, size=dim, dtype=np.uint8)
            out = ravel_index(multi_index, grid)
            try:
                assert out == np.ravel_multi_index(tuple(multi_index.tolist()), [grid] * dim)
            except Exception as e:
                print("failed dim + grid = ", dim, grid)
                raise e


def setup_unravel_index():
    for dim in range(1, 8):
        for grid in range(1, 201, 10):
            shape = [grid] * dim
            index = np.random.randint(0, grid**dim)
            out = unravel_index(index, dim, grid)
            assert np.all(out == np.unravel_index(index, shape=shape))


def setup_get_critical_points():
    np.random.seed(0)
    crit_amount = 6
    c_slope = 7
    grid = 10
    for dim in [1, 2, 3, 4, 5, 6]:
        system = Sandpile(dim, grid, c_slope, False, False)
        cfg = np.random.randint(-c_slope, c_slope, size=[grid] * dim, dtype=np.int8)
        # crit_amount = np.random.randint(0, grid**dim)
        critical_points = np.random.randint(0, grid, size=[crit_amount, dim], dtype=np.uint8)
        _, indices = np.unique(critical_points, axis=0, return_index=True)
        critical_points = critical_points[np.sort(indices)]

        for point in critical_points:
            cfg[tuple(point)] = c_slope + 1
        out = get_critical_points(cfg.reshape(-1), system)
        out = np.asarray(out)
        out = np.sort(out, axis=0)
        critical_points = np.sort(critical_points, axis=0)
        try:
            assert np.all(out == critical_points)
        except Exception as e:
            print(f"c_slope {c_slope}, dim {dim}, grid {grid}")
            print("cfg", cfg)
            print("critical points", critical_points)
            print("output", out)
            raise e


def __cl_bound_system_relax(cfg, position_index, grid_size) -> None:
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


def __op_bound_system_relax(cfg, position_index, grid_size) -> None:
    dim = len(position_index)

    if np.any(position_index == 0):
        cfg[ravel_index(position_index, grid_size)] = 0
        return

    boundary_indices = np.nonzero(position_index == (grid_size - 1))[0]
    cfg[ravel_index(position_index, grid_size)] += -2 * dim + len(boundary_indices)

    for dimension, single_index in enumerate(position_index):
        shifted_position_index = position_index.copy()

        shifted_position_index[dimension] -= 1

        cfg[ravel_index(shifted_position_index, grid_size)] += 1

        if single_index < (grid_size - 1):
            shifted_position_index[dimension] += 2
            cfg[ravel_index(shifted_position_index, grid_size)] += 1


def setup_cl_bound_system_relax():
    np.random.seed(0)

    c_slope = 3
    for grid in [10]:
        for dim in range(1, 7):
            cfg = np.random.randint(-c_slope, c_slope, size=[grid] * dim, dtype=np.int8)
            crit = np.random.randint(0, grid, size=dim, dtype=np.uint8)
            cfg[*crit] = c_slope + 1

            __cfg = cfg.copy()
            __cl_bound_system_relax(__cfg.reshape(-1), crit, grid)

            cl_bound_system_relax(cfg.reshape(-1), crit, grid)
            assert np.all(cfg == __cfg)


def setup_op_bound_system_relax():
    np.random.seed(0)

    c_slope = 3
    for grid in [10]:
        for dim in range(1, 7):
            cfg = np.random.randint(-c_slope, c_slope, size=[grid] * dim, dtype=np.int8)
            crit = np.random.randint(0, grid, size=dim, dtype=np.uint8)
            cfg[*crit] = c_slope + 1

            __cfg = cfg.copy()
            __op_bound_system_relax(__cfg.reshape(-1), crit, grid)

            op_bound_system_relax(cfg.reshape(-1), crit, grid)
            assert np.all(cfg == __cfg)


def __get_critical_points(cfg, critical_slope, dim, grid_size):

    points = np.asarray(np.nonzero(cfg > critical_slope)).swapaxes(0, 1).astype(np.uint64)
    return [unravel_index(p[0], dim, grid_size) for p in points]


def __relax_avalanche(time_step, start_cfg, start_point, system):
    dim, grid, critical_slope, closed = system
    dissipation_rate = []
    size, time, reach = 0, 0, 0.0

    max_step = 5_000
    i = 0
    if closed:
        relax = __cl_bound_system_relax
    else:
        relax = __op_bound_system_relax

    for i in range(max_step):
        critical_points = __get_critical_points(start_cfg, critical_slope, dim, grid)
        if len(critical_points) == 0:
            break

        size += len(critical_points)
        time += 1
        dissipation_rate.append(len(critical_points))

        # max_distance = np.max([np.sqrt(((_p - start_point) ** 2).sum()) for _p in critical_points])
        max_distance = np.max(
            [np.sqrt(((_p.astype(np.float64) - start_point.astype(np.float64)) ** 2).sum()) for _p in critical_points]
        )
        reach = max(max_distance, reach)

        np.random.shuffle(critical_points)
        for critical_point in critical_points:
            relax(start_cfg, critical_point, grid)

    if i == (max_step - 1):
        raise Exception("Max number of step iterations reached.")

    return (time_step, size, time, reach), np.array(dissipation_rate, dtype=np.uint16)


def setup_relax_avalanche():
    np.random.seed(0)
    c_slope = 7
    grid = 10
    b = True
    for b in [True, False]:
        for dim in range(1, 7):
            system = Sandpile(dim, grid, c_slope, b, True)
            cfg = np.random.randint(0, c_slope + 1, size=[grid] * dim, dtype=np.int8)
            crit = np.random.randint(0, grid, size=dim, dtype=np.uint8)
            cfg[*crit] = c_slope + 1

            cfg1 = cfg.copy()
            out1, dis1 = __relax_avalanche(0, cfg1.reshape(-1), crit, (dim, grid, c_slope, b))
            cfg2 = cfg.copy()
            out2 = relax_avalanche(0, cfg2.reshape(-1), crit, system)
            out2, dis2 = (out2.time_step, out2.size, out2.time, out2.reach), out2.dissipation_rate

            try:
                for x, y in zip(out1, out2):
                    assert int(x * 100) == int(y * 100)

                assert np.all(dis1 == dis2)
                assert np.all(cfg1 == cfg2)
            except Exception as e:
                print("out1", out1)
                print("out2", out2)
                print("dis1", dis1)
                print("dis2", dis2)
                print("dis diff", dis1 - dis2)
                # print("cfg before", cfg)
                # print("cfg1", cfg1)
                # print("cfg2", cfg2)
                raise e


@pytest.mark.benchmark
def test_ravel_index(benchmark: Any):
    benchmark(setup_ravel_index)


@pytest.mark.benchmark
def test_unravel_index(benchmark: Any):
    benchmark(setup_unravel_index)


@pytest.mark.benchmark
def test_get_critical_points(benchmark: Any):
    benchmark(setup_get_critical_points)


@pytest.mark.benchmark
def test_cl_bound_system_relax(benchmark: Any):
    benchmark(setup_cl_bound_system_relax)


@pytest.mark.benchmark
def test_op_bound_system_relax(benchmark: Any):
    benchmark(setup_op_bound_system_relax)


@pytest.mark.benchmark
def test_relax_avalanche(benchmark: Any):
    benchmark(setup_relax_avalanche)
