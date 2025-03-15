from ..avalanche import *
import numpy as np
import pytest
from typing import Any


def setup_ravel_index():
    # np.random.seed(0)
    for dim in range(1, 8):
        for grid in range(1, 201, 10):
            multi_index = np.random.randint(0, grid, size=dim, dtype=np.uint8)
            out = ravel_index(multi_index, grid)
            try:
                assert out == np.ravel_multi_index(tuple(multi_index.tolist()), [grid] * dim)
            except Exception as e:
                print("failed dim + grid = ", dim, grid)
                raise e


@pytest.mark.benchmark
def test_ravel_index(benchmark: Any):
    benchmark(setup_ravel_index)


def setup_unravel_index():
    for dim in range(1, 8):
        for grid in range(1, 201, 10):
            shape = [grid] * dim
            index = np.random.randint(0, grid**dim)
            out = unravel_index(index, dim, grid)
            assert np.all(out == np.unravel_index(index, shape=shape))


@pytest.mark.benchmark
def test_unravel_index(benchmark: Any):
    benchmark(setup_unravel_index)


def setup_get_critical_points():
    np.random.seed(0)
    crit_amount = 3
    c_slope = np.random.randint(1, 10)
    grid = 10
    for dim in [1, 2, 3, 4, 5, 6]:
        system = SystemMeta(dim, grid, c_slope, False)
        cfg = np.random.randint(0, c_slope, size=[grid] * dim, dtype=np.int8)
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


@pytest.mark.benchmark
def test_get_critical_points(benchmark: Any):
    benchmark(setup_get_critical_points)
