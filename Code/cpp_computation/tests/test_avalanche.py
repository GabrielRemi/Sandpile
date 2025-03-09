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
    system = SystemMeta(1, 4, 3, True)
    x = np.array([1, 4, 2, 4], dtype=np.int8)
    out = get_critical_points(x, system)
    res = [np.array([x], dtype=np.uint8) for x in [1, 3]]
    assert len(res) == len(out)
    assert np.all(out == res)


@pytest.mark.benchmark
def test_get_critical_points(benchmark: Any):
    benchmark(setup_get_critical_points)
