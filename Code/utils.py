import inspect
import os
import sys
import typing

import numpy as np
import psutil  # type: ignore
from IPython import get_ipython
from numpy.typing import NDArray


def export(function):
    module_globals = inspect.stack()[1][0].f_globals
    if "__all__" not in module_globals:
        module_globals["__all__"] = []

    module_globals["__all__"].append(function.__name__)
    # print(function.__name__, module_globals["__name__"], module_globals["__all__"])

    return function


def get_memory() -> float:
    """Return memory usage in MB"""
    process = psutil.Process(os.getpid())
    memory_usage = process.memory_info().rss  # in bytes
    return memory_usage / (1024 ** 2)


def is_notebook() -> bool:
    try:
        # Check if the environment is an IPython shell (which includes Jupyter)
        if 'ipykernel' in sys.modules or 'IPython' in sys.modules:
            if get_ipython().__class__.__name__ == 'ZMQInteractiveShell':
                # We're in a Jupyter notebook
                return True
        return False
    except NameError:
        return False  # If 'IPython' is not available, we're not in a notebook


def get_hist(sample: typing.Sequence, bins: int | typing.Iterable | None = None, **kwargs) -> tuple[
    NDArray[np.float64], [np.float64]]:
    edges = np.array(range(
        np.floor(sample).min().astype(int),
        np.ceil(sample).max().astype(int) + 1)) - 0.5

    default = {
        "density": True,
    }
    default.update(kwargs)

    bins, edges = np.histogram(sample, bins=bins or edges, **default)

    return bins, 0.5 * (edges[1:] + edges[:-1])
