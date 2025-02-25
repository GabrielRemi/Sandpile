import os
import sys

import psutil  # type: ignore
from IPython import get_ipython


__all__ = ["get_memory", "is_notebook", "do","func"]


def get_memory() -> float:
    """Return memory usage in MB"""
    process = psutil.Process(os.getpid())
    memory_usage = process.memory_info().rss  # in bytes
    return memory_usage / (1024 ** 2)


# def is_notebook() -> bool:
#     """Return True if code is executed in a notebook environment"""
#     try:
#         return '__IPYTHON__' in globals()
#     except NameError:
#         return False


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


from multiprocessing import Process, Pool
import time

if is_notebook():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

def func(x):
    print(" ", end="")
    for _ in tqdm(range(100), desc=str(x), leave=True):
        time.sleep(0.03)


def do():
    with Pool(5) as p:
        p.map(func, range(3))
