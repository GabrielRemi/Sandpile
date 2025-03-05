from .avalanche import *
from .utils import get_total_dissipation_rate
from .utils import auto_correlation as ac


def auto_correlation(x, y, fast=True):
    return ac(x, y, fast)
