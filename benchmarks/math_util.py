import math
from typing import Union

from benchmarks.settings import *


def number_to_power(value: Union[int, float], base: Union[int, float] = 2):
    """
    Finds the correct power of two, e.g. solves `value = base^n` for n
    :param value:
    :return:
    """
    return int(math.log(value) / math.log(base))


def weighted_time(x: float) -> float:
    """
    Returns an inverse scaling factor, where more time taken equals a lower factor
    :param x:
    :return:
    """
    percent = (MAX_TIME - x) / TIME_RANGE
    log = math.log(percent) if percent != 0 else 0
    value = log * SCORE_RANGE * TIME_WEIGHT
    return value
