import math
from typing import Union


def number_to_power(value: Union[int, float], base: Union[int, float] = 2):
    """
    Finds the correct power of two, e.g. solves `value = base^n` for n
    :param value:
    :return:
    """
    return int(math.log(value) / math.log(base))


def time_factor(taken: float) -> float:
    return math.log(60 * 60 * 24) - math.log(taken)
