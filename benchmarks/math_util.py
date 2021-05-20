import math
from typing import Union


def number_to_power(value: Union[int, float], base: Union[int, float] = 2):
    """
    Finds the correct power of two, e.g. solves `value = base^n` for n
    :param value:
    :return:
    """
    return int(math.log(value) / math.log(base))


def clamp(value: Union[int, float], min_: Union[int, float], max_: Union[int, float]) -> Union[int, float]:
    """
    Clamps a value between min_ and max_, such that it is coerced into the range of [min_, max_]
    :param value:
    :param min_:
    :param max_:
    :return:
    """
    return max(min_, min(max_, value))
