import math
import random

from benchmarks.math_util import number_to_power
from benchmarks.activation import ActivationFunction
from benchmarks.networks import LayerConfig

from benchmarks.settings import *


def random_power_of_2(lower: int, upper: int) -> int:
    return 2 ** random.randint(lower, upper)


def random_activation():
    return ActivationFunction(random.randint(0, len(ActivationFunction) - 1))


def random_layer():
    layer_power = random.randint(LAYER_MIN_POWER, LAYER_MAX_POWER)
    layer_size = 2 ** layer_power

    kernel_power = random.randint(1, max(1, math.floor(number_to_power(layer_size * 0.25))))
    kernel_size = 2 ** kernel_power

    return LayerConfig(output_channels=layer_size, kernel_size=kernel_size, stride=1, padding=0, activation=random_activation())
