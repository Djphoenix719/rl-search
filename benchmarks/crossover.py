import math
import random
from typing import Tuple

from benchmarks.activation import ActivationFunction
from benchmarks.individual import Individual
from benchmarks.math_util import number_to_power
from benchmarks.networks import LayerConfig
from benchmarks.settings import LAYER_MAX_POWER
from benchmarks.settings import LAYER_MIN_POWER


def mutate(individual: Individual, probability: float = 0.5) -> Tuple[Individual]:
    """
    Mutate an individual, probabilistically changing it's weights to the next or previous power of two.
    :param individual: The individual to mutate.
    :param probability: The probability of changing a given weight.
    :return: A new mutated individual.
    """

    def mutate_power(value: int) -> int:
        if random.random() < probability:
            return value

        direction = random.choice([-1, 1])
        power = number_to_power(value) + direction

        # flip direction if we're up against the bounds
        if power < LAYER_MIN_POWER:
            power += 2
        if power > LAYER_MAX_POWER:
            power -= 2

        return 2 ** power

    def mutate_kernel(kernel_size: int, output_size: int):
        if random.random() < probability:
            return kernel_size

        max_power = math.floor(number_to_power(output_size * 0.25))

        direction = random.choice([-1, 1])
        power = number_to_power(kernel_size) + direction

        if power < 1:
            power = 1
        if power > max_power:
            power = max_power - 1

        return 2 ** power

    def mutate_activation(value: ActivationFunction) -> ActivationFunction:
        options = [ActivationFunction(index) for index in range(len(ActivationFunction))]
        options.remove(value)

        return random.choice(options)

    config: LayerConfig
    for (idx, config) in enumerate(individual):
        config.output_channels = mutate_power(config.output_channels)
        config.kernel_size = mutate_kernel(config.kernel_size, config.output_channels)
        config.activation = mutate_activation(config.activation)

    return (individual,)
