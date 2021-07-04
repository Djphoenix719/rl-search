from typing import List, Callable

from benchmarks.fitness import Fitness
from benchmarks.networks import LayerConfig


class Individual:
    """
    Represents an individual set of hyper-parameters to be turned into a model.
    This class mimics a list but is hashable to allow multithreading later.
    """

    output_size: int
    weights: List[LayerConfig]
    fitness: Fitness

    def __init__(self, output_size: Callable[[], int], weights: Callable[[], List[LayerConfig]]):
        # copy the weights rather than assigning ref
        self.weights = [value for value in weights()]
        self.fitness = Fitness()
        self.output_size = output_size()

    def __getitem__(self, item):
        return self.weights[item]

    def __setitem__(self, key, value):
        self.weights[key] = value

    def __len__(self):
        return len(self.weights)

    def __hash__(self):
        return hash(tuple(self.weights))

    def __str__(self):
        return str(self.weights)

    def __repr__(self):
        return repr(self.weights)

    def encode(self):
        return "&".join([config.encode() for config in self.weights]) + f"-{self.output_size}"
