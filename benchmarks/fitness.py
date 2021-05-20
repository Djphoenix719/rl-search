from deap import base

from benchmarks.settings import *


class Fitness(base.Fitness):
    """
    Represents the fitness of our individual. Mostly this is used as a way of
    declaring the initial values within the paradigm permitted by the DEAP framework.
    """

    def __init__(self):
        self.weights = INIT_FITNESS
