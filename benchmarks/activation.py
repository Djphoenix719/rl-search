import torch.nn as nn
from enum import Enum


class ActivationFunction(Enum):
    """
    Enum representing an activation function.
    """

    RELU = 0
    GELU = 1
    CELU = 2

    def function(self):
        if self.value == 0:
            return nn.ReLU()
        elif self.value == 1:
            return nn.GELU()
        elif self.value == 2:
            return nn.CELU()
        else:
            raise ValueError("Invalid activation function type.")

    def __str__(self):
        return str(self.function())

    def __repr__(self):
        return repr(self.function())

    def __hash__(self):
        return hash(self.value)

    def __eq__(self, other):
        if not (isinstance(other, Enum)):
            return False

        return self.value == other.value
