import torch.nn as nn
from enum import Enum


class ActivationFunction(Enum):
    """
    Enum representing an activation function.
    """

    RELU = 1
    GELU = 2
    CELU = 3

    def function(self):
        if self.value == 1:
            return nn.ReLU()
        elif self.value == 2:
            return nn.GELU()
        elif self.value == 3:
            return nn.CELU()
        else:
            raise ValueError("Invalid activation function type.")

    def __str__(self):
        return str(self.function())

    def __repr__(self):
        return repr(self.function())
