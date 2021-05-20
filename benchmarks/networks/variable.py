from typing import List

import gym
from torch import nn
from dataclasses import dataclass

from benchmarks.activation import ActivationFunction
from benchmarks.networks.benchmark import Benchmark


@dataclass
class LayerConfig:
    output_channels: int
    kernel_size: int
    stride: int
    padding: int
    activation: ActivationFunction

    def __len__(self):
        return 5

    def __getitem__(self, item: int):
        if item == 0:
            return self.output_channels
        elif item == 1:
            return self.kernel_size
        elif item == 2:
            return self.stride
        elif item == 3:
            return self.padding
        elif item == 4:
            return self.activation
        raise IndexError()

    def __hash__(self):
        return hash(tuple([self.output_channels, self.kernel_size, self.stride, self.padding, self.activation]))

    def encode(self):
        return str(f"{self.output_channels}:{self.kernel_size}:{self.stride}:{self.padding}:{self.activation}")


class VariableBenchmark(Benchmark):
    """
    Variable benchmark, where a set of hyper-parameters can be configured to vary a feature extractor at runtime
    Most of the internals are handled by the package Stable Baselines
    """

    def __init__(self, observation_space: gym.spaces.Box, layers: [LayerConfig]):
        self.layer_configs: List[LayerConfig] = layers
        super().__init__(observation_space)

    def setup_model(self) -> nn.Sequential:
        layers = []

        # init last_config so we can simplify our for loop to make use of it
        last_config = LayerConfig(self.input_channels, 0, 0, 0, "")
        for config in self.layer_configs:
            layers.append(
                nn.Conv2d(
                    last_config.output_channels,
                    config.output_channels,
                    kernel_size=config.kernel_size,
                    stride=config.stride,
                    padding=config.padding,
                )
            )
            layers.append(config.activation.function())

            # activation = config.activation.lower()
            # if activation == "gelu":
            #     layers.append(nn.GELU())
            # elif activation == "relu":
            #     layers.append(nn.ReLU())
            # elif activation == "celu":
            #     layers.append(nn.CELU())
            # else:
            #     raise ValueError(f"Layer activation must be one of [GELU, RELU, CELU], but got {activation}.")

            last_config = config

        layers.append(nn.Flatten())
        return nn.Sequential(*layers)
