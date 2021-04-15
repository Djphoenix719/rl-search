from typing import List

import gym
from torch import nn
from dataclasses import dataclass
from benchmarks.networks.benchmark import Benchmark


@dataclass
class LayerConfig:
    output_channels: int
    kernel_size: int
    stride: int
    padding: int
    activation: str


def benchmark_name(layers: List[LayerConfig]) -> str:
    names = map(lambda layer: f"{layer.output_channels}-{layer.kernel_size}-{layer.stride}-{layer.padding}-{layer.activation}", layers)
    return "Variable_Benchmark_" + "_".join(names)


class VariableBenchmark(Benchmark):
    def name(self):
        return benchmark_name(self.layer_configs)

    def __init__(self, observation_space: gym.spaces.Box, layers: [LayerConfig]):
        self.layer_configs: List[LayerConfig] = layers
        super().__init__(observation_space)

    def setup_model(self) -> nn.Sequential:
        layers = []

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

            activation = config.activation.lower()
            if activation == "gelu":
                layers.append(nn.GELU())
            elif activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "celu":
                layers.append(nn.CELU())
            else:
                raise ValueError(f"Layer activation must be one of [GELU, RELU, CELU], but got {activation}.")

            last_config = config

        layers.append(nn.Flatten())
        return nn.Sequential(*layers)
