from abc import ABC
from abc import abstractmethod

from stable_baselines3.common.policies import BaseFeaturesExtractor
from stable_baselines3.common.preprocessing import is_image_space

import gym
import torch
from torch import nn


class Benchmark(BaseFeaturesExtractor, ABC):
    @property
    def input_channels(self) -> int:
        """
        Get the number of input channels from the observation space.
        :return:
        """
        return self._observation_space.shape[0]

    @property
    def output_channels(self) -> int:
        """
        Dimensionality of resulting state space the model with output.
        :return:
        """
        return 512

    def __init__(self, observation_space: gym.spaces.Box):
        super(Benchmark, self).__init__(observation_space, self.output_channels)

        assert is_image_space(observation_space), "This feature extraction policy must be used with image spaces."

        self._setup()

    @abstractmethod
    def setup_model(self) -> nn.Sequential:
        """
        Setup the benchmark network.
        :return:
        """
        raise NotImplementedError("Class must implement setup_model method")

    def setup_linear(self, model_output_dimensions: int) -> nn.Sequential:
        """
        Create a linear layer for the model output.
        :param model_output_dimensions:
        :return:
        """
        return nn.Sequential(nn.Linear(model_output_dimensions, self.output_channels), nn.ReLU())

    def _setup(self):
        self.cnn = self.setup_model()

        # Compute shape by doing one forward pass
        with torch.no_grad():
            num_output = self.cnn(torch.as_tensor(self._observation_space.sample()[None]).float()).shape[1]

        self.linear = self.setup_linear(num_output)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Single forward pass of the model
        :param observations:
        :return:
        """
        return self.linear(self.cnn(observations))
