from torch import nn

from benchmarks.networks.benchmark import Benchmark


class NatureBaseline(Benchmark):
    @classmethod
    def name(self):
        return "Baseline_Nature_CNN"

    def setup_model(self) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(self.input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )


class NatureSmall(Benchmark):
    @classmethod
    def name(self):
        return "Small_Nature_CNN"

    def setup_model(self) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(self.input_channels, 16, kernel_size=4, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=2, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )


class NatureLarge(Benchmark):
    @classmethod
    def name(self):
        return "Large_Nature_CNN"

    def setup_model(self) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(self.input_channels, 64, kernel_size=16, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=2, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=1, stride=2, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )
