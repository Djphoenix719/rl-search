from time import time
from stable_baselines3.common.callbacks import BaseCallback


class TimeLimitCallback(BaseCallback):
    max_time: int
    start_time: float

    def __init__(self, max_time: int):
        super().__init__()
        self.max_time = max_time

    def _init_callback(self) -> None:
        self.start_time = time()

    def _on_step(self) -> bool:
        return time() - self.start_time < self.max_time
