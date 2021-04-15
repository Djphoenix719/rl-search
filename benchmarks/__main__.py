import os
import random

import numpy as np
import torch
from torch import ByteTensor

from benchmarks.networks import Benchmark
from benchmarks.networks import VariableBenchmark
from benchmarks.networks import benchmark_name
from benchmarks.networks import LayerConfig
from benchmarks.networks import NatureOneLayerSmallKernel
from benchmarks.networks import NatureOneLayerLarge
from benchmarks.networks import NatureBaselineCELU
from benchmarks.networks import NatureBaselineGELU

from stable_baselines3.common.utils import get_device
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.ppo import PPO
from stable_baselines3.ppo import CnnPolicy
from stable_baselines3.common.callbacks import CheckpointCallback

from deap import base
from deap import creator
from deap import tools

from subprocess import Popen
from webbrowser import open_new_tab

RNG_SEED = 42  # Seed for pytorch
VERBOSE = 2  # 0 no output, 1 info, 2 debug
BASE_CHECKPOINT_PATH = "checkpoints/"  # path to save checkpoints to
BASE_LOG_PATH = "logs/"  # path to save tensorboard logs to
ENV_NAME = "Pong-v0"  # name of gym environment
NUM_STEPS = 1_000_000  # total training steps to take
CHECKPOINT_FREQ = 500_000  # interval between checkpoints
BATCH_SIZE = 2048  # size of batch updates
NUM_ENVS = 16  # number of parallel environments
TENSORBOARD_PORT = 6006  # port tensorboard should run on

MODELS: [Benchmark] = [NatureOneLayerSmallKernel, NatureOneLayerLarge, NatureBaselineGELU, NatureBaselineCELU]


def evaluate(individual: np.ndarray):
    print(individual)
    return individual.sum()


def main():
    device = get_device("cuda")
    print(f"Loading model onto {device}")

    tensorboard = Popen(f"tensorboard --logdir={BASE_LOG_PATH} --port={TENSORBOARD_PORT}")
    open_new_tab(f"http://localhost:{TENSORBOARD_PORT}")

    # set new seed and record initial rng state for reproducibility
    random.seed(RNG_SEED)
    torch.manual_seed(RNG_SEED)
    torch_state = torch.get_rng_state()
    random_state = random.getstate()

    for output_channels1 in range(32, 128, 16):
        for output_channels2 in range(64, 128, 16):
            for output_channels3 in range(64, 128, 16):
                layer1 = LayerConfig(output_channels1, 8, 4, 0, "ReLU")
                layer2 = LayerConfig(output_channels2, 4, 2, 0, "ReLU")
                layer3 = LayerConfig(output_channels3, 3, 1, 0, "ReLU")
                layers = [layer1, layer2, layer3]

                name = benchmark_name(layers)

                print(f"Training {name}")

                checkpoint_path = os.path.join(BASE_CHECKPOINT_PATH, "PPO", ENV_NAME, name)
                os.makedirs(checkpoint_path, exist_ok=True)

                log_path = os.path.join(BASE_LOG_PATH, "PPO", ENV_NAME, name)
                os.makedirs(log_path, exist_ok=True)

                log_name = f"{name}_{ENV_NAME}"

                env_wrapper_args = dict(
                    noop_max=30,
                    frame_skip=1,
                    screen_size=84,
                    terminal_on_life_loss=True,
                    clip_reward=True,
                )
                env = make_atari_env(ENV_NAME, n_envs=NUM_ENVS, seed=RNG_SEED, wrapper_kwargs=env_wrapper_args)

                # setup callback to save model at fixed intervals
                callback = CheckpointCallback(save_freq=CHECKPOINT_FREQ, save_path=checkpoint_path, name_prefix=log_name)
                model = PPO(
                    CnnPolicy,
                    env,
                    verbose=VERBOSE,
                    device=device,
                    batch_size=BATCH_SIZE,
                    seed=RNG_SEED * 7,
                    tensorboard_log=log_path,
                    policy_kwargs=dict(features_extractor_class=VariableBenchmark, features_extractor_kwargs=dict(layers=layers)),
                )

                with open(f"{checkpoint_path}/cnn_config", "w") as file:
                    file.write(f"{name}\n")
                    file.write(str(model.policy.features_extractor.cnn))

                model.learn(NUM_STEPS, callback=callback, tb_log_name=log_name)
                model.save(os.path.join(checkpoint_path, "final.zip"))

                del model

                torch.set_rng_state(torch.ByteTensor(torch_state))
                random.setstate(random_state)

    # TODO: Had to refactor Benchmark.name() from a class method to an instance method,
    #  the below code must be changed to adapt the name property to an instance method.
    # for benchmark_class in MODELS:
    #     print(f"Training {benchmark_class.name()}")
    #
    #     checkpoint_path = os.path.join(BASE_CHECKPOINT_PATH, "PPO", ENV_NAME, benchmark_class.name())
    #     os.makedirs(checkpoint_path, exist_ok=True)
    #
    #     log_path = os.path.join(BASE_LOG_PATH, "PPO", ENV_NAME, benchmark_class.name())
    #     os.makedirs(log_path, exist_ok=True)
    #
    #     log_name = f"{benchmark_class.name()}_{ENV_NAME}"
    #
    #     env_wrapper_args = dict(
    #         noop_max=30,
    #         frame_skip=1,
    #         screen_size=84,
    #         terminal_on_life_loss=True,
    #         clip_reward=True,
    #     )
    #     env = make_atari_env(ENV_NAME, n_envs=NUM_ENVS, seed=PYTORCH_SEED, wrapper_kwargs=env_wrapper_args)
    #
    #     # setup callback to save model at fixed intervals
    #     callback = CheckpointCallback(save_freq=CHECKPOINT_FREQ, save_path=checkpoint_path, name_prefix=log_name)
    #     model = PPO(
    #         CnnPolicy,
    #         env,
    #         verbose=VERBOSE,
    #         device=device,
    #         batch_size=BATCH_SIZE,
    #         seed=PYTORCH_SEED * 7,
    #         tensorboard_log=log_path,
    #         policy_kwargs=dict(
    #             features_extractor_class=benchmark_class,
    #         ),
    #     )
    #
    #     with open(f"{checkpoint_path}/cnn_config", "w") as file:
    #         file.write(str(model.policy.features_extractor.cnn))
    #
    #     model.learn(NUM_STEPS, callback=callback, tb_log_name=log_name)
    #     model.save(os.path.join(checkpoint_path, "final.zip"))
    #
    #     del model

    tensorboard.kill()


if __name__ == "__main__":
    main()
