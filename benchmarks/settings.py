import os
from typing import Callable

import torch
import platform


# OUTPUT SETTINGS
VERBOSE = 2  # 0 no output, 1 info, 2 debug
ROOT_PATH = "D:/rl-search/" if platform.node() == "DESKTOP-C9EHK4U" else "/mnt/checkpoints"  # root save path
BASE_CHECKPOINT_PATH = os.path.join(ROOT_PATH, "checkpoints")  # path to save checkpoints to
BASE_LOG_PATH = os.path.join(ROOT_PATH, "logs")  # path to save tensorboard logs to
TENSORBOARD_PORT = 6006  # port tensorboard should run on


# RNG SEEDS
RANDOM_SEED = 42  # seed for random.seed
# TORCH_SEED = RANDOM_SEED + 1  # seed for pytorch


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """

    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func


# GYM & BASELINES SETTINGS
ENV_NAME = "Pong-v0"  # name of gym environment
MAX_SCORE = 21  # max achievable score in the env
MIN_SCORE = -21  # min achievable score in the env
MAX_TIME = 60 * 60 * 6  # max time in seconds a single env is estimated to take to train
MIN_TIME = 60 * 15  # min time in seconds a single env is estimated tot rain
TIME_RANGE = MAX_TIME - MIN_TIME
SCORE_RANGE = MAX_SCORE - MIN_SCORE
CHECKPOINT_FREQ = 500_000  # interval between checkpoints
DEVICE_TYPE = "cuda" if torch.cuda.is_available() else "cpu"  # run on cpu or cuda
EVAL_FREQ = 10_000  # number steps between evaluations
EVAL_THRESHOLD = 0  # early stopping threshold

# n_envs: 8
N_ENVS = 20  # number of parallel environments to evaluate
# n_steps: 128
UPDATE_STEPS = 128
# n_epochs: 4
N_EPOCHS = 4
# batch_size: 256
BATCH_SIZE = 256  # size of batch updates
# n_timesteps: !!float 1e7
TRAIN_STEPS = 10_000_000  # total training steps to take
EVAL_STEPS = 10_000  # steps to evaluate a trained model
# learning_rate: lin_2.5e-4
LEARNING_RATE = linear_schedule(2.5e-4)
# clip_range: lin_0.1
CLIP_RANGE = linear_schedule(0.1)
# vf_coef: 0.5
VF_COEF = 0.5
# ent_coef: 0.01
ENT_COEF = 0.01


# GENETIC ALGORITHM SETTINGS
N_GEN = 25  # number of generations to run the algorithm for
CROSSOVER_PROB = 0.5  # probability of mating two individuals
MUTATION_PROB = 0.5  # probability of mutating an individual
N_BEST = 5  # keep the top n individuals from a round
N_CHILDREN = 20  # number of children to produce each generation
LAYER_MIN_POWER, LAYER_MAX_POWER = 1, 9  # max size of output dimensions on a layer in power of 2
N_CYCLES = 3  # number of weights to generate, functionally becomes number of layers in cnn
POPULATION_SIZE = 25  # before each round, ensure this many individuals exist, less may due to selection
N_HOF = 5  # keep the global n individuals that are the best from the entire run
SCORE_WEIGHT = 0.7  # percentage of fitness that should be score
TIME_WEIGHT = 0.3  # percentage of fitness that should be time based
assert sum([SCORE_WEIGHT, TIME_WEIGHT]) == 1, "weights must sum to 1"
MAX_FITNESS = (MAX_SCORE,)  # max fitness a model can achieve, dependent on task
INIT_FITNESS = MAX_FITNESS  # initial fitness values, deap pos/neg as better depending on what is here
