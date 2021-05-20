import os
import random
from functools import lru_cache
from time import time
from typing import Tuple

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.ppo import CnnPolicy

from benchmarks.random_util import random_slug
from benchmarks.individual import Individual
from benchmarks.networks import VariableBenchmark

from settings import *


@lru_cache(maxsize=None)
def evaluate(individual: Individual) -> Tuple[int]:
    """
    Evaluate a single individual model and return it's mean score after the training time is elapsed.
    Models are trained and evaluated for a number of timestamps as parameterized in the constants at the
    top of the file.
    :param individual: The individual to evaluate.
    :return:
    """

    t_start = time()
    layers = individual.weights
    name = random_slug(SLUG_WORDS)

    print("\nEvaluating individual")
    for layer in layers:
        print("\t", layer)

    checkpoint_path = os.path.join(BASE_CHECKPOINT_PATH, "PPO", ENV_NAME, name)
    os.makedirs(checkpoint_path, exist_ok=True)
    log_path = os.path.join(BASE_LOG_PATH, "PPO", ENV_NAME, name)
    os.makedirs(log_path, exist_ok=True)

    # Creates a gym environment for an atari game using the specified seed and number of environments
    # This is a "vectorized environment", which means Stable Baselines batches the updates into vectors
    # for improved performance..
    env = make_atari_env(
        ENV_NAME,
        n_envs=N_ENVS,
        seed=RNG_SEED,
        wrapper_kwargs=dict(
            noop_max=30,  # max sequential no-ops to take
            frame_skip=4,  # number of frames to skip before taking a new action
            screen_size=84,
            terminal_on_life_loss=True,
            clip_reward=True,
        ),
    )

    # setup callback to save model at fixed intervals
    callback = CheckpointCallback(save_freq=CHECKPOINT_FREQ, save_path=checkpoint_path, name_prefix=name)
    model = PPO(
        CnnPolicy,
        env,
        verbose=VERBOSE,
        batch_size=BATCH_SIZE,
        seed=RNG_SEED * 7,
        tensorboard_log=log_path,
        policy_kwargs=dict(features_extractor_class=VariableBenchmark, features_extractor_kwargs=dict(layers=layers)),
    )

    config_path = os.path.join(checkpoint_path, "cnn_config")
    zip_path = os.path.join(checkpoint_path, "model.zip")

    # output the model config to a file for easier viewing
    with open(config_path, "w") as file:
        file.write(f"{name}\n")
        file.write(str(model.policy.features_extractor.cnn))

    # model.learn(TRAIN_STEPS, callback=callback, tb_log_name=name)
    # model.learn(1, callback=callback, tb_log_name=name)
    model.save(zip_path)

    env = make_atari_env(ENV_NAME)
    reward_mean, reward_std = evaluate_policy(model, env)

    print(f"Evaluated {name} in {(time() - t_start):.2f}s")

    return (reward_mean,)


@lru_cache(maxsize=None)
def mock_evaluate(individual: Individual) -> Tuple[int]:
    return (random.randint(-20, 20),)
