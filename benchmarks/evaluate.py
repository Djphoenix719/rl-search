import random
from time import time
from typing import Tuple
from typing import Union

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.callbacks import StopTrainingOnRewardThreshold
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecTransposeImage
from stable_baselines3.ppo import CnnPolicy

from benchmarks.individual import Individual
from benchmarks.math_util import weighted_time
from benchmarks.networks import VariableBenchmark
from benchmarks.settings import *


def evaluate(individual: Individual, device: Union[torch.device, str] = "auto") -> Tuple[int]:
    """
    Evaluate a single individual model and return it's mean score after the training time is elapsed.
    Models are trained and evaluated for a number of timestamps as parameterized in the constants at the
    top of the file.
    :param individual: The individual to evaluate.
    :return:
    """

    t_start = time()
    layers = individual.weights
    name = individual.encode()

    checkpoint_path = os.path.join(BASE_CHECKPOINT_PATH, "PPO", ENV_NAME, name)
    os.makedirs(checkpoint_path, exist_ok=True)
    log_path = os.path.join(BASE_LOG_PATH, "PPO", ENV_NAME, name)
    os.makedirs(log_path, exist_ok=True)

    env_args = dict(
        noop_max=30,  # max sequential no-ops to take
        frame_skip=4,  # number of frames to skip before taking a new action
        screen_size=84,
        terminal_on_life_loss=True,
        clip_reward=True,
    )

    # Creates a gym environment for an atari game using the specified seed and number of environments
    # This is a "vectorized environment", which means Stable Baselines batches the updates into vectors
    # for improved performance..
    train_env = make_atari_env(ENV_NAME, n_envs=N_ENVS, seed=RANDOM_SEED, wrapper_kwargs=env_args)
    eval_env = VecTransposeImage(make_atari_env(ENV_NAME))

    # setup callback to save model at fixed intervals
    save_callback = CheckpointCallback(save_freq=CHECKPOINT_FREQ, save_path=checkpoint_path, name_prefix=name)
    stop_callback = StopTrainingOnRewardThreshold(reward_threshold=EVAL_THRESHOLD)
    best_callback = EvalCallback(
        eval_env,
        eval_freq=EVAL_FREQ,
        best_model_save_path=checkpoint_path,
        callback_on_new_best=stop_callback,
    )
    list_callback = CallbackList([save_callback, best_callback])

    model = PPO(
        CnnPolicy,
        train_env,
        verbose=VERBOSE,
        batch_size=BATCH_SIZE,
        seed=RANDOM_SEED * 7,
        tensorboard_log=log_path,
        device=device,
        policy_kwargs=dict(features_extractor_class=VariableBenchmark, features_extractor_kwargs=dict(layers=layers)),
    )

    config_path = os.path.join(checkpoint_path, "cnn_config")
    zip_path = os.path.join(checkpoint_path, "model.zip")

    # output the model config to a file for easier viewing
    with open(config_path, "w") as file:
        file.write(f"{name}\n")
        file.write(str(model.policy.features_extractor.cnn))

    print("Beginning training...")

    model.learn(TRAIN_STEPS, callback=list_callback, tb_log_name="run")
    model.save(zip_path)

    time_taken = time() - t_start

    train_env.reset()
    reward_mean, reward_std = evaluate_policy(model, make_atari_env(ENV_NAME))

    reward_mean = abs(MIN_SCORE) + reward_mean

    value = (reward_mean * weighted_time(time_taken),)

    print(f"Evaluated {name} with a score of {value}  in {(time_taken):.2f}s")

    return value


def mock_evaluate(individual: Individual, device: Union[torch.device, str] = "auto") -> Tuple[int]:

    name = individual.encode()
    time_taken = random.random() * 60 * 60 * 6

    # TODO: Finish score weighting by time
    reward_mean = random.randint(-21, 21)
    value = (reward_mean * SCORE_WEIGHT + weighted_time(time_taken) * TIME_WEIGHT,)

    print(f"Evaluated {name} {(time_taken):.2f}s -> {reward_mean}, scaled to {value}")
    print(f"reward_mean {reward_mean}, scaled {reward_mean * SCORE_WEIGHT}")
    print(f"time_taken {time_taken}, factor {weighted_time(time_taken)}, scaled {weighted_time(time_taken) * TIME_WEIGHT}")

    return (sum([layer.output_channels for layer in individual]),)
