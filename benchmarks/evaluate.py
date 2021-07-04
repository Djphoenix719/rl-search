import json
import os
import random
from time import time
from typing import Tuple
from typing import Union

import gym
from stable_baselines3 import A2C
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.callbacks import StopTrainingOnRewardThreshold
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.vec_env import VecTransposeImage
from stable_baselines3.ppo import CnnPolicy
from stable_baselines3.common.vec_env import SubprocVecEnv

from benchmarks.callbacks import TimeLimitCallback
from benchmarks.individual import Individual
from benchmarks.math_util import weighted_time
from benchmarks.networks import VariableBenchmark
from benchmarks.settings import *
from benchmarks.wrapper import AtariWrapper


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

    if os.path.exists(checkpoint_path):
        return (random.randint(MIN_SCORE, MAX_SCORE),)

    os.makedirs(checkpoint_path, exist_ok=True)
    log_path = os.path.join(BASE_LOG_PATH, "PPO", ENV_NAME, name)
    os.makedirs(log_path, exist_ok=True)

    results_path = os.path.join(checkpoint_path, "results.json")

    if not os.path.exists(results_path):
        env_args = dict(
            frame_skip=4,
            screen_size=84,
            terminal_on_life_loss=True,
            clip_reward=True,
        )

        # Creates a gym environment for an atari game using the specified seed and number of environments
        # This is a "vectorized environment", which means Stable Baselines batches the updates into vectors
        # for improved performance..
        def atari_wrapper(env: gym.Env) -> gym.Env:
            env = AtariWrapper(env, **env_args)
            return env

        def make_env(rank: int, count: int) -> VecEnv:
            return make_vec_env(
                ENV_NAME,
                n_envs=count,
                seed=RANDOM_SEED + rank,
                start_index=0,
                monitor_dir=None,
                wrapper_class=atari_wrapper,
                env_kwargs=None,
                vec_env_cls=SubprocVecEnv,
                vec_env_kwargs=None,
                monitor_kwargs=None,
            )

        train_env = make_env(0, N_ENVS)
        eval_env = make_env(1, 1)

        # required by models in baselines
        train_env = VecTransposeImage(train_env)
        eval_env = VecTransposeImage(eval_env)

        # setup callback to save model at fixed intervals
        save_callback = CheckpointCallback(save_freq=CHECKPOINT_FREQ, save_path=checkpoint_path, name_prefix=name)
        stop_callback = StopTrainingOnRewardThreshold(reward_threshold=EVAL_THRESHOLD)
        time_callback = TimeLimitCallback(max_time=60 * 5)
        best_callback = EvalCallback(
            eval_env,
            eval_freq=EVAL_FREQ,
            best_model_save_path=checkpoint_path,
            callback_on_new_best=stop_callback,
        )
        list_callback = CallbackList([save_callback, best_callback, time_callback])

        model = PPO(
            CnnPolicy,
            train_env,
            verbose=VERBOSE,
            batch_size=BATCH_SIZE,
            seed=RANDOM_SEED * 7,
            tensorboard_log=log_path,
            learning_rate=LEARNING_RATE,
            n_steps=UPDATE_STEPS,
            n_epochs=N_EPOCHS,
            ent_coef=ENT_COEF,
            vf_coef=VF_COEF,
            clip_range=CLIP_RANGE,
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

        del train_env
        del eval_env

        time_taken = time() - t_start

        print("Beginning evaluation...")

        # score of the game, standard deviation of multiple runs
        reward_mean, reward_std = evaluate_policy(model, make_env(2, 1))

        with open(results_path, "w") as handle:
            handle.write(json.dumps((reward_mean, reward_std, time_taken)))
    else:
        reward_mean, reward_std, time_taken = json.load(open(results_path, "r"))

    reward_mean = abs(MIN_SCORE) + reward_mean
    value = (reward_mean * weighted_time(time_taken),)

    print(f"Evaluated {name} with a score of {value}  in {(time_taken):.2f}s")

    return value


def mock_evaluate(individual: Individual, device: Union[torch.device, str] = "auto") -> Tuple[int]:
    import json

    name = individual.encode()

    checkpoint_path = os.path.join(BASE_CHECKPOINT_PATH, "PPO", ENV_NAME, name)
    os.makedirs(checkpoint_path, exist_ok=True)
    log_path = os.path.join(BASE_LOG_PATH, "PPO", ENV_NAME, name)
    os.makedirs(log_path, exist_ok=True)

    results_path = os.path.join(checkpoint_path, "results.json")

    if os.path.exists(results_path):
        reward_mean, reward_std, time_taken = json.load(open(results_path, "r"))
    else:
        reward_mean = random.randint(-21, 21)
        reward_std = random.randint(-21, 21) / 20
        time_taken = random.randint(60 * 5, 60 * 60 * 3)

        with open(results_path, "w") as handle:
            handle.write(json.dumps((reward_mean, reward_std, time_taken)))

    reward_mean = abs(MIN_SCORE) + reward_mean
    value = (reward_mean * weighted_time(time_taken),)

    print(f"Evaluated {name} with a score of {value}  in {(time_taken):.2f}s")

    return value


# evaluate = mock_evaluate
