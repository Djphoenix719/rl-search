import json
from time import time

import gym
import torch as th
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.callbacks import StopTrainingOnRewardThreshold
from torch import nn
from stable_baselines3 import PPO
from stable_baselines3.common.atari_wrappers import ClipRewardEnv
from stable_baselines3.common.atari_wrappers import EpisodicLifeEnv
from stable_baselines3.common.atari_wrappers import FireResetEnv
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv
from stable_baselines3.common.atari_wrappers import WarpFrame
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.vec_env import VecTransposeImage
from stable_baselines3.ppo import CnnPolicy

from benchmarks.settings import *
from benchmarks.callbacks import TimeLimitCallback
from benchmarks.wrapper import AtariWrapper


class FeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 512):
        super(FeatureExtractor, self).__init__(observation_space, features_dim)

        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 16, kernel_size=4, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(16, 256, kernel_size=32, stride=1, padding=0),
            nn.GELU(),
            nn.Conv2d(256, 2, kernel_size=2, stride=1, padding=0),
            nn.CELU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(th.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))


def main():

    set_random_seed(RANDOM_SEED)

    t_start = time()
    name = "LargeFinalLayer"

    checkpoint_path = os.path.join(BASE_CHECKPOINT_PATH, "PPO", ENV_NAME, name)
    os.makedirs(checkpoint_path, exist_ok=True)

    log_path = os.path.join(BASE_LOG_PATH, "PPO", ENV_NAME, name)
    os.makedirs(log_path, exist_ok=True)

    results_path = os.path.join(checkpoint_path, "results.json")

    env_args = dict(
        frame_skip=4,
        screen_size=84,
        terminal_on_life_loss=True,
        clip_reward=True,
    )

    # Creates a gym environment for an atari game using the specified seed and number of environments
    # This is a "vectorized environment", which means Stable Baselines batches the updates into vectors
    # for improved performance..
    # train_env = make_atari_env(ENV_NAME, n_envs=N_ENVS, seed=RANDOM_SEED, wrapper_kwargs=env_args)

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
            vec_env_cls=None,
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
    time_callback = TimeLimitCallback(max_time=TIME_LIMIT)
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
        seed=RANDOM_SEED,
        tensorboard_log=log_path,
        learning_rate=LEARNING_RATE,
        n_steps=UPDATE_STEPS,
        n_epochs=N_EPOCHS,
        ent_coef=ENT_COEF,
        vf_coef=VF_COEF,
        clip_range=CLIP_RANGE,
        device=DEVICE_TYPE,
        policy_kwargs=dict(features_extractor_class=FeatureExtractor),
    )

    config_path = os.path.join(checkpoint_path, "cnn_config")
    zip_path = os.path.join(checkpoint_path, "model.zip")

    # output the model config to a file for easier viewing
    with open(config_path, "w") as file:
        file.write(f"{name}\n")
        file.write(str(model.policy.features_extractor.cnn))

    print("Beginning training...")

    model.learn(TRAIN_STEPS, callback=list_callback, tb_log_name="run")
    # model.learn(TRAIN_STEPS, tb_log_name="run")
    model.save(zip_path)

    del train_env
    # del eval_env

    time_taken = time() - t_start

    print("Beginning evaluation...")

    # score of the game, standard deviation of multiple runs
    reward_mean, reward_std = evaluate_policy(model, make_env(2, 1))

    with open(results_path, "w") as handle:
        handle.write(json.dumps((reward_mean, reward_std, time_taken)))


if __name__ == "__main__":
    import cProfile

    cProfile.run("main()")
