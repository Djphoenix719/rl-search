import json
from time import time

import gym
from stable_baselines3 import PPO
from stable_baselines3.common.atari_wrappers import ClipRewardEnv
from stable_baselines3.common.atari_wrappers import EpisodicLifeEnv
from stable_baselines3.common.atari_wrappers import FireResetEnv
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv
from stable_baselines3.common.atari_wrappers import WarpFrame
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.torch_layers import NatureCNN
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.vec_env import VecTransposeImage
from stable_baselines3.ppo import CnnPolicy

from benchmarks.settings import *


class AtariWrapper(gym.Wrapper):
    """
    Atari 2600 preprocessings

    Specifically:

    * NoopReset: obtain initial state by taking random number of no-ops on reset.
    * Frame skipping: 4 by default
    * Max-pooling: most recent two observations
    * Termination signal when a life is lost.
    * Resize to a square image: 84x84 by default
    * Grayscale observation
    * Clip reward to {-1, 0, 1}

    :param env: gym environment
    :param noop_max: max number of no-ops
    :param frame_skip: the frequency at which the agent experiences the game.
    :param screen_size: resize Atari frame
    :param terminal_on_life_loss: if True, then step() returns done=True whenever a life is lost.
    :param clip_reward: If True (default), the reward is clip to {-1, 0, 1} depending on its sign.
    """

    def __init__(
        self,
        env: gym.Env,
        frame_skip: int = 4,
        screen_size: int = 84,
        terminal_on_life_loss: bool = True,
        clip_reward: bool = True,
    ):
        env = MaxAndSkipEnv(env, skip=frame_skip)
        if terminal_on_life_loss:
            env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = WarpFrame(env, width=screen_size, height=screen_size)
        if clip_reward:
            env = ClipRewardEnv(env)

        super(AtariWrapper, self).__init__(env)


def main():

    set_random_seed(RANDOM_SEED)

    t_start = time()
    name = "ParamTest"

    checkpoint_path = os.path.join(BASE_CHECKPOINT_PATH, "PPO", ENV_NAME, name)
    os.makedirs(checkpoint_path, exist_ok=True)
    log_path = os.path.join(BASE_LOG_PATH, "PPO", ENV_NAME, name)
    os.makedirs(log_path, exist_ok=True)
    results_path = os.path.join(checkpoint_path, "results.json")

    env_args = dict(
        # noop_max=1,  # max sequential no-ops to take
        frame_skip=1,  # number of frames to skip before taking a new action
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
    # stop_callback = StopTrainingOnRewardThreshold(reward_threshold=EVAL_THRESHOLD)
    best_callback = EvalCallback(
        eval_env,
        eval_freq=EVAL_FREQ,
        best_model_save_path=checkpoint_path,
        # callback_on_new_best=stop_callback,
    )
    list_callback = CallbackList([save_callback, best_callback])

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
        policy_kwargs=dict(features_extractor_class=NatureCNN),
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


if __name__ == "__main__":
    main()
