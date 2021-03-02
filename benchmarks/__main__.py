import os

from benchmarks.networks import Benchmark
from benchmarks.networks import NatureBaseline
from benchmarks.networks import NatureLarge
from benchmarks.networks import NatureSmall

from stable_baselines3.common.utils import get_device
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.ppo import PPO
from stable_baselines3.ppo import CnnPolicy
from stable_baselines3.common.callbacks import CheckpointCallback

PYTORCH_SEED = 42  # Seed for pytorch
VERBOSE = 2  # 0 no output, 1 info, 2 debug
CHECKPOINT_FREQ = 1_000_000  # interval between checkpoints
BASE_CHECKPOINT_PATH = "checkpoints/"  # path to save checkpoints to
BASE_LOG_PATH = "logs/"
ENV_NAME = "Breakout-v0"  # name of gym environment
NUM_STEPS = 10_000_000  # total training steps to take
BATCH_SIZE = 2048  # size of batch updates
NUM_ENVS = 16  # number of parallel environments

MODELS: [Benchmark] = [NatureBaseline, NatureSmall, NatureLarge]


def main():
    device = get_device("cuda")
    print(f"Loading model onto {device}")

    for benchmark_class in MODELS:
        print(f"Training {benchmark_class.name()}")

        checkpoint_path = os.path.join(BASE_CHECKPOINT_PATH, "PPO", ENV_NAME, benchmark_class.name())
        os.makedirs(checkpoint_path, exist_ok=True)

        log_path = os.path.join(BASE_LOG_PATH, "PPO", ENV_NAME, benchmark_class.name())
        os.makedirs(log_path, exist_ok=True)

        log_name = f"{benchmark_class.name()}_{ENV_NAME}"

        env_wrapper_args = dict(
            noop_max=30,
            frame_skip=1,
            screen_size=84,
            terminal_on_life_loss=True,
            clip_reward=True,
        )
        env = make_atari_env(ENV_NAME, n_envs=NUM_ENVS, seed=PYTORCH_SEED, wrapper_kwargs=env_wrapper_args)

        # setup callback to save model at fixed intervals
        callback = CheckpointCallback(save_freq=CHECKPOINT_FREQ, save_path=checkpoint_path, name_prefix=log_name)
        model = PPO(
            CnnPolicy,
            env,
            verbose=VERBOSE,
            device=device,
            batch_size=BATCH_SIZE,
            seed=PYTORCH_SEED * 7,
            tensorboard_log=log_path,
            policy_kwargs=dict(
                features_extractor_class=benchmark_class,
            ),
        )

        with open(f"{checkpoint_path}/cnn_config", "w") as file:
            file.write(str(model.policy.features_extractor.cnn))

        model.learn(NUM_STEPS, callback=callback, tb_log_name=log_name)
        model.save(os.path.join(checkpoint_path, "final.zip"))

        del model


if __name__ == "__main__":
    main()
