import math
import os
import random
import statistics

from functools import lru_cache
from subprocess import Popen

from deap.base import Fitness
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.ppo import CnnPolicy
from typing import List
from typing import Tuple

from google.cloud import storage

import torch
from deap import base
from deap import tools
from deap import algorithms

from benchmarks.networks import LayerConfig
from benchmarks.networks import VariableBenchmark
from benchmarks.networks import benchmark_name

# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./service-account.json"
# STORAGE_CLIENT = storage.Client(project="masters-thesis-312816")
# STORAGE_BUCKET = storage.Bucket(STORAGE_CLIENT, name="rl-search-masters-thesis")

RNG_SEED = 42  # Seed for pytorch
VERBOSE = 2  # 0 no output, 1 info, 2 debug
BASE_CHECKPOINT_PATH = "/mnt/disks/checkpoints/checkpoints/"  # path to save checkpoints to
BASE_LOG_PATH = "/mnt/disks/checkpoints/logs/"  # path to save tensorboard logs to
ENV_NAME = "Pong-v0"  # name of gym environment
TRAIN_STEPS = 1_000  # total training steps to take
EVAL_STEPS = 1_000  # steps to evaluate a trained model
CHECKPOINT_FREQ = 500_000  # interval between checkpoints
BATCH_SIZE = 1024  # size of batch updates
N_ENVS = os.cpu_count()  # number of parallel environments to evaluate
DEVICE_TYPE = "cuda" if torch.cuda.is_available() else "cpu"  # run on cpu or cuda
TENSORBOARD_PORT = 6006  # port tensorboard should run on
LAYER_MIN_POWER, LAYER_MAX_POWER = 1, 12  # max size of output dimensions on a layer in power of 2
N_CYCLES = 3  # number of weights to generate, functionally becomes number of layers in cnn
POPULATION_SIZE = 5  # before each round, ensure this many individuals exist, less may due to selection
N_BEST = 4  # keep the top n individuals from a round
MAX_FITNESS = (20.0,)  # max fitness a model can achieve, dependent on task
INIT_FITNESS = (0.0,)  # initial fitness values
N_GEN = 25  # number of generations
INDIVIDUAL_CLASS = tuple  # class individual uses in deap framework


class Fitness(base.Fitness):
    def __init__(self):
        self.weights = MAX_FITNESS


class Individual:
    _params: List[int]
    fitness: Fitness

    def __init__(self, weights):
        self._params = [value for value in weights]
        self.fitness = Fitness()

    def __getitem__(self, item):
        return self._params[item]

    def __setitem__(self, key, value):
        self._params[key] = value

    def __len__(self):
        return len(self._params)

    def __hash__(self):
        return tuple(self._params).__hash__()

    def __str__(self):
        return str(self._params)

    def __repr__(self):
        return repr(self._params)


def ind_2_str(individual: Individual) -> str:
    return "_".join([str(w) for w in individual])


def random_power_of_2(lower: int, upper: int) -> int:
    return 2 ** random.randint(lower, upper)


def construct_layers(individual: Individual) -> List[LayerConfig]:
    return [LayerConfig(n_out, 2, 1, 0, "gelu") for n_out in individual]


@lru_cache(maxsize=None)
def evaluate(individual: Individual) -> Tuple[int]:
    layers = construct_layers(individual)
    name = benchmark_name(layers)

    print("Evaluating individual: ", individual)

    print(f"{ind_2_str(individual)} now running on cuda")

    checkpoint_path = os.path.join(BASE_CHECKPOINT_PATH, "PPO", ENV_NAME, name)
    os.makedirs(checkpoint_path, exist_ok=True)

    log_path = os.path.join(BASE_LOG_PATH, "PPO", ENV_NAME, name)
    os.makedirs(log_path, exist_ok=True)

    env = make_atari_env(
        ENV_NAME,
        n_envs=N_ENVS,
        seed=RNG_SEED,
        wrapper_kwargs=dict(
            noop_max=30,
            frame_skip=1,
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

    config_path = f"{checkpoint_path}\\cnn_config"
    zip_path = f"{checkpoint_path}\\model.zip"

    with open(config_path, "w") as file:
        file.write(f"{name}\n")
        file.write(str(model.policy.features_extractor.cnn))
        print(model.policy.features_extractor.cnn)

    model.learn(TRAIN_STEPS, callback=callback, tb_log_name=name)
    model.save(zip_path)

    # print("Uploading model")
    # blob_config = STORAGE_BUCKET.blob(config_path)
    # blob_config.upload_from_filename(config_path, content_type="text/plain")
    # zip_config = STORAGE_BUCKET.blob(zip_path)
    # zip_config.upload_from_filename(zip_path, content_type="application/zip")
    # print("Upload complete")

    print("Evaluating final model")

    del env  # unallocate old memory first
    env = make_atari_env(ENV_NAME)
    reward_mean, reward_std = evaluate_policy(model, env)

    return (reward_mean,)


def mate(a: List[int], b: List[int], probability: float = 0.5) -> Tuple[List[int], List[int]]:
    """
    Mate two parents and return two children.
    :param a: The left parent.
    :param b: The right parent.
    :param probability: Probability of exchanging two weights.
    :return: Two children of the individuals.
    """
    return tools.cxUniform(a.copy(), b.copy(), probability)


def mutate(a: List[int], probability: float = 0.5) -> List[int]:
    """
    Mutate an individual, probabilistically changing it's weights to the next or previous power of two.
    :param a: The individual to mutate.
    :param probability: The probability of changing a given weight.
    :return: A new mutated individual.
    """
    # a = a.copy()
    for idx in range(len(a)):
        if random.random() < probability:
            continue

        direction = random.choice([-1, 1])
        power = int(math.log(a[idx]) / math.log(2)) + direction

        # flip direction if we're up against the bounds
        if power < LAYER_MIN_POWER:
            power += 2
        if power > LAYER_MAX_POWER:
            power -= 2

        a[idx] = 2 ** power

    return (a,)


def main():
    os.makedirs(BASE_CHECKPOINT_PATH, exist_ok=True)
    os.makedirs(BASE_LOG_PATH, exist_ok=True)

    tensorboard = Popen(f"tensorboard --logdir={BASE_LOG_PATH} --port={TENSORBOARD_PORT}")
    # open_new_tab(f"http://localhost:{TENSORBOARD_PORT}")

    # set new seed and record initial rng state for reproducibility
    random.seed(RNG_SEED)
    torch.manual_seed(RNG_SEED + 1)
    # torch_state = torch.get_rng_state()
    # random_state = random.getstate()

    sts = tools.Statistics()
    sts.register("std", statistics.stdev)
    sts.register("mean", statistics.mean)
    sts.register("median", statistics.median)
    hof = tools.HallOfFame(maxsize=N_BEST)

    toolbox = base.Toolbox()
    toolbox.register("layer_output", random_power_of_2, LAYER_MIN_POWER, LAYER_MAX_POWER)
    toolbox.register("individual", tools.initCycle, Individual, (toolbox.layer_output,), n=N_CYCLES)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxUniform, indpb=0.5)
    toolbox.register("mutate", mutate)

    print(f"Running with {torch.cuda.device_count()} GPUs")

    population = toolbox.population(POPULATION_SIZE)

    done = False
    while not done:
        mu = POPULATION_SIZE // 10
        lambda_ = POPULATION_SIZE - mu
        algorithms.eaMuPlusLambda(population, toolbox, mu=mu, lambda_=lambda_, cxpb=0.5, mutpb=0.05, ngen=N_GEN, stats=sts, halloffame=hof, verbose=VERBOSE > 0)

        print("Best of run")
        print(hof.items)

        done = True

    tensorboard.kill()


if __name__ == "__main__":
    main()
