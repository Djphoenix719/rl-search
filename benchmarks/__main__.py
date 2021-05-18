import math
import os
import random
import statistics

from functools import lru_cache

from deap.base import Fitness
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.ppo import CnnPolicy
from typing import List
from typing import Tuple

import torch
from deap import base
from deap import tools
from deap import algorithms

from benchmarks.networks import LayerConfig
from benchmarks.networks import VariableBenchmark
from benchmarks.networks import benchmark_name

RNG_SEED = 42  # Seed for pytorch
VERBOSE = 2  # 0 no output, 1 info, 2 debug
ROOT_PATH = "./"
BASE_CHECKPOINT_PATH = os.path.join(ROOT_PATH, "checkpoints")  # path to save checkpoints to
BASE_LOG_PATH = os.path.join(ROOT_PATH, "logs")  # path to save tensorboard logs to
ENV_NAME = "Pong-v0"  # name of gym environment
TRAIN_STEPS = 1_000_000  # total training steps to take
EVAL_STEPS = 10_000  # steps to evaluate a trained model
CHECKPOINT_FREQ = 500_000  # interval between checkpoints
BATCH_SIZE = 64  # size of batch updates
N_ENVS = 4  # number of parallel environments to evaluate
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
    """
    Represents the fitness of our individual. Mostly this is used as a way of
    declaring the initial values within the paradigm permitted by the DEAP framework.
    """

    def __init__(self):
        self.weights = MAX_FITNESS


class Individual:
    """
    Represents an individual set of hyper-parameters to be turned into a model.
    This class mimics a list but is hashable to allow multithreading later.
    """

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
    """
    Construct a list of layer configs to pass to the feature extractor
    :param individual: The individual to construct the configs for
    :return:
    """
    # TODO: Determine kernel size
    return [LayerConfig(n_out, 2, 1, 0, "gelu") for n_out in individual]


@lru_cache(maxsize=None)
def evaluate(individual: Individual) -> Tuple[int]:
    """
    Evaluate a single individual model and return it's mean score after the training time is elapsed.
    Models are trained and evaluated for a number of timesteps as paramterized in the constants at the
    top of the file.
    :param individual: The individual to evaluate.
    :return:
    """

    layers = construct_layers(individual)
    name = benchmark_name(layers)

    print("Evaluating individual: ", individual)

    print(f"{ind_2_str(individual)} now running on cuda")

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
            frame_skip=1,  # number of frames to skip before taking a new action
            screen_size=84,
            terminal_on_life_loss=True,  #
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

    # output the model config to a file for easier viewing
    with open(config_path, "w") as file:
        file.write(f"{name}\n")
        file.write(str(model.policy.features_extractor.cnn))
        print(model.policy.features_extractor.cnn)

    model.learn(TRAIN_STEPS, callback=callback, tb_log_name=name)
    model.save(zip_path)

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

    # tensorboard = Popen(f"/home/cuccinela5/anaconda3/envs/thesis1/bin/tensorboard --logdir={BASE_LOG_PATH} --port={TENSORBOARD_PORT}")
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
    # register a new helper function. pass it the min and max power weights every call
    toolbox.register("layer_output", random_power_of_2, LAYER_MIN_POWER, LAYER_MAX_POWER)
    # this registration is used to create an individual. We generate 3 weights using the layer_output registration above
    toolbox.register("individual", tools.initCycle, Individual, (toolbox.layer_output,), n=N_CYCLES)
    # register the population function, which we'll use to create a population of individuals
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # register the selection function the algorithm will use
    # in our case we use tournament selection, which operates as follows
    # > Select the best individual among *tournsize* randomly chosen individuals, *k* times.
    # Note we choose K below with the mu property
    toolbox.register("select", tools.selTournament, tournsize=3)
    # the evaluation function will run the model on the gpu in our case
    toolbox.register("evaluate", evaluate)
    # register our desired mating function, in our case inplace uniform crossover
    toolbox.register("mate", tools.cxUniform, indpb=0.5)
    # register our desired mutation function, in our case we move a weight up or down a power of two randomly
    toolbox.register("mutate", mutate)

    print(f"Running with {torch.cuda.device_count()} GPUs")

    population = toolbox.population(POPULATION_SIZE)

    done = False
    while not done:
        mu = POPULATION_SIZE // 10  # select the top 10% of individuals with tournament selection
        lambda_ = POPULATION_SIZE - mu  #

        # We'll use a Mu + Lambda evolutionary algorithm that runs based on the below psuedocode
        # > evaluate(population)
        # > for g in range(ngen):
        # >     offspring = varOr(population, toolbox, lambda_, cxpb, mutpb)
        # >     evaluate(offspring)
        # >     population = select(population + offspring, mu)
        # See: https://deap.readthedocs.io/en/master/api/algo.html#deap.algorithms.eaMuPlusLambda
        logbook = algorithms.eaMuPlusLambda(
            population, toolbox, mu=mu, lambda_=lambda_, cxpb=0.5, mutpb=0.05, ngen=N_GEN, stats=sts, halloffame=hof, verbose=VERBOSE > 0
        )

        print("Best of run")
        print(hof.items)

        # TODO: Stopping criteria, do we want to continue after N_GENS for any reason?
        done = True

    # tensorboard.kill()


if __name__ == "__main__":
    main()
