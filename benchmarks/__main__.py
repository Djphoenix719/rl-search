import math
import os
import random

from functools import lru_cache
from typing import Union

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.ppo import CnnPolicy
from typing import List
from typing import Tuple

import torch
from time import time
from faker import Faker

from deap import base
from deap import tools
from deap import algorithms

from benchmarks.activation import ActivationFunction
from benchmarks.individual import Individual
from benchmarks.networks import LayerConfig
from benchmarks.networks import VariableBenchmark

RNG_SEED = 42  # Seed for pytorch
VERBOSE = 2  # 0 no output, 1 info, 2 debug
ROOT_PATH = "./"  # root save path
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
LAYER_MIN_POWER, LAYER_MAX_POWER = 1, 9  # max size of output dimensions on a layer in power of 2
N_CYCLES = 3  # number of weights to generate, functionally becomes number of layers in cnn
POPULATION_SIZE = 50  # before each round, ensure this many individuals exist, less may due to selection
N_BEST = 4  # keep the top n individuals from a round
MAX_FITNESS = (20.0,)  # max fitness a model can achieve, dependent on task
INIT_FITNESS = (-20.0,)  # initial fitness values
N_GEN = 25  # number of generations
SLUG_WORDS = 8  # number of words in our slugs


def random_power_of_2(lower: int, upper: int) -> int:
    return 2 ** random.randint(lower, upper)


def random_slug(n_words: int):
    return "".join([word.capitalize() for word in Faker().words(n_words)])


def random_activation():
    return ActivationFunction(random.randint(1, 3))


def number_to_power_of_2(value: Union[int, float]):
    """
    Finds the correct power of two, e.g. solves `value = 2^n` for n
    :param value:
    :return:
    """
    return int(math.log(value) / math.log(2))


def random_layer():
    layer_power = random.randint(LAYER_MIN_POWER, LAYER_MAX_POWER)
    layer_size = 2 ** layer_power

    kernel_power = random.randint(1, max(1, math.floor(number_to_power_of_2(layer_size * 0.25))))
    kernel_size = 2 ** kernel_power

    return LayerConfig(output_channels=layer_size, kernel_size=kernel_size, stride=1, padding=0, activation=random_activation())


@lru_cache(maxsize=None)
def mock_evaluate(individual: Individual) -> Tuple[int]:
    return (random.randint(-20, 20),)


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


def mate(a: List[int], b: List[int], probability: float = 0.5) -> Tuple[List[int], List[int]]:
    """
    Mate two parents and return two children.
    :param a: The left parent.
    :param b: The right parent.
    :param probability: Probability of exchanging two weights.
    :return: Two children of the individuals.
    """
    return tools.cxUniform(a.copy(), b.copy(), probability)


def clamp(value: Union[int, float], min_: Union[int, float], max_: Union[int, float]) -> Union[int, float]:
    if value >= max_:
        value = max_
    if value <= min_:
        value = min_

    return value


def mutate(individual: Individual, probability: float = 0.5) -> Tuple[Individual]:
    """
    Mutate an individual, probabilistically changing it's weights to the next or previous power of two.
    :param individual: The individual to mutate.
    :param probability: The probability of changing a given weight.
    :return: A new mutated individual.
    """

    def mutate_power(value: int) -> int:
        if random.random() < probability:
            return value

        direction = random.choice([-1, 1])
        power = number_to_power_of_2(value) + direction

        # flip direction if we're up against the bounds
        if power < LAYER_MIN_POWER:
            power += 2
        if power > LAYER_MAX_POWER:
            power -= 2

        return 2 ** power

    def mutate_kernel(kernel_size: int, output_size: int):
        if random.random() < probability:
            return kernel_size

        max_power = math.floor(number_to_power_of_2(output_size * 0.25))

        direction = random.choice([-1, 1])
        power = number_to_power_of_2(kernel_size) + direction

        if power < 1:
            power = 1
        if power > max_power:
            power = max_power - 1

        return 2 ** power

    def mutate_activation(value: ActivationFunction) -> ActivationFunction:
        if value == ActivationFunction.RELU:
            return random.choice([ActivationFunction.GELU, ActivationFunction.CELU])
        elif value == ActivationFunction.CELU:
            return random.choice([ActivationFunction.GELU, ActivationFunction.RELU])
        elif value == ActivationFunction.GELU:
            return random.choice([ActivationFunction.CELU, ActivationFunction.RELU])
        raise ValueError()

    config: LayerConfig
    for (idx, config) in enumerate(individual):
        config.output_channels = mutate_power(config.output_channels)
        config.kernel_size = mutate_kernel(config.kernel_size, config.output_channels)
        config.activation = mutate_activation(config.activation)

    return (individual,)


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
    hof = tools.HallOfFame(maxsize=N_BEST)

    toolbox = base.Toolbox()
    # register a new helper function. pass it the min and max power weights every call
    # this registration is used to create an individual. We generate 3 weights using the layer_output registration above
    toolbox.register("individual", tools.initCycle, Individual, (random_layer,), n=N_CYCLES)
    # register the population function, which we'll use to create a population of individuals
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # register the selection function the algorithm will use
    # in our case we use tournament selection, which operates as follows
    # > Select the best individual among *tournsize* randomly chosen individuals, *k* times.
    # Note we choose K below with the mu property
    toolbox.register("select", tools.selTournament, tournsize=3)
    # the evaluation function will run the model on the gpu in our case
    toolbox.register("evaluate", mock_evaluate)
    # register our desired mating function, in our case inplace uniform crossover
    toolbox.register("mate", tools.cxUniform, indpb=0.5)
    # register our desired mutation function, in our case we move a weight up or down a power of two randomly
    toolbox.register("mutate", mutate)

    print(f"Running with {torch.cuda.device_count()} GPUs")

    population = toolbox.population(POPULATION_SIZE)

    done = False
    generation = 0
    while not done:
        mu = len(population) // 10  # The number of individuals to select for the next generation
        lambda_ = POPULATION_SIZE - mu  # The number of children to produce at each generation

        print("mu", mu)
        print("lambda_", lambda_)

        # We'll use a Mu + Lambda evolutionary algorithm that runs based on the below psuedocode
        # > evaluate(population)
        # > for g in range(ngen):
        # >     offspring = varOr(population, toolbox, lambda_, cxpb, mutpb)
        # >     evaluate(offspring)
        # >     population = select(population + offspring, mu)
        # See: https://deap.readthedocs.io/en/master/api/algo.html#deap.algorithms.eaMuPlusLambda

        population, logbook = algorithms.eaMuPlusLambda(
            population, toolbox, mu=mu, lambda_=lambda_, cxpb=0.5, mutpb=0.05, ngen=N_GEN, halloffame=hof, verbose=VERBOSE > 0
        )

        if generation >= N_GEN:
            done = True

    # tensorboard.kill()


if __name__ == "__main__":
    main()
