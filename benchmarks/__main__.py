import math
import os
import random
from torch.multiprocessing import Manager
from torch.multiprocessing import Process
from torch.multiprocessing import Queue
from typing import List
from typing import Tuple

import numpy as np
import torch
from deap import base
from deap import creator
from deap import tools

from benchmarks.networks import LayerConfig
from benchmarks.networks import benchmark_name

RNG_SEED = 42  # Seed for pytorch
VERBOSE = 2  # 0 no output, 1 info, 2 debug
BASE_CHECKPOINT_PATH = "checkpoints/"  # path to save checkpoints to
BASE_LOG_PATH = "logs/"  # path to save tensorboard logs to
ENV_NAME = "Pong-v0"  # name of gym environment
TRAIN_STEPS = 1_000  # total training steps to take
EVAL_STEPS = 1_000  # steps to evaluate a trained model
CHECKPOINT_FREQ = 500_000  # interval between checkpoints
BATCH_SIZE = 2048  # size of batch updates
NUM_ENVS = 16  # number of parallel environments ran on a SINGLE gpu
DEVICES: List[torch.cuda.device] = [torch.cuda.device(i) for i in range(torch.cuda.device_count())]  # available gpus
TENSORBOARD_PORT = 6006  # port tensorboard should run on
LAYER_MIN_POWER, LAYER_MAX_POWER = 1, 4  # max size of output dimensions on a layer in power of 2
N_CYCLES = 3  # number of weights to generate, functionally becomes number of layers in cnn
POPULATION_SIZE = 5  # before each round, ensure this many individuals exist, less may due to selection
N_BEST = 4  # keep the top n individuals from a round
MAX_FITNESS = (20.0,)  # max fitness a model can achieve, dependent on task


def ind_2_str(individual: List[int]) -> str:
    return "_".join([str(w) for w in individual])


def str_2_ind(indvidiauL: str) -> List[int]:
    return [int(p) for p in indvidiauL.split("_")]


def random_power_of_2(lower: int, upper: int) -> int:
    return 2 ** random.randint(lower, upper)


def construct_layers(individual: List[int]) -> List[LayerConfig]:
    return [LayerConfig(n_out, 2, 1, 0, "gelu") for n_out in individual]


def train(idx: int, round: int, queue: Queue, results: dict):
    while not queue.empty():
        individual = str_2_ind(queue.get())
        layers = construct_layers(individual)
        name = benchmark_name(layers)

        print("Evaluating individual: ", individual)

        if name in results:
            print(f"Previously evaluated, skipping")
            continue

        device = DEVICES[idx]

        print(f"{name} now running on cuda:{device.idx}")

        # checkpoint_path = os.path.join(BASE_CHECKPOINT_PATH, "PPO", ENV_NAME, name)
        # os.makedirs(checkpoint_path, exist_ok=True)
        #
        # log_path = os.path.join(BASE_LOG_PATH, "PPO", ENV_NAME, name)
        # os.makedirs(log_path, exist_ok=True)
        #
        # env_wrapper_args = dict(
        #     noop_max=30,
        #     frame_skip=1,
        #     screen_size=84,
        #     terminal_on_life_loss=True,
        #     clip_reward=True,
        # )
        # env = make_atari_env(ENV_NAME, n_envs=NUM_ENVS, seed=RNG_SEED, wrapper_kwargs=env_wrapper_args)

        # setup callback to save model at fixed intervals
        # callback = CheckpointCallback(save_freq=CHECKPOINT_FREQ, save_path=checkpoint_path, name_prefix=name)
        # model = PPO(
        #     CnnPolicy,
        #     env,
        #     verbose=VERBOSE,
        #     device=device.idx,
        #     batch_size=BATCH_SIZE,
        #     seed=RNG_SEED * 7,
        #     tensorboard_log=log_path,
        #     policy_kwargs=dict(features_extractor_class=VariableBenchmark, features_extractor_kwargs=dict(layers=layers)),
        # )
        #
        # with open(f"{checkpoint_path}/cnn_config", "w") as file:
        #     file.write(f"{name}\n")
        #     file.write(str(model.policy.features_extractor.cnn))
        #     print(model.policy.features_extractor.cnn)

        # model.learn(NUM_STEPS, callback=callback, tb_log_name=name)
        # model.save(os.path.join(checkpoint_path, "final.zip"))

        # env = make_atari_env(ENV_NAME)
        # reward_mean, reward_std = evaluate_policy(model, env)
        reward_mean, reward_std = random.random() * MAX_FITNESS[0], random.random() * (MAX_FITNESS[0] / 10)

        results[ind_2_str(individual)] = {"name": name, "mean": reward_mean, "std": reward_std, "round": round}
    return 0.0


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
    a = a.copy()
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

    return a


def main():
    os.makedirs(BASE_CHECKPOINT_PATH, exist_ok=True)
    os.makedirs(BASE_LOG_PATH, exist_ok=True)

    # tensorboard = Popen(f"tensorboard --logdir={BASE_LOG_PATH} --port={TENSORBOARD_PORT}")
    # open_new_tab(f"http://localhost:{TENSORBOARD_PORT}")

    # set new seed and record initial rng state for reproducibility
    # TODO: random.seed(RNG_SEED)
    # TODO: torch.manual_seed(RNG_SEED)
    # torch_state = torch.get_rng_state()
    # random_state = random.getstate()

    with Manager() as manager:
        queue = manager.Queue()
        results = manager.dict()
        rnd = 0

        creator.create("FitnessMax", base.Fitness, weights=MAX_FITNESS)
        creator.create("Individual", list, fitness=creator.FitnessMax)

        toolbox = base.Toolbox()
        toolbox.register("layer_output", random_power_of_2, LAYER_MIN_POWER, LAYER_MAX_POWER)
        toolbox.register("individual", tools.initCycle, creator.Individual, (toolbox.layer_output,), n=N_CYCLES)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        # TODO: We can get data back from the multiprocs, assuming they evaluate fine for now the issue is
        #  1: Cleaner to/from conversion of repr strings
        #  2: Correctly mapping individual objects used by deap to str reperesntations
        #  3: Ensuring the evaluation function is called, which it does not seem to be
        def fitness(individual: List[int]) -> float:
            index = benchmark_name(construct_layers(individual))
            assert results[index] is not None
            return results[index]["mean"]

        toolbox.register("fitness", fitness)
        toolbox.register("select", tools.selTournament, tournsize=3)

        evaluations = dict()
        population = toolbox.population(POPULATION_SIZE)
        for individual in population:
            key = ind_2_str(individual)
            evaluations[key] = {"Individual": individual}

        # we can't pickle lists, so we'll have to convert to a tuple and back
        # population = [(i[0], i[1], i[2]) for i in population]

        print(f"Running with {torch.cuda.device_count()} GPUs")

        done = False
        while not done:
            for individual in population:
                queue.put(ind_2_str(individual))

            processes = [Process(target=train, args=(idx, rnd, queue, results)) for idx in range(torch.cuda.device_count())]

            [p.start() for p in processes]
            [p.join() for p in processes]

            for individual in population:
                index = ind_2_str(individual)
                print(index, results[index])

            best = tools.selBest(population, int(len(population) * 0.2))
            print([(ind, ind.fitness.values) for ind in best])

            rnd += 1

            # torch.set_rng_state(torch.ByteTensor(torch_state))
            # random.setstate(random_state)

            done = True


if __name__ == "__main__":
    main()
