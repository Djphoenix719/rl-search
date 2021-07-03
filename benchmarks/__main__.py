import queue
from multiprocessing import Manager
from multiprocessing import Process
from multiprocessing import Queue
from typing import Dict
from typing import List
from typing import Tuple

from deap import base
from deap import tools
from deap.algorithms import varOr
from stable_baselines3.common.utils import set_random_seed

from benchmarks.crossover import mutate
from benchmarks.evaluate import evaluate
from benchmarks.individual import Individual
from benchmarks.misc_util import print_banner
from benchmarks.random_util import random_layer
from benchmarks.random_util import random_power_of_2
from benchmarks.settings import *


def evaluate_invalid(
    population: List[Individual],
    eval_queue: Queue,
    eval_cache: Dict[
        str,
        Tuple[
            float,
        ],
    ],
) -> None:
    eval_count = len(eval_cache)
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    [eval_queue.put(ind) for ind in invalid_ind]

    assert len(invalid_ind) == eval_queue.qsize(), "Some individuals were not inserted into the queue"

    gpu_count = torch.cuda.device_count()

    if gpu_count > 1:
        processes = [
            Process(
                target=worker,
                args=(
                    eval_queue,
                    eval_cache,
                    torch.device(idx),
                ),
            )
            for idx in range(gpu_count)
        ]

        print(f"Created {len(processes)} processes")
        assert len(processes) == gpu_count, "Did not create all gpu processes"
        print(f"Queued {eval_queue.qsize()} individuals for evaluation")

        [process.start() for process in processes]
        [process.join() for process in processes]

        print_banner(f"Done evaluating batch\n{len(eval_cache)} individual(s) evaluated\n{len(eval_cache) - eval_count} new individual(s) evaluated")
    else:
        worker(eval_queue, eval_cache, device=torch.device(0))

    for ind in invalid_ind:
        encoding = ind.encode()
        ind.fitness.values = eval_cache[encoding]


def worker(
    eval_queue: Queue,
    eval_cache: Dict[
        str,
        Tuple[
            float,
        ],
    ],
    device: torch.device,
) -> None:
    os.chdir(ROOT_PATH)
    while not eval_queue.empty():
        try:
            ind: Individual = eval_queue.get(timeout=1)
        except queue.Empty:
            return

        encoding = ind.encode()

        if encoding in eval_cache:
            continue

        print(f"Evaluating {encoding} on {device.type}:{device.index}")

        results = evaluate(ind, device)
        eval_cache[encoding] = results


def main():
    os.makedirs(BASE_CHECKPOINT_PATH, exist_ok=True)
    os.makedirs(BASE_LOG_PATH, exist_ok=True)

    # set new seed and record initial rng state for reproducibility
    set_random_seed(RANDOM_SEED)

    toolbox = base.Toolbox()
    hof = tools.HallOfFame(maxsize=N_BEST)
    toolbox.register("output", random_power_of_2, OUTPUT_MIN_POWER, OUTPUT_MAX_POWER)
    toolbox.register("layers", tools.initCycle, container=list, seq_func=(random_layer,), n=N_LAYERS)
    toolbox.register("individual", Individual, toolbox.output, toolbox.layers)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", tools.cxUniform, indpb=0.5)
    toolbox.register("mutate", mutate)

    print(f"Running with {torch.cuda.device_count()} GPUs")

    population = toolbox.population(POPULATION_SIZE)

    assert N_BEST > 0, "mu (top individuals) must be > 0"
    assert N_CHILDREN > 0, "lambda (children per generation) must be > 0"

    with Manager() as manager:
        eval_cache = manager.dict()
        eval_queue = manager.Queue()

        # Evaluate the individuals with an invalid fitness
        evaluate_invalid(population, eval_queue, eval_cache)

        assert eval_queue.qsize() == 0

        hof.update(population)

        # Begin the generational process
        for gen in range(1, N_GEN + 1):
            print_banner(f"Generation {gen}")

            # Vary the population
            offspring = varOr(population, toolbox, N_CHILDREN, CROSSOVER_PROB, MUTATION_PROB)

            print(f"Produced {len(offspring)} children")

            # Evaluate the individuals with an invalid fitness
            evaluate_invalid(offspring, eval_queue, eval_cache)

            # Update the hall of fame with the generated individuals
            hof.update(offspring)

            # Select the next generation population
            population[:] = tools.selTournament(population + offspring, N_BEST, tournsize=int(len(population + offspring) * 0.2))

    print_banner("Hall of Fame")
    for ind in hof:
        print(ind.fitness, ind)

    print_banner("Resulting Population")
    for ind in population:
        print(ind.fitness, ind)

    # TODO: Email self when run is complete


if __name__ == "__main__":
    main()
