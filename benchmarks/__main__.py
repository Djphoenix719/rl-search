import random
from typing import List

from deap import base
from deap import tools
from deap.algorithms import varOr
from deap.base import Toolbox

from benchmarks.crossover import mutate
from benchmarks.evaluate import mock_evaluate
from benchmarks.individual import Individual
from benchmarks.misc_util import print_banner
from benchmarks.random_util import random_layer

from benchmarks.settings import *


def evaluate_invalid(population: List[Individual], toolbox: Toolbox) -> None:
    """
    Evaluates individuals  in place which have an invalid fitness
    :param population:
    :param toolbox:
    :return:
    """
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit


def main():
    os.makedirs(BASE_CHECKPOINT_PATH, exist_ok=True)
    os.makedirs(BASE_LOG_PATH, exist_ok=True)

    # set new seed and record initial rng state for reproducibility
    random.seed(RANDOM_SEED)
    torch.manual_seed(TORCH_SEED)

    toolbox = base.Toolbox()
    hof = tools.HallOfFame(maxsize=N_HOF)
    toolbox.register("individual", tools.initCycle, Individual, (random_layer,), n=N_CYCLES)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", mock_evaluate)
    toolbox.register("mate", tools.cxUniform, indpb=0.5)
    toolbox.register("mutate", mutate)

    print(f"Running with {torch.cuda.device_count()} GPUs")

    population = toolbox.population(POPULATION_SIZE)

    assert N_BEST > 0, "mu (top individuals) must be > 0"
    assert N_CHILDREN > 0, "lambda (children per generation) must be > 0"

    # TODO: Clean up code.
    #   Create worker threads to consume a global job queue, once the code is inlined that is easier

    # Evaluate the individuals with an invalid fitness
    evaluate_invalid(population, toolbox)

    hof.update(population)

    # Begin the generational process
    for gen in range(1, N_GEN + 1):
        print_banner(f"Generation {gen}")

        # Vary the population
        offspring = varOr(population, toolbox, N_CHILDREN, CROSSOVER_PROB, MUTATION_PROB)

        print(f"Produced {len(offspring)} children")

        # Evaluate the individuals with an invalid fitness
        evaluate_invalid(offspring, toolbox)

        # Update the hall of fame with the generated individuals
        hof.update(offspring)

        # Select the next generation population
        population[:] = tools.selTournament(population + offspring, N_BEST, tournsize=int(len(population + offspring) * 0.2))

        print(f"Population now has {len(population)} individuals")

    # population, logbook = algorithms.eaMuPlusLambda(
    #     population, toolbox, mu=mu, lambda_=lambda_, cxpb=0.5, mutpb=0.05, ngen=1, halloffame=hof, verbose=VERBOSE > 0
    # )

    print_banner("Hall of Fame")
    for ind in hof:
        print(ind.fitness, ind)

    print_banner("Resulting Population")
    for ind in population:
        print(ind.fitness, ind)


if __name__ == "__main__":
    main()
