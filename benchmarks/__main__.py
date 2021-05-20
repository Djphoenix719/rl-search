import random

from deap import base
from deap import tools
from deap import algorithms

from benchmarks.crossover import mutate
from benchmarks.evaluate import mock_evaluate
from benchmarks.individual import Individual
from benchmarks.random_util import random_layer

from benchmarks.settings import *


def main():
    os.makedirs(BASE_CHECKPOINT_PATH, exist_ok=True)
    os.makedirs(BASE_LOG_PATH, exist_ok=True)

    # set new seed and record initial rng state for reproducibility
    random.seed(RNG_SEED)
    torch.manual_seed(RNG_SEED + 1)

    toolbox = base.Toolbox()
    hof = tools.HallOfFame(maxsize=N_BEST)
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

    mu = len(population) // 10  # The number of individuals to select for the next generation
    lambda_ = POPULATION_SIZE - mu  # The number of children to produce at each generation

    print("mu", mu)
    print("lambda_", lambda_)

    print("len(population)", len(population))

    assert mu != 0, "mu is equal to zero"
    assert lambda_ != 0, "lambda is equal to 0"

    # TODO: Clean up code. Extract settings to separate file.
    #   Inline genetic algorithm for modifiability
    #   Create worker threads to consume a global job queue, once the code is inlined that is easier

    population, logbook = algorithms.eaMuPlusLambda(
        population, toolbox, mu=mu, lambda_=lambda_, cxpb=0.5, mutpb=0.05, ngen=1, halloffame=hof, verbose=VERBOSE > 0
    )

    for ind in population:
        print(ind)


if __name__ == "__main__":
    main()
