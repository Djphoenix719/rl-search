import os
import torch

# OUTPUT SETTINGS
VERBOSE = 2  # 0 no output, 1 info, 2 debug
ROOT_PATH = "D:/rl-search"  # root save path
BASE_CHECKPOINT_PATH = os.path.join(ROOT_PATH, "checkpoints")  # path to save checkpoints to
BASE_LOG_PATH = os.path.join(ROOT_PATH, "logs")  # path to save tensorboard logs to
TENSORBOARD_PORT = 6006  # port tensorboard should run on
SLUG_WORDS = 8  # number of words in our slugs which are used as uuids for the models

# RNG SEEDS
RANDOM_SEED = 42  # seed for random.seed
TORCH_SEED = RANDOM_SEED + 1  # seed for pytorch

# GYM & BASELINES SETTINGS
ENV_NAME = "Pong-v0"  # name of gym environment
TRAIN_STEPS = 1_000_000  # total training steps to take
EVAL_STEPS = 10_000  # steps to evaluate a trained model
CHECKPOINT_FREQ = 500_000  # interval between checkpoints
BATCH_SIZE = 64  # size of batch updates
N_ENVS = 4  # number of parallel environments to evaluate
DEVICE_TYPE = "cuda" if torch.cuda.is_available() else "cpu"  # run on cpu or cuda
LAYER_MIN_POWER, LAYER_MAX_POWER = 1, 9  # max size of output dimensions on a layer in power of 2
N_CYCLES = 3  # number of weights to generate, functionally becomes number of layers in cnn
POPULATION_SIZE = 50  # before each round, ensure this many individuals exist, less may due to selection
N_HOF = 5  # keep the global n individuals that are the best from the entire run


# GENETIC ALGORITHM SETTINGS
N_GEN = 25  # number of generations to run the algorithm for
MAX_FITNESS = (20.0,)  # max fitness a model can achieve, dependent on task
INIT_FITNESS = MAX_FITNESS  # initial fitness values, deap uses the metric of "closest to init" as best, so our init is max
CROSSOVER_PROB = 0.5  # probability of mating two individuals
MUTATION_PROB = 0.5  # probability of mutating an individual
N_BEST = 5  # keep the top n individuals from a round
N_CHILDREN = 20  # number of children to produce each generation
