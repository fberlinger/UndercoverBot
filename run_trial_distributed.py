import matplotlib
import math
import numpy as np
import pickle
import time
from threading import Thread
from multiprocessing import Pool

from interaction import Interaction
from environment import Environment
from DelightFish import Fish
from channel import Channel
from observer import Observer
from discriminator import Classifier
from fitness import Fitness
from optimizer import Optimizer

from turing_learning import test_simulation
from utils import generate_distortion, generate_fish, generate_replica_fish, generate_all_fish, run_simulation

""" Use this file to run a semi-distributed algorithm for learning
fish behavior
"""

# Set up some decided constants that could be changed not in the context of learning
conn_threshold = 100
run_time = 15
total_fish = 25
k_ar = 0.003
max_speed = 9
arena_size = 300


# To do Turing Learning, we need to run a full simulation, termed test
# for both real and fake fish. The fake fish use the weights, The real
# fish do not.
def run_full_test(weights, real = False):
    arena_center = arena_size / 2.0
    initial_spread = 20
    fish_pos = initial_spread * np.random.rand(total_fish, 2) + arena_center - initial_spread / 2.0
    clock_freqs = 1
    verbose = False

    distortion = generate_distortion(type='none', n=arena_size)
    environment = Environment(
        node_pos=fish_pos,
        distortion=distortion,
        prob_type='binary',
        noise_magnitude=0,
        conn_thres=conn_threshold,
        verbose=verbose
    )
    interaction = Interaction(environment, verbose=verbose)
    channel = Channel(environment)

    # Have all real or all fake
    if real:
        n_fish = total_fish
        n_replica_fish = 0
    else:
        n_fish = 0
        n_replica_fish = total_fish

    fish = generate_all_fish(
        n_fish=n_fish,
        n_replica_fish= n_replica_fish,
        channel=channel,
        interaction=interaction,
        k_coh = 0,
        k_ar = k_ar,
        alpha = 40,
        weights = weights,
        lim_neighbors=[0, math.inf],
        neighbor_weights=1.0,
        fish_max_speeds=max_speed,
        clock_freqs=clock_freqs,
        verbose=verbose
    )
    channel.set_nodes(fish)

    observer = Observer(fish=fish, environment=environment, channel=channel)
    fish_matrix = test_simulation(fish=fish, observer=observer, run_time=run_time)
    return fish_matrix


# Before Learning, we initalize evolution parameters. These broadly are
# the number of generations to run the algorithm, the population size
# and the number of weights we want to learn, which is dependen on the
# network architecture for the fish and the classifier
opt = Optimizer()
pop_size = 100
num_generations = 200
opt.init_model(21, pop_size)
opt.init_classifier(48, pop_size)

# We could also read in the optimizer's saved state and continue
# learning from a previous trial

# with open('norm_test_dist5-radii-.pkl', 'rb') as f:
#     opt = pickle.load(f)

# The pool packages provides an easy way to
# quickly parallelize the funning of a function (here my simulations).
# pool.map will map the function on the set number of processes. This
# significantly increased the speed of learning. My computer could do
# 16 processes, but at that point the speed up was not much more and
# my other programs were slower. 8 processes also allows me to run
# ~5 trials at once and still use my (slower then) computer
pool = Pool(processes = 8)
for i in range(num_generations):
    print("Gen {}".format(i))
    # In a generation, we get new weights, then run simulations and
    # collect data regarding those fish
    model_weights = opt.get_model_weights()
    classifier_weights = opt.get_classifier_weights()
    ideal_model = run_full_test(None, real = True)
    start_time = time.time()
    replica_models = pool.map(run_full_test, model_weights)
    replica_models.insert(0, ideal_model)
    end_time = time.time()
    print("All trials took {} seconds".format(end_time - start_time))

    # The distributed simulation framework means very occasionaly one trial
    # will hit one more or fewer clock cycles and not have the right sized data.
    # This mostly happens if my computer sleeps. In this case, we just try
    # agin with the same weights, though it does cound as a generation.
    try:
        all_trials = np.stack(replica_models)
    except ValueError:
        print("error")
        continue
    start_time = time.time()

    # We initialize and run the classifiers on all data collected. This
    # could also be parallelized for a small performance improvement
    total_classifiers = [Classifier(id = 1, weights = weights).classify_all_models(all_trials) for weights in classifier_weights]
    fitness_scorer = Fitness(total_classifiers)
    end_time = time.time()
    print("scoring took {} seconds".format(end_time - start_time))
    class_scores = fitness_scorer.score_classifiers()
    model_scores = fitness_scorer.score_models()
    print(class_scores)
    print(model_scores)

    # This tells the optimizer (the evolutinary algorithm) the fitness
    # scores of all current agents and classifiers. The optimizer than
    # updates the weights for the next use
    opt.give_model_scores(model_scores)
    opt.give_classifier_scores(class_scores)

    # Saving state every ten trials allows us to go abck and look at
    # learned behavior over the generations
    if (i % 10 == 0):
        filename = "norm_test_dist5-iter{}-radii-only.pkl".format(i)
        with open(filename, "wb") as output:
            pickle.dump(opt, output, pickle.HIGHEST_PROTOCOL)

        # also save scores
        filename = "norm_test_dist5-iter{}-scores-radii-only.pkl".format(i)
        with open(filename, "wb") as output:
            pickle.dump((class_scores, model_scores), output, pickle.HIGHEST_PROTOCOL)
pool.close()
pool.join()

# save final weights and scores to file for evalution.
with open("norm_test_dist5-radii-only-end.pkl", 'wb') as output:
    pickle.dump(opt, output, pickle.HIGHEST_PROTOCOL)

with open("norm_test_dist5_scores-radii-only-end.pkl", "wb") as output:
    pickle.dump((class_scores, model_scores), output, pickle.HIGHEST_PROTOCOL)
