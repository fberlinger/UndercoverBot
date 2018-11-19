import matplotlib
import math
import numpy as np
import pickle

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

conn_threshold = 100
run_time = 3
total_fish = 25
k_ar = 0.003
max_speed = 9
arena_size = 300


def run_full_test(weights, result, index, real = False):
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
    if not real:
        result[index] = fish_matrix
    return fish_matrix



from threading import Thread
from time import sleep

# go through ten generations
opt = Optimizer()
opt.init_model(21, 50)
opt.init_classifier(46, 50)
for i in range(100):
    model_weights = opt.get_model_weights()
    classifier_weights = opt.get_classifier_weights()
    ideal_model = run_full_test(None, [], 0, real = True)
    replica_models = [{} for i in model_weights]
    replica_models_threads = []
    for i, weights in enumerate(model_weights):
        replica_models_threads.append(Thread(target = run_full_test, args = [weights, replica_models, i]))
        replica_models_threads[i].start()
        sleep(0.5)
    for thread in replica_models_threads:
        thread.join()
    replica_models.insert(0, ideal_model)
    try:
        all_trials = np.stack(replica_models)
    except ValueError: ## occasionally a model runs slighly different number of iterations, in which case just move on
        continue
    total_classifiers = [Classifier(id = 1, weights = weights).classify_all_models(all_trials) for weights in classifier_weights]
    fitness_scorer = Fitness(total_classifiers)

    class_scores = fitness_scorer.score_classifiers()
    model_scores = fitness_scorer.score_models()
    print(class_scores)
    print(model_scores)

    opt.give_model_scores(model_scores)
    opt.give_classifier_scores(class_scores)

    if (i % 10 == 0):
        filename = "test_dist1-iter{}.pkl".format(i)
        with open(filename, "wb") as output:
            pickle.dump(opt, output, pickle.HIGHEST_PROTOCOL)

# save final weights to file
with open("test_dist1.pkl", 'wb') as output:  # Overwrites any existing file.
    pickle.dump(opt, output, pickle.HIGHEST_PROTOCOL)