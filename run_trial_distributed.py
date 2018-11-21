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
run_time = 15
total_fish = 25
k_ar = 0.003
max_speed = 9
arena_size = 300



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



from threading import Thread
import time
from multiprocessing import Pool

# go through ten generations
opt = Optimizer()
pop_size = 100
num_generations = 100
opt.init_model(21, pop_size)
opt.init_classifier(46, pop_size)
pool = Pool(processes = 4)
for i in range(num_generations):
    model_weights = opt.get_model_weights()
    classifier_weights = opt.get_classifier_weights()
    ideal_model = run_full_test(None, real = True)
    #replica_models = [Simulation() for i in model_weights]
    replica_models_threads = []
    sim_in_group = 20
    start_time = time.time()
    replica_models = pool.map(run_full_test, model_weights)
    # for start in range(0, pop_size, sim_in_group):
    #     start_time = time.time()
    #     for index in range(start, min(start + sim_in_group, pop_size)):
    #         weights = model_weights[index]
    #         replica_models_threads.append(Process(target = replica_models[index].run_full_test, args = [weights]))
    #         replica_models_threads[index].start()
    #     for index in range(start, min(start + sim_in_group, pop_size)):
    #         replica_models_threads[index].join()
    #     end_time = time.time()
    #     print("finished first {} trials in generation {}".format(index, i))
    #     print("this took {} seconds".format(end_time - start_time))
    #print(replica_models)
    #replica_models = [sim.result for sim in replica_models]
    replica_models.insert(0, ideal_model)
    end_time = time.time()
    print("All trials took {} seconds".format(end_time - start_time))
    try:
        all_trials = np.stack(replica_models)
    except ValueError: ## occasionally a model runs slighly different number of iterations, in which case just move on
        print("error")
        for model in replica_models:
            print(replica_models)
            print(model.shape)
        continue
    start_time = time.time()
    total_classifiers = [Classifier(id = 1, weights = weights).classify_all_models(all_trials) for weights in classifier_weights]
    fitness_scorer = Fitness(total_classifiers)
    end_time = time.time()
    print("scoring took {} seconds".format(end_time - start_time))
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
pool.close()
pool.join()
# from multiprocessing import Process, Queue

# def run_simulation(simulations, results):
#     while simulations.qsize() > 0:
#         simulation_params = simulations.get()
#         # run simulation
#         results.put(simulation_result)
#         simulations.task_done()

# if __name__ == '__main__':
#     simulations_to_run = Queue()
#     simulations_to_run.put({}) # simulation parameters go in this dict, add all simulations, one per line (could be done in a loop, with a list of dicts)
#     results = Queue()
#     for i in range(8): #number processes you want to run
#         p = Process(target=run_simulation, args=(simulations_to_run, results))
#         p.start()

#     simulations_to_run.join()
#     # now, all results shoud be in the results Queue

# save final weights to file
with open("test_dist1.pkl", 'wb') as output:  # Overwrites any existing file.
    pickle.dump(opt, output, pickle.HIGHEST_PROTOCOL)