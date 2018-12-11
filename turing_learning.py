import numpy as np
import matplotlib.pyplot as plt
import threading
import math

from channel import Channel
from environment import Environment
from replica_fish import ReplicaFish
from DelightFish import Fish
from observer import Observer
from utils import generate_distortion, generate_fish, generate_replica_fish, generate_all_fish, run_simulation
from interaction import Interaction
from environment import Environment
def test_simulation(
    fish,
    observer,
    run_time=5,
):
    """Run a simulation and format data from it for use by classifiers


    Arguments:
        fish {list} -- List of fish instances
        observer {Observer} -- Observer instance

    Keyword Arguments:
        run_time {number} -- Total run time in seconds (default: {5})

    """
    def stop():
        for f in fish:
            f.stop()
        observer.stop()


    # Start the fish
    fish_threads = []
    for f in fish:
        threading.Thread(target=f.start).start()


    observer_thread = threading.Thread(target=observer.start)
    observer_thread.start()

    # Wait for the simulation to end, so data can be collected
    # from the observer
    fish_matrixes = []
    threading.Timer(run_time, stop).start()
    observer_thread.join()

    # merge each fish's linear speed, angular speed, and neighbor
    # distances into a single matrix. This will
    # utlimately be a N x (N + 1) matrix, where N is the number
    # of fish.
    for fish_index in range(observer.num_nodes):
        single_fish = np.column_stack((observer.lin_speed[fish_index],
            observer.ang_speed[fish_index],
            observer.neighbor_distances[fish_index]))
        fish_matrixes.append(single_fish)
    return np.stack(fish_matrixes, axis = 0)



def run_full_test(weights,
    conn_threshold,
    run_time,
    total_fish,
    k_ar,
    max_speed,
    arena_size,
    real = False):
    """
    Start and run a simulation and collect data for Turing Learning. This function
    initializes other objects needed for simulation, rather than just
    starting and stopping everything

    Arguments:
        weights {float|list} --- weights used by Neural Network in imposter fish
        conn_threshold {float} -- Distance at which fish can no longer detect other fish
        run_time {int} -- Length of time to run simulation
        total_fish {int} -- Number of fish to be in the school
        k_ar {float} -- parameter for delight fish
        max_speed {float} -- Max speed of a single fish
        arena_size {int} -- boundaries of arena to create distortion
        real {bool} -- Should this test have real or imposter fish (default : {False})
    """
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


