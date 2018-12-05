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
    """Start the simulation.


    Arguments:
        fish {list} -- List of fish instances
        observer {Observer} -- Observer instance

    Keyword Arguments:
        run_time {number} -- Total run time in seconds (default: {10})
        dark {bool} -- If `True` plot a dark chart (default: {False})
        white_axis {bool} -- If `True` plot white axes (default: {False})
        no_legend {bool} -- If `True` do not plot a legend (default: {False})
        no_star {bool} -- If `True` do not plot a star (default: {False})
    """
    def stop():
        for f in fish:
            f.stop()
        observer.stop()


    #print('Please wait patiently {} seconds. Thanks.'.format(run_time))

    # Start the fish
    fish_threads = []
    for f in fish:
        threading.Thread(target=f.start).start()


    observer_thread = threading.Thread(target=observer.start)
    observer_thread.start()

    # Ciao stops run time
    fish_matrixes = []
    threading.Timer(run_time, stop).start()
    observer_thread.join()
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


