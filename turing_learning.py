import numpy as np
import matplotlib.pyplot as plt
import threading

from channel import Channel
from environment import Environment
from replica_fish import ReplicaFish
from DelightFish import Fish
from observer import Observer

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


    print('Please wait patiently {} seconds. Thanks.'.format(run_time))

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
        single_fish = np.column_stack((observer.lin_speed[fish_index], observer.ang_speed[fish_index]))
        fish_matrixes.append(single_fish)
    return np.stack(fish_matrixes, axis = 0)

