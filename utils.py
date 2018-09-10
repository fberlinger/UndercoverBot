"""Helper methods to run the experiment
"""

import numpy as np
import matplotlib.pyplot as plt
import threading

from channel import Channel
from environment import Environment
from interaction import Interaction
from fish import Fish
from observer import Observer


def generate_distortion(type='linear', n=10, show=False):
    """Generates a distortion model represented as a vector field
    """

    X, Y = np.mgrid[0:n, 0:n]
    distortion = np.zeros((n, n, 2))

    if type == 'none':
        distortion[:, :, 0] = 0
        distortion[:, :, 1] = 0
        return distortion

    elif type == 'linear':
        X_new = 1
        Y_new = 0

    elif type == 'aggregation':
        theta = np.arctan2(Y-(n-1)/2, X-(n-1)/2)
        X_new = -np.cos(theta)
        Y_new = -np.sin(theta)

    elif type == 'dispersion':
        theta = np.arctan2(Y-(n-1)/2, X-(n-1)/2)
        X_new = np.cos(theta)
        Y_new = np.sin(theta)

    elif type == 'curl':
        X_new = -(Y-(n-1)/2)
        Y_new = X-(n-1)/2

    unit_magnitude = 1/(np.sqrt(X_new**2 + Y_new**2))
    distortion[:, :, 0] = unit_magnitude * X_new
    distortion[:, :, 1] = unit_magnitude * Y_new

    if show:
        plt.quiver(X, Y, distortion[:, :, 0], distortion[:, :, 1])
        plt.show()

    return distortion


def generate_fish(
    n,
    channel,
    interaction,
    lim_neighbors,
    neighbor_weights=None,
    fish_max_speeds=None,
    clock_freqs=None,
    verbose=False,
    names=None
):
    """Generate some fish

    Arguments:
        n {int} -- Number of fish to generate
        channel {Channel} -- Channel instance
        interaction {Interaction} -- Interaction instance
        lim_neighbors {list} -- Tuple of min and max neighbors
        neighbor_weight {float|list} -- List of neighbor weights
        fish_max_speeds {float|list} -- List of max speeds
        clock_freqs {int|list} -- List of clock speeds
        names {list} -- List of names for your fish
    """

    if neighbor_weights is None:
        neighbor_weights = [1.0] * n
    elif not isinstance(neighbor_weights, list):
        neighbor_weights = [neighbor_weights] * n

    if fish_max_speeds is None:
        fish_max_speeds = [1.0] * n
    elif not isinstance(fish_max_speeds, list):
        fish_max_speeds = [fish_max_speeds] * n

    if clock_freqs is None:
        clock_freqs = [1] * n
    elif not isinstance(clock_freqs, list):
        clock_freqs = [clock_freqs] * n

    if names is None:
        names = ['Unnamed'] * n

    fish = []
    for i in range(n):
        fish.append(Fish(
            id=i,
            channel=channel,
            interaction=interaction,
            lim_neighbors=lim_neighbors,
            neighbor_weight=neighbor_weights[i],
            fish_max_speed=fish_max_speeds[i],
            clock_freq=clock_freqs[i],
            verbose=verbose,
            name=names[i]
        ))

    return fish


def init_simulation(
    clock_freq,
    single_time,
    offset_time,
    num_trials,
    final_buffer,
    run_time,
    num_fish,
    size_dist,
    center,
    spread,
    fish_pos,
    lim_neighbors,
    neighbor_weights,
    fish_max_speeds,
    noise_magnitude,
    conn_thres,
    prob_type,
    dist_type,
    verbose,
    conn_drop=1.0,
):
    """Initialize all the instances needed for a simulation

    Arguments:
        clock_freq {int} -- Clock frequency for each fish.
        single_time {float} -- Number clock cycles per individual run.
        offset_time {float} -- Initial clock offset time
        num_trials {int} -- Number of trials per experiment.
        final_buffer {float} -- Final clock buffer (because the clocks don't
            sync perfectly).
        run_time {float} -- Total run time in seconds.
        num_fish {int} -- Number of fish.
        size_dist {int} -- Distortion field size.
        center {float} -- Distortion field center.
        spread {float} -- Initial fish position spread.
        fish_pos {np.array} -- Initial fish position.
        lim_neighbors {list} -- Min. and max. desired neighbors. If too few
            neighbors start aggregation, if too many neighbors disperse!
        neighbor_weights {float} -- Distance-depending neighbor weight.
        fish_max_speeds {float} -- Max fish speed.
        noise_magnitude {float} -- Amount of white noise added to each move.
        conn_thres {float} -- Distance at which the connection either cuts off
            or starts dropping severely.
        prob_type {str} -- Probability type. Can be `binary`, `quadratic`, or
            `sigmoid`.
        dist_type {str} -- Position distortion type
        verbose {bool} -- If `true` print a lot of stuff

    Keyword Arguments:
        conn_drop {number} -- Defined the connection drop for the sigmoid
            (default: {1.0})

    Returns:
        tuple -- Quintuple holding the `channel`, `environment`, `fish,
            `interaction`, and `observer`
    """
    distortion = generate_distortion(type=dist_type, n=size_dist)
    environment = Environment(
        node_pos=fish_pos,
        distortion=distortion,
        prob_type=prob_type,
        noise_magnitude=noise_magnitude,
        conn_thres=conn_thres,
        conn_drop=conn_drop,
        verbose=verbose
    )
    interaction = Interaction(environment, verbose=verbose)
    channel = Channel(environment)

    fish = generate_fish(
        n=num_fish,
        channel=channel,
        interaction=interaction,
        lim_neighbors=lim_neighbors,
        neighbor_weights=neighbor_weights,
        fish_max_speeds=fish_max_speeds,
        clock_freqs=clock_freq,
        verbose=verbose
    )
    channel.set_nodes(fish)

    observer = Observer(
        fish=fish,
        environment=environment,
        channel=channel,
        clock_freq=clock_freq,
        fish_pos=np.copy(fish_pos)
    )
    channel.intercept(observer)

    return channel, environment, fish, interaction, observer


def run_simulation(
    fish,
    observer,
    run_time=10,
    dark=False,
    white_axis=False,
    no_legend=False,
    no_star=False
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

        print('It\'s time to say bye bye!')

        observer.stop()
        observer.plot(
            dark=dark,
            white_axis=white_axis,
            no_legend=no_legend,
            no_star=no_star
        )

    print('Please wait patiently {} seconds. Thanks.'.format(run_time))

    # Start the fish
    for f in fish:
        threading.Thread(target=f.start).start()

    threading.Thread(target=observer.start).start()

    # Ciao stops run time
    threading.Timer(run_time, stop).start()
