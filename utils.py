"""Helper methods to run the experiment
"""

import numpy as np
import matplotlib.pyplot as plt
import threading

from channel import Channel
from environment import Environment
from interaction import Interaction
from replica_fish import ReplicaFish
from DelightFish import Fish
from observer import Observer


def generate_distortion(type='linear', n=10, show=False):
    """Generates a distortion model represented as a vector field
    Distortions are ignored during Turing Learning
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
    k_coh,
    k_ar,
    alpha,
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
            k_coh = k_coh,
            k_ar = k_ar,
            alpha = alpha,
            lim_neighbors=lim_neighbors,
            neighbor_weight=neighbor_weights[i],
            fish_max_speed=fish_max_speeds[i],
            clock_freq=clock_freqs[i],
            verbose=verbose,
            name=names[i]
        ))

    return fish

def generate_replica_fish(
    n_fish,
    channel, 
    interaction,
    weights,
    fish_max_speeds=None,
    clock_freqs=None,
    verbose=False,
    names=None
):
    """Generate some fish

    Arguments:
        n_fish {int} -- Number of fish to generate
        channel {Channel} -- Channel instance
        interaction {Interaction} -- Interaction instance
        weights {float|list} -- List of weights for NN controller
        fish_max_speeds {float|list} -- List of max speeds for fish
        clock_freqs {int|list} -- List of clock speeds
        names {list} -- List of names for your fish
    Returns:
        {ReplicaFish|list} -- List of initialized ReplicaFish objects 
    """

    if fish_max_speeds is None:
        fish_max_speeds = [1.0] * n_fish

    elif not isinstance(fish_max_speeds, list):
        fish_max_speeds = [fish_max_speeds] * n_fish

    if clock_freqs is None:
        clock_freqs = [1] * n_fish

    elif not isinstance(clock_freqs, list):
        clock_freqs = [clock_freqs] * n_fish

    if names is None:
        names = ['Unnamed'] * n_fish

    fish = []

    for i in range(0, n_fish):
        fish.append(ReplicaFish(
            id=i,
            channel=channel,
            interaction=interaction,
            weights = weights,
            fish_max_speed=fish_max_speeds[i],
            clock_freq=clock_freqs[i],
            name=names[i],
            verbose=verbose
        ))
    return fish

def generate_all_fish(
    n_fish,
    n_replica_fish,
    channel,
    interaction,
    k_coh,
    k_ar,
    alpha,
    lim_neighbors,
    weights = [1],
    neighbor_weights=None,
    fish_max_speeds=None,
    clock_freqs=None,
    verbose=False,
    names=None
):
    """Generate both replica and regular fish

    Arguments:
        n_fish {int} -- Number of ideal fish to generate
        n_replica_fish {int} -- Number of replica fish to generate
        channel {Channel} -- Channel instance
        interaction {Interaction} -- Interaction instance
        k_coh {float} -- Parameter to Delight Fish
        k_ar {float} -- Weighting of neighbors in Delight Fish
        alpha {int} -- Goal distance from neighbor for Delight Fish
        lim_neighbors {list} -- Tuple of min and max neighbors
        weights {float|list} -- List of weights for replica fish learned function
        neighbor_weight {float|list} -- List of neighbor weights
        fish_max_speeds {float|list} -- List of max speeds
        clock_freqs {int|list} -- List of clock speeds
        names {list} -- List of names for your replica fish
    """
    n = n_fish + n_replica_fish
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

    all_fish = []
    for i in range(n_fish):
        all_fish.append(Fish(
            id=i,
            channel=channel,
            interaction=interaction,
            k_coh = k_coh,
            k_ar = k_ar,
            alpha = alpha,
            lim_neighbors=lim_neighbors,
            neighbor_weight=neighbor_weights[i],
            fish_max_speed=fish_max_speeds[i],
            clock_freq=clock_freqs[i],
            verbose=verbose,
            name=names[i]
        ))

    for i in range(n_fish, n_fish + n_replica_fish):
        all_fish.append(ReplicaFish(
            id=i,
            channel=channel,
            interaction=interaction,
            weights = weights,
            fish_max_speed=fish_max_speeds[i],
            clock_freq=clock_freqs[i],
            name=names[i],
            verbose=verbose
        ))

    return all_fish


def init_simulation(
    clock_freq,
    single_time,
    offset_time,
    num_trials,
    final_buffer,
    run_time,
    num_fish,
    num_replica_fish,
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
        num_replica_fish {int} -- Number of replica fish.
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
    replica_fish = generate_replica_fish(
        n=num_fish,
        channel=channel,
        interaction=interaction,
        lim_neighbors=lim_neighbors,
        neighbor_weights=neighbor_weights,
        fish_max_speeds=fish_max_speeds,
        clock_freqs=clock_freq,
        verbose=verbose
    )
    all_fish = fish + replica_fish
    channel.set_nodes(all_fish)


    observer = Observer(
        fish=all_fish,
        environment=environment,
        channel=channel,
        clock_freq=clock_freq,
        fish_pos=np.copy(fish_pos)
    )
    channel.intercept(observer)

    return channel, environment, all_fish, interaction, observer


def run_simulation(
    fish,
    observer,
    run_time=10,
    dark=False,
    white_axis=False,
    no_legend=False,
    no_star=False,
    show_dist_plot=False,
    plot=True
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

        #print('It\'s time to say bye bye!', flush = True)

        observer.stop()
        if plot:
            observer.plot(
                dark=dark,
                white_axis=white_axis,
                no_legend=no_legend,
                no_star=no_star,
                show_dist_plot=show_dist_plot
            )



    #print('Please wait patiently {} seconds. Thanks.'.format(run_time))

    # Start the fish
    for f in fish:
        threading.Thread(target=f.start).start()

    observer_thread = threading.Thread(target=observer.start)
    observer_thread.start()

    # Ciao stops run time
    threading.Timer(run_time, stop).start()
    if not plot:
        observer_thread.join()

# 21 categorical colors. Used for plotting
colors = [
    [230/255, 25/255, 75/255, 1.0],
    [60/255, 180/255, 75/255, 1.0],
    [255/255, 225/255, 25/255, 1.0],
    [0/255, 130/255, 200/255, 1.0],
    [245/255, 130/255, 48/255, 1.0],
    [145/255, 30/255, 180/255, 1.0],
    [70/255, 240/255, 240/255, 1.0],
    [240/255, 50/255, 230/255, 1.0],
    [210/255, 245/255, 60/255, 1.0],
    [250/255, 190/255, 190/255, 1.0],
    [0/255, 128/255, 128/255, 1.0],
    [230/255, 190/255, 255/255, 1.0],
    [170/255, 110/255, 40/255, 1.0],
    [255/255, 250/255, 200/255, 1.0],
    [128/255, 0/255, 0/255, 1.0],
    [170/255, 255/255, 195/255, 1.0],
    [128/255, 128/255, 0/255, 1.0],
    [255/255, 215/255, 180/255, 1.0],
    [0/255, 0/255, 128/255, 1.0],
    [128/255, 128/255, 128/255, 1.0],
    [0/255, 0/255, 0/255, 1.0],
]


def run_replica_trial(run_time, num_fish, initial_spread, weights):
    """
    Run a single simulation for the purposes of illustrating it. 

    Arguments:
    run_time {int} -- Length of time to run simulation
    num_fish {int} -- Number of fish in swarm, must be replicas
    initial_spread {int} -- Initial spread of fish's randomly initialized positions.
        This is essentially the max diameter of the school, where we to encircle it,
        at the start of the simulation
    weights {float|list} -- list of weights governing fish behavior (replica fish)

    Returns:
        fish_xs {list of int lists} - x positions of each fish at each timestep.
        fish_ys {list of int lists} - y positions of each fish at each timestep
        neighbor_distances {float list} -- the average distance between a fish
            and its detected neighbors across all time steps
        avg_speeds {float list} -- the average speed of all fish at each time step
    """

    run_time = run_time # in seconds
    num_fish = num_fish
    arena_size = 200
    arena_center = arena_size / 2.0
    initial_spread = initial_spread
    fish_pos = initial_spread * np.random.rand(num_fish, 2) + arena_center - initial_spread / 2.0
    clock_freqs = 1
    verbose = False

    distortion = generate_distortion(type='none', n=arena_size)
    environment = Environment(
        node_pos=fish_pos,
        distortion=distortion,
        prob_type='binary',
        noise_magnitude=0,
        conn_thres=100,
        verbose=verbose
    )
    interaction = Interaction(environment, verbose=verbose)
    channel = Channel(environment)

    fish = generate_replica_fish(
        n_fish=num_fish,
        channel=channel,
        interaction=interaction,
        weights=weights,
        fish_max_speeds=9,
        clock_freqs=clock_freqs,
        verbose=verbose
    )
    channel.set_nodes(fish)

    observer = Observer(fish=fish, environment=environment, channel=channel)
    run_simulation(fish=fish, observer=observer, run_time=run_time,
        dark=True, white_axis=False, no_legend=True, no_star=False,
        show_dist_plot=True, plot=False)

    fish_xs = observer.x
    fish_ys = observer.y
    neighbor_distances = observer.avg_dist
    avg_speeds = observer.avg_speed
    return fish_xs, fish_ys, neighbor_distances, avg_speeds


def plot_fish(fish_xs, fish_ys, filename):

    """
    Generate a visualization of the fish in simulation and save the file

    Arguments:
    fish_xs {list of int lists} -- list of each fish's x position over time
        indexed by fish id
    fish_ys {list of int lists} -- list of each fish's y position over time.
        Indexed by fish id
    filename {string} -- file in which to save visualization
    """

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    num_fish = len(fish_xs)
    for i in range(num_fish):
            c = colors[i%20]
            if i != 0 and not i % 20:
                c = [1.0, 1.0, 1.0, 1.0]

            # Plot fish trajectories
            ax.plot(fish_xs[i], fish_ys[i], c=c,
                linewidth=2.0, alpha=0.4)
            ax.scatter(fish_xs[i], fish_ys[i], c=c,
                marker='o', alpha=0.2)

            # plot fish start
            ax.scatter(fish_xs[i][0], fish_ys[i][0], c=c,
                        marker='>', s=200, alpha=0.5)

            # plot fish final
            ax.scatter(fish_xs[i][-1], fish_ys[i][-1], c=c,
                        marker='s', s=200,alpha=1, zorder = 100)

    # format black background, white axis
    ax.set_facecolor((0, 0, 0))
    ax.spines['top'].set_color('black')
    ax.spines['right'].set_color('black')

    # save visualization
    plt.savefig(filename)


def plot_dist(distances, filename):
    """
    Plot the average distance between a fish and its neighbors
    over the course of a simulation and save this graph

    Arguements:
        distances {flot list} -- the average distance at each timestep
        filename {string} -- name of file in which to save graph
    """
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    ax.plot(range(len(distances)), distances)
    ax.scatter(range(len(distances)), distances)
    ax.set_xlabel("Time")
    ax.set_ylabel("Mean neighbor spacing")

    plt.savefig(filename)

def plot_speed(speeds, filename):
    """
    Plot the average speed of a fish over the course of a simulation.

    Arguments:
        distances {float | list} -- the average speed at each timestep
        filename {string} -- name of file in which to save graph
    """
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    ax.plot(range(len(speeds)), speeds)
    ax.scatter(range(len(speeds)), speeds)
    ax.set_xlabel("Time")
    ax.set_ylabel("Mean swarm speed")
    plt.savefig(filename)


def visualize_replica_fish(run_time, num_fish, trial_type, weights, root_filename):
    """
    Runs a single simulation and saves visualizations of it. Assumes all replica fish. 
    Arguments:
        run_time {int} -- Length of time to run simulation
        num_fish {int} -- Number of fish in swarm
        trial_type {string} -- Type of trial being trained. Must be aggregation or dispersion
        root_filename {string} -- preface of all filenames. Will be appended with specific visualization

    Saves a visualization of fish trajectories, neighbor distance, and speeeds
    """

    if trial_type == "dispersion":
        initial_spread = 20
    elif trial_type == "aggregation":
        initial_spread = 100
    else:
        initial_spread = 40
        print("bad trial type")

    xs, ys, distances, speeds = run_replica_trial(run_time, num_fish, initial_spread, weights)
    plot_fish(xs, ys, "{}-fish.png".format(root_filename))
    plot_dist(distances, "{}-distances.png".format(root_filename))
    plot_speed(speeds, "{}-speeds.png".format(root_filename))
