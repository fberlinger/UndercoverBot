import matplotlib

matplotlib.rcParams['figure.figsize'] = [12, 8]

import math
import numpy as np
from queue import Queue, PriorityQueue
import time
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

from interaction import Interaction
from environment import Environment
from DelightFish import Fish
from channel import Channel
from observer import Observer

from utils import generate_distortion, generate_fish, generate_replica_fish, generate_all_fish, run_simulation
"""
This file runs tests used to set the parameter for the Delight Fish.
When run, it will output graphs of varying values of K_ar parameter.

"""
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


def run_trial(run_time, num_fish, initial_spread, k_ar, alpha):
    """
    Run a single simulation.

    Arguments:
    run_time {int} -- Length of time to run simulation
    num_fish {int} -- Number of fish in swarm
    initial_spread {int} -- Initial spread of fish's randomly initialized positions.
        This is essentially the max diameter of the school, where we to encircle it,
        at the start of the simulation
    k_ar {float} -- Paramter used by delight fish to weight importance of each
        neighbor's contribution to final velocity
    alpha {float} -- Equilibrium distance for Delight Fish's neighbors

    Returns:
        fish_xs {list of int lists} - x positions of each fish at each timestep.
        fish_ys {list of int lists} - y positions of each fish at each timestep
        neighbor_distances {float list} -- the average distance between a fish
            and its detected neighbors across all time steps
        avg_speeds {float list} -- the average speed of all fish at each time step


    """
    run_time = run_time # in seconds
    num_fish = num_fish
    num_replica_fish = 0
    arena_size = 200
    arena_center = arena_size / 2.0
    initial_spread = initial_spread
    fish_pos = initial_spread * np.random.rand(num_fish + num_replica_fish, 2) + arena_center - initial_spread / 2.0
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

    fish = generate_all_fish(
        n_fish=num_fish,
        n_replica_fish= num_replica_fish,
        channel=channel,
        interaction=interaction,
        k_coh = 0,
        k_ar = k_ar,
        alpha = alpha,
        lim_neighbors=[0, math.inf],
        neighbor_weights=1.0,
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


def plot_fish(ax, fish_xs, fish_ys, title):
    """
    Generate a visualization of the fish in simulation

    Arguements:
    ax {matplotlib Axis} -- axis to plot the fish on
    fish_xs {list of int lists} -- list of each fish's x position over time
        indexed by fish id
    fish_ys {list of int lists} -- list of each fish's y position over time.
        Indexed by fish id
    title {string} -- Title to add to top of plot.

    """
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


    ax.set_title(title)

def plot_dist(ax, distances):
    """
    Plot the average distance between a fish and its neighbors
    over the course of a simulation.

    Arguements:
        ax {Matplotlib Axis} -- the axis to make the plot on
        distances {flot list} -- the average distance at each timestep

    """
    ax.plot(range(len(distances)), distances)
    ax.scatter(range(len(distances)), distances)
    ax.set_xlabel("Time")
    ax.set_ylabel("Mean neighbor spacing")
    ax.set_title("Mean neighbor spacing over time")

def plot_speed(ax, speeds):
    """
    Plot the average speed of a fish over the course of a simulation.

    Arguements:
        ax {Matplotlib Axis} -- the axis to make the plot on
        distances {flot list} -- the average speed at each timestep
    """
    ax.plot(range(len(speeds)), speeds)
    ax.scatter(range(len(speeds)), speeds)
    ax.set_xlabel("Time")
    ax.set_ylabel("Mean swarm speed")
    ax.set_title("Mean swarm speed over time")


def main():
    """
    Search through varying preset parameters for k_ar. You can also
    vary alpha, the goal fish distance, to see how the weight parameter's
    effectiveness changes. At conclusion, the program will output a plot
    for each inputs with the simulation visualized and average neighbor distance
    and average speed over time. There are also two plots for neighbor distance
    and average speed with data from all parameter values aggregated into
    a single graph. This graph is most useful for deciding on a final value for k_ar
    """
    _, (dist_ax, speed_ax) = plt.subplots(2,1)
    ks = [0.03, 0.01, 0.005, 0.003]
    #alphas = [0.5, 1, 2, 3.5, 4]
    fish = 25
    alpha = 40
    initial_spread = 20
    time = 20

    for k in ks:
        #for alpha in alphas:
        xs, ys, neighbors, speeds = run_trial(time, fish, initial_spread, k, alpha)

        # create figure for this trial
        fig = plt.figure(figsize=(12, 8))
        gridsize = (3, 2)
        fish_ax = plt.subplot2grid(gridsize, (0, 0), colspan=2, rowspan=2)
        trial_dist_ax = plt.subplot2grid(gridsize, (2, 0))
        trial_speed_ax = plt.subplot2grid(gridsize, (2, 1))


        title = "{} fish, {} initial spread, {} k_ar, {} time, {} alpha".format(fish, initial_spread, k, time, alpha)
        plot_fish(fish_ax, xs, ys, title)
        plot_dist(trial_dist_ax, neighbors)
        plot_speed(trial_speed_ax, speeds)

        # Add to all parameter search figure
        dist_ax.plot(range(len(neighbors)), neighbors, label = "k = {}, alpha = {}".format(k, alpha))
        speed_ax.plot(range(len(speeds)), speeds, label = "k = {}, alpha = {}".format(k, alpha))

    # add titles and formatting to stability fig
    dist_ax.set_xlabel("Time")
    dist_ax.set_ylabel("Mean neighbor spacing")
    dist_ax.legend()
    dist_ax.set_title("Spacing over time for varying values of k")

    speed_ax.set_xlabel("Time")
    speed_ax.set_ylabel("Mean swarm speed")
    speed_ax.legend()
    speed_ax.set_title("Swarm speed over time for varying values of k")

    plt.show()


if __name__ == '__main__':
    main()
