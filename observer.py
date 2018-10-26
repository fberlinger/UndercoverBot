import numpy as np
from queue import Queue, PriorityQueue
import time
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

from eventcodes import (
    INFO_EXTERNAL, INFO_INTERNAL, START_HOP_COUNT, HOP_COUNT,
    START_LEADER_ELECTION, LEADER_ELECTION
)

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


class Observer():
    """The god-like observer keeps track of the fish movement for analysis.
    """
    def __init__(
        self,
        environment,
        fish,
        channel,
        clock_freq=1,
        fish_pos=None,
        verbose=False
    ):
        """Create a god-like observer!

        We always wanted to be god! This will be marvelous!

        Arguments:
            environment {Environment} -- Environment to observer
            fish {list} -- Fish instances to observe
            channel {Channel} -- Channel instance to observer

        Keyword Arguments:
            clock_freq {number} -- Same clock frequency as the
                fish (default: {1})
            fish_pos {np.array} -- Initial fish positions (default: {None})
            verbose {bool} -- If `true` log out some stuff (default: {False})
        """
        self.environment = environment
        self.fish = fish
        self.channel = channel
        self.clock_freq = clock_freq
        self.object = np.zeros((2, 1))
        self.fish_pos = fish_pos

        self.clock_speed = 1 / self.clock_freq
        self.clock = 0

        self.num_nodes = self.environment.node_pos.shape[0]
        self.x = []
        self.y = []
        self.c = []
        self.status = []
        self.avg_dist = []
        self.avg_speed = []
        self.reset = False

        self.node_colors = []
        for i in range(self.num_nodes):
            ii = i % 20
            self.node_colors.append(colors[ii])
            self.x.append([])
            self.y.append([])
            self.status.append([])

        self.is_started = False

        self.track_info = None
        self.not_saw_info = 0

        self.transmissions = Queue()
        self.instructions = PriorityQueue()

        self.study_info_consistency = False
        self.study_hop_count = False
        self.study_leader_election = False
        self.study_data = []

        self.study_hop_count = False
        self.track_hop_count = False
        self.track_hop_count_num_events = 0
        self.track_hop_count_started = False
        self.hop_count_source_id = -1

        self.track_leader_election = False
        self.track_leader_election_num_events = 0

        self.is_instructed = False

        self.verbose = verbose

    def instruct(
        self,
        event,
        rel_clock=0,
        fish_id=None,
        pos=np.zeros(2,),
        fish_all=False
    ):
        """Make the observer instruct the fish swarm.

        This will effectively trigger an event in the fish environment, like an
        instruction or some kind of obstacle.

        Arguments:
            event {*} -- Some event instance.

        Keyword Arguments:
            rel_clock {number} -- Number of relative clock cycles from now when
                to broadcast the event (default: {0})
            fish_id {int} -- If not `None` directly put the event on the
                fish with this id. (default: {None})
            pos {np.array} -- Imaginary event position. Used to determine the
                probability that fish will hear the
                event. (default: {np.zeros(2,)})
            fish_all {bool} -- If `true` all fish will immediately receive the
                event, i.e., no probabilistic event anymore. (default: {False})
        """
        self.instructions.put((
            self.clock + rel_clock, event, fish_id, pos, fish_all
        ))
        self.object = pos
        self.is_instructed = True

        if fish_id is not None:
            self.object = self.environment.node_pos[fish_id]

    def start(self):
        """Start the process

        This sets `is_started` to true and invokes `run()`.
        """
        self.is_started = True
        self.run()

    def stop(self):
        """Stop the process

        This sets `is_started` to false.
        """
        self.is_started = False

        if self.track_hop_count:
            self.study_data[0].append(self.track_hop_count_num_events)
            self.study_data[1].append(
                self.fish[self.hop_count_source_id].hop_count
            )

        if self.track_leader_election:
            self.study_data[0].append(
                [f.leader_election_max_id for f in self.fish]
            )
            self.study_data[1].append(
                self.track_leader_election_num_events
            )

    def run(self):
        """Run the process recursively

        This method simulates the fish and calls `eval` on every clock tick as
        long as the fish `is_started`.
        """

        if not self.is_started:
            return

        time.sleep(self.clock_speed / 2)

        start_time = time.time()

        self.eval()

        time_elapsed = time.time() - start_time
        sleep_time = (self.clock_speed / 2) - time_elapsed
        time.sleep(max(0, sleep_time))

        self.run()

    def activate_reset(self):
        """Activate automatic resetting of the fish positions on a new
        instruction.
        """
        self.reset = True

    def deactivate_reset(self):
        """Deactivate automatic resetting of the fish positions on a new
        instruction.
        """
        self.reset = False

    def check_instructions(self):
        """Check external instructions to be broadcasted.

        If we reach the clock cycle in which they should be broadcasted, send
        them out.
        """
        if self.instructions.empty():
            return

        when, _, _, _, _ = self.instructions.queue[0]

        if when == self.clock:
            if self.track_info is not None:
                self.check_info_consistency()

            self.track_info = None
            self.not_saw_info = 0

            if self.reset and self.fish_pos is not None:
                self.environment.node_pos = np.copy(self.fish_pos)

            _, event, fish_id, pos, fish_all = self.instructions.get()
            if fish_id is not None:
                self.fish[fish_id].queue.put((event, pos))
            elif fish_all:
                for fish in self.fish:
                    fish.queue.put((event, pos))
            else:
                if event.opcode == START_LEADER_ELECTION:
                    for fish in self.fish:
                        fish.leader_election_max_id = -1
                        fish.last_leader_election_clock = -1

                self.channel.transmit(
                    source=self,
                    event=event,
                    pos=pos,
                    is_observer=True
                )

            if event.opcode == INFO_EXTERNAL and event.track:
                self.track_info = event.message

            if event.opcode == START_HOP_COUNT:
                if self.track_hop_count:
                    self.study_data[0].append(self.track_hop_count_num_events)
                    self.study_data[1].append(
                        self.fish[self.hop_count_source_id].hop_count
                    )

                self.track_hop_count = True
                self.track_hop_count_num_events = 0
                self.track_hop_count_started = True

            if event.opcode == START_LEADER_ELECTION:
                if self.track_leader_election:
                    self.study_data[0].append(
                       [f.leader_election_max_id for f in self.fish]
                    )
                    self.study_data[1].append(
                        self.track_leader_election_num_events
                    )

                self.track_leader_election = True
                self.track_leader_election_num_events = 0

            self.check_instructions()

    def study(self, prop):
        if prop == 'info':
            self.study_info_consistency = True
            self.study_data = [[], []]

        if prop == 'hop_count':
            self.study_hop_count = True
            self.study_data = [[], []]

        if prop == 'leader':
            self.study_leader_election = True
            self.study_data = [[], []]

    def check_info_consistency(self):
        """Check consistency of a tracked information
        """

        correct = 0
        max_hops = 0

        for fish in self.fish:
            if fish.info == self.track_info:
                correct += 1
                max_hops = max(fish.info_hops, max_hops)

        if self.study_info_consistency:
            self.study_data[0].append(correct)
            self.study_data[1].append(max_hops)

        if self.verbose:
            print(
                '{} out of {} got the message. Max hops: {} (clock {})'.format(
                    correct, len(self.fish), max_hops, self.clock
                )
            )

    def check_transmissions(self):
        """Check intercepted transmission from the channel
        """

        not_saw_info = True

        while not self.transmissions.empty():
            event = self.transmissions.get()

            if event.opcode == INFO_INTERNAL:
                not_saw_info = False

            if event.opcode == HOP_COUNT:
                if self.track_hop_count_started:
                    self.hop_count_source_id = event.source_id
                    self.track_hop_count_started = False

                self.track_hop_count_num_events += 1

            if event.opcode == LEADER_ELECTION:
                self.track_leader_election_num_events += 1

        if not_saw_info:
            self.not_saw_info += 1
        else:
            self.not_saw_info = 0

        if self.track_info is not None:
            if self.not_saw_info > 1:
                self.check_info_consistency()
                self.track_info = None
                self.not_saw_info = 0

    def eval(self):
        """Save the position and connectivity status of the fish.
        """

        self.check_transmissions()
        self.check_instructions()

        neighbor_distances = 0
        num_neighbor_pairs = 0
        total_speed = 0

        for i in range(self.num_nodes):
            self.x[i].append(self.environment.node_pos[i, 0])
            self.y[i].append(self.environment.node_pos[i, 1])

            n = len(self.fish[i].neighbors)

            if n < self.fish[i].lim_neighbors[0]:
                self.status[i].append(-1)
            elif n > self.fish[i].lim_neighbors[1]:
                self.status[i].append(1)
            else:
                self.status[i].append(0)

            neighbor_distances += sum(self.fish[i].neighbor_spacing)
            num_neighbor_pairs += len(self.fish[i].neighbor_spacing)
            if self.fish[i].speed != None:
                total_speed += self.fish[i].speed
        if num_neighbor_pairs > 0:
            self.avg_dist.append(neighbor_distances / num_neighbor_pairs)
            self.avg_speed.append(total_speed / self.num_nodes)
        self.clock += 1

    def time_plot(self,
                  dark=False,
                  white_axis=False,
                  no_legend=False,
                  show_bar_char=False,
                  no_star=False):
        """
        Create a widget that allows you to see fish movement over time
        """
        ax = plt.gca()
        fig = plt.figure(figsize=(8,3))
        base_plot = plt.scatter(
                self.object[0],
                self.object[1],
                marker=(5, 1, 0),
                facecolors='white',
                edgecolors='white',
                s=2000,
                alpha = 0.5
            )

        for i in range(self.num_nodes):
            # start mark
            plt.scatter(
                self.x[i][0],
                self.y[i][0],
                c=colors[-2],
                marker='s',
                s=100,
                alpha=0.6
            )
            # end mark
            plt.scatter(
                self.x[i][-1],
                self.y[i][-1],
                c=colors[0],
                marker='>',
                s=200,
                alpha=1
            )
        ## add slider
        slider_ax = plt.axes([0.25, 0.15, 0.65, 0.03])
        time_slider = Slider(slider_ax, 'Time', 0, len(self.x[0]), valinit=0)

        def update(time):
            xs = []
            ys = []
            time = int(time)
            for i in range(self.num_nodes):
                xs.append(self.x[i][time])
                ys.append(self.y[i][time])
            base_plot.set_ydata(ys) # set new y-coordinates of the plotted points
            base_plot.set_xdata(xs)
            fig.canvas.draw_idle()          # redraw the plot

        leg = []
        leg.append(mpatches.Patch(color=colors[-2], label='Time 0'))

        for i in range(1, len(self.status[0])):
            leg.append(mpatches.Patch(
                color=colors[i % len(colors)], label='Time {}'.format(i - 1)
            ))
        leg.append(mpatches.Patch(
                color=colors[0], label='Final State'
            ))
        if not no_legend:
            legend = plt.legend(handles=leg)

        if dark:
            ax.set_facecolor((0, 0, 0))
            ax.spines['top'].set_color('black')
            ax.spines['right'].set_color('black')

            # Axis
            if white_axis:
                ax.spines['bottom'].set_color('white')
                ax.spines['left'].set_color('white')
                ax.tick_params(axis='x', colors='white')
                ax.tick_params(axis='y', colors='white')
                ax.yaxis.label.set_color('white')
                ax.xaxis.label.set_color('white')
                ax.title.set_color('white')

            # Legend
            if not no_legend:
                legend.get_frame().set_facecolor('black')
                for text in legend.get_texts():
                    plt.setp(text, color='white')
        plt.title("alpha: {}, k_ar: {}, fish: {}".format(self.fish[0].alpha, self.fish[0].k_ar, len(self.fish)))
        time_slider.on_changed(update)
        plt.show()


    def plot(
        self,
        dark=False,
        white_axis=False,
        no_legend=False,
        show_bar_chart=False,
        no_star=False,
        show_dist_plot=False,
    ):
        """Plot the fish movement
        """
        ax = plt.gca()

        if self.is_instructed and not no_star:
            plt.scatter(
                self.object[0],
                self.object[1],
                marker=(5, 1, 0),
                facecolors='white',
                edgecolors='white',
                s=2000,
                alpha=0.5
            )

        for i in range(self.num_nodes):
            c = self.node_colors[i]
            if i != 0 and not i % 20 and dark:
                c = [1.0, 1.0, 1.0, 1.0]

            plt.plot(
                self.x[i],
                self.y[i],
                c=c,
                linewidth=2.0,
                alpha=0.4
            )

            plt.scatter(
                self.x[i],
                self.y[i],
                marker='o',
                c = c,
                s= 10,
                alpha=0.3
            )

            plt.scatter(
                self.x[i][0],
                self.y[i][0],
                c=c,
                marker='>',
                s=200,
                alpha=0.5
            )

        # draw final position on top of other elements
        for i in range(self.num_nodes):
            c = self.node_colors[i]

            plt.scatter(
                self.x[i][-1],
                self.y[i][-1],
                c=c,
                marker='s',
                s=200,
                alpha=1
            )

        leg = []
        leg.append(mpatches.Patch(
                color=colors[-2], label='Time 0'
            ))
        for i in range(1, len(self.status[0])):
            leg.append(mpatches.Patch(
                color=colors[i % len(colors)], label='Time {}'.format(i - 1)
            ))
        leg.append(mpatches.Patch(
                color=colors[0], label='Final State'
            ))
        if not no_legend:
            legend = plt.legend(handles=leg)

        if dark:
            ax.set_facecolor((0, 0, 0))
            ax.spines['top'].set_color('black')
            ax.spines['right'].set_color('black')

            # Axis
            if white_axis:
                ax.spines['bottom'].set_color('white')
                ax.spines['left'].set_color('white')
                ax.tick_params(axis='x', colors='white')
                ax.tick_params(axis='y', colors='white')
                ax.yaxis.label.set_color('white')
                ax.xaxis.label.set_color('white')
                ax.title.set_color('white')

            # Legend
            if not no_legend:
                legend.get_frame().set_facecolor('black')
                for text in legend.get_texts():
                    plt.setp(text, color='white')
        plt.title("alpha: {}, k_ar: {}, fish: {}".format(self.fish[0].alpha, self.fish[0].k_ar, len(self.fish)))

        plt.show()

        if self.study_info_consistency:
            print('Num. Fish with Correct Info:', self.study_data[0])
            print('Num. Hops:', self.study_data[1])

        if self.study_hop_count:
            print('Num. Messages:', self.study_data[0])
            print('Num. Hops:', self.study_data[1])

        if self.study_leader_election:
            print('Leader:', self.study_data[0])
            print('Num. Messages:', self.study_data[1])

        # assess dist change for aggregation / dispersion
        if (show_dist_plot):
            fig, (dist_ax, speed_ax) = plt.subplots(1,2)
            dist_ax.plot(range(len(self.avg_dist)), self.avg_dist)
            dist_ax.scatter(range(len(self.avg_dist)), self.avg_dist)
            dist_ax.set_xlabel("Time")
            dist_ax.set_ylabel("Mean neighbor spacing")
            dist_ax.set_title("Mean neighbor spacing over time")

            speed_ax.plot(range(len(self.avg_speed)), self.avg_speed)
            speed_ax.scatter(range(len(self.avg_speed)), self.avg_speed)
            speed_ax.set_xlabel("Time")
            speed_ax.set_ylabel("Average Speed")
            speed_ax.set_title("Average Swarm Speed over time")

            plt.show()

        if (
            show_bar_chart and
            (self.study_info_consistency or self.study_hop_count)
        ):
            # Consistency
            ax = plt.gca()
            plt.hist(
                self.study_data[0],
                facecolor='#ff00ff' if dark else 'black'
            )

            print(self.study_data[0], self.study_data[1])

            ax.set_xlabel('# Fish')
            ax.set_ylabel('# Trials')
            ax.set_title('Total Fish with correct information')

            if dark:
                ax.set_facecolor((0, 0, 0))
                ax.spines['top'].set_color('black')
                ax.spines['right'].set_color('black')

                # Axis
                if white_axis:
                    ax.spines['bottom'].set_color('white')
                    ax.spines['left'].set_color('white')
                    ax.tick_params(axis='x', colors='white')
                    ax.tick_params(axis='y', colors='white')
                    ax.yaxis.label.set_color('white')
                    ax.xaxis.label.set_color('white')
                    ax.title.set_color('white')

            plt.show()

            # Max Hops
            ax = plt.gca()
            plt.hist(
                self.study_data[1],
                facecolor='#eeff41' if dark else 'black'
            )

            ax.set_xlabel('# Hops')
            ax.set_ylabel('# Trials')
            ax.set_title('Number of hops')

            if dark:
                ax.set_facecolor((0, 0, 0))
                ax.spines['top'].set_color('black')
                ax.spines['right'].set_color('black')

                # Axis
                if white_axis:
                    ax.spines['bottom'].set_color('white')
                    ax.spines['left'].set_color('white')
                    ax.tick_params(axis='x', colors='white')
                    ax.tick_params(axis='y', colors='white')
                    ax.yaxis.label.set_color('white')
                    ax.xaxis.label.set_color('white')
                    ax.title.set_color('white')

            plt.show()
