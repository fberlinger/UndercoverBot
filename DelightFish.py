import math
import numpy as np
from queue import Queue
import time
import datetime

from events import HopCount, Ping, InfoInternal, LeaderElection
from eventcodes import (
    PING, HOMING, HOP_COUNT, INFO_EXTERNAL, INFO_INTERNAL, START_HOP_COUNT,
    START_LEADER_ELECTION, LEADER_ELECTION, MOVE
)


class Fish():
    """This class models each fish robot node in the network from the fish'
    perspective. The fish's behavior is based on the model created by
    Delight et al in "Developing Robotic Swarms for Ocean Surface Mapping"

    Each fish has an ID, communicates over the channel, and perceives its
    neighbors and takes actions accordingly. In taking actions, the fish can
    weight information from neighbors based on their distance. The fish aims to
    stay between a lower and upper limit of neighbors to maintain a cohesive
    collective. It can moves at a maximal speed and updates its behavior on
    every clock tick.
    """

    def __init__(
        self,
        id,
        channel,
        interaction,
        k_coh,
        k_ar,
        alpha,
        lim_neighbors=[0, math.inf],
        fish_max_speed=9,
        clock_freq=1,
        neighbor_weight=1.0,
        name='Unnamed',
        verbose=False,
        cohesion=False
    ):
        """Create a new fish

        Arguments:
            id {int} -- UUID.
            channel {class} -- Communication channel.
            interaction {class} -- Interactions which include perception of
                neighbors and own movement.
            k_coh {number} -- weight given to cohesion behavior
            k_ar {number} -- weight given to attraction-repulsion behavior
            alpha {number} -- desired distance between fish

        Keyword Arguments:
            lim_neighbors {int, int} -- Lower and upper limit of neighbors each
                fish aims to be connected to.
                (default: {0, math.inf})
            fish_max_speed {number} -- Max speed of each fish. Defines by how
                much it can change its position in one simulation step.
                (default: {1})
            clock_freq {number} -- Behavior update rate in Hertz (default: {1})
            neighbor_weight {number} -- A weight based on distance that defines
                how much each of a fish's neighbor affects its next move.
                (default: {1.0})
            name {str} -- Unique name of the fish. (default: {'Unnamed'})
            verbose {bool} -- If `true` log out some stuff (default: {False})
        """

        self.id = id
        self.channel = channel
        self.interaction = interaction
        self.neighbor_weight = neighbor_weight
        self.lim_neighbors = lim_neighbors
        self.fish_max_speed = fish_max_speed
        self.clock_freq = clock_freq
        self.name = name
        self.verbose = verbose
        self.k_coh = k_coh
        self.k_ar = k_ar
        self.alpha = alpha

        self.clock_speed = 1 / self.clock_freq
        self.clock = 0
        self.queue = Queue()
        self.target_pos = np.zeros((2,))
        self.is_started = False
        self.neighbors = set()

        self.status = None

        self.info = None  # Some information
        self.info_clock = 0  # Time stamp of the information, i.e., the clock
        self.info_hops = 0  # Number of hops until the information arrived
        self.last_hop_count_clock = -math.inf
        self.hop_count = 0
        self.hop_distance = 0
        self.hop_count_initiator = False
        self.initial_hop_count_clock = 0
        self.neighbor_spacing = [] # track neighbor distances
        self.orientation = 0 # track orientation, which is same as velocity
        self.speed = None

        self.leader_election_max_id = -1
        self.last_leader_election_clock = -1

        now = datetime.datetime.now()

        # Stores messages to be send out at the end of the clock cycle
        self.messages = []

        # Logger instance
        # with open('{}_{}.log'.format(self.name, self.id), 'w') as f:
        #     f.truncate()
        #     f.write('TIME  ::  #NEIGHBORS  ::  INFO  ::  ({})\n'.format(
        #         datetime.datetime.now())
        #     )

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

    def log(self, neighbors=set()):
        """Log current state
        """

        with open('{}_{}.log'.format(self.name, self.id), 'a+') as f:
            f.write(
                '{:05}    {:04}    {}    {}\n'.format(
                    self.clock,
                    len(neighbors),
                    self.info,
                    self.info_hops
                )
            )

    def run(self):
        """Run the process recursively

        This method simulates the fish and calls `eval` on every clock tick as
        long as the fish `is_started`.
        """

        while  self.is_started:
            start_time = time.time()
            self.eval()
            time_elapsed = time.time() - start_time

            sleep_time = (self.clock_speed / 2) - time_elapsed

            # print(time_elapsed, sleep_time, self.clock_speed / 2)
            time.sleep(max(0, sleep_time))
            if sleep_time < 0 and self.verbose:
                print('Warning frequency too high or computer too slow')

            start_time = time.time()
            self.communicate()
            time_elapsed = time.time() - start_time

            sleep_time = (self.clock_speed / 2) - time_elapsed
            time.sleep(max(0, sleep_time))
            if sleep_time < 0 and self.verbose:
                print('Warning frequency too high or computer too slow')

    def move_handler(self, event):
        """Handle move events, i.e., update the target position.

        Arguments:
            event {Move} -- Event holding an x and y target position
        """
        self.target_pos[0] = event.x
        self.target_pos[1] = event.y

    def ping_handler(self, neighbors, rel_pos, event):
        """Handle ping events

        Adds the

        Arguments:
            neighbors {set} -- Set of active neighbors, i.e., nodes from which
                this fish received a ping event.
            rel_pos {dict} -- Dictionary of relative positions from this fish
                to the source of the ping event.
            event {Ping} -- The ping event instance
        """
        neighbors.add(event.source_id)

        # When the other fish is not perceived its relative position is [0,0]
        rel_pos[event.source_id] = self.interaction.perceive_pos(
            self.id, event.source_id
        )

        if self.verbose:
            print('Fish #{}: saw friend #{} at {}'.format(
                self.id, event.source_id, rel_pos[event.source_id]
            ))

    def homing_handler(self, event, pos):
        """Homing handler, i.e., make fish aggregated extremely

        Arguments:
            event {Homing} -- Homing event
            pos {np.array} -- Position of the homing event initialtor
        """
        self.info = 'signal_aircraft'  # Very bad practice. Needs to be fixed!
        self.info_clock = self.clock

        self.messages.append(
            (self, InfoInternal(self.id, self.clock, self.info))
        )

        # update behavior based on external event
        self.status = 'wait'
        self.target_pos = self.interaction.perceive_object(self.id, pos)

        if self.verbose:
            print('Fish #{} got external info {}'.format(
                self.id, event.message
            ))

    def info_ext_handler(self, event):
        """External information handler

        Always accept the external information and spread the news.

        Arguments:
            event {InfoExternal} -- InfoExternal event
        """
        self.info = event.message
        self.info_clock = self.clock

        self.messages.append(
            (self, InfoInternal(self.id, self.clock, self.info))
        )

        if self.verbose:
            print('Fish #{} got external info {}'.format(
                self.id, event.message
            ))

    def info_int_handler(self, event):
        """Internal information event handler.

        Only accept the information of the clock is higher than from the last
        information

        Arguments:
            event {InfoInternal} -- Internal information event instance
        """
        if self.info_clock >= event.clock:
            return

        self.info = event.message
        self.info_clock = event.clock
        self.info_hops = event.hops + 1

        self.messages.append((
            self,
            InfoInternal(self.id, self.info_clock, self.info, self.info_hops)
        ))

        if self.verbose:
            print('Fish #{} got info: {} from #{}'.format(
                self.id, event.message, event.source_id
            ))



    def weight_neighbor(self, rel_pos_to_neighbor):
        """Weight neighbors by the relative position to them

        Currently only returns a static value but this could be tweaked in the
        future to calculate a weighted center point.

        Arguments:
            rel_pos_to_neighbor {np.array} -- Relative position to a neighbor

        Returns:
            float -- Weight for this neighbor
        """
        return self.neighbor_weight

    def start_leader_election_handler(self, event):
        """Leader election start handler

        Always accept a new start event for a leader election

        Arguments:
            event {StartLeaderElection} -- Leader election start event
        """
        self.last_leader_election_clock = self.clock
        self.leader_election_max_id = self.id

        self.messages.append((
            self,
            LeaderElection(self.id, self.id)
        ))

    def comp_center(self, rel_pos):
        """Compute the (potentially weighted) centroid of the fish neighbors

        Arguments:
            rel_pos {dict} -- Dictionary of relative positions to the
                neighboring fish.

        Returns:
            np.array -- 2D centroid
        """
        center = np.zeros((2,))
        n = max(1, len(rel_pos))

        for key, value in rel_pos.items():
            weight = self.weight_neighbor(value)
            center += value * weight

        center /= n

        if self.verbose:
            print('Fish #{}: swarm centroid {}'.format(self.id, center))

        return center

    def attraction_repulsion(self, rel_pos_to_neighbor):
        """ Compute the attraction-repulsion component of neighbor influnce on velocity

        Arguments:
            rel_pos_to_neighbor {np.array} -- Relative position to a neighbor

        Returns:
            np.array -- neighbor's influence on velocity
        """
        dist_neighbor = max(0.00001, np.linalg.norm(rel_pos_to_neighbor))
        self.neighbor_spacing.append(np.abs(dist_neighbor))
        diff_alpha = dist_neighbor - self.alpha
        if self.verbose:
            print("fish #{}, rel_pos_to_neighbor {}, dist_neighbor {}, diff_alpha {}".format(self.id, rel_pos_to_neighbor, dist_neighbor, diff_alpha))
        return self.k_ar * np.sign(diff_alpha) * np.power(diff_alpha, 2) * rel_pos_to_neighbor / dist_neighbor


    def move(self, neighbors, rel_pos):
        """Make a attraction repulsion move

        The move is determined by the atraction-repulsion component of
        velocity change limited by the maximum fish speed.

        Arguments:
            neighbors {set} -- Set of active neighbors, i.e., other fish that
                responded to the most recent ping event.
            rel_pos {dict} -- Relative positions to all neighbors

        Returns:
            np.array -- Move direction as a 2D vector
        """
        n = len(neighbors)
        # Get the centroid of the swarm
        new_velocity = np.zeros((2,))

        # reset neighbors:
        self.neighbor_spacing = []
        for neighbor in neighbors:
            attract = self.attraction_repulsion(rel_pos[neighbor])
            if self.verbose:
                print("fish #{}, attract result = {}".format(self.id, attract))
            new_velocity += attract
        if self.verbose:
            print("fish #{}, new velocity after neighbors {}".format(self.id, new_velocity))
        # Cap the length of the move
        magnitude = np.linalg.norm(new_velocity)
        if magnitude > 0:
            direction = new_velocity / magnitude
            final_move = direction * min(magnitude, self.fish_max_speed)
        else:
            final_move = new_velocity
        if self.verbose:
            print('Fish #{}: move to {}'.format(self.id, final_move))

        self.speed = np.linalg.norm(final_move)

        # set orientation in direction of velocity
        if (final_move[0] == 0):
            self.orientation = (np.pi / 2) * np.sign(final_move[1])
        else:
            self.orientation = np.arctan(final_move[1] / final_move[0])

        return final_move

    def update_behavior(self):
        """Update the fish behavior.

        This actively changes the cohesion strategy to either 'wait', i.e, do
        not care about any neighbors or 'signal_aircraft', i.e., aggregate with
        as many fish friends as possible.

        In robotics 'signal_aircraft' is a secret key word for robo-fish-nerds
        to gather in a secret lab until some robo fish finds a robo aircraft.
        """
        if self.status == 'wait':
            self.lim_neighbors = [0, math.inf]
        elif self.info == 'signal_aircraft':
            self.lim_neighbors = [math.inf, math.inf]

    def eval(self):
        """The fish evaluates its state

        Currently the fish checks all responses to previous pings and evaluates
        its relative position to all neighbors. Neighbors are other fish that
        received the ping element.
        """

        # Set of neighbors at this point. Will be reconstructed every time
        neighbors = set()
        rel_pos = {}

        self.saw_hop_count = False

        while not self.queue.empty():
            (event, pos) = self.queue.get()

            if event.opcode == PING:
                self.ping_handler(neighbors, rel_pos, event)

            if event.opcode == HOMING:
                self.homing_handler(event, pos)

            if event.opcode == START_HOP_COUNT:
                self.start_hop_count_handler(event)

            if event.opcode == HOP_COUNT:
                self.hop_count_handler(event)

            if event.opcode == INFO_EXTERNAL:
                self.info_ext_handler(event)

            if event.opcode == INFO_INTERNAL:
                self.info_int_handler(event)

            if event.opcode == START_LEADER_ELECTION:
                self.start_leader_election_handler(event)

            if event.opcode == LEADER_ELECTION:
                self.leader_election_handler(event)

            if event.opcode == MOVE:
                self.move_handler(event)

        if self.clock > 1:
            # Move around (or just stay where you are)
            self.interaction.move(self.id, self.move(neighbors, rel_pos))

        # Update behavior based on status and information - update behavior
        self.update_behavior()

        self.neighbors = neighbors

        # self.log(neighbors)
        self.clock += 1

    def communicate(self):
        """Broadcast all collected event messages.

        This method is called as part of the second clock cycle.
        """
        for message in self.messages:
            self.channel.transmit(*message)

        self.messages = []

        # Always send out a ping to other fish
        self.channel.transmit(self, Ping(self.id))