import math
import numpy as np
from queue import Queue
import time
import datetime

#from bob.learn.mlp import Machine
#from bob.learn.activation import Logistic

from events import HopCount, Ping, InfoInternal, LeaderElection
from eventcodes import (
    PING, HOMING, HOP_COUNT, INFO_EXTERNAL, INFO_INTERNAL, START_HOP_COUNT,
    START_LEADER_ELECTION, LEADER_ELECTION, MOVE
)


class ReplicaFish():
    """This class models each fish robot node in the network from the fish'
    perspective.

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
        weights,
        fish_max_speed=9,
        clock_freq=1,
        name='Unnamed',
        verbose=False
    ):
        """Create a new fish

        Arguments:
            id {int} -- UUID.
            channel {class} -- Communication channel.
            interaction {class} -- Interactions which include perception of
                neighbors and own movement.
            weights {array} -- Weights for mlp governming neighbor weight

        Keyword Arguments:
            fish_max_speed {number} -- Max speed of each fish. Defines by how
                much it can change its position in one simulation step.
                (default: {1})
            clock_freq {number} -- Behavior update rate in Hertz (default: {1})
            name {str} -- Unique name of the fish. (default: {'Unnamed'})
            verbose {bool} -- If `true` log out some stuff (default: {False})
        """

        self.id = id
        self.channel = channel
        self.interaction = interaction
        self.fish_max_speed = fish_max_speed
        self.clock_freq = clock_freq
        self.name = name
        self.verbose = verbose

        self.clock_speed = 1 / self.clock_freq
        self.clock = 0
        self.queue = Queue()
        self.is_started = False
        self.neighbors = set()
        self.neighbor_spacing = [] # track neighbor distances
        self.orientation = 0
        self.speed = None

        now = datetime.datetime.now()

        # model with 2 input neurons, 1 hidden layer with 3 neurons, and 3 outputs
        # The input is the relative position of the neighbor
        # the output is the velocity direction (x, y) and speed
        # We appropriately scale output
        # Both hidden and output layer have bias
        # activation function is the

        self.input_to_hidden = np.reshape(weights[:6], (2, 3))
        self.bias_hidden = np.reshape(weights[6:9], (1, 3))
        self.hidden_to_output = np.reshape(weights[9:18], (3, 3))
        self.bias_output = np.reshape(weights[18:21], (1, 3))


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
                '{:05}    {:04}\n'.format(
                    self.clock,
                    len(neighbors)
                )
            )

    def run(self):
        """Run the process recursively

        This method simulates the fish and calls `eval` on every clock tick as
        long as the fish `is_started`.
        """

        while self.is_started:

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


    # def move_handler(self, event):
    #     """Handle move events, i.e., update the target position.

    #     Arguments:
    #         event {Move} -- Event holding an x and y target position
    #     """
    #     self.target_pos[0] = event.x
    #     self.target_pos[1] = event.y

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


    def logistic_sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def weight_neighbor(self, rel_pos_to_neighbor):
        """Weight neighbors by the relative position to them

        Uses multi-layer perceptron with weights learned over time via evolutionary strategy

        Arguments:
            rel_pos_to_neighbor {np.array} -- Relative position to a neighbor

        Returns:
            float -- Weight for this neighbor
        """
        #print(rel_pos_to_neighbor)
        hidden = np.dot(rel_pos_to_neighbor, self.input_to_hidden) + \
                    self.bias_hidden
        hidden = self.logistic_sigmoid(hidden)


        # run output layer - only need to do this with final input, as we discard previous outputs
        # figure out how to scale appropriately
        output = self.logistic_sigmoid(np.dot(hidden, self.hidden_to_output) + self.bias_output)
        #print(output)
        output = np.reshape(output, (3, ))
        output += np.array([-0.5, -0.5, 0])
        direction = output[:2]
        direction_magnitude = np.linalg.norm(direction)
        if direction_magnitude > 0:
            # scale speed output by fish max speed
            final_velocity_vector = (direction / direction_magnitude) * output[2] * self.fish_max_speed

        else:
            final_velocity_vector = output[0:2]
        #print(final_velocity_vector)
        return np.reshape(final_velocity_vector, (2,))


    def move(self, neighbors, rel_pos):
        """Make a cohesion and target-driven move

        The move is determined by the relative position of the centroid and a
        target position and is limited by the maximum fish speed.
        This move is random, but limited by maximum fish speed

        Arguments:
            neighbors {set} -- Set of active neighbors, i.e., other fish that
                responded to the most recent ping event.
            rel_pos {dict} -- Relative positions to all neighbors

        Returns:
            np.array -- Move direction as a 2D vector
        """

        n = len(neighbors)
        # Get the centroid of the swarm
        new_velocity = np.zeros(2,)

        # reset neighbors:
        self.neighbor_spacing = []
        for neighbor in neighbors:
            neighbor_vector = self.weight_neighbor(rel_pos[neighbor])
            if self.verbose:
                print("fish #{}, attract result = {}".format(self.id, attract))
            new_velocity += neighbor_vector
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
        #print(self.orientation)
        #print(final_move)

        return final_move


    def eval(self):
        """The fish evaluates its state

        Currently the fish checks all responses to previous pings and evaluates
        its relative position to all neighbors. Neighbors are other fish that
        received the ping element.
        """

        # Set of neighbors at this point. Will be reconstructed every time
        neighbors = set()
        rel_pos = {}


        while not self.queue.empty():
            (event, pos) = self.queue.get()

            if event.opcode == PING:
                self.ping_handler(neighbors, rel_pos, event)


        if self.clock > 1:
            # Move around (or just stay where you are)
            self.interaction.move(self.id, self.move(neighbors, rel_pos))

        # Update behavior based on status and information - update behavior
        self.neighbors = neighbors

        # self.log(neighbors)
        self.clock += 1

    def communicate(self):
        """Broadcast all collected event messages.

        This method is called as part of the second clock cycle.
        """
        # Always send out a ping to other fish
        self.channel.transmit(self, Ping(self.id))

