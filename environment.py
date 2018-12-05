import math
import numpy as np

from scipy.spatial.distance import cdist


class Environment():
    """The dynamic network of robot nodes in the underwater environment

    This class keeps track of the network dynamics by storing the positions of
    all nodes. It contains functions to derive the distorted position from a
    target position by adding a distortion and noise, to update the position of
    a node, to update the distance between nodes, to derive the probability of
    receiving a message from a node based on that distance, and to get the
    relative position from one node to another node.
    """

    def __init__(
        self,
        node_pos,
        distortion,
        prob_type='quadratic',
        conn_thres=math.inf,
        conn_drop=1,
        noise_magnitude=0.1,
        verbose=False,
        see_neighbor=math.inf
    ):
        """Create a new environment for the fish

        Arguments:
            node_pos {np.array} -- Initial positions of all nodes.
            distortion {np.array} -- Static distortion model, e.g. pull/push
                to/from origin. Describes velocity vectors that act on each
                position.

        Keyword Arguments:
            prob_type {str} -- Type of probability for receiving a message;
                binary based on distance threshold, or quadratic decay with
                distance. (default: {'quadratic'})
            conn_thres {float} -- Threshold at which a connection between 2
                nodes brakes (for binary or quadratic) or where the probability
                of connectedness is 0.5 for sigmoid. (default: {math.inf})
            conn_drop {number} -- Determines the drop of connectivity for
                sigmoid (default: {1})
            noise_magnitude {number} -- Magnitude of noise that is added in
                each simulation round to each target position  of a node.
                (default: {0.1})
            verbose {bool} -- If `true` print some information during
                simulation (default: {False})
            see_neighbor {float} -- Tracks number of neighbors in given distance,
                used for Turing Learning
        """
        # Params
        self.node_pos = node_pos
        self.distortion = distortion
        self.conn_thres = conn_thres
        self.conn_drop = conn_drop
        self.noise_magnitude = noise_magnitude
        self.verbose = verbose
        self.prob_type = prob_type
        self.record_neighbor = see_neighbor

        # Init
        self.num_nodes = node_pos.size
        self.update_distance()

    def get_distorted_pos(self, source_index, target_pos):
        """Calculate the distorted target position of a node.

        This method adds random noise and the position-based distortion onto
        the ideal target position to calculate the final position of the node.

        Arguments:
            source_index {int} -- Index of the source node which position is to
                be distorted.
            target_pos {np.array} -- Ideal target position to be distorted

        Returns:
            np.array -- Final position of the node.
        """
        # Get indices for the distortion vector field
        indices = target_pos.astype(int)

        # Simulate random noise in [-1,1]
        noise = (
            np.random.rand(2,) * 2 - np.ones((2,))
        ) * self.noise_magnitude

        return target_pos + noise #self.distortion[indices[0], indices[1]] + noise

    def set_pos(self, source_index, new_pos):
        """Set the new position

        Save the new position into the positions array.

        Arguments:
            source_index {int} -- Index of the node position to be set
            new_pos {np.array} -- New node position ([x, y]) to be set.
        """

        self.node_pos[source_index] = new_pos

        self.update_distance()

        if self.verbose:
            print('Env: {} is now at {}'.format(
                source_index, new_pos, self.node_pos
            ))

    def update_distance(self):
        """Calculate pairwise distances of every node

        Calculate and saves the pairwise distance of every node.
        """
        self.node_dist = cdist(self.node_pos, self.node_pos)

    def prob(self, node_a_index, node_b_index):
        """Calculate the probability of connectivity of two points based on
        their Eucledian distance.

        Arguments:
            node_a_index {int} -- Node A index
            node_b_index {int} -- Node B index

        Returns:
            float -- probability of connectivity
        """
        distance = self.node_dist[node_a_index, node_b_index]
        return self.prob_dist(distance)

    def prob_dist(self, distance):
        """Calls the approriate probability functions

        The returned probability depends on prob_type

        Arguments:
            distance {float} -- Eucledian distance

        Returns:
            float -- probability of connectivity
        """
        if self.prob_type == 'quadratic':
            return self.prob_quadratic(distance)
        if self.prob_type == 'sigmoid':
            return self.prob_sigmoid(distance)

        # Binary connectivity by default
        return self.prob_binary(distance)

    def prob_binary(self, distance):
        """Simulate binary connectivity probability

        This function either returns 1 or 0 if the distance of two nodes is
        smaller (or larger) than the user defined threshold.

        Arguments:
            distance {float} -- Eucledian distance

        Returns:
            float -- probability of connectivity. The probability is either 1
                or 0 depending on the distance threshold.
        """
        if distance > self.conn_thres:
            return 0

        return 1

    def neighbor_distance(self, node_a_index, node_b_index):
        # get distance of a given neighbor for all neighbors
        distance = self.node_dist[node_a_index, node_b_index]

        return distance

    def prob_quadratic(self, distance):
        """Simulate quadradic connectivity probability

        Arguments:
            distance {float} -- Eucledian distance

        Returns:
            float -- probability of connectivity as a function of the distance.
                The probability drops quadratically.
        """

        if distance > self.conn_thres:
            return 0

        return max(self.conn_thres, (distance + 1)**-2)

        return 1 / (math.exp(distance) + 1)

    def prob_sigmoid(self, distance):
        """Simulate sigmoid connectivity probability

        Arguments:
            distance {float} -- Eucledian distance

        Returns:
            float -- probability of connectivity as a sigmoid function of the
                distance.
        """

        return 1 / (1 + np.exp(self.conn_drop * (distance - self.conn_thres)))

    def get_rel_pos(self, source_index, target_index):
        """Calculate the relative position of two nodes

        Calculate the vector pointing from the source node to the target node.

        Arguments:
            source_index {int} -- Index of the source node, i.e., the node for
                which the relative position to target is specified.
            target_index {int} -- Index of the target node, i.e., the node to
                which source is relatively positioned to.

        Returns:
            np,array -- Vector pointing from source to target
        """
        return self.node_pos[target_index] - self.node_pos[source_index]
