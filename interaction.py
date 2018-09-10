import numpy as np
import random


class Interaction():
    """Underwater interactions

    This class models interactions of the fish with their environment, e.g.,
    to perceive other fish or to change their position.
    """

    def __init__(self, environment, verbose=False):
        """Constructor

        Initializes the channel

        Arguments:
            nodes {list} -- List of fish instances
        """

        self.environment = environment
        self.verbose = verbose

    def perceive_object(self, source_id, pos):
        """Perceive the relative position to an object

        This simulates the fish's perception of external sources and targets.

        Arguments:
            source_id {int} -- Index of the fish that wants to know its
                location
            pos {np.array} -- X and Y position of the object
        """

        return pos - self.environment.node_pos[source_id]

    def perceive_pos(self, source_id, target_id):
        """Perceive the relative position to another fish

        This simulates the fish's perception of neighbors.

        Arguments:
            source_id {int} -- Index of the fish to be perceived
            target_id {int} -- Index of the fish to be perceived
        """

        if source_id == target_id:
            # You are very close to yourself!
            return np.zeros((2,))

        prob = self.environment.prob(source_id, target_id)

        success = random.random() <= prob

        if self.verbose:
            print('Interaction: {} perceived {}: {} (prob: {:0.2f})'.format(
                source_id, target_id, success, prob
            ))

        if success:
            return self.environment.get_rel_pos(source_id, target_id)
        else:
            return np.zeros((2,))

    def move(self, source_id, target_direction):
        """Move a fish

        Moves the fish relatively into the given direction and adds
        target-based distortion to the fish position.

        Arguments:
            source_id {int} -- Fish identifier
            target_direction {np.array} -- Relative direction to move to
        """
        node_pos = self.environment.node_pos[source_id]
        target_pos = node_pos + target_direction
        final_pos = self.environment.get_distorted_pos(source_id, target_pos)

        self.environment.set_pos(source_id, final_pos)

        if self.verbose:
            print('Interaction: {} moved to {}'.format(
                source_id, final_pos
            ))
