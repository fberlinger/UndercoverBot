import numpy as np
import random

from eventcodes import INFO_INTERNAL


class Channel():
    """Underwater wireless communication channel

    This class models the underwater communication between fish instances and
    connects fish to the environmental network.
    """

    def __init__(self, environment, verbose=False):
        """Initialize the channel

        Arguments:
            environment {Environment} -- Instance of the environment
        """

        self.environment = environment
        self.verbose = verbose
        self.observer = None

    def set_nodes(self, nodes):
        """This method just stores a references to all nodes

        Arguments:
            nodes {list} -- List of node instances
        """
        self.nodes = nodes

    def intercept(self, observer):
        """Let an observer intercept all messages.

        It's really unfortunate but there are not just holes in Swiss cheese.
        Our channel is no exception and a god-like observer is able to listen
        to all transmitted messages in the name of research. Please don't tell
        anyone.

        Arguments:
            observer {Observer} -- The all mighty observer
        """
        self.observer = observer

    def transmit(self, source, event, pos=np.zeros((2,)), is_observer=False):
        """Transmit a broadcasted event to node instances

        This method gets the probability of connectedness between two nodes
        from the environment and adds the events on the node instances given
        that probability.

        Arguments:
            source {*} -- Node instance
            event {Event} -- Some event to be broadcasted
        """

        if self.observer:
            self.observer.transmissions.put(event)

        for target in self.nodes:
            if source == target:
                # Sorry no monologs
                continue

            source_log = ''

            if is_observer:
                dist = np.sqrt(
                    np.sum(
                        (self.environment.node_pos[target.id] - pos) ** 2
                    )
                )
                prob = self.environment.prob_dist(dist)
                source_log = 'observer'
            else:
                prob = self.environment.prob(source.id, target.id)
                source_log = source.id

            success = random.random() <= prob

            if success:
                target.queue.put((event, pos))

            if self.verbose:
                print(
                    'Channel: transmitted event from {} to {}: {} '
                    '(prob: {:0.2f})'.format(
                        source_log, target.id, success, prob
                    )
                )
